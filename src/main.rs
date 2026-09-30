// Release builds link as a GUI app so double-clicking offgrid.exe does not
// open a console window behind the UI. Debug builds keep the console
// subsystem, so `cargo run` still prints without any of the plumbing below.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod agent;
mod app;
mod bridge;
mod config;
mod hardware;
mod hub;
#[cfg(feature = "images")]
mod imagegen;
mod llm;
mod models;
mod server;
mod session;
mod theme;
mod tui;
mod webchat;

/// Terminate the process immediately, skipping C/C++ static destructors.
///
/// llama.cpp / ggml register global teardown that libc's `exit()` runs via
/// `__cxa_finalize`. On macOS the ggml-metal backend's static destructor
/// aborts (SIGABRT) while a ggml worker thread is still alive, so a perfectly
/// clean session crashes on the way out, *after* the window has closed and all
/// our real work is done. `_exit()` ends the process without running any of
/// that, which is exactly what we want at the very end of `main`. Anything that
/// needs flushing (stdout/stderr) is flushed here first.
pub fn hard_exit(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    unsafe {
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// Reattach to the terminal that launched us (Windows only).
///
/// A GUI-subsystem binary starts with no console at all, so `--tui`,
/// `--smoke` and friends would print into the void when run from cmd or
/// PowerShell. `AttachConsole(ATTACH_PARENT_PROCESS)` borrows the parent's
/// console when there is one and fails harmlessly when there isn't (launched
/// from Explorer) — which is exactly the case where we want no window.
///
/// The attach itself does not necessarily fill in the process' standard
/// handles, so we point any unset handle at the console device. Handles the
/// parent already gave us are left alone: that is how `offgrid --smoke > log`
/// keeps redirecting to the file.
#[cfg(windows)]
fn attach_parent_console() {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

    unsafe extern "system" {
        fn AttachConsole(process_id: u32) -> i32;
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn SetStdHandle(which: u32, handle: *mut c_void) -> i32;
    }

    if unsafe { AttachConsole(ATTACH_PARENT_PROCESS) } == 0 {
        return; // no parent console — stay quiet, as a GUI app should
    }

    let bind = |which: u32, device: &str, read: bool| {
        let existing = unsafe { GetStdHandle(which) };
        if !existing.is_null() && existing != INVALID_HANDLE_VALUE {
            return; // inherited or redirected by the caller; don't clobber it
        }
        if let Ok(file) = std::fs::OpenOptions::new()
            .read(read)
            .write(!read)
            .open(device)
        {
            unsafe { SetStdHandle(which, file.as_raw_handle() as *mut c_void) };
            // The handle now belongs to the process' stdio; closing the File
            // at the end of this scope would close it out from under us.
            std::mem::forget(file);
        }
    };
    bind(STD_INPUT_HANDLE, "CONIN$", true);
    bind(STD_OUTPUT_HANDLE, "CONOUT$", false);
    bind(STD_ERROR_HANDLE, "CONOUT$", false);
}

fn main() -> eframe::Result {
    #[cfg(windows)]
    attach_parent_console();

    // Headless check of the core plumbing (download → load → generate → serve).
    if std::env::args().any(|a| a == "--smoke") {
        smoke(false);
        hard_exit(0);
    }
    // Diagnostic: run the web-chat router turn once and print what the model
    // emits, so we can see whether it produces a parseable tool call.
    // Usage: offgrid --web-probe <model-substring> <question…>
    if let Some(i) = std::env::args().position(|a| a == "--web-probe") {
        let args: Vec<String> = std::env::args().collect();
        let model_match = args.get(i + 1).cloned().unwrap_or_default();
        let question = args[i + 2..].join(" ");
        web_probe(&model_match, &question);
        hard_exit(0);
    }
    // Image generation without the GUI: fetches the weights if needed, runs the
    // pipeline, writes a PNG next to the working directory and prints what each
    // stage cost. Usage: offgrid --image-probe <prompt…>   (steps: IMAGE_STEPS)
    #[cfg(feature = "images")]
    if let Some(i) = std::env::args().position(|a| a == "--image-probe") {
        let args: Vec<String> = std::env::args().collect();
        image_probe(&args[i + 1..].join(" "));
        hard_exit(0);
    }
    // Same, but exercises the coding-agent loop instead of serving.
    // Headless: no display, or asked for explicitly.
    let headless = cfg!(target_os = "linux")
        && std::env::var_os("DISPLAY").is_none()
        && std::env::var_os("WAYLAND_DISPLAY").is_none();
    if std::env::args().any(|a| a == "--tui") || headless {
        if let Err(e) = tui::run() {
            eprintln!("tui: {e}");
            hard_exit(1);
        }
        hard_exit(0);
    }

    if std::env::args().any(|a| a == "--smoke-agent") {
        smoke(true);
        hard_exit(0);
    }
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icons/Alert_Idea.png"))
        .unwrap_or_default();
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 650.0])
            .with_title("offgrid")
            .with_icon(icon),
        ..Default::default()
    };
    let result = eframe::run_native(
        "offgrid",
        options,
        Box::new(|cc| Ok(Box::new(app::OffgridApp::new(cc)))),
    );
    if let Err(e) = &result {
        eprintln!("offgrid: {e}");
    }
    // Bypass ggml/llama.cpp static destructors, which abort on exit (macOS).
    hard_exit(if result.is_ok() { 0 } else { 1 });
}

/// Load a model whose filename contains `model_match` and run exactly one
/// web-chat router turn for `question`, printing the raw model output and
/// whether `parse_tool_call` found a call. This is the pre-pass that decides
/// whether chat searches the web — if the model does not emit a tool call
/// here, chat silently answers from memory.
fn web_probe(model_match: &str, question: &str) {
    let dir = config::models_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no models dir at {}", dir.display());
        return;
    };
    let path = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.to_lowercase().contains(&model_match.to_lowercase()))
        });
    let Some(path) = path else {
        eprintln!("no .gguf in {} matching {model_match:?}", dir.display());
        return;
    };
    println!("model: {}", path.display());
    println!("question: {question}\n");

    // Keep KV small so big models don't thrash while we probe.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let handle = llm::spawn_worker(hardware::HardwareProfile::detect().physical_cores);
    handle
        .cmd_tx
        .send(llm::LlmCmd::Load {
            path,
            n_ctx: llm::DEFAULT_N_CTX,
        })
        .unwrap();
    loop {
        match handle.event_rx.recv().unwrap() {
            llm::LlmEvent::Loaded(n) => {
                println!("loaded: {n}\n");
                break;
            }
            llm::LlmEvent::Error(e) => {
                eprintln!("load failed: {e}");
                return;
            }
            _ => {}
        }
    }

    // Drive the real web-chat path end to end, exactly as the TUI/GUI do.
    let conversation = vec![llm::ChatMessage {
        role: llm::Role::User,
        content: question.to_string(),
    }];
    let (tx, rx) = std::sync::mpsc::channel();
    webchat::spawn(
        conversation,
        handle.cmd_tx.clone(),
        tx,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        0.7,
        4096,
    );
    let mut searched = false;
    let mut answer = String::new();
    for event in rx {
        match event {
            llm::LlmEvent::Info(note) => {
                searched = true;
                println!("[info] {note}");
            }
            llm::LlmEvent::Token(t) => answer.push_str(&t),
            llm::LlmEvent::GenDone => break,
            llm::LlmEvent::Error(e) => {
                eprintln!("error: {e}");
                break;
            }
            _ => {}
        }
    }
    println!("\n--- answer ---\n{answer}\n--- end ---");
    println!(
        "\nRESULT: {}",
        if searched {
            "searched the web ✓"
        } else {
            "did NOT search — answered from memory"
        }
    );
}

/// Headless image generation, for measuring what the candle prototype costs on
/// a given machine without a display in the way.
#[cfg(feature = "images")]
fn image_probe(prompt: &str) {
    let prompt = if prompt.is_empty() {
        "a rusty robot walking on a sandy beach"
    } else {
        prompt
    };
    // IMAGE_MODEL indexes imagegen::MODELS; steps default to what that model
    // was distilled for.
    let model: usize = std::env::var("IMAGE_MODEL")
        .ok()
        .and_then(|m| m.parse().ok())
        .filter(|m: &usize| *m < imagegen::MODELS.len())
        .unwrap_or(0);
    let steps: usize = std::env::var("IMAGE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(imagegen::MODELS[model].steps);
    // IMAGE_SIZE=768 or IMAGE_SIZE=512x768.
    let (width, height) = std::env::var("IMAGE_SIZE")
        .ok()
        .and_then(|spec| {
            let (w, h) = spec
                .split_once('x')
                .unwrap_or((spec.as_str(), spec.as_str()));
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or(imagegen::DEFAULT_SIZE);
    // IMAGE_TRANSPARENT=1 asks for a cutout, where the model can do it.
    let prompt = if std::env::var("IMAGE_TRANSPARENT")
        .map(|v| v != "0")
        .unwrap_or(false)
        && imagegen::MODELS[model].transparency
    {
        imagegen::transparent_prompt(prompt)
    } else {
        prompt.to_string()
    };
    // IMAGE_REF=<path>[:<path>…] hands the model pictures to compose from,
    // which is the only way to exercise reference images without a UI.
    let references: Vec<_> = std::env::var("IMAGE_REF")
        .unwrap_or_default()
        .split(':')
        .filter(|p| !p.is_empty())
        .map(|path| {
            let reference = imagegen::load_reference(std::path::Path::new(path))
                .unwrap_or_else(|e| panic!("reference image: {e}"));
            println!(
                "reference: {path} ({}x{})",
                reference.width, reference.height
            );
            std::sync::Arc::new(reference)
        })
        .collect();
    let references_present = !references.is_empty();
    println!(
        "model: {}\nprompt: {prompt}\nsteps: {steps}\nsize: {width}x{height}",
        imagegen::MODELS[model].name
    );
    // What the GPU decision is made on, since it is made silently otherwise.
    match crate::hardware::gpu() {
        Some(gpu) => println!(
            "gpu: {} — {} free of {}, needs {}",
            gpu.name,
            crate::hardware::fmt_bytes(gpu.vram_free),
            crate::hardware::fmt_bytes(gpu.vram_total),
            crate::hardware::fmt_bytes(imagegen::MODELS[model].device_memory_needed(width, height))
        ),
        None => println!("gpu: none"),
    }

    let handle = imagegen::spawn_worker();
    handle
        .cmd_tx
        .send(imagegen::ImageCmd::Generate {
            model,
            // IMAGE_OFFLOAD=0 turns it off; it is on by default, which is what
            // a GPU build with a big model needs and a CPU build ignores.
            offload: std::env::var("IMAGE_OFFLOAD")
                .map(|v| v != "0")
                .unwrap_or(true),
            prompt: prompt.clone(),
            steps,
            seed: 42,
            width,
            height,
            references,
            // IMAGE_REF_FIT=crop trims to fill; the default keeps all of it.
            fit: match std::env::var("IMAGE_REF_FIT").as_deref() {
                Ok("crop") => imagegen::RefFit::Crop,
                _ => imagegen::RefFit::Letterbox,
            },
            // The reference figure when there is a reference, since that is
            // the case the default is wrong for; IMAGE_CFG overrides either.
            cfg: if references_present {
                imagegen::MODELS[model].cfg_reference
            } else {
                imagegen::MODELS[model].cfg
            },
        })
        .unwrap();

    // IMAGE_STOP_AFTER=<seconds> sets the same flag the Stop button sets, which
    // is the only way to exercise cancellation without a UI: `generate_image`
    // blocks for the whole run, so the flag has to arrive from elsewhere.
    if let Some(after) = std::env::var("IMAGE_STOP_AFTER")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        let stop = handle.stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(after));
            println!("--- asking it to stop ---");
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    let start = std::time::Instant::now();
    let mut first_step: Option<std::time::Instant> = None;
    let mut previews = 0usize;
    for event in handle.event_rx {
        match event {
            imagegen::ImageEvent::Note(note) => {
                println!("[{:.1}s] {note}", start.elapsed().as_secs_f32())
            }
            imagegen::ImageEvent::Step { done, total } => {
                let now = std::time::Instant::now();
                let per_step = match first_step {
                    Some(t0) if done > 1 => (now - t0).as_secs_f32() / (done - 1) as f32,
                    _ => {
                        first_step = Some(now);
                        0.0
                    }
                };
                println!(
                    "[{:.1}s] step {done}/{total}{}",
                    start.elapsed().as_secs_f32(),
                    if per_step > 0.0 {
                        format!(" — {per_step:.1}s/step")
                    } else {
                        String::new()
                    }
                );
            }
            imagegen::ImageEvent::Image {
                width,
                height,
                channels,
                pixels,
            } => {
                let recipe = imagegen::Recipe {
                    model: imagegen::MODELS[model].name,
                    prompt: prompt.clone(),
                    steps,
                    cfg: imagegen::MODELS[model].cfg,
                    seed: 42,
                    width,
                    height,
                };
                let path = std::path::Path::new("offgrid-image-probe.png");
                match imagegen::save_png(path, width, height, channels, &pixels, &recipe) {
                    Ok(()) => println!(
                        "wrote {} ({width}x{height}, {channels} channels)",
                        path.display()
                    ),
                    Err(e) => eprintln!("{e}"),
                }
            }
            // Nothing to show headless, but worth reporting: whether previews
            // arrive at all is a per-model question, and a silent probe cannot
            // answer it.
            imagegen::ImageEvent::Preview {
                width,
                height,
                channels,
                ..
            } => {
                previews += 1;
                println!(
                    "[{:.1}s] preview {width}x{height}, {channels} channels",
                    start.elapsed().as_secs_f32()
                );
            }
            imagegen::ImageEvent::Error(e) => eprintln!("error: {e}"),
            imagegen::ImageEvent::Done => break,
        }
    }
    println!(
        "total: {:.1}s, {previews} preview(s)",
        start.elapsed().as_secs_f32()
    );
}

fn smoke(agent_mode: bool) {
    use std::sync::{Arc, Mutex};

    let dir = config::models_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let entry = &models::catalog()[0]; // smallest model
    let path = dir.join(entry.file);

    if !path.exists() {
        println!("downloading {} …", entry.file);
        let dl = hub::start_download(entry.repo, entry.file, entry.size, &dir);
        for event in dl.rx {
            match event {
                hub::DownloadEvent::Progress { bytes, total } => {
                    print!("\r{} / {} ", bytes / 1_000_000, total / 1_000_000);
                }
                hub::DownloadEvent::Done => break,
                hub::DownloadEvent::Error(e) => panic!("download failed: {e}"),
            }
        }
        println!("done");
    }

    let handle = llm::spawn_worker(hardware::HardwareProfile::detect().physical_cores);
    handle
        .cmd_tx
        .send(llm::LlmCmd::Load {
            path,
            n_ctx: llm::DEFAULT_N_CTX,
        })
        .unwrap();
    let name = loop {
        match handle.event_rx.recv().unwrap() {
            llm::LlmEvent::Loaded(n) => break n,
            llm::LlmEvent::Error(e) => panic!("load failed: {e}"),
            _ => {}
        }
    };
    println!("loaded: {name}");

    if agent_mode {
        let ws = std::env::temp_dir().join("offgrid-agent-smoke");
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();
        println!("workspace: {}", ws.display());
        let run = agent::start(
            ws.clone(),
            "Create a file named hello.txt containing exactly: hello offgrid".into(),
            handle.cmd_tx.clone(),
            true,
            false,
            llm::DEFAULT_N_CTX,
        );
        for event in run.rx {
            match event {
                agent::AgentEvent::Token(t) => print!("{t}"),
                agent::AgentEvent::Info(t) => println!("[info] {t}"),
                agent::AgentEvent::Ctx(used) => println!("[ctx] {used} tokens"),
                agent::AgentEvent::TurnDone => println!("\n---"),
                agent::AgentEvent::ToolCall { name, summary } => {
                    println!("[tool call] {name}: {summary}");
                }
                agent::AgentEvent::ToolResult { output, .. } => {
                    println!("[tool result] {output}");
                }
                agent::AgentEvent::NeedsApproval { .. } => println!("[unexpected approval req]"),
                agent::AgentEvent::Done { iterations } => {
                    println!("[done after {iterations} turn(s)]");
                    break;
                }
                agent::AgentEvent::Error(e) => {
                    println!("[agent error] {e}");
                    break;
                }
            }
        }
        let content = std::fs::read_to_string(ws.join("hello.txt")).unwrap_or_default();
        println!("hello.txt content: {content:?}");
        return;
    }

    let (reply_tx, reply_rx) = std::sync::mpsc::channel();
    handle
        .cmd_tx
        .send(llm::LlmCmd::Generate {
            messages: Arc::new(vec![llm::ChatMessage {
                role: llm::Role::User,
                content: "Reply with exactly: hello from offgrid".into(),
            }]),
            reply: reply_tx,
            temp: 0.7,
            n_ctx: llm::DEFAULT_N_CTX,
        })
        .unwrap();
    print!("chat: ");
    for event in reply_rx {
        match event {
            llm::LlmEvent::Token(t) => print!("{t}"),
            llm::LlmEvent::GenDone => break,
            llm::LlmEvent::Error(e) => panic!("generate failed: {e}"),
            _ => {}
        }
    }
    println!();

    let loaded = Arc::new(Mutex::new(Some(name)));
    let _server = server::start(
        server::DEFAULT_PORT,
        false,
        handle.cmd_tx.clone(),
        dir,
        loaded,
        Arc::new(std::sync::atomic::AtomicU32::new(llm::DEFAULT_N_CTX)),
        None,
        agent::active_run(),
    )
    .expect("server start");
    println!(
        "server on http://127.0.0.1:{} — press Ctrl+C to quit",
        server::DEFAULT_PORT
    );
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
