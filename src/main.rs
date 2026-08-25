mod agent;
mod app;
mod bridge;
mod config;
mod hardware;
mod hub;
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

fn main() -> eframe::Result {
    // Headless check of the core plumbing (download → load → generate → serve).
    if std::env::args().any(|a| a == "--smoke") {
        smoke(false);
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
    handle.cmd_tx.send(llm::LlmCmd::Load(path)).unwrap();
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
            messages: vec![llm::ChatMessage {
                role: llm::Role::User,
                content: "Reply with exactly: hello from offgrid".into(),
            }],
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
        llm::DEFAULT_N_CTX,
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
