mod agent;
mod app;
mod config;
mod hardware;
mod hub;
mod llm;
mod models;
mod server;
mod theme;

fn main() -> eframe::Result {
    // Headless check of the core plumbing (download → load → generate → serve).
    if std::env::args().any(|a| a == "--smoke") {
        smoke(false);
        return Ok(());
    }
    // Same, but exercises the coding-agent loop instead of serving.
    if std::env::args().any(|a| a == "--smoke-agent") {
        smoke(true);
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([900.0, 650.0])
            .with_title("offgrid"),
        ..Default::default()
    };
    eframe::run_native(
        "offgrid",
        options,
        Box::new(|cc| Ok(Box::new(app::OffgridApp::new(cc)))),
    )
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

    let handle = llm::spawn_worker(hardware::HardwareProfile::detect().cores);
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
        );
        for event in run.rx {
            match event {
                agent::AgentEvent::Token(t) => print!("{t}"),
                agent::AgentEvent::TurnDone => println!("\n---"),
                agent::AgentEvent::ToolCall { name, summary } => {
                    println!("[tool call] {name}: {summary}");
                }
                agent::AgentEvent::ToolResult { output } => {
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
    let _server = server::start(server::DEFAULT_PORT, handle.cmd_tx.clone(), dir, loaded)
        .expect("server start");
    println!(
        "server on http://127.0.0.1:{} — press Ctrl+C to quit",
        server::DEFAULT_PORT
    );
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
