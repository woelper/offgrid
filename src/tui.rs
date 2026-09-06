//! Minimal terminal UI for headless machines: the same tabs as the desktop
//! app, drawn by hand.
//!
//! Deliberately small — crossterm for raw input and cursor control, no
//! widget framework. Everything above the drawing (the conversation, the
//! command vocabulary, the agent run slot) comes from `session` and
//! `agent`, so this frontend only reads keys and paints text.

use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::{cursor, execute, style, terminal};

use crate::agent::{self, AgentEvent, AgentRun};
use crate::config::{Config, models_dir};
use crate::llm::{self, LlmCmd, LlmEvent, Role};
use crate::models::{self, LocalModel};
use crate::server::{self, ApiServer};
use crate::session::{self, ChatBusy, Command, Conversation, Mode};

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
    Chat,
    Code,
    Serve,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Models, Tab::Chat, Tab::Code, Tab::Serve];

    fn label(self) -> &'static str {
        match self {
            Tab::Models => "Models",
            Tab::Chat => "Chat",
            Tab::Code => "Code",
            Tab::Serve => "Serve",
        }
    }
}

struct Tui {
    tab: Tab,
    mode: Mode,
    input: String,
    /// Lines shown in the Code tab: tool calls, steering, agent info.
    transcript: Vec<String>,
    models: Vec<LocalModel>,
    selected: usize,
    loaded: Option<String>,
    loading: bool,
    generating: bool,
    status: String,
    scroll: usize,
    chat: Conversation,
    busy: ChatBusy,
    active: agent::ActiveRun,
    run: Option<AgentRun>,
    llm: llm::LlmHandle,
    loaded_shared: Arc<Mutex<Option<String>>>,
    server: Option<ApiServer>,
    config: Config,
}

/// Wrap text to the terminal width, keeping existing newlines.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(20);
    let mut out = Vec::new();
    for line in text.split('\n') {
        if line.chars().count() <= width {
            out.push(line.to_string());
            continue;
        }
        let mut current = String::new();
        for word in line.split(' ') {
            let extra = if current.is_empty() { 0 } else { 1 };
            if current.chars().count() + extra + word.chars().count() > width {
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                // A single word longer than the line: hard-split it.
                let mut rest = word;
                while rest.chars().count() > width {
                    let cut = rest
                        .char_indices()
                        .nth(width)
                        .map(|(i, _)| i)
                        .unwrap_or(rest.len());
                    out.push(rest[..cut].to_string());
                    rest = &rest[cut..];
                }
                current = rest.to_string();
            } else {
                if extra == 1 {
                    current.push(' ');
                }
                current.push_str(word);
            }
        }
        out.push(current);
    }
    out
}

impl Tui {
    /// Everything the content pane should show for the current tab.
    fn body(&self, width: usize) -> Vec<String> {
        match self.tab {
            Tab::Models => {
                let mut lines = vec![
                    "↑/↓ select · Enter load · u unload".to_string(),
                    String::new(),
                ];
                if self.models.is_empty() {
                    lines.push("No models on disk — download one in the desktop app.".into());
                }
                for (i, m) in self.models.iter().enumerate() {
                    let marker = if self.loaded.as_deref() == Some(m.name.as_str()) {
                        "●"
                    } else if i == self.selected {
                        "›"
                    } else {
                        " "
                    };
                    lines.push(format!(
                        "{marker} {:<48} {}",
                        m.name.chars().take(48).collect::<String>(),
                        crate::hardware::fmt_bytes(m.size)
                    ));
                }
                lines
            }
            Tab::Chat => {
                let mut lines = Vec::new();
                for m in session::snapshot(&self.chat) {
                    let who = match m.role {
                        Role::User => "you",
                        Role::Assistant => "model",
                        Role::System => "system",
                    };
                    let body = session::strip_think(&m.content);
                    if body.trim().is_empty() {
                        continue;
                    }
                    lines.push(format!("{who}:"));
                    lines.extend(wrap(&body, width));
                    lines.push(String::new());
                }
                if lines.is_empty() {
                    lines.push("Type to talk to the model. /last, /new, /help.".into());
                }
                lines
            }
            Tab::Code => {
                let mut lines = Vec::new();
                if let Some(summary) = agent::run_summary(&self.active) {
                    lines.extend(wrap(&summary, width));
                    lines.push(String::new());
                }
                for line in &self.transcript {
                    lines.extend(wrap(line, width));
                }
                if lines.is_empty() {
                    lines.push(format!(
                        "Type a task to run the agent in {}.",
                        self.config
                            .workspace
                            .as_ref()
                            .map(|w| w.display().to_string())
                            .unwrap_or_else(|| "(no workspace set — use /workspace <path>)".into())
                    ));
                }
                lines
            }
            Tab::Serve => {
                let port = self.config.server_port.unwrap_or(server::DEFAULT_PORT);
                let host = if self.config.server_lan {
                    server::lan_ip().unwrap_or_else(|| "0.0.0.0".into())
                } else {
                    "127.0.0.1".into()
                };
                vec![
                    format!(
                        "API server: {}",
                        if self.server.is_some() {
                            format!("running on http://{host}:{port}/v1")
                        } else {
                            "stopped".into()
                        }
                    ),
                    String::new(),
                    "/serve on · /serve off · /serve lan".to_string(),
                ]
            }
        }
    }

    fn draw(&mut self) -> std::io::Result<()> {
        // A pty that reports no size (or a comically small window) would
        // otherwise collapse the layout to nothing.
        let (cols, rows) = terminal::size().unwrap_or((80, 24));
        let (cols, rows) = ((cols as usize).max(40), (rows as usize).max(10));
        let mut out = std::io::stdout();
        execute!(
            out,
            terminal::Clear(terminal::ClearType::All),
            cursor::MoveTo(0, 0)
        )?;

        // Header: model and context, like the desktop title area.
        let header = format!(
            " offgrid — {} ",
            self.loaded.clone().unwrap_or_else(|| "no model".into())
        );
        writeln!(
            out,
            "{}\r",
            style::Attribute::Reverse.to_string()
                + &format!("{header:<cols$}")
                + &style::Attribute::Reset.to_string()
        )?;

        // Tab bar.
        let mut bar = String::new();
        for t in Tab::ALL {
            if t == self.tab {
                bar.push_str(&format!(
                    "{}[ {} ]{}",
                    style::Attribute::Reverse,
                    t.label(),
                    style::Attribute::Reset
                ));
            } else {
                bar.push_str(&format!("  {}  ", t.label()));
            }
        }
        writeln!(out, "{bar}\r")?;
        writeln!(out, "{}\r", "─".repeat(cols))?;

        // Content pane, scrolled to the bottom by default.
        let pane = rows.saturating_sub(6);
        let lines = self.body(cols.saturating_sub(1));
        let max_scroll = lines.len().saturating_sub(pane);
        let start = max_scroll.saturating_sub(self.scroll);
        for line in lines.iter().skip(start).take(pane) {
            writeln!(out, "{}\r", line.chars().take(cols).collect::<String>())?;
        }
        for _ in lines.len().saturating_sub(start)..pane {
            writeln!(out, "\r")?;
        }

        // Status + input.
        writeln!(out, "{}\r", "─".repeat(cols))?;
        let hint = if self.loading {
            "loading model…".to_string()
        } else if self.generating {
            "generating… (esc stops)".to_string()
        } else {
            format!("{} mode · tab switches · ctrl-c quits", self.mode.label())
        };
        writeln!(
            out,
            "{}\r",
            format!("{hint} — {}", self.status)
                .chars()
                .take(cols)
                .collect::<String>()
        )?;
        write!(out, "> {}\r", self.input)?;
        execute!(
            out,
            cursor::MoveToColumn((self.input.chars().count() + 2) as u16)
        )?;
        out.flush()
    }

    /// Drain worker events; returns true if anything changed.
    fn pump(&mut self) -> bool {
        let mut dirty = false;
        while let Ok(event) = self.llm.event_rx.try_recv() {
            dirty = true;
            match event {
                LlmEvent::Loaded(name) => {
                    self.loading = false;
                    *self.loaded_shared.lock().unwrap() = Some(name.clone());
                    self.status = format!("loaded {name}");
                    self.loaded = Some(name);
                }
                LlmEvent::Unloaded => {
                    self.loaded = None;
                    *self.loaded_shared.lock().unwrap() = None;
                    self.status = "model unloaded".into();
                }
                LlmEvent::Token(t) => session::append_assistant(&self.chat, &t),
                LlmEvent::GenDone => {
                    self.generating = false;
                    self.busy.release();
                    self.llm.stop.store(false, Ordering::Relaxed);
                }
                LlmEvent::Error(e) => {
                    self.loading = false;
                    if self.generating {
                        self.generating = false;
                        self.busy.release();
                        session::pop_unanswered(&self.chat);
                    }
                    self.status = format!("error: {e}");
                }
                LlmEvent::Stats { .. } => {}
            }
        }
        let mut finished = None;
        if let Some(run) = &self.run {
            while let Ok(event) = run.rx.try_recv() {
                dirty = true;
                match event {
                    AgentEvent::ToolCall { name, summary } => {
                        agent::note_activity(&self.active, &name, &summary);
                        self.transcript.push(format!("▸ {name}: {summary}"));
                    }
                    AgentEvent::ToolResult { ok: false, output } => {
                        let first = output.lines().next().unwrap_or_default();
                        self.transcript.push(format!("  ✗ {first}"));
                    }
                    AgentEvent::TurnDone => agent::note_turn(&self.active),
                    AgentEvent::Info(note) => self.transcript.push(format!("· {note}")),
                    AgentEvent::Error(e) => finished = Some(format!("error: {e}")),
                    AgentEvent::Done { iterations } => {
                        finished = Some(format!("finished after {iterations} turns"))
                    }
                    _ => {}
                }
            }
        }
        if let Some(note) = finished {
            self.transcript.push(format!("· {note}"));
            agent::release(&self.active);
            self.run = None;
        }
        dirty
    }

    fn submit(&mut self) {
        let text = std::mem::take(&mut self.input);
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        // Local-only commands first; the rest is the shared vocabulary.
        if let Some(rest) = text.strip_prefix("/workspace") {
            let path = std::path::PathBuf::from(rest.trim());
            if path.is_dir() {
                self.config.workspace = Some(path.clone());
                self.config.save();
                self.status = format!("workspace: {}", path.display());
            } else {
                self.status = "no such directory".into();
            }
            return;
        }
        if let Some(rest) = text.strip_prefix("/serve") {
            self.toggle_server(rest.trim());
            return;
        }
        if text == "/quit" || text == "/exit" {
            self.status = "quit".into();
            self.tab = Tab::Models;
            let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen);
            std::process::exit(0);
        }

        let run_active = self.active.lock().unwrap().is_some();
        let mode = if self.tab == Tab::Code {
            Mode::Code
        } else {
            self.mode
        };
        match session::parse(text, mode, run_active, true) {
            Command::Empty => {}
            Command::Help => self.status = session::help(self.mode, true).replace('\n', " · "),
            Command::SwitchMode(m) => {
                self.mode = m;
                self.tab = if m == Mode::Code {
                    Tab::Code
                } else {
                    Tab::Chat
                };
                self.status = format!("{} mode", m.label());
            }
            Command::New => {
                session::clear(&self.chat);
                self.status = "new conversation".into();
            }
            Command::Last => self.status = "see the Chat tab".into(),
            Command::Status => {
                self.status = agent::run_summary(&self.active)
                    .unwrap_or_else(|| format!("idle · {} turns", session::turns(&self.chat)))
            }
            Command::Stop => {
                if let Some(state) = self.active.lock().unwrap().as_ref() {
                    state.stop.store(true, Ordering::Relaxed);
                    self.status = "stopping the run…".into();
                } else {
                    self.llm.stop.store(true, Ordering::Relaxed);
                    self.status = "stopped".into();
                }
            }
            Command::Steer(text) => {
                agent::steer(&self.active, &text);
                self.transcript.push(format!("↪ you: {text}"));
            }
            Command::CodeDisabled => self.status = "code mode is unavailable".into(),
            Command::Code(task) => self.start_run(task, false),
            Command::Resume => self.start_run(String::new(), true),
            Command::Chat(text) => self.send_chat(&text),
        }
    }

    fn send_chat(&mut self, text: &str) {
        if self.loaded.is_none() {
            self.status = "no model loaded — pick one in the Models tab".into();
            return;
        }
        if !self.busy.claim() {
            self.status = "the model is answering another message".into();
            return;
        }
        self.tab = Tab::Chat;
        session::push_user(&self.chat, text);
        let _ = self.llm.cmd_tx.send(LlmCmd::Generate {
            messages: session::snapshot(&self.chat),
            reply: self.llm.event_tx.clone(),
            temp: 0.7,
            n_ctx: self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX),
        });
        session::push_assistant(&self.chat);
        self.generating = true;
    }

    fn start_run(&mut self, task: String, resuming: bool) {
        let Some(ws) = self.config.workspace.clone().filter(|w| w.is_dir()) else {
            self.status = "no workspace — use /workspace <path>".into();
            return;
        };
        if self.loaded.is_none() {
            self.status = "no model loaded".into();
            return;
        }
        if let Some(summary) = agent::run_summary(&self.active) {
            self.status = format!("busy — {summary}");
            return;
        }
        let n_ctx = self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX);
        let run = if resuming {
            match agent::resume(
                ws,
                self.llm.cmd_tx.clone(),
                true,
                self.config.web_tools,
                n_ctx,
            ) {
                Some(run) => run,
                None => {
                    self.status = "nothing to resume".into();
                    return;
                }
            }
        } else {
            agent::start(
                ws,
                task.clone(),
                self.llm.cmd_tx.clone(),
                true,
                self.config.web_tools,
                n_ctx,
            )
        };
        agent::claim(&self.active, agent::RunSource::Tui, &task, &run);
        self.transcript.push(format!("▶ {task}"));
        self.tab = Tab::Code;
        self.run = Some(run);
    }

    fn toggle_server(&mut self, arg: &str) {
        match arg {
            "off" => {
                self.server = None;
                self.config.server_enabled = false;
                self.status = "server stopped".into();
            }
            on @ ("on" | "lan") => {
                self.config.server_lan = on == "lan";
                self.config.server_enabled = true;
                match server::start(
                    self.config.server_port.unwrap_or(server::DEFAULT_PORT),
                    self.config.server_lan,
                    self.llm.cmd_tx.clone(),
                    models_dir(),
                    self.loaded_shared.clone(),
                    self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX),
                    self.config.workspace.clone(),
                    self.active.clone(),
                ) {
                    Ok(s) => {
                        self.server = Some(s);
                        self.status = "server started".into();
                    }
                    Err(e) => self.status = format!("server: {e}"),
                }
            }
            _ => self.status = "usage: /serve on|off|lan".into(),
        }
        self.config.save();
    }

    fn key(&mut self, key: KeyEvent) -> bool {
        match (key.code, key.modifiers) {
            (KeyCode::Char('c' | 'd'), KeyModifiers::CONTROL) => return false,
            (KeyCode::Tab, _) => {
                let i = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
                self.tab = Tab::ALL[(i + 1) % Tab::ALL.len()];
                self.scroll = 0;
            }
            (KeyCode::BackTab, _) => {
                let i = Tab::ALL.iter().position(|t| *t == self.tab).unwrap_or(0);
                self.tab = Tab::ALL[(i + Tab::ALL.len() - 1) % Tab::ALL.len()];
                self.scroll = 0;
            }
            (KeyCode::Enter, _) => {
                if self.tab == Tab::Models && self.input.is_empty() {
                    self.load_selected();
                } else {
                    self.submit();
                }
            }
            (KeyCode::Esc, _) => {
                self.llm.stop.store(true, Ordering::Relaxed);
                if let Some(state) = self.active.lock().unwrap().as_ref() {
                    state.stop.store(true, Ordering::Relaxed);
                }
                self.status = "stopped".into();
            }
            (KeyCode::Up, _) if self.tab == Tab::Models => {
                self.selected = self.selected.saturating_sub(1)
            }
            (KeyCode::Down, _) if self.tab == Tab::Models => {
                self.selected = (self.selected + 1).min(self.models.len().saturating_sub(1))
            }
            (KeyCode::Char('u'), _) if self.tab == Tab::Models && self.input.is_empty() => {
                let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
            }
            (KeyCode::PageUp, _) => self.scroll += 5,
            (KeyCode::PageDown, _) => self.scroll = self.scroll.saturating_sub(5),
            (KeyCode::Backspace, _) => {
                self.input.pop();
            }
            // Only real typing reaches the input: without this, Ctrl+J and
            // friends insert their bare letter.
            (KeyCode::Char(c), m) if m.is_empty() || m == KeyModifiers::SHIFT => self.input.push(c),
            _ => {}
        }
        true
    }

    fn load_selected(&mut self) {
        if let Some(m) = self.models.get(self.selected) {
            self.loading = true;
            self.status = format!("loading {}…", m.name);
            let _ = self.llm.cmd_tx.send(LlmCmd::Load(m.path.clone()));
            self.config.last_model = Some(m.path.clone());
            self.config.save();
        }
    }
}

/// Run the terminal UI until the user quits.
pub fn run() -> Result<(), String> {
    let config = Config::load();
    let hardware = crate::hardware::HardwareProfile::detect();
    let llm = llm::spawn_worker(hardware.physical_cores);
    let models = models::scan_local(&models_dir());
    let loaded_shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let mut tui = Tui {
        tab: Tab::Chat,
        mode: Mode::Chat,
        input: String::new(),
        transcript: Vec::new(),
        selected: 0,
        loaded: None,
        loading: config.last_model.is_some(),
        generating: false,
        status: "ready".into(),
        scroll: 0,
        chat: session::conversation(),
        busy: ChatBusy::new(),
        active: agent::active_run(),
        run: None,
        loaded_shared,
        server: None,
        models,
        llm,
        config,
    };
    // Pick up where the desktop app left off.
    if let Some(path) = tui.config.last_model.clone() {
        let _ = tui.llm.cmd_tx.send(LlmCmd::Load(path));
    }

    // llama.cpp writes its loader dump straight to stderr, which would
    // shred the screen. Harmless in the GUI (it lands in the launching
    // terminal); fatal here.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));

    terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    execute!(std::io::stdout(), terminal::EnterAlternateScreen).map_err(|e| e.to_string())?;
    let result = event_loop(&mut tui);
    let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen);
    let _ = terminal::disable_raw_mode();
    result
}

fn event_loop(tui: &mut Tui) -> Result<(), String> {
    let mut dirty = true;
    loop {
        if dirty {
            tui.draw().map_err(|e| e.to_string())?;
        }
        dirty = tui.pump();
        if event::poll(std::time::Duration::from_millis(120)).map_err(|e| e.to_string())? {
            match event::read().map_err(|e| e.to_string())? {
                Event::Key(key) if key.is_press() => {
                    if !tui.key(key) {
                        return Ok(());
                    }
                    dirty = true;
                }
                Event::Resize(..) => dirty = true,
                _ => {}
            }
        }
        // While tokens stream in, keep repainting.
        if tui.generating || tui.run.is_some() {
            dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_long_lines_and_keeps_breaks() {
        // Width is clamped to 20, so a tiny terminal still wraps sanely.
        let wrapped = wrap("one two three four five six seven eight", 9);
        assert!(
            wrapped.iter().all(|l| l.chars().count() <= 20),
            "{wrapped:?}"
        );
        assert_eq!(wrapped.join(" "), "one two three four five six seven eight");
        assert_eq!(
            wrap("the quick brown fox jumps over the lazy dog", 24),
            vec!["the quick brown fox", "jumps over the lazy dog"]
        );
        // Existing newlines survive.
        assert_eq!(wrap("a\nb", 40), vec!["a", "b"]);
        // A word longer than the line is hard-split rather than dropped.
        let long = wrap(&"x".repeat(45), 20);
        assert_eq!(long.len(), 3);
        assert_eq!(long.concat(), "x".repeat(45));
    }
}
