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
use crate::hub::{self, ActiveDownload, DownloadEvent, HubEvent, RepoFile, RepoResult};
use crate::llm::{self, LlmCmd, LlmEvent, Role};
use crate::hardware::HardwareProfile;
use crate::models::{self, Fit, LocalModel};
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

/// What the Models tab is currently showing: the local list, Hugging Face
/// search results, or the GGUF files of one repo.
#[derive(Clone, Copy, PartialEq)]
enum ModelsView {
    Local,
    Search,
    Files,
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
    /// Context fill, in tokens, for the Chat and Code conversations — shown as
    /// a percentage in the header. The desktop app tracks the same two.
    chat_ctx_used: usize,
    agent_ctx_used: usize,
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
    view: ModelsView,
    search_query: String,
    search_results: Vec<RepoResult>,
    search_pending: bool,
    /// Files of the repo picked from the search results.
    repo_files: Option<(String, Vec<RepoFile>, bool)>,
    hub_tx: std::sync::mpsc::Sender<HubEvent>,
    hub_rx: std::sync::mpsc::Receiver<HubEvent>,
    downloads: Vec<ActiveDownload>,
    hardware: HardwareProfile,
    /// A local model armed for deletion, awaiting a confirming second `d`.
    confirm_delete: Option<String>,
    /// Chat may search the web before answering (`/web on`). Off by default —
    /// a query leaves the machine.
    chat_web: bool,
}

/// Compact token count for the header: 512 → "512", 6912 → "6.9k", 16384 → "16k".
fn fmt_tokens(n: usize) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let k = n as f32 / 1000.0;
    if k < 10.0 {
        format!("{k:.1}k")
    } else {
        format!("{k:.0}k")
    }
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
            Tab::Models => self.models_body(),
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

    /// The context length that will actually be used to load models — the
    /// configured value or the default. Fit depends on it, so the badges and
    /// guards all read it from here.
    fn n_ctx(&self) -> u32 {
        self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX)
    }

    /// What a message would actually do right now. The Code tab always routes
    /// to the agent regardless of `self.mode`, so this — not `self.mode` — is
    /// what the hint line and `/help` should reflect.
    fn effective_mode(&self) -> Mode {
        if self.tab == Tab::Code {
            Mode::Code
        } else {
            self.mode
        }
    }

    /// Context fill (tokens) to display for the current tab. Chat and Code each
    /// carry their own conversation; Models and Serve have none.
    fn ctx_used_for_tab(&self) -> Option<usize> {
        match self.tab {
            Tab::Chat => Some(self.chat_ctx_used),
            Tab::Code => Some(self.agent_ctx_used),
            _ => None,
        }
    }

    /// The same verdict the desktop app shows as badges: RAM fit and
    /// estimated speed, e.g. "fits · ~12 t/s".
    fn gauge(&self, name: &str, size: u64) -> String {
        format!(
            "{} · {}",
            Fit::of(size, self.hardware.total_ram, self.n_ctx()).label(),
            models::fmt_tok_s(models::est_tokens_per_sec(
                name,
                size,
                self.hardware.mem_bandwidth
            ))
        )
    }

    fn models_body(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for dl in &self.downloads {
            let note = match &dl.failed {
                Some(e) => format!("failed: {e} — download it again to resume"),
                None => {
                    let pct = if dl.total > 0 {
                        dl.bytes * 100 / dl.total
                    } else {
                        0
                    };
                    format!(
                        "{pct}% · {} of {}",
                        crate::hardware::fmt_bytes(dl.bytes),
                        crate::hardware::fmt_bytes(dl.total)
                    )
                }
            };
            lines.push(format!("⇣ {} — {note}", dl.file));
        }
        if !lines.is_empty() {
            lines.push(String::new());
        }
        match self.view {
            ModelsView::Local => {
                lines.push("↑/↓ select · Enter load · u unload · d delete · /search <query> for Hugging Face".into());
                // Dynamic proposals: what this machine can comfortably run.
                let p = models::propose(self.hardware.total_ram, self.n_ctx());
                if p.chat.is_some() || p.code.is_some() {
                    lines.push(String::new());
                    lines.push("Recommended for your hardware (/get chat · /get code):".into());
                    if let Some(c) = &p.chat {
                        lines.push(format!("  chat  {}  ({})", c.name, self.gauge(c.file, c.size)));
                    }
                    if let Some(c) = &p.code {
                        lines.push(format!("  code  {}  ({})", c.name, self.gauge(c.file, c.size)));
                    }
                }
                lines.push(String::new());
                if self.models.is_empty() {
                    lines.push("No models on disk — /search <query> finds one to download.".into());
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
                        "{marker} {:<48} {:>9}  {}",
                        m.name.chars().take(48).collect::<String>(),
                        crate::hardware::fmt_bytes(m.size),
                        self.gauge(&m.name, m.size),
                    ));
                }
            }
            ModelsView::Search => {
                lines.push(format!(
                    "Hugging Face “{}” · Enter shows files · Esc goes back",
                    self.search_query
                ));
                lines.push(String::new());
                if self.search_pending {
                    lines.push("searching…".into());
                } else if self.search_results.is_empty() {
                    lines.push("No GGUF repos found.".into());
                }
                for (i, r) in self.search_results.iter().enumerate() {
                    let marker = if i == self.selected { "›" } else { " " };
                    lines.push(format!(
                        "{marker} {:<56} {} downloads",
                        r.id.chars().take(56).collect::<String>(),
                        r.downloads
                    ));
                }
            }
            ModelsView::Files => {
                let Some((repo, files, only_multipart)) = &self.repo_files else {
                    return lines;
                };
                lines.push(format!("{repo} · Enter downloads · Esc goes back"));
                lines.push(String::new());
                if files.is_empty() {
                    lines.push(if *only_multipart {
                        "Only multi-part GGUFs here — offgrid can't download those.".into()
                    } else {
                        "No usable GGUF files in this repo.".into()
                    });
                }
                for (i, f) in files.iter().enumerate() {
                    let marker = if i == self.selected { "›" } else { " " };
                    lines.push(format!(
                        "{marker} {:<56} {:>9}  {}",
                        f.name.chars().take(56).collect::<String>(),
                        crate::hardware::fmt_bytes(f.size),
                        self.gauge(&f.name, f.size),
                    ));
                }
            }
        }
        lines
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

        // Header: model on the left, context fill on the right, like the
        // desktop title area.
        let left = format!(
            " offgrid — {} ",
            self.loaded.clone().unwrap_or_else(|| "no model".into())
        );
        let right = match self.ctx_used_for_tab() {
            Some(used) if self.loaded.is_some() => {
                let n = self.n_ctx().max(1) as usize;
                let pct = (used * 100 / n).min(100);
                // Prefer the detailed form; drop the counts if the header is
                // too narrow to hold both it and the model name.
                let full = format!("ctx {pct}% ({}/{}) ", fmt_tokens(used), fmt_tokens(n));
                if left.chars().count() + full.chars().count() <= cols {
                    full
                } else {
                    format!("ctx {pct}% ")
                }
            }
            _ => String::new(),
        };
        // The context indicator wins the space race: truncate the (long) model
        // name rather than letting the model name push the percentage off-screen.
        let rw = right.chars().count();
        // Keep at least one space between a truncated name and the indicator.
        let keep = cols.saturating_sub(rw + usize::from(rw > 0));
        let left: String = left.chars().take(keep).collect();
        let gap = cols.saturating_sub(left.chars().count() + rw);
        let header: String = format!("{left}{}{right}", " ".repeat(gap))
            .chars()
            .take(cols)
            .collect();
        writeln!(
            out,
            "{}\r",
            style::Attribute::Reverse.to_string()
                + &header
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
        } else if self.run.is_some() {
            "agent running… (esc stops)".to_string()
        } else if self.generating {
            "generating… (esc stops)".to_string()
        } else {
            format!(
                "{} mode · tab switches · ctrl-c quits",
                self.effective_mode().label()
            )
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
                LlmEvent::Info(note) => self.status = note,
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
                LlmEvent::Stats {
                    prompt_tokens,
                    gen_tokens,
                    ..
                } => self.chat_ctx_used = prompt_tokens + gen_tokens,
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
                    AgentEvent::Ctx(used) => self.agent_ctx_used = used,
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
            // The run has truly ended now — reflect it in the status bar,
            // which until this point read "stopping…" or "agent running…".
            self.status = note;
            agent::release(&self.active);
            self.run = None;
        }
        while let Ok(event) = self.hub_rx.try_recv() {
            dirty = true;
            match event {
                HubEvent::SearchResults(results) => {
                    self.search_pending = false;
                    self.status = format!("{} matching repos", results.len());
                    self.search_results = results;
                }
                HubEvent::Files {
                    repo,
                    mut files,
                    only_multipart,
                } => {
                    // Only jump forward if the user is still on the results;
                    // a late reply must not yank them out of another view.
                    if self.view == ModelsView::Search {
                        files.sort_by_key(|f| f.size);
                        self.repo_files = Some((repo, files, only_multipart));
                        self.selected = 0;
                        self.scroll = 0;
                        self.view = ModelsView::Files;
                    }
                }
                HubEvent::Error(e) => {
                    self.search_pending = false;
                    self.status = format!("hub: {e}");
                }
            }
        }
        let mut done = false;
        for dl in &mut self.downloads {
            while let Ok(event) = dl.rx.try_recv() {
                dirty = true;
                match event {
                    DownloadEvent::Progress { bytes, total } => {
                        dl.bytes = bytes;
                        dl.total = total;
                    }
                    DownloadEvent::Done => {
                        dl.bytes = u64::MAX; // mark finished
                        done = true;
                    }
                    DownloadEvent::Error(e) => dl.failed = Some(e),
                }
            }
        }
        if done {
            self.downloads.retain(|d| d.bytes != u64::MAX);
            self.models = models::scan_local(&models_dir());
            self.status = "download finished".into();
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
        if let Some(rest) = text.strip_prefix("/search") {
            self.start_search(rest.trim());
            return;
        }
        if let Some(rest) = text.strip_prefix("/get") {
            self.get_proposal(rest.trim());
            return;
        }
        if let Some(rest) = text.strip_prefix("/web") {
            self.chat_web = match rest.trim() {
                "on" => true,
                "off" => false,
                "" => !self.chat_web, // bare /web toggles
                other => {
                    self.status = format!("usage: /web on|off (got \"{other}\")");
                    return;
                }
            };
            self.status = if self.chat_web {
                "web search in chat: on — queries leave this machine".into()
            } else {
                "web search in chat: off".into()
            };
            return;
        }
        if text == "/quit" || text == "/exit" {
            self.status = "quit".into();
            self.tab = Tab::Models;
            let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen);
            let _ = terminal::disable_raw_mode();
            crate::hard_exit(0);
        }

        let run_active = self.active.lock().unwrap().is_some();
        let mode = self.effective_mode();
        match session::parse(text, mode, run_active, true) {
            Command::Empty => {}
            Command::Help => self.status = session::help(mode, true).replace('\n', " · "),
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
        let n_ctx = self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX);
        if self.chat_web {
            self.status = "searching the web…".into();
            crate::webchat::spawn(
                session::snapshot(&self.chat),
                self.llm.cmd_tx.clone(),
                self.llm.event_tx.clone(),
                self.llm.stop.clone(),
                0.7,
                n_ctx,
            );
        } else {
            let _ = self.llm.cmd_tx.send(LlmCmd::Generate {
                messages: session::snapshot(&self.chat),
                reply: self.llm.event_tx.clone(),
                temp: 0.7,
                n_ctx,
            });
        }
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
        if !resuming {
            self.agent_ctx_used = 0;
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

    /// Download the proposed chat or coding model for this hardware.
    fn get_proposal(&mut self, kind: &str) {
        let p = models::propose(self.hardware.total_ram, self.n_ctx());
        let entry = match kind {
            "chat" => p.chat,
            "code" | "coding" => p.code,
            _ => {
                self.status = "usage: /get chat|code".into();
                return;
            }
        };
        let Some(entry) = entry else {
            self.status = "nothing in the catalog fits this machine — try /search".into();
            return;
        };
        self.tab = Tab::Models;
        self.view = ModelsView::Local;
        if self.models.iter().any(|m| m.name == entry.file.trim_end_matches(".gguf"))
            || models_dir().join(entry.file).exists()
        {
            self.status = format!("{} is already downloaded", entry.name);
            return;
        }
        if self.downloads.iter().any(|d| d.path == entry.file) {
            self.status = format!("{} is already downloading", entry.name);
            return;
        }
        self.downloads
            .push(hub::start_download(entry.repo, entry.file, entry.size, &models_dir()));
        self.status = format!("downloading {}", entry.name);
    }

    fn start_search(&mut self, query: &str) {
        if query.is_empty() {
            self.status = "usage: /search <query>".into();
            return;
        }
        hub::spawn_search(query.to_string(), self.hub_tx.clone());
        self.tab = Tab::Models;
        self.view = ModelsView::Search;
        self.search_query = query.to_string();
        self.search_results.clear();
        self.search_pending = true;
        self.selected = 0;
        self.scroll = 0;
        self.status = format!("searching Hugging Face for “{query}”…");
    }

    /// How many rows the current Models view has, for selection clamping.
    fn list_len(&self) -> usize {
        match self.view {
            ModelsView::Local => self.models.len(),
            ModelsView::Search => self.search_results.len(),
            ModelsView::Files => self
                .repo_files
                .as_ref()
                .map(|(_, f, _)| f.len())
                .unwrap_or(0),
        }
    }

    /// Esc in the Models tab walks back up: files → results → local list.
    fn pop_view(&mut self) {
        match self.view {
            ModelsView::Files => {
                self.view = ModelsView::Search;
                // Put the cursor back on the repo the files came from.
                self.selected = self
                    .repo_files
                    .as_ref()
                    .and_then(|(repo, ..)| self.search_results.iter().position(|r| &r.id == repo))
                    .unwrap_or(0);
            }
            ModelsView::Search => {
                self.view = ModelsView::Local;
                self.selected = 0;
            }
            ModelsView::Local => {}
        }
        self.scroll = 0;
    }

    fn activate_selected(&mut self) {
        match self.view {
            ModelsView::Local => self.load_selected(),
            ModelsView::Search => {
                if let Some(r) = self.search_results.get(self.selected) {
                    hub::spawn_list_files(r.id.clone(), self.hub_tx.clone());
                    self.status = format!("listing files in {}…", r.id);
                }
            }
            ModelsView::Files => {
                if let Some((repo, files, _)) = &self.repo_files
                    && let Some(f) = files.get(self.selected)
                {
                    self.downloads
                        .push(hub::start_download(repo, &f.name, f.size, &models_dir()));
                    self.status = format!("downloading {}", f.name);
                }
            }
        }
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
        // Any keypress cancels a pending delete; only a second `d` (below)
        // reads this back to confirm, so anything else is a silent abort.
        let armed_delete = self.confirm_delete.take();
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
                    self.activate_selected();
                } else {
                    self.submit();
                }
            }
            (KeyCode::Esc, _) => {
                if self.tab == Tab::Models && self.view != ModelsView::Local {
                    self.pop_view();
                } else {
                    // Signal both the chat worker and any agent run. The stop
                    // flag is only checked between tokens/turns, so a run does
                    // not end the instant Esc is pressed — say "stopping…" and
                    // let pump() report the actual finish, rather than claiming
                    // "stopped" while the model is still grinding out a turn.
                    self.llm.stop.store(true, Ordering::Relaxed);
                    let run_active = self.active.lock().unwrap().as_ref().is_some_and(|s| {
                        s.stop.store(true, Ordering::Relaxed);
                        true
                    });
                    self.status = if run_active || self.run.is_some() {
                        "stopping the run… (it halts at the next step)".into()
                    } else if self.generating {
                        "stopping…".into()
                    } else {
                        "stopped".into()
                    };
                }
            }
            (KeyCode::Up, _) if self.tab == Tab::Models => {
                self.selected = self.selected.saturating_sub(1)
            }
            (KeyCode::Down, _) if self.tab == Tab::Models => {
                self.selected = (self.selected + 1).min(self.list_len().saturating_sub(1))
            }
            (KeyCode::Char('u'), _)
                if self.tab == Tab::Models
                    && self.view == ModelsView::Local
                    && self.input.is_empty() =>
            {
                let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
            }
            (KeyCode::Char('d'), _)
                if self.tab == Tab::Models
                    && self.view == ModelsView::Local
                    && self.input.is_empty() =>
            {
                self.delete_selected(armed_delete);
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

    /// Delete the selected local model. The first press arms it (passing the
    /// previously armed name in `armed`); a second press on the same model
    /// confirms and removes the file, unloading it first if it is loaded.
    fn delete_selected(&mut self, armed: Option<String>) {
        let Some(m) = self.models.get(self.selected).cloned() else {
            return;
        };
        if armed.as_deref() != Some(m.name.as_str()) {
            self.confirm_delete = Some(m.name.clone());
            self.status = format!(
                "delete {} ({})? press d again to confirm",
                m.name,
                crate::hardware::fmt_bytes(m.size)
            );
            return;
        }
        if self.loaded.as_deref() == Some(m.name.as_str()) {
            let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
            self.loaded = None;
        }
        if self.config.last_model.as_deref() == Some(m.path.as_path()) {
            self.config.last_model = None;
            self.config.save();
        }
        match std::fs::remove_file(&m.path) {
            Ok(()) => self.status = format!("deleted {}", m.name),
            Err(e) => self.status = format!("delete failed: {e}"),
        }
        self.models = models::scan_local(&models_dir());
        self.selected = self.selected.min(self.models.len().saturating_sub(1));
    }

    fn load_selected(&mut self) {
        if let Some(m) = self.models.get(self.selected) {
            // Loading past physical RAM kills the whole process (llama.cpp
            // aborts), so refuse instead of terminating a headless session.
            if Fit::of(m.size, self.hardware.total_ram, self.n_ctx()) == Fit::TooBig {
                self.status = format!(
                    "{} won't fit: needs more than the {} of RAM here",
                    m.name,
                    crate::hardware::fmt_bytes(self.hardware.total_ram)
                );
                return;
            }
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
    let (hub_tx, hub_rx) = std::sync::mpsc::channel();

    // Only auto-load the last model if it still fits — re-loading a too-big
    // one is what got the process OOM-killed ("Killed") on the last run.
    let n_ctx = config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX);
    let autoload = config
        .last_model
        .clone()
        .filter(|p| p.exists() && models::safe_to_load(p, hardware.total_ram, n_ctx));
    let startup_status = match &config.last_model {
        Some(p) if autoload.is_none() && p.exists() => format!(
            "{} won't fit the {} of RAM here — not auto-loaded; pick another in Models.",
            p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            crate::hardware::fmt_bytes(hardware.total_ram)
        ),
        _ => "ready".into(),
    };

    let mut tui = Tui {
        tab: Tab::Chat,
        mode: Mode::Chat,
        input: String::new(),
        transcript: Vec::new(),
        selected: 0,
        loaded: None,
        loading: autoload.is_some(),
        generating: false,
        chat_ctx_used: 0,
        agent_ctx_used: 0,
        status: startup_status,
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
        view: ModelsView::Local,
        search_query: String::new(),
        search_results: Vec::new(),
        search_pending: false,
        repo_files: None,
        hub_tx,
        hub_rx,
        downloads: Vec::new(),
        hardware,
        confirm_delete: None,
        chat_web: false,
    };
    // Pick up where the desktop app left off — but only if it fits.
    if let Some(path) = autoload {
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
        // While tokens stream in or bytes come down, keep repainting.
        if tui.generating || tui.run.is_some() || !tui.downloads.is_empty() {
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

    #[test]
    fn tokens_are_formatted_compactly() {
        assert_eq!(fmt_tokens(0), "0");
        assert_eq!(fmt_tokens(512), "512");
        assert_eq!(fmt_tokens(6912), "6.9k");
        assert_eq!(fmt_tokens(16384), "16k");
    }
}
