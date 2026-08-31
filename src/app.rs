use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use crate::agent::{self, AgentEvent, AgentRun};
use crate::bridge;
use crate::config::{Config, models_dir};
use crate::hardware::{self, HardwareProfile, fmt_bytes, fmt_bytes_precise};
use crate::hub::{self, ActiveDownload, DownloadEvent, HubEvent, RepoFile, RepoResult};
use crate::llm::{self, ChatMessage, LlmCmd, LlmEvent, LlmHandle, Role};
use crate::models::{self, Fit, LocalModel};
use crate::server::{self, ApiServer};
use crate::theme;

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
    Chat,
    Code,
    Serve,
    Settings,
}

fn tool_icon(name: &str) -> egui::ImageSource<'static> {
    match name {
        "run_command" => theme::icons().code.clone(),
        "web_search" => theme::icons().search.clone(),
        "fetch_url" => theme::icons().serve.clone(),
        "list_files" => theme::icons().folder.clone(),
        "read_file" | "write_file" => theme::icons().file.clone(),
        _ => theme::icons().disk.clone(),
    }
}

/// Cheap virtualization for long, variable-height lists: rows scrolled out
/// of view are replaced by spacers of their last measured height, so
/// markdown parsing and syntax highlighting only run for visible rows.
#[derive(Default)]
struct RowCuller {
    heights: Vec<f32>,
    width: f32,
}

impl RowCuller {
    fn begin(&mut self, ui: &egui::Ui, len: usize) {
        // Any width change re-wraps text, invalidating every height.
        if (self.width - ui.available_width()).abs() > 1.0 {
            self.heights.clear();
            self.width = ui.available_width();
        }
        self.heights.resize(len, 0.0);
    }

    /// `hot` rows (recently changed, e.g. still streaming) always render.
    fn row(&mut self, ui: &mut egui::Ui, i: usize, hot: bool, render: impl FnOnce(&mut egui::Ui)) {
        let h = self.heights.get(i).copied().unwrap_or(0.0);
        if !hot && h > 0.0 {
            // Generous margin: rows near the viewport edge stay rendered, so
            // small height corrections don't shift the list (bottom flicker),
            // and scrolling has pre-laid-out content in both directions.
            const MARGIN: f32 = 400.0;
            let clip = ui.clip_rect();
            let top = ui.cursor().top();
            if top + h < clip.min.y - MARGIN || top > clip.max.y + MARGIN {
                // A rendered widget advances the cursor by height PLUS
                // item_spacing; add_space advances by exactly the amount.
                // Without the compensation, total content height changes with
                // the culled-row count, which made scrolling jump and flicker.
                ui.add_space(h + ui.spacing().item_spacing.y);
                return;
            }
        }
        let resp = ui.scope(render);
        if let Some(slot) = self.heights.get_mut(i) {
            *slot = resp.response.rect.height();
        }
    }

    fn clear(&mut self) {
        self.heights.clear();
    }
}

enum AgentItem {
    Task(String),
    Assistant(String),
    Tool {
        name: String,
        summary: String,
        output: Option<String>,
        ok: Option<bool>,
    },
    Info(String),
}

pub struct OffgridApp {
    hardware: HardwareProfile,
    /// Free space where models live. Cached: querying the mount table every
    /// frame would be wasteful; refreshed whenever the model list changes.
    free_space: Option<u64>,
    config: Config,
    tab: Tab,
    models_dir: PathBuf,
    local_models: Vec<LocalModel>,

    // Hub browsing
    hub_tx: Sender<HubEvent>,
    hub_rx: Receiver<HubEvent>,
    search_query: String,
    search_pending: bool,
    last_search: Option<String>,
    search_results: Vec<RepoResult>,
    repo_files: HashMap<String, (Vec<RepoFile>, bool)>,
    files_pending: HashSet<String>,
    downloads: Vec<ActiveDownload>,
    interrupted: Vec<hub::PartInfo>,

    // LLM
    llm: LlmHandle,
    loaded_model: Option<String>,
    // Same value, shared with the API server thread.
    loaded_model_shared: Arc<Mutex<Option<String>>>,
    model_loading: bool,

    // API server for external tools (opencode etc.)
    api_server: Option<ApiServer>,
    bridge: Option<bridge::Bridge>,
    /// Cached at server start — resolving it opens a UDP socket, too costly
    /// per frame.
    lan_ip: Option<String>,

    // Chat
    messages: Vec<ChatMessage>,
    input: String,
    generating: bool,
    md_cache: CommonMarkCache,
    gen_stats: Option<String>,
    live_tokens: usize,
    live_start: Option<std::time::Instant>,

    // Code agent
    workspace_input: String,
    agent_task: String,
    agent_run: Option<AgentRun>,
    agent_transcript: Vec<AgentItem>,
    agent_current: String,
    agent_auto_approve: bool,
    agent_approval: Option<(String, Sender<bool>)>,

    confirm_delete: Option<LocalModel>,
    last_error: Option<String>,
    chat_culler: RowCuller,
    agent_culler: RowCuller,
    chat_ctx_used: usize,
    agent_ctx_used: usize,
    hl_memo: HighlightMemo,
}

impl OffgridApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = Config::load();
        if let Some(skin) = &config.skin {
            theme::set_kind(theme::SkinKind::from_id(skin));
        }
        theme::apply(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let hardware = HardwareProfile::detect();
        let models_dir = models_dir();
        let _ = std::fs::create_dir_all(&models_dir);
        let (hub_tx, hub_rx) = std::sync::mpsc::channel();
        let llm = llm::spawn_worker(hardware.physical_cores);

        let mut model_loading = false;
        if let Some(last) = &config.last_model
            && last.exists()
        {
            let _ = llm.cmd_tx.send(LlmCmd::Load(last.clone()));
            model_loading = true;
        }

        let loaded_model_shared = Arc::new(Mutex::new(None));
        let workspace_input = config
            .workspace
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let interrupted = hub::scan_parts(&models_dir);
        let mut app = Self {
            local_models: models::scan_local(&models_dir),
            hardware,
            free_space: hardware::free_space(&models_dir),
            config,
            tab: Tab::Models,
            models_dir,
            hub_tx,
            hub_rx,
            search_query: String::new(),
            search_pending: false,
            last_search: None,
            search_results: Vec::new(),
            repo_files: HashMap::new(),
            files_pending: HashSet::new(),
            downloads: Vec::new(),
            interrupted,
            llm,
            loaded_model: None,
            loaded_model_shared,
            model_loading,
            api_server: None,
            bridge: None,
            lan_ip: None,
            messages: Vec::new(),
            input: String::new(),
            generating: false,
            md_cache: CommonMarkCache::default(),
            gen_stats: None,
            live_tokens: 0,
            live_start: None,
            workspace_input,
            agent_task: String::new(),
            agent_run: None,
            agent_transcript: Vec::new(),
            agent_current: String::new(),
            agent_auto_approve: false,
            agent_approval: None,
            confirm_delete: None,
            last_error: None,
            chat_culler: RowCuller::default(),
            agent_culler: RowCuller::default(),
            chat_ctx_used: 0,
            agent_ctx_used: 0,
            hl_memo: HighlightMemo::default(),
        };
        if app.config.server_enabled {
            app.start_server();
        }
        if app.config.bridge_enabled {
            app.start_bridge();
        }
        app
    }

    fn n_ctx(&self) -> u32 {
        self.config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX)
    }

    fn server_port(&self) -> u16 {
        self.config.server_port.unwrap_or(server::DEFAULT_PORT)
    }

    fn start_server(&mut self) {
        if self.api_server.is_some() {
            return;
        }
        self.lan_ip = server::lan_ip();
        match server::start(
            self.server_port(),
            self.config.server_lan,
            self.llm.cmd_tx.clone(),
            self.models_dir.clone(),
            self.loaded_model_shared.clone(),
            self.n_ctx(),
            self.config.workspace.clone(),
        ) {
            Ok(s) => self.api_server = Some(s),
            Err(e) => {
                self.last_error = Some(format!("could not start server: {e}"));
                self.config.server_enabled = false;
                self.config.save();
            }
        }
    }

    fn set_loaded(&mut self, name: Option<String>) {
        self.loaded_model = name.clone();
        *self.loaded_model_shared.lock().unwrap() = name;
    }

    fn rescan(&mut self) {
        self.local_models = models::scan_local(&self.models_dir);
        self.interrupted = hub::scan_parts(&self.models_dir);
        self.free_space = hardware::free_space(&self.models_dir);
    }

    /// End the current agent run and note why in the transcript.
    fn agent_finished(&mut self, note: String) {
        self.agent_transcript.push(AgentItem::Info(note));
        self.agent_run = None;
        self.agent_approval = None;
        self.llm.stop.store(false, Ordering::Relaxed);
    }

    /// Track a streamed token for the live tok/s display.
    fn note_token(&mut self) {
        self.live_tokens += 1;
        if self.live_start.is_none() {
            self.live_start = Some(std::time::Instant::now());
        }
    }

    fn drain_events(&mut self) {
        loop {
            match self.hub_rx.try_recv() {
                Ok(HubEvent::SearchResults(results)) => {
                    self.search_results = results;
                    self.search_pending = false;
                }
                Ok(HubEvent::Files {
                    repo,
                    mut files,
                    only_multipart,
                }) => {
                    self.files_pending.remove(&repo);
                    files.sort_by_key(|f| f.size);
                    self.repo_files.insert(repo, (files, only_multipart));
                }
                Ok(HubEvent::Error(e)) => {
                    self.search_pending = false;
                    self.last_error = Some(e);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }

        let mut finished = false;
        for dl in &mut self.downloads {
            loop {
                match dl.rx.try_recv() {
                    Ok(DownloadEvent::Progress { bytes, total }) => {
                        dl.bytes = bytes;
                        dl.total = total;
                    }
                    Ok(DownloadEvent::Done) => {
                        dl.bytes = u64::MAX; // mark finished
                        finished = true;
                    }
                    Ok(DownloadEvent::Error(e)) => {
                        // Keep the row so the user can resume; the .part file
                        // and its metadata are still on disk.
                        dl.failed = Some(e);
                    }
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            }
        }
        if finished {
            self.downloads.retain(|d| d.bytes != u64::MAX);
            self.rescan();
        }

        if let Some(run) = &self.agent_run {
            let events: Vec<AgentEvent> = std::iter::from_fn(|| run.rx.try_recv().ok()).collect();
            for event in events {
                match event {
                    AgentEvent::Token(t) => {
                        self.note_token();
                        self.agent_current.push_str(&t);
                    }
                    AgentEvent::TurnDone => {
                        let text = std::mem::take(&mut self.agent_current);
                        if !text.trim().is_empty() {
                            self.agent_transcript.push(AgentItem::Assistant(text));
                        }
                        self.live_tokens = 0;
                        self.live_start = None;
                    }
                    AgentEvent::ToolCall { name, summary } => {
                        self.agent_transcript.push(AgentItem::Tool {
                            name,
                            summary,
                            output: None,
                            ok: None,
                        });
                    }
                    AgentEvent::ToolResult { output, ok } => {
                        if let Some(AgentItem::Tool {
                            output: slot,
                            ok: ok_slot,
                            ..
                        }) = self.agent_transcript.last_mut()
                        {
                            *slot = Some(output);
                            *ok_slot = Some(ok);
                        }
                    }
                    AgentEvent::Info(text) => {
                        self.agent_transcript.push(AgentItem::Info(text));
                    }
                    AgentEvent::Ctx(used) => {
                        self.agent_ctx_used = used;
                    }
                    AgentEvent::NeedsApproval { command, reply } => {
                        self.agent_approval = Some((command, reply));
                    }
                    AgentEvent::Done { iterations } => {
                        self.agent_finished(format!("finished after {iterations} turn(s)"));
                    }
                    AgentEvent::Error(e) => {
                        self.agent_finished(format!("error: {e}"));
                    }
                }
            }
        }

        loop {
            match self.llm.event_rx.try_recv() {
                Ok(LlmEvent::Loaded(name)) => {
                    self.model_loading = false;
                    self.set_loaded(Some(name));
                    self.config.save();
                }
                Ok(LlmEvent::Unloaded) => {
                    self.set_loaded(None);
                }
                Ok(LlmEvent::Token(text)) => {
                    self.note_token();
                    if let Some(last) = self.messages.last_mut()
                        && last.role == Role::Assistant
                    {
                        last.content.push_str(&text);
                    }
                }
                Ok(LlmEvent::Stats {
                    prompt_tokens,
                    prompt_secs,
                    gen_tokens,
                    gen_secs,
                }) => {
                    self.chat_ctx_used = prompt_tokens + gen_tokens;
                    self.gen_stats = Some(format!(
                        "{:.1} tok/s · {} tokens · prompt: {} tok in {:.1}s",
                        gen_tokens as f32 / gen_secs.max(0.001),
                        gen_tokens,
                        prompt_tokens,
                        prompt_secs
                    ));
                }
                Ok(LlmEvent::GenDone) => {
                    self.generating = false;
                    self.llm.stop.store(false, Ordering::Relaxed);
                }
                Ok(LlmEvent::Error(e)) => {
                    self.model_loading = false;
                    self.generating = false;
                    self.last_error = Some(if e.starts_with("context window full") {
                        format!("{e} — press Clear (next to Send) to start a new conversation")
                    } else {
                        e
                    });
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    fn is_downloaded(&self, file: &str) -> bool {
        self.models_dir.join(file_basename(file)).exists()
    }

    fn is_downloading(&self, file: &str) -> bool {
        let name = file_basename(file);
        self.downloads.iter().any(|d| d.file == name)
    }

    fn start_download(&mut self, repo: &str, file: &str, size: u64) {
        if self.is_downloaded(file) || self.is_downloading(file) {
            return;
        }
        self.downloads
            .push(hub::start_download(repo, file, size, &self.models_dir));
    }

    fn load_model(&mut self, path: PathBuf) {
        self.config.last_model = Some(path.clone());
        let _ = self.llm.cmd_tx.send(LlmCmd::Load(path));
        self.model_loading = true;
    }

    fn send_chat(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.generating || self.loaded_model.is_none() {
            return;
        }
        self.input.clear();
        self.messages.push(ChatMessage {
            role: Role::User,
            content: text,
        });
        let _ = self.llm.cmd_tx.send(LlmCmd::Generate {
            messages: self.messages.clone(),
            reply: self.llm.event_tx.clone(),
            temp: 0.7,
            n_ctx: self.n_ctx(),
        });
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: String::new(),
        });
        self.generating = true;
        self.live_tokens = 0;
        self.live_start = None;
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        // Plain grey strip above the tabs, like Haiku's window layouts.
        ui.add_space(14.0);
        theme::tab_bar(
            ui,
            &mut self.tab,
            &[
                (Tab::Models, theme::icons().models.clone(), "Models"),
                (Tab::Chat, theme::icons().chat.clone(), "Chat"),
                (Tab::Code, theme::icons().code.clone(), "Code"),
                (Tab::Serve, theme::icons().serve.clone(), "Serve"),
                (Tab::Settings, theme::icons().settings.clone(), "Settings"),
            ],
        );
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(
            ui,
            "Appearance",
            Some(theme::icons().appearance.clone()),
            |ui| {
                let row_h = theme::skin().control_height;
                ui.horizontal(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(70.0, row_h),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| ui.label("UI style:"),
                    );
                    let mut selected = theme::kind();
                    egui::ComboBox::from_id_salt("skin_select")
                        .selected_text(selected.label())
                        .show_ui(ui, |ui| {
                            for kind in theme::SkinKind::ALL {
                                ui.selectable_value(&mut selected, kind, kind.label());
                            }
                        });
                    if selected != theme::kind() {
                        theme::set_kind(selected);
                        theme::apply(ui.ctx());
                        // Fonts changed — cached row heights are stale.
                        self.chat_culler.clear();
                        self.agent_culler.clear();
                        self.config.skin = Some(selected.id().to_string());
                        self.config.save();
                    }
                });
                ui.weak(
                    "Haiku is offgrid's native look. \"egui default\" is the stock egui \
                 dark theme with its default fonts.",
                );
            },
        );

        theme::group(ui, "Model", Some(theme::icons().chat.clone()), |ui| {
            let row_h = theme::skin().control_height;
            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(70.0, row_h),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| ui.label("Context:"),
                );
                let mut n_ctx = self.n_ctx();
                egui::ComboBox::from_id_salt("n_ctx_select")
                    .selected_text(format!("{n_ctx} tokens"))
                    .show_ui(ui, |ui| {
                        for v in [4096u32, 8192, 16384, 32768] {
                            ui.selectable_value(&mut n_ctx, v, format!("{v} tokens"));
                        }
                    });
                if n_ctx != self.n_ctx() {
                    self.config.n_ctx = Some(n_ctx);
                    self.config.save();
                }
            });
            ui.weak(
                "Larger context windows let agent tasks run longer before compaction, \
                 at the cost of RAM (KV cache) and slower long-context generation.",
            );
        });

        theme::group(ui, "System", Some(theme::icons().disk.clone()), |ui| {
            ui.label(format!(
                "{} · {} cores ({} physical) · {} RAM",
                self.hardware.cpu_brand,
                self.hardware.cores,
                self.hardware.physical_cores,
                fmt_bytes(self.hardware.total_ram)
            ));
            ui.weak(format!(
                "Measured memory bandwidth: {}/s — this drives the tok/s estimates \
                 in the model lists.",
                fmt_bytes(self.hardware.mem_bandwidth)
            ));
        });

        // Debug builds only: quick access to agent session logs.
        if cfg!(debug_assertions) {
            theme::group(
                ui,
                "Session logs (debug)",
                Some(theme::icons().file.clone()),
                |ui| {
                    let dir = crate::config::logs_dir();
                    ui.horizontal(|ui| {
                        ui.monospace(dir.display().to_string());
                        if theme::button(
                            ui,
                            Some((theme::icons().folder.clone(), 16.0)),
                            "Open folder",
                        )
                        .clicked()
                            && let Err(e) = open::that_detached(&dir)
                        {
                            self.last_error = Some(format!("could not open folder: {e}"));
                        }
                        if theme::button(
                            ui,
                            Some((theme::icons().trash.clone(), 16.0)),
                            "Clear logs",
                        )
                        .clicked()
                        {
                            // Only remove our own agent logs, nothing else
                            // that might live in the directory.
                            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                                let name = entry.file_name().to_string_lossy().to_string();
                                if name.starts_with("agent-")
                                    && name.ends_with(".log")
                                    && let Err(e) = std::fs::remove_file(entry.path())
                                {
                                    self.last_error = Some(format!("could not delete {name}: {e}"));
                                }
                            }
                        }
                    });
                    let mut logs: Vec<(String, std::path::PathBuf, u64, std::time::SystemTime)> =
                        std::fs::read_dir(&dir)
                            .into_iter()
                            .flatten()
                            .flatten()
                            .filter_map(|e| {
                                let meta = e.metadata().ok()?;
                                Some((
                                    e.file_name().to_string_lossy().to_string(),
                                    e.path(),
                                    meta.len(),
                                    meta.modified().ok()?,
                                ))
                            })
                            .collect();
                    logs.sort_by_key(|l| std::cmp::Reverse(l.3)); // newest first
                    if logs.is_empty() {
                        ui.weak("No session logs yet — run an agent task first.");
                    }
                    for (name, path, size, _) in logs.iter().take(10) {
                        ui.horizontal(|ui| {
                            ui.monospace(name);
                            ui.weak(fmt_bytes(*size));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if theme::button(ui, None, "Open").clicked()
                                        && let Err(e) = open::that_detached(path)
                                    {
                                        self.last_error = Some(format!("could not open log: {e}"));
                                    }
                                    if theme::button(ui, None, "Copy contents").clicked() {
                                        match std::fs::read_to_string(path) {
                                            Ok(text) => ui.ctx().copy_text(text),
                                            Err(e) => {
                                                self.last_error =
                                                    Some(format!("could not read log: {e}"));
                                            }
                                        }
                                    }
                                },
                            );
                        });
                    }
                },
            );
        }
    }

    fn fit_badge(&self, ui: &mut egui::Ui, size: u64) {
        let (label, color) = Fit::of(size, self.hardware.total_ram).badge();
        ui.colored_label(color, label);
    }

    fn download_button(ui: &mut egui::Ui) -> bool {
        theme::button(
            ui,
            Some((theme::icons().download.clone(), 22.0)),
            "Download",
        )
        .clicked()
    }

    fn models_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            theme::group(
                ui,
                "Current model",
                Some(theme::icons().model.clone()),
                |ui| {
                    ui.horizontal(|ui| {
                        if self.model_loading {
                            ui.spinner();
                            ui.label("loading model…");
                        } else if let Some(name) = self.loaded_model.clone() {
                            theme::icon(ui, theme::icons().model.clone(), 18.0);
                            ui.label(&name);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if theme::button(ui, None, "Unload").clicked() {
                                        let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
                                    }
                                },
                            );
                        } else {
                            ui.weak("No model loaded — pick one below.");
                        }
                    });
                },
            );

            let on_disk = match self.free_space {
                Some(free) => format!("On disk  ({} free)", fmt_bytes(free)),
                None => "On disk".to_string(),
            };
            theme::group(ui, &on_disk, Some(theme::icons().disk.clone()), |ui| {
                if self.local_models.is_empty() {
                    ui.weak("No models yet — download one below.");
                }
                let locals = self.local_models.clone();
                for (i, model) in locals.iter().enumerate() {
                    let loaded = self.loaded_model.as_deref() == Some(model.name.as_str());
                    let can_load = !loaded && !self.model_loading;
                    let badge = Fit::of(model.size, self.hardware.total_ram).badge();
                    let mut clicked_load = false;
                    let mut clicked_delete = false;
                    list_row(
                        ui,
                        i % 2 == 1,
                        |ui| {
                            theme::icon(ui, theme::icons().disk.clone(), 16.0);
                            ui.add(egui::Label::new(&model.name).truncate());
                            if loaded {
                                ui.colored_label(theme::skin().good, "•");
                            }
                        },
                        model.size,
                        models::fmt_tok_s(models::est_tokens_per_sec(
                            &model.name,
                            model.size,
                            self.hardware.mem_bandwidth,
                        )),
                        badge,
                        |ui| {
                            // right-to-left: first added sits at the right edge
                            clicked_delete =
                                theme::button(ui, Some((theme::icons().trash.clone(), 18.0)), "Delete").clicked();
                            let load = ui.add_enabled(
                                can_load,
                                egui::Button::new("Load").min_size(egui::vec2(60.0, 0.0)),
                            );
                            theme::gloss(ui, load.rect);
                            clicked_load = load.clicked();
                        },
                    );
                    if clicked_load {
                        self.load_model(model.path.clone());
                    }
                    if clicked_delete {
                        self.confirm_delete = Some(model.clone());
                    }
                }

                let mut resume: Option<(String, String, u64)> = None;
                let mut discard: Option<usize> = None;
                for (i, dl) in self.downloads.iter().enumerate() {
                    let frac = if dl.total > 0 {
                        dl.bytes as f32 / dl.total as f32
                    } else {
                        0.0
                    };
                    if let Some(err) = &dl.failed {
                        ui.horizontal(|ui| {
                            theme::icon(ui, theme::icons().download.clone(), 16.0);
                            ui.add(egui::Label::new(&dl.file).truncate());
                            ui.colored_label(
                                theme::skin().bad,
                                format!("interrupted: {err}"),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if theme::button(ui, None, "Discard").clicked() {
                                        discard = Some(i);
                                    }
                                    if theme::button(ui, None, "Resume").clicked() {
                                        resume =
                                            Some((dl.repo.clone(), dl.path.clone(), dl.total));
                                    }
                                },
                            );
                        });
                        theme::progress_bar(ui, frac);
                        ui.add_space(4.0);
                        continue;
                    }
                    let elapsed = dl.started.elapsed().as_secs_f32();
                    let speed = dl.bytes.saturating_sub(dl.resumed_from) as f32 / elapsed.max(0.1);
                    let eta = if speed > 1.0 && dl.total > dl.bytes {
                        fmt_eta((dl.total - dl.bytes) as f32 / speed)
                    } else {
                        "—".to_string()
                    };
                    // Haiku Installer layout: status line above, bar below.
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().download.clone(), 16.0);
                        ui.add(egui::Label::new(&dl.file).truncate());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.weak(format!(
                                "{} / {} · {} · {}",
                                fmt_bytes_precise(dl.bytes),
                                fmt_bytes_precise(dl.total),
                                fmt_bitrate(speed),
                                eta
                            ));
                        });
                    });
                    theme::progress_bar(ui, frac);
                    ui.add_space(4.0);
                }
                if let Some(i) = discard {
                    let dl = self.downloads.remove(i);
                    hub::discard_part(&self.models_dir, &dl.file);
                    self.rescan();
                }
                if let Some((repo, path, size)) = resume {
                    self.downloads
                        .retain(|d| file_basename(&d.path) != file_basename(&path));
                    self.downloads
                        .push(hub::start_download(&repo, &path, size, &self.models_dir));
                }

                // Partial downloads left over from earlier sessions.
                let mut resume_part: Option<hub::PartMeta> = None;
                let mut discard_part: Option<String> = None;
                for part in &self.interrupted {
                    if self.is_downloading(&part.file) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().download.clone(), 16.0);
                        ui.add(egui::Label::new(&part.file).truncate());
                        ui.colored_label(
                            theme::skin().warn,
                            format!(
                                "interrupted — {} of {} downloaded",
                                fmt_bytes(part.bytes),
                                fmt_bytes(part.meta.size)
                            ),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if theme::button(ui, None, "Discard").clicked() {
                                discard_part = Some(part.file.clone());
                            }
                            if theme::button(ui, None, "Resume").clicked() {
                                resume_part = Some(part.meta.clone());
                            }
                        });
                    });
                }
                if let Some(file) = discard_part {
                    hub::discard_part(&self.models_dir, &file);
                    self.rescan();
                }
                if let Some(meta) = resume_part {
                    self.downloads.push(hub::start_download(
                        &meta.repo,
                        &meta.path,
                        meta.size,
                        &self.models_dir,
                    ));
                    self.rescan();
                }
            });

            theme::group(ui, "Get models", Some(theme::icons().depot.clone()), |ui| {
                if let Some(rec) = models::recommended(self.hardware.total_ram) {
                    ui.horizontal(|ui| {
                        ui.label("Recommended for your hardware:");
                        ui.strong(rec.name);
                    });
                    ui.separator();
                }
                let catalog = models::catalog();
                for (i, entry) in catalog.iter().enumerate() {
                    let badge = Fit::of(entry.size, self.hardware.total_ram).badge();
                    let downloaded = self.is_downloaded(entry.file);
                    let downloading = self.is_downloading(entry.file);
                    let tooltip = models::quant_tooltip(entry.file);
                    let mut clicked_download = false;
                    list_row(
                        ui,
                        i % 2 == 1,
                        |ui| {
                            ui.add(egui::Label::new(entry.name).truncate())
                                .on_hover_text(tooltip);
                        },
                        entry.size,
                        models::fmt_tok_s(models::est_tokens_per_sec(
                            entry.file,
                            entry.size,
                            self.hardware.mem_bandwidth,
                        )),
                        badge,
                        |ui| {
                            if downloaded {
                                ui.weak("downloaded");
                            } else if downloading {
                                ui.spinner();
                            } else {
                                clicked_download = Self::download_button(ui);
                            }
                        },
                    );
                    if clicked_download {
                        self.start_download(entry.repo, entry.file, entry.size);
                    }
                }
            });

            theme::group(ui, "Search Hugging Face", Some(theme::icons().search.clone()), |ui| {
                ui.horizontal(|ui| {
                    let h = theme::skin().control_height;
                    let resp = ui.add_sized(
                        [320.0, h],
                        egui::TextEdit::singleline(&mut self.search_query),
                    );
                    let submitted =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let search = ui.add_sized(
                        [0.0, h],
                        egui::Button::image_and_text(
                            egui::Image::new(theme::icons().search.clone()).fit_to_exact_size(egui::vec2(16.0, 16.0)),
                            "Search",
                        ),
                    );
                    theme::gloss(ui, search.rect);
                    let search_clicked = search.clicked();
                    if (search_clicked || submitted) && !self.search_query.trim().is_empty() {
                        self.search_pending = true;
                        self.search_results.clear();
                        self.last_search = Some(self.search_query.trim().to_string());
                        hub::spawn_search(
                            self.search_query.trim().to_string(),
                            self.hub_tx.clone(),
                        );
                    }
                    if self.search_pending {
                        ui.spinner();
                    }
                });
                ui.weak(
                    "Q4_K_M is the sweet spot for most machines — higher Q means better but \
                 bigger and slower, Q2 and below degrade noticeably. \"best pick\" marks \
                 the highest-quality quant that fits your RAM.",
                );
                if !self.search_pending
                    && self.search_results.is_empty()
                    && let Some(q) = &self.last_search
                {
                    ui.weak(format!(
                        "No GGUF repositories found for \"{q}\" — try fewer or different words."
                    ));
                }
                let results = self.search_results.clone();
                for repo in &results {
                    let open = egui::CollapsingHeader::new(format!(
                        "{}  ({} downloads)",
                        repo.id,
                        fmt_count(repo.downloads)
                    ))
                    .icon(theme::caret_icon)
                    .show(ui, |ui| {
                        match self.repo_files.get(&repo.id).cloned() {
                            Some((files, only_multipart)) => {
                                if files.is_empty() {
                                    if only_multipart {
                                        ui.weak(
                                            "This repo only contains multi-part models                                              (too large to download as a single file) —                                              offgrid can't use them.",
                                        );
                                    } else {
                                        ui.weak("No usable GGUF model files in this repo.");
                                    }
                                }
                                let best = files
                                    .iter()
                                    .filter(|f| {
                                        Fit::of(f.size, self.hardware.total_ram) == Fit::Fits
                                    })
                                    .min_by_key(|f| (models::quant_tag(&f.name).pref, f.size))
                                    .map(|f| f.name.clone());
                                egui::Grid::new(("repo_files", &repo.id))
                                    .num_columns(6)
                                    .spacing([16.0, 6.0])
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for f in &files {
                                            let tip = models::quant_tooltip(&f.name);
                                            // Bounded + truncating: unbounded
                                            // names widen the grid and push the
                                            // download button off screen.
                                            ui.scope(|ui| {
                                                ui.set_max_width(COL_NAME);
                                                ui.add(egui::Label::new(&f.name).truncate())
                                                    .on_hover_text(&tip);
                                            });
                                            ui.weak(fmt_bytes(f.size));
                                            ui.weak(models::fmt_tok_s(
                                                models::est_tokens_per_sec(
                                                    &f.name,
                                                    f.size,
                                                    self.hardware.mem_bandwidth,
                                                ),
                                            ));
                                            self.fit_badge(ui, f.size);
                                            ui.horizontal(|ui| {
                                                let tag = models::quant_tag(&f.name);
                                                if !tag.label.is_empty() {
                                                    ui.colored_label(tag.color, tag.label)
                                                        .on_hover_text(&tip);
                                                }
                                                if best.as_deref() == Some(f.name.as_str()) {
                                                    ui.strong("• best pick").on_hover_text(
                                                        "The highest-quality quant of this repo \
                                                     that fits your RAM.",
                                                    );
                                                }
                                            });
                                            if self.is_downloaded(&f.name) {
                                                ui.weak("downloaded");
                                            } else if self.is_downloading(&f.name) {
                                                ui.spinner();
                                            } else if Self::download_button(ui) {
                                                self.start_download(&repo.id, &f.name, f.size);
                                            }
                                            ui.end_row();
                                        }
                                    });
                            }
                            None => {
                                ui.spinner();
                            }
                        }
                    });
                    if open.body_response.is_some()
                        && !self.repo_files.contains_key(&repo.id)
                        && !self.files_pending.contains(&repo.id)
                    {
                        self.files_pending.insert(repo.id.clone());
                        hub::spawn_list_files(repo.id.clone(), self.hub_tx.clone());
                    }
                }
            });
        });
    }

    fn chat_ui(&mut self, ui: &mut egui::Ui) {
        if self.loaded_model.is_none() && !self.model_loading {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.weak("Load a model in the Models tab to start chatting.");
            });
            return;
        }

        egui::Panel::bottom("chat_input")
            .frame(egui::Frame::side_top_panel(ui.style()).inner_margin(8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if self.generating {
                        ui.spinner();
                        if let Some(start) = self.live_start {
                            let secs = start.elapsed().as_secs_f32().max(0.001);
                            ui.weak(format!(
                                "generating… {:.1} tok/s · {} tokens",
                                self.live_tokens as f32 / secs,
                                self.live_tokens
                            ));
                        }
                    } else if let Some(stats) = &self.gen_stats {
                        ui.weak(stats.clone());
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if !self.generating
                            && !self.messages.is_empty()
                            && theme::button(
                                ui,
                                Some((theme::icons().trash.clone(), 14.0)),
                                "Clear history",
                            )
                            .on_hover_text("Start a fresh conversation")
                            .clicked()
                        {
                            self.messages.clear();
                            self.chat_culler.clear();
                            self.chat_ctx_used = 0;
                        }
                        theme::context_meter(ui, self.chat_ctx_used, self.n_ctx() as usize);
                    });
                });
                let input_h = 60.0;
                ui.horizontal(|ui| {
                    let resp = ui.add_sized(
                        [ui.available_width() - 88.0, input_h],
                        egui::TextEdit::multiline(&mut self.input)
                            .desired_rows(2) // keep intrinsic height below add_sized's
                            .hint_text("Type a message… (Enter to send, Shift+Enter for newline)"),
                    );
                    let send_key = resp.has_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    // The button matches the input's full height.
                    let label = if self.generating { "Stop" } else { "Send" };
                    let action = ui.add_sized([80.0, input_h], egui::Button::new(label));
                    theme::gloss(ui, action.rect);
                    if self.generating {
                        if action.clicked() {
                            self.llm.stop.store(true, Ordering::Relaxed);
                        }
                    } else if action.clicked() || send_key {
                        self.send_chat();
                    }
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut culler = std::mem::take(&mut self.chat_culler);
                    culler.begin(ui, self.messages.len());
                    let len = self.messages.len();
                    for (i, msg) in self.messages.iter().enumerate() {
                        // The last messages may still stream — always render.
                        let hot = i + 2 >= len;
                        culler.row(ui, i, hot, |ui| {
                            let (label, color) = match msg.role {
                                Role::User => ("You", theme::skin().accent),
                                Role::Assistant => ("Model", theme::skin().good),
                                Role::System => ("System", egui::Color32::GRAY),
                            };
                            ui.colored_label(color, label);
                            if msg.content.is_empty() {
                                ui.label("…");
                            }
                            let mut memo = std::mem::take(&mut self.hl_memo);
                            render_message(ui, &mut self.md_cache, &mut memo, &msg.content, i);
                            self.hl_memo = memo;
                            ui.add_space(8.0);
                        });
                    }
                    self.chat_culler = culler;
                });
        });
    }

    fn opencode_snippet(&self) -> String {
        let models: serde_json::Map<String, serde_json::Value> = self
            .local_models
            .iter()
            .map(|m| (m.name.clone(), serde_json::json!({"name": m.name.clone()})))
            .collect();
        let snippet = serde_json::json!({
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "offgrid": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "offgrid (local)",
                    "options": {
                        "baseURL": format!("http://127.0.0.1:{}/v1", self.server_port())
                    },
                    "models": models
                }
            }
        });
        serde_json::to_string_pretty(&snippet).unwrap_or_default()
    }

    fn workspace_path(&self) -> Option<PathBuf> {
        let trimmed = self.workspace_input.trim();
        if trimmed.is_empty() {
            return None;
        }
        let path = PathBuf::from(shellexpand_home(trimmed));
        path.is_dir().then_some(path)
    }

    fn workspace_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Folder:");
            match &self.config.workspace {
                Some(p) => {
                    ui.monospace(p.display().to_string());
                }
                None => {
                    ui.weak("no folder selected");
                }
            }
            if theme::button(ui, Some((theme::icons().folder.clone(), 16.0)), "Browse…").clicked()
            {
                let start = self
                    .workspace_path()
                    .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
                    .unwrap_or_else(|| PathBuf::from("."));
                if let Some(dir) = rfd::FileDialog::new().set_directory(start).pick_folder() {
                    self.workspace_input = dir.display().to_string();
                    self.config.workspace = Some(dir);
                    self.config.save();
                }
            }
            match self.workspace_path() {
                Some(ws) => {
                    ui.colored_label(theme::skin().good, "✔");
                    if !ws.join("AGENTS.md").exists()
                        && theme::button(ui, None, "Create AGENTS.md")
                            .on_hover_text(
                                "A project instructions file the agent reads before every task",
                            )
                            .clicked()
                        && let Err(e) =
                            std::fs::write(ws.join("AGENTS.md"), agent::AGENTS_MD_TEMPLATE)
                    {
                        self.last_error = Some(format!("could not create AGENTS.md: {e}"));
                    }
                }
                None => {
                    if !self.workspace_input.trim().is_empty() {
                        ui.colored_label(theme::skin().bad, "not a folder");
                    }
                }
            }
        });
    }

    fn code_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(ws) = self.workspace_path() {
            // A valid workspace collapses to a single line; expand to change.
            egui::CollapsingHeader::new(format!("Workspace: {}", ws.display()))
                .id_salt("workspace_section")
                .icon(theme::caret_icon)
                .default_open(false)
                .show(ui, |ui| self.workspace_controls(ui));
            ui.add_space(4.0);
        } else {
            theme::group(ui, "Workspace", Some(theme::icons().code.clone()), |ui| {
                self.workspace_controls(ui);
            });
        }

        theme::group(ui, "Task", None, |ui| {
            let task_resp = ui.add(
                egui::TextEdit::multiline(&mut self.agent_task)
                    .hint_text(
                        "Describe a task, e.g. \"write a python script that prints the first \
                         20 primes, run it, and fix any errors\" (Enter to run, Shift+Enter \
                         for newline)",
                    )
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            let submit = task_resp.has_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
            ui.horizontal(|ui| {
                let running = self.agent_run.is_some();
                let ready = !running
                    && self.loaded_model.is_some()
                    && self.workspace_path().is_some()
                    && !self.agent_task.trim().is_empty();
                let run_resp = ui.add_enabled(ready, egui::Button::new("▶ Run"));
                theme::gloss(ui, run_resp.rect);
                if (run_resp.clicked() || submit) && ready {
                    let ws = self.workspace_path().unwrap();
                    let task = self.agent_task.trim().to_string();
                    self.agent_task = task.clone(); // drop the submit newline
                    self.agent_transcript.push(AgentItem::Task(task.clone()));
                    self.agent_current.clear();
                    self.live_tokens = 0;
                    self.live_start = None;
                    self.agent_run = Some(agent::start(
                        ws,
                        task,
                        self.llm.cmd_tx.clone(),
                        self.agent_auto_approve,
                        self.config.web_tools,
                        self.n_ctx(),
                    ));
                }
                if running {
                    if theme::button(ui, None, "Stop").clicked() {
                        if let Some(run) = &self.agent_run {
                            run.stop.store(true, Ordering::Relaxed);
                        }
                        self.llm.stop.store(true, Ordering::Relaxed);
                        if let Some((_, reply)) = self.agent_approval.take() {
                            let _ = reply.send(false);
                        }
                    }
                    ui.spinner();
                    if let Some(start) = self.live_start {
                        let secs = start.elapsed().as_secs_f32().max(0.001);
                        ui.weak(format!("{:.1} tok/s", self.live_tokens as f32 / secs));
                    }
                }
                if theme::checkbox(ui, &mut self.agent_auto_approve, "auto-approve commands")
                    .on_hover_text("Run shell commands without asking")
                    .changed()
                {
                    if let Some(run) = &self.agent_run {
                        run.auto_approve
                            .store(self.agent_auto_approve, Ordering::Relaxed);
                    }
                    // Turning it on also answers a prompt that is already open.
                    if self.agent_auto_approve
                        && let Some((_, reply)) = self.agent_approval.take()
                    {
                        let _ = reply.send(true);
                    }
                }
                if theme::checkbox(ui, &mut self.config.web_tools, "allow web tools")
                    .on_hover_text(
                        "Give the agent web_search and fetch_url. Fails gracefully when \
                         offline — the agent falls back to local knowledge.",
                    )
                    .changed()
                {
                    self.config.save();
                }
                if !self.agent_transcript.is_empty()
                    && !running
                    && theme::button(
                        ui,
                        Some((theme::icons().trash.clone(), 14.0)),
                        "Clear history",
                    )
                    .on_hover_text("Clear the task transcript")
                    .clicked()
                {
                    self.agent_transcript.clear();
                    self.agent_culler.clear();
                    self.agent_ctx_used = 0;
                }
                theme::context_meter(ui, self.agent_ctx_used, self.n_ctx() as usize);
            });
            if self.loaded_model.is_none() {
                ui.colored_label(theme::skin().warn, "Load a model in the Models tab first.");
            }
        });

        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut approve_clicked: Option<bool> = None;
                let mut culler = std::mem::take(&mut self.agent_culler);
                culler.begin(ui, self.agent_transcript.len());
                let len = self.agent_transcript.len();
                for (i, item) in self.agent_transcript.iter().enumerate() {
                    // Recent rows can still change (streaming, tool results).
                    let hot = i + 3 >= len;
                    culler.row(ui, i, hot, |ui| match item {
                        AgentItem::Task(t) => {
                            ui.colored_label(theme::skin().accent, "Task");
                            ui.label(t);
                            ui.add_space(6.0);
                        }
                        AgentItem::Assistant(text) => {
                            ui.colored_label(theme::skin().good, "Model");
                            let mut memo = std::mem::take(&mut self.hl_memo);
                            render_message(ui, &mut self.md_cache, &mut memo, text, i);
                            self.hl_memo = memo;
                            ui.add_space(6.0);
                        }
                        AgentItem::Tool {
                            name,
                            summary,
                            output,
                            ok,
                        } => {
                            ui.horizontal(|ui| {
                                theme::icon(ui, tool_icon(name), 22.0);
                                ui.strong(name);
                                ui.weak(summary);
                                match ok {
                                    Some(true) => {
                                        ui.colored_label(theme::skin().good, "\u{2714}");
                                    }
                                    Some(false) => {
                                        ui.colored_label(theme::skin().bad, "\u{2716} failed");
                                    }
                                    None => {}
                                }
                            });
                            if let Some(out) = output {
                                if out.lines().count() > 5 {
                                    egui::CollapsingHeader::new("output")
                                        .id_salt(("tool_output", i))
                                        .icon(theme::caret_icon)
                                        .default_open(false)
                                        .show(ui, |ui| {
                                            ui.monospace(out);
                                        });
                                } else {
                                    ui.monospace(out);
                                }
                            }
                            ui.add_space(4.0);
                        }
                        AgentItem::Info(text) => {
                            ui.weak(text);
                            ui.add_space(4.0);
                        }
                    });
                }
                self.agent_culler = culler;
                if !self.agent_current.is_empty() {
                    ui.colored_label(theme::skin().good, "Model");
                    let mut memo = std::mem::take(&mut self.hl_memo);
                    render_message(
                        ui,
                        &mut self.md_cache,
                        &mut memo,
                        &self.agent_current,
                        usize::MAX,
                    );
                    self.hl_memo = memo;
                }
                if let Some((command, _)) = &self.agent_approval {
                    egui::Frame::new()
                        .stroke(egui::Stroke::new(1.0, theme::skin().warn))
                        .corner_radius(egui::CornerRadius::same(3))
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                theme::icon(ui, theme::icons().code.clone(), 22.0);
                                ui.strong("The agent wants to run a command:");
                            });
                            ui.monospace(command);
                            ui.horizontal(|ui| {
                                if theme::button(ui, None, "Approve").clicked() {
                                    approve_clicked = Some(true);
                                }
                                if theme::button(ui, None, "Deny").clicked() {
                                    approve_clicked = Some(false);
                                }
                            });
                        });
                }
                if let Some(answer) = approve_clicked
                    && let Some((_, reply)) = self.agent_approval.take()
                {
                    let _ = reply.send(answer);
                }
            });
    }

    fn start_bridge(&mut self) {
        if self.bridge.is_some() || self.config.bridge_token.trim().is_empty() {
            return;
        }
        self.bridge = Some(bridge::start(
            self.config.bridge_token.trim().to_string(),
            self.config.bridge_allowed.clone(),
            self.llm.cmd_tx.clone(),
            self.loaded_model_shared.clone(),
            self.n_ctx(),
            self.config.workspace.clone(),
            self.config.web_tools,
            self.config.bridge_code,
        ));
    }

    /// Restart the worker so a changed token or allowlist takes effect.
    fn restart_bridge(&mut self) {
        if let Some(b) = self.bridge.take() {
            b.stop();
        }
        if self.config.bridge_enabled {
            self.start_bridge();
        }
    }

    fn bridge_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(
            ui,
            "Telegram bridge",
            Some(theme::icons().chat.clone()),
            |ui| {
                ui.label(
                    "Chat with the loaded model from your phone. The model still runs \
                     here — but messages travel through Telegram's servers, so this is \
                     the one part of offgrid that is not offline.",
                );
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Bot token:");
                    let field = egui::TextEdit::singleline(&mut self.config.bridge_token)
                        .password(true)
                        .hint_text("from @BotFather")
                        .desired_width(260.0);
                    if ui.add(field).lost_focus() {
                        self.config.save();
                        if self.config.bridge_enabled {
                            self.restart_bridge();
                        }
                    }
                });

                let mut enabled = self.config.bridge_enabled;
                if theme::checkbox(ui, &mut enabled, "Enable bridge").changed() {
                    self.config.bridge_enabled = enabled;
                    self.config.save();
                    self.restart_bridge();
                }

                let mut code = self.config.bridge_code;
                if theme::checkbox(ui, &mut code, "Allow /code (agent runs)").changed() {
                    self.config.bridge_code = code;
                    self.config.save();
                    self.restart_bridge();
                }
                if self.config.bridge_code {
                    ui.colored_label(
                        theme::skin().warn,
                        "Approved chats can run the coding agent in your workspace, \
                         executing shell commands with auto-approve.",
                    );
                    match &self.config.workspace {
                        Some(w) => ui.weak(format!("workspace: {}", w.display())),
                        None => ui.colored_label(
                            theme::skin().warn,
                            "No workspace set — pick one in the Code tab first.",
                        ),
                    };
                }

                if let Some(b) = &self.bridge {
                    let status = b.status.lock().unwrap().clone();
                    let color = if status == "connected" {
                        theme::skin().good
                    } else {
                        theme::skin().warn
                    };
                    ui.colored_label(color, format!("• {status}"));
                    // Unknown senders queue up here for one-click approval.
                    let pending: Vec<(i64, String)> = b.pending.lock().unwrap().clone();
                    let mut allow: Option<i64> = None;
                    for (id, from) in &pending {
                        ui.horizontal(|ui| {
                            ui.label(format!("{from} ({id}) wants to chat"));
                            if theme::button(ui, None, "Allow").clicked() {
                                allow = Some(*id);
                            }
                        });
                    }
                    if let Some(id) = allow {
                        self.config.bridge_allowed.push(id);
                        self.config.save();
                        if let Some(b) = &self.bridge {
                            b.pending.lock().unwrap().retain(|(i, _)| *i != id);
                        }
                        self.restart_bridge();
                    }
                }

                if self.config.bridge_allowed.is_empty() {
                    ui.weak("No chats allowed yet — message the bot once and approve it here.");
                } else {
                    let mut remove: Option<i64> = None;
                    for id in self.config.bridge_allowed.clone() {
                        ui.horizontal(|ui| {
                            ui.monospace(format!("chat {id}"));
                            if theme::button(ui, Some((theme::icons().trash.clone(), 14.0)), "")
                                .clicked()
                            {
                                remove = Some(id);
                            }
                        });
                    }
                    if let Some(id) = remove {
                        self.config.bridge_allowed.retain(|i| *i != id);
                        self.config.save();
                        self.restart_bridge();
                    }
                }
            },
        );
    }

    fn serve_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(ui, "API server", Some(theme::icons().serve.clone()), |ui| {
            ui.label(
                "Expose your local models over an OpenAI-compatible API so other tools \
                 (opencode, aider, editors, scripts) can use them — still fully local.",
            );
            ui.add_space(4.0);

            let mut enabled = self.config.server_enabled;
            if theme::checkbox(ui, &mut enabled, "Enable server").changed() {
                self.config.server_enabled = enabled;
                if enabled {
                    self.start_server();
                } else if let Some(s) = self.api_server.take() {
                    s.stop();
                }
                self.config.save();
            }

            let mut lan = self.config.server_lan;
            if theme::checkbox(ui, &mut lan, "Allow LAN access").changed() {
                self.config.server_lan = lan;
                self.config.save();
                if let Some(s) = self.api_server.take() {
                    // Rebind on the new address; give the old listener a
                    // moment to release the port (its accept loop ticks
                    // every 200ms).
                    s.stop();
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    self.start_server();
                }
            }
            if self.config.server_lan {
                ui.colored_label(
                    theme::skin().warn,
                    "Anyone on your network can use the model, read agent session logs, \
                     and start agent runs that execute shell commands with auto-approve.",
                );
            }

            if self.api_server.is_some() {
                ui.horizontal(|ui| {
                    ui.colored_label(theme::skin().good, "• running");
                    let host = if self.config.server_lan {
                        self.lan_ip.clone().unwrap_or_else(|| "0.0.0.0".into())
                    } else {
                        "127.0.0.1".into()
                    };
                    ui.monospace(format!("http://{host}:{}/v1", self.server_port()));
                });
                if self.loaded_model.is_none() {
                    ui.colored_label(
                        theme::skin().warn,
                        "No model loaded — requests will fail until you load one.",
                    );
                }
            }
        });

        self.bridge_ui(ui);

        theme::group(ui, "opencode setup", None, |ui| {
            ui.label(
                "Add this to opencode.json (globally in ~/.config/opencode/, or per project), \
                 then pick a model from the 'offgrid (local)' provider:",
            );
            let snippet = self.opencode_snippet();
            ui.horizontal(|ui| {
                if theme::button(ui, None, "Copy").clicked() {
                    ui.ctx().copy_text(snippet.clone());
                }
                ui.weak("Works the same for any tool that accepts an OpenAI-compatible base URL.");
            });
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    CommonMarkViewer::new().show(
                        ui,
                        &mut self.md_cache,
                        &format!("```json\n{snippet}\n```"),
                    );
                });
        });
    }

    fn modals(&mut self, ctx: &egui::Context) {
        if let Some(model) = self.confirm_delete.clone() {
            egui::Window::new("Delete model?")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().trash.clone(), 24.0);
                        ui.label(format!(
                            "Permanently delete {} ({})?",
                            model.name,
                            fmt_bytes(model.size)
                        ));
                    });
                    ui.horizontal(|ui| {
                        if theme::button(ui, None, "Delete").clicked() {
                            if self.loaded_model.as_deref() == Some(model.name.as_str()) {
                                let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
                            }
                            if let Err(e) = std::fs::remove_file(&model.path) {
                                self.last_error = Some(format!("delete failed: {e}"));
                            }
                            self.rescan();
                            self.confirm_delete = None;
                        }
                        if theme::button(ui, None, "Cancel").clicked() {
                            self.confirm_delete = None;
                        }
                    });
                });
        }
    }
}

impl eframe::App for OffgridApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        let ctx = ui.ctx().clone();

        egui::Panel::top("top")
            .show_separator_line(false)
            .show(ui, |ui| {
                self.top_bar(ui);
            });

        if let Some(err) = self.last_error.clone() {
            egui::Panel::bottom("error_bar").show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(theme::skin().bad, &err);
                    if ui.small_button("✕").clicked() {
                        self.last_error = None;
                    }
                });
            });
        }

        self.modals(&ctx);

        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Models => self.models_ui(ui),
            Tab::Chat => self.chat_ui(ui),
            Tab::Code => self.code_ui(ui),
            Tab::Serve => self.serve_ui(ui),
            Tab::Settings => self.settings_ui(ui),
        });

        let busy = self.generating
            || self.model_loading
            || self.search_pending
            || self.agent_run.is_some()
            || !self.downloads.is_empty()
            || !self.files_pending.is_empty();
        if busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

const COL_NAME: f32 = 340.0;
const COL_SIZE: f32 = 80.0;
const COL_TOKS: f32 = 70.0;
const COL_BADGE: f32 = 60.0;
const ROW_H: f32 = 26.0;

/// A model-list row with fixed-width columns (shared between "On disk" and
/// "Get models" so the two tables line up) and right-aligned actions.
fn list_row(
    ui: &mut egui::Ui,
    stripe: bool,
    name: impl FnOnce(&mut egui::Ui),
    size: u64,
    est: String,
    badge: (&'static str, egui::Color32),
    actions: impl FnOnce(&mut egui::Ui),
) {
    let fill = if stripe {
        ui.visuals().faint_bg_color
    } else {
        egui::Color32::TRANSPARENT
    };
    egui::Frame::new()
        .fill(fill)
        .inner_margin(egui::Margin::symmetric(4, 2))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let cell = egui::Layout::left_to_right(egui::Align::Center);
                ui.allocate_ui_with_layout(egui::vec2(COL_NAME, ROW_H), cell, |ui| {
                    ui.set_width(COL_NAME);
                    name(ui);
                });
                ui.allocate_ui_with_layout(egui::vec2(COL_SIZE, ROW_H), cell, |ui| {
                    ui.set_width(COL_SIZE);
                    ui.weak(fmt_bytes(size));
                });
                ui.allocate_ui_with_layout(egui::vec2(COL_TOKS, ROW_H), cell, |ui| {
                    ui.set_width(COL_TOKS);
                    ui.weak(est).on_hover_text(
                        "Estimated generation speed on this machine \
                         (from measured memory bandwidth)",
                    );
                });
                let (label, color) = badge;
                ui.allocate_ui_with_layout(egui::vec2(COL_BADGE, ROW_H), cell, |ui| {
                    ui.set_width(COL_BADGE);
                    ui.colored_label(color, label);
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), actions);
            });
        });
}

fn fmt_eta(secs: f32) -> String {
    if !secs.is_finite() || secs > 359_999.0 {
        return "—".into();
    }
    let s = secs as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02} left", s / 3600, (s % 3600) / 60, s % 60)
    } else {
        format!("{}:{:02} left", s / 60, s % 60)
    }
}

/// Bytes/second shown as a line rate.
fn fmt_bitrate(bytes_per_sec: f32) -> String {
    let mbit = bytes_per_sec * 8.0 / 1_000_000.0;
    if mbit >= 1.0 {
        format!("{mbit:.1} Mbit/s")
    } else {
        format!("{:.0} kbit/s", mbit * 1000.0)
    }
}

/// Repo files may live in subfolders; local files are always flat.
fn file_basename(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

fn fmt_count(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1_000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

fn shellexpand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return Path::new(&home).join(rest).display().to_string();
    }
    path.to_string()
}

enum Segment<'a> {
    Text(&'a str),
    Think(&'a str),
    ToolCall(&'a str),
}

/// Split message content into normal text, `<think>…</think>` and
/// `<tool_call>…</tool_call>` segments. An unclosed tag (mid-stream) claims
/// the rest of the text.
fn split_segments(s: &str) -> Vec<Segment<'_>> {
    const TAGS: [(&str, &str); 2] = [("<think>", "</think>"), ("<tool_call>", "</tool_call>")];
    let mut out = Vec::new();
    let mut rest = s;
    loop {
        let next = TAGS
            .iter()
            .enumerate()
            .filter_map(|(i, (open, _))| rest.find(open).map(|pos| (pos, i)))
            .min();
        let Some((start, tag)) = next else { break };
        let (open, close) = TAGS[tag];
        let make = if tag == 0 {
            Segment::Think
        } else {
            Segment::ToolCall
        };
        if start > 0 {
            out.push(Segment::Text(&rest[..start]));
        }
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => {
                out.push(make(&after[..end]));
                rest = &after[end + close.len()..];
            }
            None => {
                out.push(make(after));
                rest = "";
                break;
            }
        }
    }
    if !rest.is_empty() {
        out.push(Segment::Text(rest));
    }
    out
}

/// Render one chat/agent message: markdown text, think blocks as quotes,
/// tool calls as pretty-printed JSON code blocks.
/// Session-lifetime cache of highlighted code. egui's FrameCache evicts
/// entries as soon as a block scrolls out of view, so scrolling back would
/// re-run syntect from scratch (brutal in debug builds). This memo keeps
/// every block highlighted exactly once.
#[derive(Default)]
struct HighlightMemo {
    map: HashMap<u64, std::sync::Arc<egui::text::LayoutJob>>,
}

impl HighlightMemo {
    fn job(
        &mut self,
        ui: &egui::Ui,
        code: &str,
        lang: &str,
    ) -> std::sync::Arc<egui::text::LayoutJob> {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        theme::kind().id().hash(&mut h);
        lang.hash(&mut h);
        code.hash(&mut h);
        let key = h.finish();
        if self.map.len() > 512 {
            self.map.clear();
        }
        self.map
            .entry(key)
            .or_insert_with(|| {
                let theme = egui_extras::syntax_highlighting::CodeTheme::from_style(ui.style());
                std::sync::Arc::new(egui_extras::syntax_highlighting::highlight(
                    ui.ctx(),
                    ui.style(),
                    &theme,
                    code,
                    lang,
                ))
            })
            .clone()
    }
}

/// Code block rendered from the persistent highlight memo.
fn cached_code_block(ui: &mut egui::Ui, memo: &mut HighlightMemo, code: &str, lang: &str) {
    let job = memo.job(ui, code, lang);
    egui::Frame::new()
        .fill(ui.visuals().extreme_bg_color)
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let mut job = (*job).clone();
            job.wrap.max_width = ui.available_width();
            ui.add(egui::Label::new(job));
        });
}

/// Heuristic for a tool call the model emitted as bare JSON, possibly still
/// streaming in: a JSON object mentioning "name" and "arguments".
fn looks_like_tool_json(t: &str) -> bool {
    t.starts_with('{') && t.contains("\"name\"") && t.contains("\"arguments\"")
}

fn render_message(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    memo: &mut HighlightMemo,
    text: &str,
    seed: usize,
) {
    // Position-based salt: identical tool calls (e.g. the same write_file
    // resent after a rejected overwrite) must still get distinct widget IDs.
    let mut block = 0usize;
    for segment in split_segments(text) {
        match segment {
            Segment::Text(t) => {
                let trimmed = t.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Bare tool-call JSON (no <tool_call> wrapper) must not go
                // through markdown: it eats the escapes and mangles the code.
                if looks_like_tool_json(trimmed) {
                    render_tool_call_block(ui, cache, memo, trimmed, (seed, block));
                    block += 1;
                } else {
                    CommonMarkViewer::new().show(ui, cache, t);
                }
            }
            Segment::Think(t) => {
                let t = t.trim();
                if !t.is_empty() {
                    render_think_block(ui, t);
                }
            }
            Segment::ToolCall(t) => {
                let t = t.trim();
                if !t.is_empty() {
                    render_tool_call_block(ui, cache, memo, t, (seed, block));
                    block += 1;
                }
            }
        }
    }
}

/// Render a `<tool_call>` for humans: a long `content` argument (write_file)
/// is pulled out of the JSON and shown as its own code block with real
/// newlines, highlighted by the target file's extension.
fn render_tool_call_block(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    memo: &mut HighlightMemo,
    t: &str,
    salt: (usize, usize),
) {
    let parsed = serde_json::from_str::<serde_json::Value>(t).or_else(|_| {
        serde_json::from_str::<serde_json::Value>(&agent::escape_control_chars_in_strings(t))
    });
    match parsed {
        Ok(mut v) => {
            let lang = v["arguments"]["path"]
                .as_str()
                .and_then(|p| p.rsplit('.').next())
                .unwrap_or("")
                .to_string();
            let content = v["arguments"]
                .as_object_mut()
                .and_then(|args| args.remove("content"))
                .and_then(|c| c.as_str().map(String::from));
            let head = serde_json::to_string_pretty(&v).unwrap_or_else(|_| t.to_string());
            CommonMarkViewer::new().show(ui, cache, &format!("```json\n{head}\n```"));
            if let Some(content) = content {
                let lines = content.lines().count();
                if lines > 30 {
                    // Big blocks collapse: syntax highlighting is expensive
                    // and re-runs every frame while a block is visible.
                    egui::CollapsingHeader::new(format!("file content ({lines} lines)"))
                        .id_salt(("tc_content", salt))
                        .icon(theme::caret_icon)
                        .default_open(false)
                        .show(ui, |ui| {
                            cached_code_block(ui, memo, &content, &lang);
                        });
                } else {
                    cached_code_block(ui, memo, &content, &lang);
                }
            }
        }
        Err(_) => {
            // Mid-stream, the JSON is incomplete — show it raw, but unescape
            // the common sequences so code stays readable while it streams.
            let display = t
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"");
            CommonMarkViewer::new().show(ui, cache, &format!("````json\n{display}\n````"));
        }
    }
}

/// Reasoning block, styled like a quote: gray bar on the left, italic gray text.
fn render_think_block(ui: &mut egui::Ui, text: &str) {
    let response = egui::Frame::new()
        .inner_margin(egui::Margin {
            left: 12,
            right: 4,
            top: 4,
            bottom: 4,
        })
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .italics()
                    .color(ui.visuals().weak_text_color()),
            );
        })
        .response;
    let rect = response.rect;
    ui.painter().rect_filled(
        egui::Rect::from_min_max(rect.min, egui::pos2(rect.min.x + 3.0, rect.max.y)),
        1.0,
        ui.visuals().weak_text_color().gamma_multiply(0.5),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEMO_RAM: u64 = 32 * 1024 * 1024 * 1024;
    const DEMO_BW: u64 = 22_000_000_000;

    /// A very long model name must not widen the search-results grid and push
    /// the download button off screen — it truncates with an ellipsis instead.
    #[test]
    fn long_model_name_truncates_in_search_grid() {
        let long = "unsloth_Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL-with-a-silly-long-name.gguf";
        let seen = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let recorder = seen.clone();
        let mut harness = egui_kittest::Harness::new_ui(move |ui| {
            egui::Grid::new("g").num_columns(2).show(ui, |ui| {
                ui.scope(|ui| {
                    ui.set_max_width(COL_NAME);
                    ui.add(egui::Label::new(long).truncate());
                });
                let x = ui.button("Download").rect.min.x;
                recorder.store(x as u32, std::sync::atomic::Ordering::Relaxed);
                ui.end_row();
            });
        });
        harness.run();
        let button_x = seen.load(std::sync::atomic::Ordering::Relaxed) as f32;
        assert!(
            button_x > 0.0 && button_x < COL_NAME + 40.0,
            "download button at x={button_x}, expected within the {COL_NAME}px name column"
        );
    }

    /// Wrap the demo screen in a faked Haiku desktop: blue backdrop, a window
    /// with the yellow title tab, border and drop shadow — for a README
    /// screenshot that looks like a real desktop capture.
    fn desktop_ui(ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        theme::apply(&ctx);
        egui_extras::install_image_loaders(&ctx);

        let desktop = ui.max_rect();
        ui.painter()
            .rect_filled(desktop, 0.0, egui::Color32::from_rgb(51, 102, 152));

        let margin = 46.0;
        let tab_h = 30.0;
        let win = egui::Rect::from_min_max(
            desktop.min + egui::vec2(margin, margin + tab_h),
            desktop.max - egui::vec2(margin, margin),
        );

        // Haiku window tab: yellow, rounded top, close box + bold title.
        let tab = egui::Rect::from_min_size(
            egui::pos2(win.min.x, win.min.y - tab_h + 1.0),
            egui::vec2(170.0, tab_h),
        );
        let tab_stroke = egui::Stroke::new(1.0, theme::skin().title_border);
        ui.painter().rect(
            tab,
            egui::CornerRadius {
                nw: 4,
                ne: 4,
                sw: 0,
                se: 0,
            },
            theme::skin().title,
            tab_stroke,
            egui::StrokeKind::Inside,
        );
        let close = egui::Rect::from_center_size(
            egui::pos2(tab.min.x + 18.0, tab.center().y),
            egui::vec2(13.0, 13.0),
        );
        ui.painter().rect(
            close,
            2.0,
            egui::Color32::from_rgb(255, 226, 100),
            tab_stroke,
            egui::StrokeKind::Inside,
        );
        ui.painter().text(
            egui::pos2(tab.min.x + 34.0, tab.center().y),
            egui::Align2::LEFT_CENTER,
            "offgrid",
            egui::FontId::proportional(15.0),
            egui::Color32::BLACK,
        );

        let frame = egui::Frame::new()
            .fill(theme::skin().panel)
            .stroke(egui::Stroke::new(1.0, theme::skin().window_border))
            .shadow(egui::Shadow {
                offset: [4, 6],
                blur: 18,
                spread: 0,
                color: egui::Color32::from_black_alpha(110),
            });
        ui.scope_builder(egui::UiBuilder::new().max_rect(win), |ui| {
            frame.show(ui, |ui| {
                ui.set_min_size(win.size() - egui::vec2(16.0, 16.0));
                demo_ui(ui);
            });
        });
    }

    /// A deterministic replica of the main screen (canned data, no threads,
    /// no config/disk access) rendered with the real theme, icons and widgets.
    fn demo_ui(ui: &mut egui::Ui) {
        egui::Panel::top("top")
            .show_separator_line(false)
            .show(ui, |ui| {
                ui.add_space(14.0);
                let mut tab = Tab::Models;
                theme::tab_bar(
                    ui,
                    &mut tab,
                    &[
                        (Tab::Models, theme::icons().models.clone(), "Models"),
                        (Tab::Chat, theme::icons().chat.clone(), "Chat"),
                        (Tab::Code, theme::icons().code.clone(), "Code"),
                        (Tab::Serve, theme::icons().serve.clone(), "Serve"),
                        (Tab::Settings, theme::icons().settings.clone(), "Settings"),
                    ],
                );
            });

        egui::CentralPanel::default().show(ui, |ui| {
            theme::group(
                ui,
                "Current model",
                Some(theme::icons().model.clone()),
                |ui| {
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().model.clone(), 18.0);
                        ui.label("Qwen_Qwen3-4B-Instruct-2507-Q4_K_M");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let _ = theme::button(ui, None, "Unload");
                        });
                    });
                },
            );
            // Fixed free-space figure: the real one varies per machine and
            // would make the snapshot test non-deterministic.
            theme::group(
                ui,
                "On disk  (412.7 GB free)",
                Some(theme::icons().disk.clone()),
                |ui| {
                    let rows: [(&str, u64, bool); 3] = [
                        ("Qwen3-0.6B-Q4_K_M", 396_705_472, false),
                        ("Qwen_Qwen3-4B-Instruct-2507-Q4_K_M", 2_497_280_736, true),
                        ("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", 18_556_689_568, false),
                    ];
                    for (i, (name, size, loaded)) in rows.into_iter().enumerate() {
                        list_row(
                            ui,
                            i % 2 == 1,
                            |ui| {
                                theme::icon(ui, theme::icons().disk.clone(), 16.0);
                                ui.add(egui::Label::new(name).truncate());
                                if loaded {
                                    ui.colored_label(theme::skin().good, "•");
                                }
                            },
                            size,
                            models::fmt_tok_s(models::est_tokens_per_sec(name, size, DEMO_BW)),
                            Fit::of(size, DEMO_RAM).badge(),
                            |ui| {
                                let _ = theme::button(
                                    ui,
                                    Some((theme::icons().trash.clone(), 18.0)),
                                    "Delete",
                                );
                                let load = ui.add_enabled(
                                    !loaded,
                                    egui::Button::new("Load").min_size(egui::vec2(60.0, 0.0)),
                                );
                                theme::gloss(ui, load.rect);
                            },
                        );
                    }
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().download.clone(), 16.0);
                        ui.add(egui::Label::new("Qwen3.8-27B-UD-Q4_K_M.gguf").truncate());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.weak("9.87 GB / 15.30 GB · 65.6 Mbit/s · 11:04 left");
                        });
                    });
                    theme::progress_bar(ui, 0.645);
                },
            );

            theme::group(ui, "Get models", Some(theme::icons().depot.clone()), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Recommended for your hardware:");
                    ui.strong("Qwen3 Coder 30B-A3B (Q4_K_M)");
                });
                ui.separator();
                let rows: [(&str, u64); 3] = [
                    ("Qwen3 1.7B (Q4_K_M)", 1_107_409_472),
                    ("Gemma 3 4B Instruct (Q4_K_M)", 2_489_758_112),
                    ("Mistral 7B Instruct v0.3 (Q4_K_M)", 4_372_812_000),
                ];
                for (i, (name, size)) in rows.into_iter().enumerate() {
                    list_row(
                        ui,
                        i % 2 == 1,
                        |ui| {
                            ui.add(egui::Label::new(name).truncate());
                        },
                        size,
                        models::fmt_tok_s(models::est_tokens_per_sec(name, size, DEMO_BW)),
                        Fit::of(size, DEMO_RAM).badge(),
                        |ui| {
                            let _ = theme::button(
                                ui,
                                Some((theme::icons().download.clone(), 22.0)),
                                "Download",
                            );
                        },
                    );
                }
            });
        });
    }

    #[test]
    fn main_screen_snapshot() {
        let mut harness = egui_kittest::Harness::builder()
            .with_size(egui::vec2(1000.0, 700.0))
            .build_ui(desktop_ui);
        harness.run();
        harness.snapshot("offgrid");
    }
}
