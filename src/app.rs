use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use crate::config::{Config, models_dir};
use crate::hardware::{HardwareProfile, fmt_bytes};
use crate::hub::{self, ActiveDownload, DownloadEvent, HubEvent, RepoFile, RepoResult};
use crate::llm::{self, ChatMessage, LlmCmd, LlmEvent, LlmHandle, Role};
use crate::agent::{self, AgentEvent, AgentRun};
use crate::models::{self, Fit, LocalModel};
use crate::server::{self, ApiServer};
use crate::theme;

const ICON_LOGO: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/App_Haiku3d.png");
const ICON_DISK: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Device_Harddisk.png");
const ICON_CHAT: egui::ImageSource<'static> = egui::include_image!("../assets/icons/App_Chat.png");
const ICON_SERVE: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Server_Net.png");
const ICON_DOWNLOAD: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Action_Download.png");
const ICON_SEARCH: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Action_Search.png");
const ICON_TRASH: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Trash_Empty.png");
const ICON_DEPOT: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/App_HaikuDepot.png");
const ICON_CODE: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/App_Terminal.png");

#[derive(PartialEq)]
enum Tab {
    Models,
    Chat,
    Code,
    Serve,
}

enum AgentItem {
    Task(String),
    Assistant(String),
    Tool {
        name: String,
        summary: String,
        output: Option<String>,
    },
    Info(String),
}

pub struct OffgridApp {
    hardware: HardwareProfile,
    config: Config,
    tab: Tab,
    models_dir: PathBuf,
    local_models: Vec<LocalModel>,

    // Hub browsing
    hub_tx: Sender<HubEvent>,
    hub_rx: Receiver<HubEvent>,
    search_query: String,
    search_pending: bool,
    search_results: Vec<RepoResult>,
    repo_files: HashMap<String, Vec<RepoFile>>,
    files_pending: HashSet<String>,
    downloads: Vec<ActiveDownload>,

    // LLM
    llm: LlmHandle,
    loaded_model: Option<String>,
    // Same value, shared with the API server thread.
    loaded_model_shared: Arc<Mutex<Option<String>>>,
    model_loading: bool,

    // API server for external tools (opencode etc.)
    api_server: Option<ApiServer>,

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
}

impl OffgridApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let hardware = HardwareProfile::detect();
        let config = Config::load();
        let models_dir = models_dir();
        let _ = std::fs::create_dir_all(&models_dir);
        let (hub_tx, hub_rx) = std::sync::mpsc::channel();
        let llm = llm::spawn_worker(hardware.cores);

        let mut model_loading = false;
        if let Some(last) = &config.last_model {
            if last.exists() {
                let _ = llm.cmd_tx.send(LlmCmd::Load(last.clone()));
                model_loading = true;
            }
        }

        let loaded_model_shared = Arc::new(Mutex::new(None));
        let workspace_input = config
            .workspace
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let mut app = Self {
            local_models: models::scan_local(&models_dir),
            hardware,
            config,
            tab: Tab::Models,
            models_dir,
            hub_tx,
            hub_rx,
            search_query: String::new(),
            search_pending: false,
            search_results: Vec::new(),
            repo_files: HashMap::new(),
            files_pending: HashSet::new(),
            downloads: Vec::new(),
            llm,
            loaded_model: None,
            loaded_model_shared,
            model_loading,
            api_server: None,
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
        };
        if app.config.server_enabled {
            app.start_server();
        }
        app
    }

    fn server_port(&self) -> u16 {
        self.config.server_port.unwrap_or(server::DEFAULT_PORT)
    }

    fn start_server(&mut self) {
        if self.api_server.is_some() {
            return;
        }
        match server::start(
            self.server_port(),
            self.llm.cmd_tx.clone(),
            self.models_dir.clone(),
            self.loaded_model_shared.clone(),
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
    }

    fn drain_events(&mut self) {
        loop {
            match self.hub_rx.try_recv() {
                Ok(HubEvent::SearchResults(results)) => {
                    self.search_results = results;
                    self.search_pending = false;
                }
                Ok(HubEvent::Files { repo, files }) => {
                    self.files_pending.remove(&repo);
                    self.repo_files.insert(repo, files);
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
                        dl.bytes = u64::MAX;
                        finished = true;
                        self.last_error = Some(format!("download of {} failed: {e}", dl.file));
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
                        self.live_tokens += 1;
                        if self.live_start.is_none() {
                            self.live_start = Some(std::time::Instant::now());
                        }
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
                        });
                    }
                    AgentEvent::ToolResult { output } => {
                        if let Some(AgentItem::Tool { output: slot, .. }) =
                            self.agent_transcript.last_mut()
                        {
                            *slot = Some(output);
                        }
                    }
                    AgentEvent::NeedsApproval { command, reply } => {
                        self.agent_approval = Some((command, reply));
                    }
                    AgentEvent::Done { iterations } => {
                        self.agent_transcript
                            .push(AgentItem::Info(format!("finished after {iterations} turn(s)")));
                        self.agent_run = None;
                        self.agent_approval = None;
                        self.llm.stop.store(false, Ordering::Relaxed);
                    }
                    AgentEvent::Error(e) => {
                        self.agent_transcript
                            .push(AgentItem::Info(format!("error: {e}")));
                        self.agent_run = None;
                        self.agent_approval = None;
                        self.llm.stop.store(false, Ordering::Relaxed);
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
                    self.live_tokens += 1;
                    if self.live_start.is_none() {
                        self.live_start = Some(std::time::Instant::now());
                    }
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == Role::Assistant {
                            last.content.push_str(&text);
                        }
                    }
                }
                Ok(LlmEvent::Stats {
                    prompt_tokens,
                    prompt_secs,
                    gen_tokens,
                    gen_secs,
                }) => {
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
                    self.last_error = Some(e);
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
    }

    fn is_downloaded(&self, file: &str) -> bool {
        self.models_dir.join(file).exists()
    }

    fn is_downloading(&self, file: &str) -> bool {
        self.downloads.iter().any(|d| d.file == file)
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
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            theme::icon(ui, ICON_LOGO, 22.0);
            ui.heading(egui::RichText::new("offgrid").strong());
            ui.separator();
            ui.label(format!(
                "{} · {} cores · {} RAM",
                self.hardware.cpu_brand,
                self.hardware.cores,
                fmt_bytes(self.hardware.total_ram)
            ));
            ui.separator();
            if self.model_loading {
                ui.spinner();
                ui.label("loading model…");
            } else if let Some(name) = &self.loaded_model {
                ui.colored_label(theme::GOOD_GREEN, name);
                if ui.small_button("Unload").clicked() {
                    let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
                }
            } else {
                ui.weak("no model loaded");
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            if theme::tab(ui, ICON_DISK, "Models", self.tab == Tab::Models).clicked() {
                self.tab = Tab::Models;
            }
            if theme::tab(ui, ICON_CHAT, "Chat", self.tab == Tab::Chat).clicked() {
                self.tab = Tab::Chat;
            }
            if theme::tab(ui, ICON_CODE, "Code", self.tab == Tab::Code).clicked() {
                self.tab = Tab::Code;
            }
            if theme::tab(ui, ICON_SERVE, "Serve", self.tab == Tab::Serve).clicked() {
                self.tab = Tab::Serve;
            }
        });
    }

    fn fit_badge(&self, ui: &mut egui::Ui, size: u64) {
        let (label, color) = Fit::of(size, self.hardware.total_ram).badge();
        ui.colored_label(color, label);
    }

    fn download_button(ui: &mut egui::Ui) -> bool {
        ui.add(egui::Button::image_and_text(
            egui::Image::new(ICON_DOWNLOAD).fit_to_exact_size(egui::vec2(14.0, 14.0)),
            "Download",
        ))
        .clicked()
    }

    fn models_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            theme::group(ui, "On disk", Some(ICON_DISK), |ui| {
                if self.local_models.is_empty() {
                    ui.weak("No models yet — download one below.");
                }
                let locals = self.local_models.clone();
                egui::Grid::new("local_models")
                    .num_columns(4)
                    .spacing([16.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for model in &locals {
                            let loaded =
                                self.loaded_model.as_deref() == Some(model.name.as_str());
                            ui.horizontal(|ui| {
                                theme::icon(ui, ICON_DISK, 16.0);
                                ui.label(&model.name);
                                if loaded {
                                    ui.colored_label(theme::GOOD_GREEN, "●");
                                }
                            });
                            ui.weak(fmt_bytes(model.size));
                            self.fit_badge(ui, model.size);
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        !loaded && !self.model_loading,
                                        egui::Button::new("Load")
                                            .min_size(egui::vec2(60.0, 0.0)),
                                    )
                                    .clicked()
                                {
                                    self.load_model(model.path.clone());
                                }
                                if ui
                                    .add(egui::Button::image_and_text(
                                        egui::Image::new(ICON_TRASH)
                                            .fit_to_exact_size(egui::vec2(14.0, 14.0)),
                                        "Delete",
                                    ))
                                    .clicked()
                                {
                                    self.confirm_delete = Some(model.clone());
                                }
                            });
                            ui.end_row();
                        }
                    });

                for dl in &self.downloads {
                    ui.horizontal(|ui| {
                        theme::icon(ui, ICON_DOWNLOAD, 16.0);
                        ui.label(&dl.file);
                        let frac = if dl.total > 0 {
                            dl.bytes as f32 / dl.total as f32
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(frac)
                                .desired_width(200.0)
                                .fill(theme::DESKTOP_BLUE)
                                .text(format!(
                                    "{} / {}",
                                    fmt_bytes(dl.bytes),
                                    fmt_bytes(dl.total)
                                )),
                        );
                    });
                }
            });

            theme::group(ui, "Get models", Some(ICON_DEPOT), |ui| {
                if let Some(rec) = models::recommended(self.hardware.total_ram) {
                    ui.horizontal(|ui| {
                        ui.label("Recommended for your hardware:");
                        ui.strong(rec.name);
                    });
                    ui.separator();
                }
                let catalog = models::catalog();
                egui::Grid::new("catalog")
                    .num_columns(4)
                    .spacing([16.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for entry in &catalog {
                            ui.label(entry.name);
                            ui.weak(fmt_bytes(entry.size));
                            self.fit_badge(ui, entry.size);
                            if self.is_downloaded(entry.file) {
                                ui.weak("downloaded");
                            } else if self.is_downloading(entry.file) {
                                ui.spinner();
                            } else if Self::download_button(ui) {
                                self.start_download(entry.repo, entry.file, entry.size);
                            }
                            ui.end_row();
                        }
                    });
            });

            theme::group(ui, "Search Hugging Face", Some(ICON_SEARCH), |ui| {
            ui.horizontal(|ui| {
                let resp = ui.text_edit_singleline(&mut self.search_query);
                let submitted =
                    resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let search_clicked = ui
                    .add(egui::Button::image_and_text(
                        egui::Image::new(ICON_SEARCH).fit_to_exact_size(egui::vec2(14.0, 14.0)),
                        "Search",
                    ))
                    .clicked();
                if (search_clicked || submitted) && !self.search_query.trim().is_empty() {
                    self.search_pending = true;
                    self.search_results.clear();
                    hub::spawn_search(self.search_query.trim().to_string(), self.hub_tx.clone());
                }
                if self.search_pending {
                    ui.spinner();
                }
            });
            let results = self.search_results.clone();
            for repo in &results {
                let open = egui::CollapsingHeader::new(format!(
                    "{}  ({} downloads)",
                    repo.id, repo.downloads
                ))
                .show(ui, |ui| {
                    match self.repo_files.get(&repo.id).cloned() {
                        Some(files) => {
                            if files.is_empty() {
                                ui.weak("no .gguf files in this repo");
                            }
                            egui::Grid::new(("repo_files", &repo.id))
                                .num_columns(4)
                                .spacing([16.0, 6.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    for f in &files {
                                        ui.label(&f.name);
                                        ui.weak(fmt_bytes(f.size));
                                        self.fit_badge(ui, f.size);
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
                if self.generating {
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
                ui.horizontal(|ui| {
                    let resp = ui.add_sized(
                        [ui.available_width() - 80.0, 60.0],
                        egui::TextEdit::multiline(&mut self.input)
                            .hint_text("Type a message… (Enter to send, Shift+Enter for newline)"),
                    );
                    let send_key = resp.has_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    ui.vertical(|ui| {
                        if self.generating {
                            if ui.button("Stop").clicked() {
                                self.llm.stop.store(true, Ordering::Relaxed);
                            }
                            ui.spinner();
                        } else {
                            if ui.button("Send").clicked() || send_key {
                                self.send_chat();
                            }
                            if !self.messages.is_empty() && ui.small_button("Clear").clicked() {
                                self.messages.clear();
                            }
                        }
                    });
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for msg in &self.messages {
                        let (label, color) = match msg.role {
                            Role::User => ("You", theme::DESKTOP_BLUE),
                            Role::Assistant => ("Model", theme::GOOD_GREEN),
                            Role::System => ("System", egui::Color32::GRAY),
                        };
                        ui.colored_label(color, label);
                        if msg.content.is_empty() {
                            ui.label("…");
                        }
                        for segment in split_think(&msg.content) {
                            match segment {
                                Segment::Text(t) => {
                                    if !t.trim().is_empty() {
                                        CommonMarkViewer::new().show(
                                            ui,
                                            &mut self.md_cache,
                                            t,
                                        );
                                    }
                                }
                                Segment::Think(t) => {
                                    let t = t.trim();
                                    if !t.is_empty() {
                                        render_think_block(ui, t);
                                    }
                                }
                            }
                        }
                        ui.add_space(8.0);
                    }
                });
        });
    }

    fn opencode_snippet(&self) -> String {
        let models: serde_json::Map<String, serde_json::Value> = self
            .local_models
            .iter()
            .map(|m| {
                (
                    m.name.clone(),
                    serde_json::json!({"name": m.name.clone()}),
                )
            })
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

    fn code_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(ui, "Workspace", Some(ICON_CODE), |ui| {
            ui.horizontal(|ui| {
                ui.label("Folder:");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.workspace_input)
                        .hint_text("/path/to/project")
                        .desired_width(320.0),
                );
                if resp.changed() {
                    self.config.workspace = self.workspace_path();
                    self.config.save();
                }
                match self.workspace_path() {
                    Some(ws) => {
                        ui.colored_label(theme::GOOD_GREEN, "✔");
                        if !ws.join("AGENTS.md").exists()
                            && ui
                                .button("Create AGENTS.md")
                                .on_hover_text(
                                    "A project instructions file the agent reads before every task",
                                )
                                .clicked()
                        {
                            if let Err(e) =
                                std::fs::write(ws.join("AGENTS.md"), agent::AGENTS_MD_TEMPLATE)
                            {
                                self.last_error = Some(format!("could not create AGENTS.md: {e}"));
                            }
                        }
                    }
                    None => {
                        ui.colored_label(theme::BAD_RED, "not a folder");
                    }
                }
            });
        });

        theme::group(ui, "Task", None, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.agent_task)
                    .hint_text(
                        "Describe a task, e.g. \"write a python script that prints the first \
                         20 primes, run it, and fix any errors\"",
                    )
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                let running = self.agent_run.is_some();
                let ready = !running
                    && self.loaded_model.is_some()
                    && self.workspace_path().is_some()
                    && !self.agent_task.trim().is_empty();
                if ui
                    .add_enabled(ready, egui::Button::new("▶ Run"))
                    .clicked()
                {
                    let ws = self.workspace_path().unwrap();
                    let task = self.agent_task.trim().to_string();
                    self.agent_transcript.push(AgentItem::Task(task.clone()));
                    self.agent_current.clear();
                    self.live_tokens = 0;
                    self.live_start = None;
                    self.agent_run = Some(agent::start(
                        ws,
                        task,
                        self.llm.cmd_tx.clone(),
                        self.agent_auto_approve,
                    ));
                }
                if running {
                    if ui.button("Stop").clicked() {
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
                        ui.weak(format!(
                            "{:.1} tok/s",
                            self.live_tokens as f32 / secs
                        ));
                    }
                }
                ui.checkbox(&mut self.agent_auto_approve, "auto-approve commands")
                    .on_hover_text("Run shell commands without asking (applies to the next run)");
                if !self.agent_transcript.is_empty()
                    && !running
                    && ui.small_button("Clear").clicked()
                {
                    self.agent_transcript.clear();
                }
            });
            if self.loaded_model.is_none() {
                ui.colored_label(theme::WARN_AMBER, "Load a model in the Models tab first.");
            }
        });

        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut approve_clicked: Option<bool> = None;
                for (i, item) in self.agent_transcript.iter().enumerate() {
                    match item {
                        AgentItem::Task(t) => {
                            ui.colored_label(theme::DESKTOP_BLUE, "Task");
                            ui.label(t);
                            ui.add_space(6.0);
                        }
                        AgentItem::Assistant(text) => {
                            ui.colored_label(theme::GOOD_GREEN, "Model");
                            for segment in split_think(text) {
                                match segment {
                                    Segment::Text(t) => {
                                        if !t.trim().is_empty() {
                                            CommonMarkViewer::new().show(
                                                ui,
                                                &mut self.md_cache,
                                                t,
                                            );
                                        }
                                    }
                                    Segment::Think(t) => {
                                        let t = t.trim();
                                        if !t.is_empty() {
                                            render_think_block(ui, t);
                                        }
                                    }
                                }
                            }
                            ui.add_space(6.0);
                        }
                        AgentItem::Tool {
                            name,
                            summary,
                            output,
                        } => {
                            ui.horizontal(|ui| {
                                theme::icon(ui, ICON_CODE, 14.0);
                                ui.strong(name);
                                ui.weak(summary);
                            });
                            if let Some(out) = output {
                                egui::CollapsingHeader::new("output")
                                    .id_salt(("tool_output", i))
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        ui.monospace(out);
                                    });
                            }
                            ui.add_space(4.0);
                        }
                        AgentItem::Info(text) => {
                            ui.weak(text);
                            ui.add_space(4.0);
                        }
                    }
                }
                if !self.agent_current.is_empty() {
                    ui.colored_label(theme::GOOD_GREEN, "Model");
                    for segment in split_think(&self.agent_current) {
                        match segment {
                            Segment::Text(t) => {
                                if !t.trim().is_empty() {
                                    CommonMarkViewer::new().show(ui, &mut self.md_cache, t);
                                }
                            }
                            Segment::Think(t) => {
                                let t = t.trim();
                                if !t.is_empty() {
                                    render_think_block(ui, t);
                                }
                            }
                        }
                    }
                }
                if let Some((command, _)) = &self.agent_approval {
                    egui::Frame::new()
                        .stroke(egui::Stroke::new(1.0, theme::WARN_AMBER))
                        .corner_radius(egui::CornerRadius::same(3))
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.strong("The agent wants to run a command:");
                            ui.monospace(command);
                            ui.horizontal(|ui| {
                                if ui.button("Approve").clicked() {
                                    approve_clicked = Some(true);
                                }
                                if ui.button("Deny").clicked() {
                                    approve_clicked = Some(false);
                                }
                            });
                        });
                }
                if let Some(answer) = approve_clicked {
                    if let Some((_, reply)) = self.agent_approval.take() {
                        let _ = reply.send(answer);
                    }
                }
            });
    }

    fn serve_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(ui, "API server", Some(ICON_SERVE), |ui| {
            ui.label(
                "Expose your local models over an OpenAI-compatible API so other tools \
                 (opencode, aider, editors, scripts) can use them — still fully local.",
            );
            ui.add_space(4.0);

            let mut enabled = self.config.server_enabled;
            if ui
                .checkbox(&mut enabled, "Enable server (127.0.0.1 only)")
                .changed()
            {
                self.config.server_enabled = enabled;
                if enabled {
                    self.start_server();
                } else if let Some(s) = self.api_server.take() {
                    s.stop();
                }
                self.config.save();
            }

            if self.api_server.is_some() {
                ui.horizontal(|ui| {
                    ui.colored_label(theme::GOOD_GREEN, "● running");
                    ui.monospace(format!("http://127.0.0.1:{}/v1", self.server_port()));
                });
                if self.loaded_model.is_none() {
                    ui.colored_label(
                        theme::WARN_AMBER,
                        "No model loaded — requests will fail until you load one.",
                    );
                }
            }
        });

        theme::group(ui, "opencode setup", None, |ui| {
            ui.label(
                "Add this to opencode.json (globally in ~/.config/opencode/, or per project), \
                 then pick a model from the 'offgrid (local)' provider:",
            );
            let snippet = self.opencode_snippet();
            ui.horizontal(|ui| {
                if ui.button("Copy").clicked() {
                    ui.ctx().copy_text(snippet.clone());
                }
                ui.weak("Works the same for any tool that accepts an OpenAI-compatible base URL.");
            });
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                ui.code(&snippet);
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
                        theme::icon(ui, ICON_TRASH, 24.0);
                        ui.label(format!(
                            "Permanently delete {} ({})?",
                            model.name,
                            fmt_bytes(model.size)
                        ));
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Delete").clicked() {
                            if self.loaded_model.as_deref() == Some(model.name.as_str()) {
                                let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
                            }
                            if let Err(e) = std::fs::remove_file(&model.path) {
                                self.last_error = Some(format!("delete failed: {e}"));
                            }
                            self.rescan();
                            self.confirm_delete = None;
                        }
                        if ui.button("Cancel").clicked() {
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

        egui::Panel::top("top").show(ui, |ui| {
            self.top_bar(ui);
        });

        if let Some(err) = self.last_error.clone() {
            egui::Panel::bottom("error_bar").show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(theme::BAD_RED, &err);
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

fn shellexpand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest).display().to_string();
        }
    }
    path.to_string()
}

enum Segment<'a> {
    Text(&'a str),
    Think(&'a str),
}

/// Split message content into normal text and `<think>…</think>` segments.
/// An unclosed `<think>` (mid-stream) is treated as a think segment.
fn split_think(s: &str) -> Vec<Segment<'_>> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(start) = rest.find(OPEN) {
        if start > 0 {
            out.push(Segment::Text(&rest[..start]));
        }
        let after = &rest[start + OPEN.len()..];
        match after.find(CLOSE) {
            Some(end) => {
                out.push(Segment::Think(&after[..end]));
                rest = &after[end + CLOSE.len()..];
            }
            None => {
                out.push(Segment::Think(after));
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
