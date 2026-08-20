use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use crate::agent::{self, AgentEvent, AgentRun};
use crate::config::{Config, models_dir};
use crate::hardware::{HardwareProfile, fmt_bytes, fmt_bytes_precise};
use crate::hub::{self, ActiveDownload, DownloadEvent, HubEvent, RepoFile, RepoResult};
use crate::llm::{self, ChatMessage, LlmCmd, LlmEvent, LlmHandle, Role};
use crate::models::{self, Fit, LocalModel};
use crate::server::{self, ApiServer};
use crate::theme;

const ICON_LOGO: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Alert_Idea.png");
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
const ICON_FILE: egui::ImageSource<'static> = egui::include_image!("../assets/icons/File_Text.png");
const ICON_FOLDER: egui::ImageSource<'static> =
    egui::include_image!("../assets/icons/Folder_generic.png");

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
    Chat,
    Code,
    Serve,
}

fn tool_icon(name: &str) -> egui::ImageSource<'static> {
    match name {
        "run_command" => ICON_CODE,
        "web_search" => ICON_SEARCH,
        "fetch_url" => ICON_SERVE,
        "list_files" => ICON_FOLDER,
        "read_file" | "write_file" => ICON_FILE,
        _ => ICON_DISK,
    }
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
                Ok(HubEvent::Files { repo, mut files }) => {
                    self.files_pending.remove(&repo);
                    files.sort_by_key(|f| f.size);
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
                        });
                    }
                    AgentEvent::ToolResult { output } => {
                        if let Some(AgentItem::Tool { output: slot, .. }) =
                            self.agent_transcript.last_mut()
                        {
                            *slot = Some(output);
                        }
                    }
                    AgentEvent::Info(text) => {
                        self.agent_transcript.push(AgentItem::Info(text));
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
            temp: 0.7,
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
                ui.colored_label(theme::skin().good, name);
                let unload = ui.small_button("Unload");
                theme::gloss(ui, unload.rect);
                if unload.clicked() {
                    let _ = self.llm.cmd_tx.send(LlmCmd::Unload);
                }
            } else {
                ui.weak("no model loaded");
            }
        });
        ui.add_space(4.0);
        theme::tab_bar(
            ui,
            &mut self.tab,
            &[
                (Tab::Models, ICON_DISK, "Models"),
                (Tab::Chat, ICON_CHAT, "Chat"),
                (Tab::Code, ICON_CODE, "Code"),
                (Tab::Serve, ICON_SERVE, "Serve"),
            ],
        );
    }

    fn fit_badge(&self, ui: &mut egui::Ui, size: u64) {
        let (label, color) = Fit::of(size, self.hardware.total_ram).badge();
        ui.colored_label(color, label);
    }

    fn download_button(ui: &mut egui::Ui) -> bool {
        theme::button(ui, Some((ICON_DOWNLOAD, 22.0)), "Download").clicked()
    }

    fn models_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            theme::group(ui, "On disk", Some(ICON_DISK), |ui| {
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
                            theme::icon(ui, ICON_DISK, 16.0);
                            ui.add(egui::Label::new(&model.name).truncate());
                            if loaded {
                                ui.colored_label(theme::skin().good, "•");
                            }
                        },
                        model.size,
                        badge,
                        |ui| {
                            // right-to-left: first added sits at the right edge
                            clicked_delete =
                                theme::button(ui, Some((ICON_TRASH, 18.0)), "Delete").clicked();
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

                for dl in &self.downloads {
                    let frac = if dl.total > 0 {
                        dl.bytes as f32 / dl.total as f32
                    } else {
                        0.0
                    };
                    let elapsed = dl.started.elapsed().as_secs_f32();
                    let speed = dl.bytes as f32 / elapsed.max(0.1);
                    let eta = if speed > 1.0 && dl.total > dl.bytes {
                        fmt_eta((dl.total - dl.bytes) as f32 / speed)
                    } else {
                        "—".to_string()
                    };
                    // Haiku Installer layout: status line above, bar below.
                    ui.horizontal(|ui| {
                        theme::icon(ui, ICON_DOWNLOAD, 16.0);
                        ui.add(egui::Label::new(&dl.file).truncate());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.weak(format!(
                                "{} / {} · {}/s · {}",
                                fmt_bytes_precise(dl.bytes),
                                fmt_bytes_precise(dl.total),
                                fmt_bytes(speed as u64),
                                eta
                            ));
                        });
                    });
                    theme::progress_bar(ui, frac);
                    ui.add_space(4.0);
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

            theme::group(ui, "Search Hugging Face", Some(ICON_SEARCH), |ui| {
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
                            egui::Image::new(ICON_SEARCH).fit_to_exact_size(egui::vec2(16.0, 16.0)),
                            "Search",
                        ),
                    );
                    theme::gloss(ui, search.rect);
                    let search_clicked = search.clicked();
                    if (search_clicked || submitted) && !self.search_query.trim().is_empty() {
                        self.search_pending = true;
                        self.search_results.clear();
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
                let results = self.search_results.clone();
                for repo in &results {
                    let open = egui::CollapsingHeader::new(format!(
                        "{}  ({} downloads)",
                        repo.id,
                        fmt_count(repo.downloads)
                    ))
                    .show(ui, |ui| {
                        match self.repo_files.get(&repo.id).cloned() {
                            Some(files) => {
                                if files.is_empty() {
                                    ui.weak("no .gguf files in this repo");
                                }
                                let models: Vec<_> = files
                                    .iter()
                                    .filter(|f| models::is_model_file(&f.name))
                                    .collect();
                                let best = models
                                    .iter()
                                    .filter(|f| {
                                        Fit::of(f.size, self.hardware.total_ram) == Fit::Fits
                                    })
                                    .min_by_key(|f| (models::quant_tag(&f.name).pref, f.size))
                                    .map(|f| f.name.clone());
                                egui::Grid::new(("repo_files", &repo.id))
                                    .num_columns(5)
                                    .spacing([16.0, 6.0])
                                    .striped(true)
                                    .show(ui, |ui| {
                                        for f in &models {
                                            let tip = models::quant_tooltip(&f.name);
                                            ui.label(&f.name).on_hover_text(&tip);
                                            ui.weak(fmt_bytes(f.size));
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
                            && theme::button(ui, Some((ICON_TRASH, 14.0)), "Clear history")
                                .on_hover_text("Start a fresh conversation")
                                .clicked()
                        {
                            self.messages.clear();
                        }
                    });
                });
                let input_h = 60.0;
                ui.horizontal(|ui| {
                    let resp = ui.add_sized(
                        [ui.available_width() - 88.0, input_h],
                        egui::TextEdit::multiline(&mut self.input)
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
                    for msg in &self.messages {
                        let (label, color) = match msg.role {
                            Role::User => ("You", theme::skin().accent),
                            Role::Assistant => ("Model", theme::skin().good),
                            Role::System => ("System", egui::Color32::GRAY),
                        };
                        ui.colored_label(color, label);
                        if msg.content.is_empty() {
                            ui.label("…");
                        }
                        render_message(ui, &mut self.md_cache, &msg.content);
                        ui.add_space(8.0);
                    }
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

    fn code_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(ui, "Workspace", Some(ICON_CODE), |ui| {
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
                if theme::button(ui, Some((ICON_FOLDER, 16.0)), "Browse…").clicked() {
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
        });

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
                theme::checkbox(ui, &mut self.agent_auto_approve, "auto-approve commands")
                    .on_hover_text("Run shell commands without asking (applies to the next run)");
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
                    && theme::button(ui, Some((ICON_TRASH, 14.0)), "Clear history")
                        .on_hover_text("Clear the task transcript")
                        .clicked()
                {
                    self.agent_transcript.clear();
                }
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
                for (i, item) in self.agent_transcript.iter().enumerate() {
                    match item {
                        AgentItem::Task(t) => {
                            ui.colored_label(theme::skin().accent, "Task");
                            ui.label(t);
                            ui.add_space(6.0);
                        }
                        AgentItem::Assistant(text) => {
                            ui.colored_label(theme::skin().good, "Model");
                            render_message(ui, &mut self.md_cache, text);
                            ui.add_space(6.0);
                        }
                        AgentItem::Tool {
                            name,
                            summary,
                            output,
                        } => {
                            ui.horizontal(|ui| {
                                theme::icon(ui, tool_icon(name), 22.0);
                                ui.strong(name);
                                ui.weak(summary);
                            });
                            if let Some(out) = output {
                                if out.lines().count() > 5 {
                                    egui::CollapsingHeader::new("output")
                                        .id_salt(("tool_output", i))
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
                    }
                }
                if !self.agent_current.is_empty() {
                    ui.colored_label(theme::skin().good, "Model");
                    render_message(ui, &mut self.md_cache, &self.agent_current);
                }
                if let Some((command, _)) = &self.agent_approval {
                    egui::Frame::new()
                        .stroke(egui::Stroke::new(1.0, theme::skin().warn))
                        .corner_radius(egui::CornerRadius::same(3))
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                theme::icon(ui, ICON_CODE, 22.0);
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

    fn serve_ui(&mut self, ui: &mut egui::Ui) {
        theme::group(ui, "API server", Some(ICON_SERVE), |ui| {
            ui.label(
                "Expose your local models over an OpenAI-compatible API so other tools \
                 (opencode, aider, editors, scripts) can use them — still fully local.",
            );
            ui.add_space(4.0);

            let mut enabled = self.config.server_enabled;
            if theme::checkbox(ui, &mut enabled, "Enable server (127.0.0.1 only)").changed() {
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
                    ui.colored_label(theme::skin().good, "• running");
                    ui.monospace(format!("http://127.0.0.1:{}/v1", self.server_port()));
                });
                if self.loaded_model.is_none() {
                    ui.colored_label(
                        theme::skin().warn,
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
                        theme::icon(ui, ICON_TRASH, 24.0);
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
const COL_BADGE: f32 = 60.0;
const ROW_H: f32 = 26.0;

/// A model-list row with fixed-width columns (shared between "On disk" and
/// "Get models" so the two tables line up) and right-aligned actions.
fn list_row(
    ui: &mut egui::Ui,
    stripe: bool,
    name: impl FnOnce(&mut egui::Ui),
    size: u64,
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
fn render_message(ui: &mut egui::Ui, cache: &mut CommonMarkCache, text: &str) {
    for segment in split_segments(text) {
        match segment {
            Segment::Text(t) => {
                if !t.trim().is_empty() {
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
                    render_tool_call_block(ui, cache, t);
                }
            }
        }
    }
}

/// Render a `<tool_call>` for humans: a long `content` argument (write_file)
/// is pulled out of the JSON and shown as its own code block with real
/// newlines, highlighted by the target file's extension.
fn render_tool_call_block(ui: &mut egui::Ui, cache: &mut CommonMarkCache, t: &str) {
    match serde_json::from_str::<serde_json::Value>(t) {
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
                // Four-backtick fence so content containing ``` stays intact.
                CommonMarkViewer::new().show(ui, cache, &format!("````{lang}\n{content}\n````"));
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
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    theme::icon(ui, ICON_LOGO, 22.0);
                    ui.heading(egui::RichText::new("offgrid").strong());
                    ui.separator();
                    ui.label("AMD Ryzen 7 4800H · 16 cores · 32.0 GB RAM");
                    ui.separator();
                    ui.colored_label(theme::skin().good, "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M");
                    let unload = ui.small_button("Unload");
                    theme::gloss(ui, unload.rect);
                });
                ui.add_space(4.0);
                let mut tab = Tab::Models;
                theme::tab_bar(
                    ui,
                    &mut tab,
                    &[
                        (Tab::Models, ICON_DISK, "Models"),
                        (Tab::Chat, ICON_CHAT, "Chat"),
                        (Tab::Code, ICON_CODE, "Code"),
                        (Tab::Serve, ICON_SERVE, "Serve"),
                    ],
                );
            });

        egui::CentralPanel::default().show(ui, |ui| {
            theme::group(ui, "On disk", Some(ICON_DISK), |ui| {
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
                            theme::icon(ui, ICON_DISK, 16.0);
                            ui.add(egui::Label::new(name).truncate());
                            if loaded {
                                ui.colored_label(theme::skin().good, "•");
                            }
                        },
                        size,
                        Fit::of(size, DEMO_RAM).badge(),
                        |ui| {
                            let _ = theme::button(ui, Some((ICON_TRASH, 18.0)), "Delete");
                            let load = ui.add_enabled(
                                !loaded,
                                egui::Button::new("Load").min_size(egui::vec2(60.0, 0.0)),
                            );
                            theme::gloss(ui, load.rect);
                        },
                    );
                }
                ui.horizontal(|ui| {
                    theme::icon(ui, ICON_DOWNLOAD, 16.0);
                    ui.add(egui::Label::new("Qwen3.8-27B-UD-Q4_K_M.gguf").truncate());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak("9.87 GB / 15.30 GB · 8.2 MB/s · 11:04 left");
                    });
                });
                theme::progress_bar(ui, 0.645);
            });

            theme::group(ui, "Get models", Some(ICON_DEPOT), |ui| {
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
                        Fit::of(size, DEMO_RAM).badge(),
                        |ui| {
                            let _ = theme::button(ui, Some((ICON_DOWNLOAD, 22.0)), "Download");
                        },
                    );
                }
            });
        });
    }

    #[test]
    fn main_screen_snapshot() {
        let mut harness = egui_kittest::Harness::builder()
            .with_size(egui::vec2(1000.0, 640.0))
            .build_ui(desktop_ui);
        harness.run();
        harness.snapshot("offgrid");
    }
}
