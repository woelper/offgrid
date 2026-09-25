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
#[cfg(feature = "images")]
use crate::imagegen;
use crate::llm::{self, LlmCmd, LlmEvent, LlmHandle, Role};
use crate::models::{self, Fit, LocalModel};
use crate::server::{self, ApiServer};
use crate::session;
use crate::theme;
use crate::webchat;

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
    Chat,
    Code,
    Serve,
    #[cfg(feature = "images")]
    Images,
    Settings,
}

/// "2m 30s" / "45s" — a rough wait, not a stopwatch.
#[cfg(feature = "images")]
fn fmt_secs(secs: f32) -> String {
    let secs = secs.max(0.0).round() as u64;
    if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
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

fn launch_error_text(e: agent::LaunchError) -> String {
    match e {
        agent::LaunchError::Busy(summary) => format!("Another run is active ({summary})"),
        agent::LaunchError::NothingToResume => "Nothing to resume".into(),
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

/// One finished image and what made it. The pixels are kept rather than the
/// texture alone, because saving needs them and a texture cannot be read back.
#[cfg(feature = "images")]
struct Generated {
    width: usize,
    height: usize,
    /// 3 or 4; Qwen-Image returns alpha, the others do not.
    channels: usize,
    pixels: Vec<u8>,
    recipe: imagegen::Recipe,
    took: Option<f32>,
    /// Uploaded on first sight, then kept: re-uploading every frame would be
    /// wasteful, and the history is small.
    texture: Option<egui::TextureHandle>,
}

#[cfg(feature = "images")]
impl Generated {
    fn color_image(&self) -> egui::ColorImage {
        color_image(self.width, self.height, self.channels, &self.pixels)
    }
}

/// egui wants its own image type, and the channel count decides which — models
/// that make alpha return four channels for previews as well as results.
#[cfg(feature = "images")]
fn color_image(width: usize, height: usize, channels: usize, pixels: &[u8]) -> egui::ColorImage {
    let size = [width, height];
    if channels == 4 {
        egui::ColorImage::from_rgba_unmultiplied(size, pixels)
    } else {
        egui::ColorImage::from_rgb(size, pixels)
    }
}

/// A generated image is a picture, not a widget: it sits centred with a little
/// shadow under it, so it reads as something produced rather than as part of
/// the panel it happens to be in.
#[cfg(feature = "images")]
fn image_frame() -> egui::Frame {
    egui::Frame::NONE.shadow(egui::epaint::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: egui::Color32::from_black_alpha(60),
    })
}

/// How many generations a session keeps before the oldest falls off.
#[cfg(feature = "images")]
const HISTORY_LIMIT: usize = 24;

/// Everything the images tab needs between frames. The worker is spawned on
/// first use, not at startup: most sessions never open this tab, and the model
/// is a 2 GB download nobody should pay for by accident.
#[cfg(feature = "images")]
struct ImagesState {
    worker: Option<imagegen::ImageHandle>,
    /// Index into `imagegen::MODELS`.
    model: usize,
    /// Ask for a transparent background, where the model can do it.
    transparent: bool,
    /// Keep weights in RAM rather than filling VRAM. Pointless on a CPU-only
    /// build, and the difference between running and not on a small card.
    offload: bool,
    prompt: String,
    steps: usize,
    seed: u64,
    /// Output size. Cost follows the pixel count, and past 512 SD 1.5 starts
    /// repeating subjects, so this is a short list rather than free numbers.
    size: (usize, usize),
    busy: bool,
    /// What the worker last said it was doing (downloading, loading, decoding).
    note: String,
    progress: Option<(usize, usize)>,
    started: Option<std::time::Instant>,
    /// Wall-clock seconds the last finished image took, the number this whole
    /// prototype exists to produce.
    took: Option<f32>,
    /// This session's generations, newest last. Bounded: at half a megabyte a
    /// picture the cost is small, but it is not nothing, and a session that
    /// ran all afternoon should not be holding all of it.
    history: Vec<Generated>,
    /// Which of them the result panel is showing; the newest by default.
    shown: Option<usize>,
    /// The image as it forms: width, height, channels, pixels.
    preview: Option<(usize, usize, usize, Vec<u8>)>,
    preview_texture: Option<egui::TextureHandle>,
    /// When the first step landed. Loading the model dominates the first
    /// minute, so steps have to be timed from their own start for an estimate
    /// to mean anything.
    first_step: Option<std::time::Instant>,
}

#[cfg(feature = "images")]
impl Default for ImagesState {
    fn default() -> Self {
        Self {
            worker: None,
            // The quick one first: a five-minute wait is a poor introduction,
            // and the picker says what the slower one buys.
            model: 0,
            transparent: false,
            offload: true,
            prompt: String::new(),
            steps: imagegen::MODELS[0].steps,
            seed: 42,
            size: imagegen::DEFAULT_SIZE,
            busy: false,
            note: String::new(),
            progress: None,
            started: None,
            took: None,
            history: Vec::new(),
            shown: None,
            preview: None,
            preview_texture: None,
            first_step: None,
        }
    }
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
    /// Live copy of the context size for the API server.
    n_ctx_shared: Arc<std::sync::atomic::AtomicU32>,
    model_loading: bool,

    // API server for external tools (opencode etc.)
    api_server: Option<ApiServer>,
    bridge: Option<bridge::Bridge>,
    /// Cached at server start — resolving it opens a UDP socket, too costly
    /// per frame.
    lan_ip: Option<String>,

    // Chat
    /// Shared with the bridge (and future frontends) so a chat started
    /// here can be continued from a phone.
    chat: session::Conversation,
    chat_busy: session::ChatBusy,
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
    /// Shared with the bridge and the API server: one agent run at a time,
    /// visible to whoever asks.
    active_run: agent::ActiveRun,
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
    /// Chat may search the web before answering. Off by default — a query
    /// leaves the machine.
    chat_web: bool,
    /// Latest web-chat progress note ("searching the web…"), shown while the
    /// pre-pass runs and no tokens are streaming yet.
    web_note: Option<String>,
    hl_memo: HighlightMemo,
    /// The conversation as of `chat_fp`. Refreshed only when the fingerprint
    /// changes, so unchanged frames do not deep-clone every message.
    chat_snapshot: Vec<llm::ChatMessage>,
    chat_fp: u64,
    #[cfg(feature = "images")]
    images: ImagesState,
}

impl OffgridApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = Config::load();
        if let Some(skin) = &config.skin {
            theme::set_kind(theme::SkinKind::from_id(skin));
        }
        theme::apply(&cc.egui_ctx);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut hardware = HardwareProfile::detect();
        if let Some(stored) = &config.gpu_bandwidth {
            hardware.adopt_gpu_bandwidth(stored);
        }
        let models_dir = models_dir();
        let _ = std::fs::create_dir_all(&models_dir);
        let (hub_tx, hub_rx) = std::sync::mpsc::channel();
        let llm = llm::spawn_worker(hardware.physical_cores);

        let mut model_loading = false;
        let mut startup_error = None;
        if let Some(last) = &config.last_model
            && last.exists()
        {
            let n_ctx = config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX);
            if models::safe_to_load(last, hardware.total_ram, n_ctx) {
                let _ = llm.cmd_tx.send(LlmCmd::Load {
                    path: last.clone(),
                    n_ctx,
                });
                model_loading = true;
            } else {
                // Don't re-load a model that won't fit — that is what crashed
                // us last time. Leave it selectable; just don't auto-load it.
                startup_error = Some(format!(
                    "{} needs more than the {} of RAM here — not auto-loaded.",
                    last.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    hardware::fmt_bytes(hardware.total_ram)
                ));
            }
        }

        let loaded_model_shared = Arc::new(Mutex::new(None));
        let n_ctx_shared = Arc::new(std::sync::atomic::AtomicU32::new(
            config.n_ctx.unwrap_or(llm::DEFAULT_N_CTX),
        ));
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
            n_ctx_shared,
            model_loading,
            api_server: None,
            bridge: None,
            lan_ip: None,
            chat: session::conversation(),
            chat_busy: session::ChatBusy::new(),
            input: String::new(),
            generating: false,
            md_cache: CommonMarkCache::default(),
            gen_stats: None,
            live_tokens: 0,
            live_start: None,
            workspace_input,
            agent_task: String::new(),
            agent_run: None,
            active_run: agent::active_run(),
            agent_transcript: Vec::new(),
            agent_current: String::new(),
            agent_auto_approve: false,
            agent_approval: None,
            confirm_delete: None,
            #[cfg(feature = "images")]
            images: ImagesState::default(),
            last_error: startup_error,
            chat_culler: RowCuller::default(),
            agent_culler: RowCuller::default(),
            chat_ctx_used: 0,
            agent_ctx_used: 0,
            chat_web: false,
            web_note: None,
            hl_memo: HighlightMemo::default(),
            chat_snapshot: Vec::new(),
            chat_fp: 0,
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
            self.n_ctx_shared.clone(),
            self.config.workspace.clone(),
            self.active_run.clone(),
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
        agent::release(&self.active_run);
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
                        // Every few tokens is plenty for a remote viewer and
                        // keeps the lock cheap.
                        if self.live_tokens.is_multiple_of(16) {
                            agent::note_text(&self.active_run, &self.agent_current);
                        }
                    }
                    AgentEvent::TurnDone => {
                        agent::note_turn(&self.active_run);
                        let text = std::mem::take(&mut self.agent_current);
                        if !text.trim().is_empty() {
                            self.agent_transcript.push(AgentItem::Assistant(text));
                        }
                        self.live_tokens = 0;
                        self.live_start = None;
                    }
                    AgentEvent::ToolCall { name, summary } => {
                        agent::note_activity(&self.active_run, &name, &summary);
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

        #[cfg(feature = "images")]
        self.drain_image_events();

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
                    // The grounded answer is now streaming; drop the pre-pass note.
                    self.web_note = None;
                    self.note_token();
                    session::append_assistant(&self.chat, &text);
                }
                Ok(LlmEvent::Info(note)) => {
                    self.web_note = Some(note);
                }
                Ok(LlmEvent::Stats {
                    prompt_tokens,
                    prompt_secs,
                    gen_tokens,
                    gen_secs,
                }) => {
                    self.chat_ctx_used = prompt_tokens + gen_tokens;
                    self.calibrate_gpu_bandwidth(gen_tokens, gen_secs);
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
                    self.web_note = None;
                    self.chat_busy.release();
                    self.llm.stop.store(false, Ordering::Relaxed);
                }
                Ok(LlmEvent::Error(e)) => {
                    self.model_loading = false;
                    self.web_note = None;
                    if self.generating {
                        self.generating = false;
                        self.chat_busy.release();
                        session::pop_unanswered(&self.chat);
                    }
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
        let _ = self.llm.cmd_tx.send(LlmCmd::Load {
            path,
            n_ctx: self.n_ctx(),
        });
        self.model_loading = true;
    }

    fn send_chat(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.generating || self.loaded_model.is_none() {
            return;
        }
        // The bridge may be answering the same conversation right now.
        if !self.chat_busy.claim() {
            self.last_error = Some("The model is answering another message — one moment.".into());
            return;
        }
        self.input.clear();
        session::push_user(&self.chat, &text);
        if self.chat_web {
            self.web_note = Some("searching the web…".into());
            webchat::spawn(
                session::snapshot(&self.chat),
                self.llm.cmd_tx.clone(),
                self.llm.event_tx.clone(),
                self.llm.stop.clone(),
                0.7,
                self.n_ctx(),
            );
        } else {
            let _ = self.llm.cmd_tx.send(LlmCmd::Generate {
                messages: Arc::new(session::snapshot(&self.chat)),
                reply: self.llm.event_tx.clone(),
                temp: 0.7,
                n_ctx: self.n_ctx(),
            });
        }
        session::push_assistant(&self.chat);
        self.generating = true;
        self.live_tokens = 0;
        self.live_start = None;
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        // Plain grey strip above the tabs, like Haiku's window layouts.
        ui.add_space(14.0);
        let mut tabs = vec![
            (Tab::Models, theme::icons().models.clone(), "Models"),
            (Tab::Chat, theme::icons().chat.clone(), "Chat"),
            (Tab::Code, theme::icons().code.clone(), "Code"),
            (Tab::Serve, theme::icons().serve.clone(), "Serve"),
        ];
        #[cfg(feature = "images")]
        tabs.push((Tab::Images, theme::icons().appearance.clone(), "Images"));
        tabs.push((Tab::Settings, theme::icons().settings.clone(), "Settings"));
        theme::tab_bar(ui, &mut self.tab, &tabs);
    }

    #[cfg(feature = "images")]
    fn drain_image_events(&mut self) {
        let Some(worker) = &self.images.worker else {
            return;
        };
        loop {
            match worker.event_rx.try_recv() {
                Ok(imagegen::ImageEvent::Note(note)) => self.images.note = note,
                Ok(imagegen::ImageEvent::Step { done, total }) => {
                    if done == 1 {
                        self.images.first_step = Some(std::time::Instant::now());
                    }
                    self.images.progress = Some((done, total));
                    self.images.note.clear();
                }
                Ok(imagegen::ImageEvent::Preview {
                    width,
                    height,
                    channels,
                    pixels,
                }) => {
                    self.images.preview = Some((width, height, channels, pixels));
                    self.images.preview_texture = None;
                }
                Ok(imagegen::ImageEvent::Image {
                    width,
                    height,
                    channels,
                    pixels,
                }) => {
                    // Keep the pixels; the texture is uploaded in the tab,
                    // which is the only place with a Context to hand.
                    let (model, spec) = (self.images.model, &imagegen::MODELS[self.images.model]);
                    let _ = model;
                    self.images.history.push(Generated {
                        width,
                        height,
                        channels,
                        pixels,
                        recipe: imagegen::Recipe {
                            model: spec.name,
                            // What the model was actually given, so the file's
                            // own metadata reproduces the image.
                            prompt: self.image_prompt(),
                            steps: self.images.steps,
                            cfg: spec.cfg,
                            seed: self.images.seed,
                            width,
                            height,
                        },
                        took: self.images.first_step.map(|t| t.elapsed().as_secs_f32()),
                        texture: None,
                    });
                    if self.images.history.len() > HISTORY_LIMIT {
                        self.images.history.remove(0);
                        if let Some(shown) = self.images.shown.as_mut() {
                            *shown = shown.saturating_sub(1);
                        }
                    }
                    self.images.shown = Some(self.images.history.len() - 1);
                    self.images.preview = None;
                    self.images.preview_texture = None;
                }
                Ok(imagegen::ImageEvent::Done) => {
                    self.images.busy = false;
                    self.images.progress = None;
                    self.images.first_step = None;
                    self.images.note.clear();
                    self.images.took = self
                        .images
                        .started
                        .take()
                        .map(|t| t.elapsed().as_secs_f32());
                }
                Ok(imagegen::ImageEvent::Error(e)) => {
                    self.last_error = Some(format!("image: {e}"));
                }
                Err(_) => break,
            }
        }
    }

    #[cfg(feature = "images")]
    fn images_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            theme::group(
                ui,
                "Image models",
                Some(theme::icons().depot.clone()),
                |ui| {
                    let mut fetch = None;
                    for (i, model) in imagegen::MODELS.iter().enumerate() {
                        let missing = model.missing_bytes();
                        ui.horizontal(|ui| {
                            let selected = i == self.images.model;
                            if ui.selectable_label(selected, model.name).clicked() && !selected {
                                self.images.model = i;
                                // Each model is distilled for its own step
                                // count, and the last one's is wrong here.
                                self.images.steps = model.steps;
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if missing == 0 {
                                        ui.weak("on disk");
                                    } else if self.images.busy {
                                        ui.weak(format!("{} to fetch", fmt_bytes(missing)));
                                    } else if Self::download_button(ui) {
                                        fetch = Some(i);
                                    }
                                    ui.weak(fmt_bytes(model.total_bytes()));
                                },
                            );
                        });
                    }
                    ui.weak(imagegen::MODELS[self.images.model].note);
                    if let Some(model) = fetch {
                        self.fetch_image_model(model);
                    }
                },
            );

            theme::group(
                ui,
                "Prompt",
                Some(theme::icons().appearance.clone()),
                |ui| {
                    if imagegen::MODELS[self.images.model].transparency {
                        ui.checkbox(&mut self.images.transparent, "Transparent background")
                            .on_hover_text(
                                "Asks the model for an RGBA image with the background cut \
                                 out — for a button or a badge. Qwen-Image decides this from \
                                 the prompt, so this rewords yours; the other models cannot \
                                 do it at all.",
                            );
                    }
                    ui.checkbox(&mut self.images.offload, "Offload weights to RAM")
                        .on_hover_text(
                            "Keeps the weights in RAM and streams them to the GPU as needed, \
                             so a model larger than VRAM still runs. No effect on a build \
                             without a GPU backend.",
                        );
                    ui.add(
                        egui::TextEdit::multiline(&mut self.images.prompt)
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                            .hint_text("a rusty robot walking on a sandy beach"),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Steps:");
                        ui.add(egui::Slider::new(&mut self.images.steps, 1..=50));
                        ui.label("Seed:");
                        ui.add(egui::DragValue::new(&mut self.images.seed));
                    });
                    // Turning steps down is the obvious way to wait less, and
                    // on a model that is not distilled for it the result is not
                    // a rougher picture but a broken one — a lattice of
                    // half-resolved patches. Say so before the wait, not after.
                    let spec = &imagegen::MODELS[self.images.model];
                    if self.images.steps < spec.steps {
                        ui.colored_label(
                            theme::skin().bad,
                            format!(
                                "{} needs {} steps — fewer leaves the image unfinished, \
                                 not merely rougher",
                                spec.name, spec.steps
                            ),
                        );
                    }
                    ui.horizontal(|ui| {
                        ui.label("Size:");
                        let (w, h) = self.images.size;
                        // Labelled against the selected model: the same 512 is
                        // native to SD 1.5 and too small for Qwen, and a label
                        // that says "native" while the line below calls it
                        // incoherent is worse than no label.
                        let spec = &imagegen::MODELS[self.images.model];
                        egui::ComboBox::from_id_salt("image_size")
                            .selected_text(imagegen::size_label(spec, w, h))
                            .show_ui(ui, |ui| {
                                for (sw, sh) in imagegen::SIZES {
                                    ui.selectable_value(
                                        &mut self.images.size,
                                        (*sw, *sh),
                                        imagegen::size_label(spec, *sw, *sh),
                                    );
                                }
                            });
                        if w.min(h) < spec.min_size {
                            ui.colored_label(
                                theme::skin().bad,
                                format!(
                                    "{} was trained at {}px and needs at least {}px here",
                                    spec.name, spec.native, spec.min_size
                                ),
                            );
                        } else {
                            // Measured against the smallest size this model can
                            // use, not a fixed one: the cheapest usable size
                            // differs per model, and a ratio against someone
                            // else's baseline says nothing useful. Attention
                            // costs the square of the latent area, so the wait
                            // climbs faster than the pixels do.
                            let base = spec.min_size * spec.min_size;
                            let ratio = (w * h) as f32 / base as f32;
                            if ratio > 1.01 {
                                ui.weak(format!(
                                    "{ratio:.1}× the pixels of {0} × {0}, and rather more than \
                                     that in time",
                                    spec.min_size
                                ));
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        let ready = !self.images.busy && !self.images.prompt.trim().is_empty();
                        if ui
                            .add_enabled(ready, egui::Button::new("Generate"))
                            .on_disabled_hover_text(if self.images.busy {
                                "an image is already generating"
                            } else {
                                // The grey line in the box is a placeholder,
                                // not text: nothing is sent until you type.
                                "type a prompt first"
                            })
                            .clicked()
                        {
                            self.start_image();
                        }
                        if self.images.busy
                            && theme::button(ui, None, "Stop").clicked()
                            && let Some(worker) = &self.images.worker
                        {
                            worker.stop.store(true, Ordering::Relaxed);
                        }
                        if let Some(took) = self.images.took {
                            ui.weak(format!("last image: {took:.1}s"));
                        }
                    });
                    let missing = imagegen::MODELS[self.images.model].missing_bytes();
                    if missing > 0 && !self.images.busy {
                        ui.weak(format!(
                        "Stable Diffusion 1.5 is not on disk yet — the first image downloads {} \
                         of weights. candle runs them in f16, not a GGUF quant.",
                        fmt_bytes(missing)
                    ));
                    }
                },
            );

            if self.images.busy {
                theme::group(ui, "Working", Some(theme::icons().model.clone()), |ui| {
                    match self.images.progress {
                        Some((done, total)) => {
                            // A step is slow enough on a CPU that the wait is
                            // worth quantifying rather than animating.
                            let label = match self.images.first_step {
                                Some(t0) if done > 0 => {
                                    let per_step = t0.elapsed().as_secs_f32() / done as f32;
                                    format!(
                                        "step {done}/{total} — {per_step:.0}s/step, {} left",
                                        fmt_secs(per_step * (total - done) as f32)
                                    )
                                }
                                _ => format!("step {done}/{total}"),
                            };
                            ui.add(egui::ProgressBar::new(done as f32 / total as f32).text(label));
                        }
                        None => {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(if self.images.note.is_empty() {
                                    "starting…"
                                } else {
                                    &self.images.note
                                });
                            });
                        }
                    }
                    // The latents on their way to being an image: an eighth of
                    // the resolution and blurry, but it moves every step.
                    if let Some((width, height, channels, pixels)) = self.images.preview.clone() {
                        let texture = self.images.preview_texture.get_or_insert_with(|| {
                            let image = color_image(width, height, channels, &pixels);
                            ui.ctx()
                                .load_texture("preview", image, egui::TextureOptions::LINEAR)
                        });
                        // Drawn at the size the finished image will be, so the
                        // tab does not jump when the real one arrives.
                        let (w, h) = self.images.size;
                        ui.vertical_centered(|ui| {
                            image_frame().show(ui, |ui| {
                                ui.add(
                                    egui::Image::new((texture.id(), texture.size_vec2()))
                                        .fit_to_exact_size(egui::vec2(w as f32, h as f32)),
                                );
                            });
                        });
                    }
                });
            }

            // The result panel shows one generation and the strip below it the
            // rest of the session, so a prompt worth keeping is not lost to the
            // next Generate.
            if let Some(shown) = self.images.shown.filter(|i| *i < self.images.history.len()) {
                theme::group(ui, "Result", Some(theme::icons().file.clone()), |ui| {
                    let entry = &mut self.images.history[shown];
                    // Uploaded once, on first sight: building the ColorImage
                    // first keeps the closure from borrowing what it assigns.
                    if entry.texture.is_none() {
                        let image = entry.color_image();
                        entry.texture = Some(ui.ctx().load_texture(
                            "generated",
                            image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                    if let Some(texture) = &entry.texture {
                        let (id, size) = (texture.id(), texture.size_vec2());
                        ui.vertical_centered(|ui| {
                            image_frame().show(ui, |ui| ui.image((id, size)));
                        });
                    }
                    ui.vertical_centered(|ui| {
                        ui.horizontal(|ui| {
                            if theme::button(
                                ui,
                                Some((theme::icons().disk.clone(), 18.0)),
                                "Save as…",
                            )
                            .clicked()
                                && let Some(path) = rfd::FileDialog::new()
                                    .set_file_name("offgrid.png")
                                    .save_file()
                            {
                                let entry = &self.images.history[shown];
                                if let Err(e) = imagegen::save_png(
                                    &path,
                                    entry.width,
                                    entry.height,
                                    entry.channels,
                                    &entry.pixels,
                                    &entry.recipe,
                                ) {
                                    self.last_error = Some(format!("saving the image: {e}"));
                                }
                            }
                            let entry = &self.images.history[shown];
                            ui.weak(match entry.took {
                                Some(took) => format!(
                                    "{} · {} steps · seed {} · {:.0}s",
                                    entry.recipe.model, entry.recipe.steps, entry.recipe.seed, took
                                ),
                                None => format!(
                                    "{} · {} steps · seed {}",
                                    entry.recipe.model, entry.recipe.steps, entry.recipe.seed
                                ),
                            });
                        });
                    });
                });
            }

            if self.images.history.len() > 1 {
                theme::group(
                    ui,
                    "This session",
                    Some(theme::icons().models.clone()),
                    |ui| {
                        // Newest first: the one just made is the one being looked
                        // for. Thumbnails share the full-size textures, which the
                        // history already holds.
                        let mut pick = None;
                        egui::ScrollArea::horizontal().show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for i in (0..self.images.history.len()).rev() {
                                    let entry = &mut self.images.history[i];
                                    if entry.texture.is_none() {
                                        let image = entry.color_image();
                                        entry.texture = Some(ui.ctx().load_texture(
                                            format!("generated{i}"),
                                            image,
                                            egui::TextureOptions::LINEAR,
                                        ));
                                    }
                                    let Some(texture) = &entry.texture else {
                                        continue;
                                    };
                                    let thumb = egui::Button::image(
                                        egui::Image::new((texture.id(), texture.size_vec2()))
                                            .fit_to_exact_size(egui::vec2(96.0, 96.0)),
                                    )
                                    .selected(Some(i) == self.images.shown);
                                    let prompt = entry.recipe.prompt.clone();
                                    if ui.add(thumb).on_hover_text(prompt).clicked() {
                                        pick = Some(i);
                                    }
                                }
                            });
                        });
                        if let Some(i) = pick {
                            self.images.shown = Some(i);
                        }
                    },
                );
            }
        });
    }

    /// Download a model's weights without generating, so the list behaves like
    /// the model tab's: pick it now, wait once, generate later.
    #[cfg(feature = "images")]
    fn fetch_image_model(&mut self, model: usize) {
        let worker = self
            .images
            .worker
            .get_or_insert_with(imagegen::spawn_worker);
        if worker
            .cmd_tx
            .send(imagegen::ImageCmd::Fetch { model })
            .is_ok()
        {
            self.images.busy = true;
            self.images.started = Some(std::time::Instant::now());
            self.images.progress = None;
            self.images.note = "starting…".into();
        }
    }

    /// The prompt as the model will see it: asking for transparency means
    /// rewording, since Qwen-Image takes that from the prompt and not a flag.
    #[cfg(feature = "images")]
    fn image_prompt(&self) -> String {
        let prompt = self.images.prompt.clone();
        if self.images.transparent && imagegen::MODELS[self.images.model].transparency {
            imagegen::transparent_prompt(&prompt)
        } else {
            prompt
        }
    }

    #[cfg(feature = "images")]
    fn start_image(&mut self) {
        // Built before the worker is borrowed: it reads the rest of self.
        let prompt = self.image_prompt();
        let worker = self
            .images
            .worker
            .get_or_insert_with(imagegen::spawn_worker);
        let (width, height) = self.images.size;
        let cmd = imagegen::ImageCmd::Generate {
            model: self.images.model,
            offload: self.images.offload,
            prompt,
            steps: self.images.steps,
            seed: self.images.seed,
            width,
            height,
        };
        if worker.cmd_tx.send(cmd).is_ok() {
            self.images.busy = true;
            self.images.started = Some(std::time::Instant::now());
            self.images.took = None;
            self.images.progress = None;
            self.images.first_step = None;
            self.images.preview = None;
            self.images.preview_texture = None;
            self.images.note = "starting…".into();
        }
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
                    self.n_ctx_shared.store(n_ctx, Ordering::Relaxed);
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
            ui.label(format!("GPU: {}", self.hardware.gpu_summary()));
            ui.weak(format!(
                "{} — this, and how much of a model fits in VRAM, drives the tok/s \
                 estimates in the model lists.",
                self.hardware.bandwidth_summary()
            ));
            if let Some(vram) = self.hardware.vram_summary(self.n_ctx()) {
                ui.weak(vram);
            }
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

    /// Turn a finished generation into a bandwidth measurement for the card.
    ///
    /// The startup assumption in `hardware.rs` is deliberately low, so on a
    /// fast card every tok/s figure in the UI reads several times too slow
    /// until a real run corrects it. A run that had the whole model on the
    /// card is that measurement, for free, from work already done.
    fn calibrate_gpu_bandwidth(&mut self, gen_tokens: usize, gen_secs: f32) {
        let all_on_gpu = self.llm.accel.lock().map(|a| a.all_on_gpu).unwrap_or(false);
        let Some(model) = self
            .local_models
            .iter()
            .find(|m| Some(m.name.as_str()) == self.loaded_model.as_deref())
        else {
            return;
        };
        let (name, size) = (model.name.clone(), model.size);
        if let Some(measured) = self
            .hardware
            .calibrate_gpu(&name, size, gen_tokens, gen_secs, all_on_gpu)
        {
            self.config.gpu_bandwidth = Some(measured);
            self.config.save();
        }
    }

    /// Where a model of this size would actually run on this machine.
    fn placement(&self, size: u64, kv_per_token: Option<u64>) -> models::Placement {
        models::Placement::of(
            size,
            kv_per_token,
            self.hardware.total_ram,
            self.hardware.dedicated_vram(),
            self.n_ctx(),
        )
    }

    fn fit_badge(&self, ui: &mut egui::Ui, size: u64) {
        let placement = self.placement(size, None);
        let (label, color) = placement.badge();
        ui.colored_label(color, label)
            .on_hover_text(placement.tooltip());
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
                            theme::spinner(ui);
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
                    // Where the weights actually ended up — the only place
                    // that reports a partial offload, which is the difference
                    // between 40 tok/s and 4.
                    let accel = self
                        .llm
                        .accel
                        .lock()
                        .map(|a| a.note.clone())
                        .unwrap_or_default();
                    if !accel.is_empty() && !self.model_loading && self.loaded_model.is_some() {
                        ui.weak(accel);
                    }
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
                    let badge = self.placement(model.size, model.kv_per_token);
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
                            self.hardware.bandwidth_for(model.size, self.n_ctx()),
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

                // A transfer that failed this session and an orphaned `.part`
                // from an earlier one need the same decision, so they share one
                // list instead of two near-identical UI blocks.
                struct NeedsAction {
                    file: String,
                    repo: String,
                    path: String,
                    size: u64,
                    status: String,
                    /// True for a failure (`bad` colour) vs. an interrupt (`warn`).
                    failed: bool,
                    /// Index into `self.downloads`, to drop it on discard.
                    active: Option<usize>,
                }
                let mut needs_action: Vec<NeedsAction> = Vec::new();
                for (i, dl) in self.downloads.iter().enumerate() {
                    if let Some(err) = &dl.failed {
                        needs_action.push(NeedsAction {
                            file: dl.file.clone(),
                            repo: dl.repo.clone(),
                            path: dl.path.clone(),
                            size: dl.total,
                            status: format!("interrupted: {err}"),
                            failed: true,
                            active: Some(i),
                        });
                    }
                }
                for part in &self.interrupted {
                    if self.is_downloading(&part.file) {
                        continue;
                    }
                    needs_action.push(NeedsAction {
                        file: part.file.clone(),
                        repo: part.meta.repo.clone(),
                        path: part.meta.path.clone(),
                        size: part.meta.size,
                        status: format!(
                            "interrupted — {} of {} downloaded",
                            fmt_bytes(part.bytes),
                            fmt_bytes(part.meta.size)
                        ),
                        failed: false,
                        active: None,
                    });
                }

                let mut resume: Option<(String, String, u64)> = None;
                let mut discard: Option<usize> = None;
                // Live downloads first: name, rate and ETA over a progress bar.
                for dl in self.downloads.iter().filter(|d| d.failed.is_none()) {
                    let frac = if dl.total > 0 {
                        dl.bytes as f32 / dl.total as f32
                    } else {
                        0.0
                    };
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
                for (i, row) in needs_action.iter().enumerate() {
                    ui.horizontal(|ui| {
                        theme::icon(ui, theme::icons().download.clone(), 16.0);
                        ui.add(egui::Label::new(&row.file).truncate());
                        let color = if row.failed {
                            theme::skin().bad
                        } else {
                            theme::skin().warn
                        };
                        ui.colored_label(color, &row.status);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if theme::button(ui, None, "Discard").clicked() {
                                discard = Some(i);
                            }
                            if theme::button(ui, None, "Resume").clicked() {
                                resume = Some((row.repo.clone(), row.path.clone(), row.size));
                            }
                        });
                    });
                }
                if let Some(i) = discard {
                    let row = &needs_action[i];
                    if let Some(active) = row.active {
                        self.downloads.remove(active);
                    }
                    hub::discard_part(&self.models_dir, &row.file);
                    self.rescan();
                }
                if let Some((repo, path, size)) = resume {
                    self.downloads
                        .retain(|d| file_basename(&d.path) != file_basename(&path));
                    self.downloads
                        .push(hub::start_download(&repo, &path, size, &self.models_dir));
                    self.rescan();
                }
            });

            theme::group(ui, "Get models", Some(theme::icons().depot.clone()), |ui| {
                let proposals = models::propose(
                    self.hardware.total_ram,
                    self.hardware.dedicated_vram(),
                    self.n_ctx(),
                );
                if !proposals.is_empty() {
                    // Two recommendations on a GPU box, because they are two
                    // different machines: what the card can hold generates
                    // several times faster, what RAM can hold answers better.
                    // Picking one for the user hides the trade they own.
                    let split = proposals.ram_adds_anything();
                    ui.label(if split {
                        "Recommended for your hardware — fastest, entirely on the GPU:"
                    } else {
                        "Recommended for your hardware:"
                    });
                    let pick = |ui: &mut egui::Ui, label: &str, entry: Option<&models::CatalogEntry>| {
                        if let Some(entry) = entry {
                            ui.horizontal(|ui| {
                                ui.label(label);
                                ui.label(theme::bold(entry.name));
                                ui.weak(fmt_bytes(entry.size));
                            });
                        }
                    };
                    if split {
                        pick(ui, "Chat:", proposals.gpu_chat.as_ref());
                        pick(ui, "Coding:", proposals.gpu_code.as_ref());
                        ui.label("Bigger, but split with system RAM and much slower:");
                        pick(ui, "Chat:", proposals.ram_chat.as_ref());
                        pick(ui, "Coding:", proposals.ram_code.as_ref());
                    } else {
                        pick(ui, "Chat:", proposals.chat());
                        pick(ui, "Coding:", proposals.code());
                    }
                    if let Some(vram) = self.hardware.vram_summary(self.n_ctx()) {
                        ui.weak(vram);
                    }
                    ui.separator();
                }
                let catalog = models::catalog();
                for (i, entry) in catalog.iter().enumerate() {
                    let badge = self.placement(entry.size, None);
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
                            self.hardware.bandwidth_for(entry.size, self.n_ctx()),
                        )),
                        badge,
                        |ui| {
                            if downloaded {
                                ui.weak("downloaded");
                            } else if downloading {
                                theme::spinner(ui);
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
                        theme::spinner(ui);
                    }
                });
                ui.weak(if self.hardware.dedicated_vram().is_some() {
                    "Q4_K_M is the sweet spot for most machines — higher Q means better but \
                     bigger and slower, Q2 and below degrade noticeably. \"best on GPU\" marks \
                     the best quant that fits your card whole, \"best in RAM\" the best that \
                     fits memory at all — bigger and better, but split with the CPU and \
                     several times slower."
                } else {
                    "Q4_K_M is the sweet spot for most machines — higher Q means better but \
                     bigger and slower, Q2 and below degrade noticeably. \"best pick\" marks \
                     the highest-quality quant that fits your RAM."
                });
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
                                // Two picks, not one: the best quant that runs
                                // entirely on the card, and the best that fits
                                // RAM at all. On a GPU box those are usually
                                // different files, and a single "best pick"
                                // sized against RAM steers people straight into
                                // a split model.
                                let vram = self.hardware.dedicated_vram();
                                let best_of = |on_card: bool| {
                                    files
                                        .iter()
                                        .filter(|f| {
                                            Fit::of(f.size, self.hardware.total_ram, self.n_ctx())
                                                == Fit::Fits
                                        })
                                        .filter(|f| {
                                            !on_card
                                                || vram.is_some_and(|v| {
                                                    models::fits_vram(f.size, None, self.n_ctx(), v)
                                                })
                                        })
                                        .min_by_key(|f| (models::quant_tag(&f.name).pref, f.size))
                                        .map(|f| f.name.clone())
                                };
                                let best_ram = best_of(false);
                                let best_gpu = vram.and_then(|_| best_of(true));
                                // Only worth two markers when they differ.
                                let best = if best_gpu == best_ram { None } else { best_ram };
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
                                                    self.hardware.bandwidth_for(f.size, self.n_ctx()),
                                                ),
                                            ));
                                            self.fit_badge(ui, f.size);
                                            ui.horizontal(|ui| {
                                                let tag = models::quant_tag(&f.name);
                                                if !tag.label.is_empty() {
                                                    ui.colored_label(tag.color, tag.label)
                                                        .on_hover_text(&tip);
                                                }
                                                if best_gpu.as_deref() == Some(f.name.as_str()) {
                                                    ui.label(theme::bold("• best on GPU"))
                                                        .on_hover_text(
                                                            "The highest-quality quant of this \
                                                             repo that fits your card whole. \
                                                             This is the fast one.",
                                                        );
                                                } else if best.as_deref() == Some(f.name.as_str()) {
                                                    ui.label(theme::bold(if best_gpu.is_some() {
                                                        "• best in RAM"
                                                    } else {
                                                        "• best pick"
                                                    }))
                                                    .on_hover_text(
                                                        "The highest-quality quant of this repo \
                                                         that fits your RAM. Better answers than \
                                                         the GPU pick, at a fraction of the speed \
                                                         when it does not fit the card.",
                                                    );
                                                }
                                            });
                                            if self.is_downloaded(&f.name) {
                                                ui.weak("downloaded");
                                            } else if self.is_downloading(&f.name) {
                                                theme::spinner(ui);
                                            } else if Self::download_button(ui) {
                                                self.start_download(&repo.id, &f.name, f.size);
                                            }
                                            ui.end_row();
                                        }
                                    });
                            }
                            None => {
                                theme::spinner(ui);
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
                        theme::spinner(ui);
                        if let Some(note) = &self.web_note {
                            // Pre-pass in flight: no tokens yet, so report the
                            // web activity instead of a misleading 0 tok/s.
                            ui.weak(note.clone());
                        } else if let Some(start) = self.live_start {
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
                            && session::turns(&self.chat) > 0
                            && theme::button(
                                ui,
                                Some((theme::icons().trash.clone(), 14.0)),
                                "Clear history",
                            )
                            .on_hover_text("Start a fresh conversation")
                            .clicked()
                        {
                            session::clear(&self.chat);
                            self.chat_culler.clear();
                            self.chat_ctx_used = 0;
                        }
                        theme::context_meter(ui, self.chat_ctx_used, self.n_ctx() as usize);
                        ui.checkbox(&mut self.chat_web, "🌐 Web").on_hover_text(
                            "Let the model search the web before answering. \
                                 Your query leaves this machine.",
                        );
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
            // Refresh the cached snapshot only when the conversation actually
            // changed; the bridge mutating it mid-draw is fine, we pick the
            // change up on the next frame via the fingerprint.
            let fp = session::fingerprint(&self.chat);
            if fp != self.chat_fp {
                self.chat_snapshot = session::snapshot(&self.chat);
                self.chat_fp = fp;
            }
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let mut culler = std::mem::take(&mut self.chat_culler);
                    let messages = std::mem::take(&mut self.chat_snapshot);
                    culler.begin(ui, messages.len());
                    let len = messages.len();
                    for (i, msg) in messages.iter().enumerate() {
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
                    self.chat_snapshot = messages;
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
                if running {
                    // Mid-run the task box steers instead of starting: the
                    // agent reads it at the next turn boundary.
                    let can_send = !self.agent_task.trim().is_empty();
                    let send = ui.add_enabled(can_send, egui::Button::new("↪ Send"));
                    theme::gloss(ui, send.rect);
                    if (send.clicked() || submit) && can_send {
                        let note = self.agent_task.trim().to_string();
                        if agent::steer(&self.active_run, &note) {
                            self.agent_transcript
                                .push(AgentItem::Info(format!("you: {note}")));
                            self.agent_task.clear();
                        }
                    }
                }
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
                    match agent::launch(
                        &self.active_run,
                        agent::RunSource::Ui,
                        ws,
                        Some(task),
                        self.llm.cmd_tx.clone(),
                        self.agent_auto_approve,
                        self.config.web_tools,
                        self.n_ctx(),
                    ) {
                        Ok(run) => self.agent_run = Some(run),
                        Err(e) => self.last_error = Some(launch_error_text(e)),
                    }
                }
                // An interrupted run left its transcript behind: offer to
                // pick it up instead of re-explaining the task.
                if !running
                    && let Some(ws) = self.workspace_path()
                    && let Some(saved) = agent::saved_run(&ws)
                {
                    let first = saved.task.lines().next().unwrap_or_default().to_string();
                    let label = format!("⏵ Resume ({} turns)", saved.turns);
                    if ui
                        .add_enabled(self.loaded_model.is_some(), egui::Button::new(label))
                        .on_hover_text(format!("Continue the interrupted run:\n{first}"))
                        .clicked()
                    {
                        match agent::launch(
                            &self.active_run,
                            agent::RunSource::Ui,
                            ws,
                            None,
                            self.llm.cmd_tx.clone(),
                            self.agent_auto_approve,
                            self.config.web_tools,
                            self.n_ctx(),
                        ) {
                            Ok(run) => {
                                self.agent_transcript
                                    .push(AgentItem::Info(format!("resuming: {first}")));
                                self.agent_current.clear();
                                self.live_tokens = 0;
                                self.live_start = None;
                                self.agent_run = Some(run);
                            }
                            Err(e) => self.last_error = Some(launch_error_text(e)),
                        }
                    }
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
                    theme::spinner(ui);
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
                                ui.label(theme::bold(name));
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
                                ui.label(theme::bold("The agent wants to run a command:"));
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
            self.active_run.clone(),
            self.llm.stop.clone(),
            self.chat.clone(),
            self.chat_busy.clone(),
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
            #[cfg(feature = "images")]
            Tab::Images => self.images_ui(ui),
            Tab::Settings => self.settings_ui(ui),
        });

        #[cfg(feature = "images")]
        let images_busy = self.images.busy;
        #[cfg(not(feature = "images"))]
        let images_busy = false;
        let busy = images_busy
            || self.generating
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
    placement: models::Placement,
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
                        "Estimated generation speed on this machine: how fast its \
                         memory can stream the model's weights, split between VRAM \
                         and system RAM the way this model would be.",
                    );
                });
                let (label, color) = placement.badge();
                ui.allocate_ui_with_layout(egui::vec2(COL_BADGE, ROW_H), cell, |ui| {
                    ui.set_width(COL_BADGE);
                    ui.colored_label(color, label)
                        .on_hover_text(placement.tooltip());
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
        // The skin's fonts land at the next frame start (egui_kittest renders
        // a frame during construction, before any test code runs), so the
        // first frame would lay out bold text without the bold face bound.
        // Draw nothing and come back once the fonts are in effect.
        if !theme::fonts_ready(&ctx) {
            ctx.request_repaint();
            return;
        }

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
                            models::Placement::of(size, None, DEMO_RAM, None, llm::DEFAULT_N_CTX),
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
                    ui.label(theme::bold("Qwen3 Coder 30B-A3B (Q4_K_M)"));
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
                        models::Placement::of(size, None, DEMO_RAM, None, llm::DEFAULT_N_CTX),
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

    /// `**bold**` in markdown must select the Bold face, not merely a darker
    /// colour (egui's `strong()` alone). A real bold face is measurably wider
    /// than the regular one; a colour change is not.
    #[test]
    fn markdown_bold_uses_bold_face() {
        use egui_kittest::kittest::Queryable;

        fn rendered_width(markdown: &'static str) -> f32 {
            let mut cache = CommonMarkCache::default();
            let mut harness = egui_kittest::Harness::builder()
                .with_size(egui::vec2(400.0, 100.0))
                .build_ui(move |ui| {
                    let ctx = ui.ctx().clone();
                    theme::apply(&ctx);
                    if !theme::fonts_ready(&ctx) {
                        ctx.request_repaint();
                        return;
                    }
                    CommonMarkViewer::new().show(ui, &mut cache, markdown);
                });
            harness.run();
            harness.get_by_label("Weighty words").rect().width()
        }
        let regular = rendered_width("Weighty words");
        let bold = rendered_width("**Weighty words**");
        assert!(
            bold > regular + 2.0,
            "bold markdown should use the Bold face: bold {bold}px vs regular {regular}px"
        );
    }
}
