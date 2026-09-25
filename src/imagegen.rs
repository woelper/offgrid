//! Local image generation, the same shape as `llm`: a worker thread owns the
//! model, the UI talks to it over channels and never blocks on it.
//!
//! The engine is stable-diffusion.cpp, sharing one ggml with llama.cpp — see
//! `scripts/images-build.sh` for why that needs a build step outside cargo.
//!
//! The alternative considered was candle: pure Rust, so no shared-ggml problem
//! to solve. It lost on the numbers. On the same CPU it took roughly five
//! times as long per step, and Intel MKL behind its matmuls only recovered a
//! third of that — the gap is in the convolutions, which is most of a
//! diffusion model. ggml's kernels are also what make chat usable on a CPU,
//! it reads quantized weights where candle's diffusion models want f16, and it
//! has a Vulkan backend for the AMD and Intel GPUs candle cannot touch.

use std::ffi::{CString, c_char, c_int, c_uchar, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};

mod ffi {
    use super::{c_char, c_int, c_uchar, c_void};

    pub type LogCb = extern "C" fn(level: c_int, text: *const c_char, data: *mut c_void);
    pub type ProgressCb = extern "C" fn(step: c_int, steps: c_int, data: *mut c_void);
    pub type PreviewCb = extern "C" fn(
        width: c_int,
        height: c_int,
        channels: c_int,
        pixels: *const c_uchar,
        data: *mut c_void,
    );

    unsafe extern "C" {
        pub fn offgrid_sd_new(
            model_path: *const c_char,
            diffusion_path: *const c_char,
            vae_path: *const c_char,
            llm_path: *const c_char,
            n_threads: c_int,
            flash_attn: c_int,
            offload_to_cpu: c_int,
        ) -> *mut c_void;
        pub fn offgrid_sd_free(ctx: *mut c_void);
        pub fn offgrid_sd_cancel(ctx: *mut c_void, reset: c_int);
        pub fn offgrid_sd_generate(
            ctx: *mut c_void,
            prompt: *const c_char,
            negative: *const c_char,
            steps: c_int,
            width: c_int,
            height: c_int,
            cfg: f32,
            seed: i64,
            sampler: c_int,
            out_pixels: *mut *mut c_uchar,
            out_width: *mut c_int,
            out_height: *mut c_int,
            out_channels: *mut c_int,
        ) -> c_int;
        pub fn offgrid_sd_free_buf(buf: *mut c_uchar);
        pub fn offgrid_sd_set_log(cb: Option<LogCb>, data: *mut c_void);
        pub fn offgrid_sd_set_progress(cb: Option<ProgressCb>, data: *mut c_void);
        pub fn offgrid_sd_set_preview(
            cb: Option<PreviewCb>,
            mode: c_int,
            interval: c_int,
            data: *mut c_void,
        );
    }
}

/// The sizes on offer. Every value is a multiple of 64, which the UNet's
/// eight-fold downsampling requires. Deliberately bare: whether one of these
/// is native, tolerable or too small is a property of the model selected, not
/// of the number, so the labels are built per model by `size_label`.
pub const SIZES: &[(usize, usize)] = &[(384, 384), (512, 512), (512, 768), (768, 512), (768, 768)];

/// How a size reads for a given model: its own resolution, one it copes with,
/// or one it will answer with a lattice of unresolved patches.
pub fn size_label(spec: &ImageModel, width: usize, height: usize) -> String {
    let size = format!("{width} × {height}");
    if width.min(height) < spec.min_size {
        format!("{size} — too small for this model")
    } else if (width, height) == (spec.native, spec.native) {
        format!("{size} — native")
    } else {
        size
    }
}

/// The size the estimates compare against.
pub const DEFAULT_SIZE: (usize, usize) = (512, 512);

/// `PREVIEW_NONE`, `PREVIEW_PROJ` and `PREVIEW_VAE` from sd.cpp's preview_t.
const PREVIEW_PROJECTION: i32 = 1;
const PREVIEW_VAE: i32 = 3;

/// One weight file of an image model, and where it comes from.
pub struct WeightFile {
    pub repo: &'static str,
    /// Path inside the repo; the basename is what lands on disk.
    pub path: &'static str,
    pub size: u64,
    pub role: Role,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Role {
    /// A whole model in one file, the way SD 1.x and SDXL ship.
    Checkpoint,
    /// The diffusion model alone; the newer models split their parts.
    Diffusion,
    Vae,
    /// The text encoder, which for Z-Image is a Qwen3 4B — the same file
    /// offgrid's own model catalog offers for chat.
    Llm,
}

pub struct ImageModel {
    pub name: &'static str,
    /// What the choice costs, in the terms that matter at ten seconds a step.
    pub note: &'static str,
    pub files: &'static [WeightFile],
    /// Steps and guidance the model was distilled for. Z-Image-Turbo wants 8
    /// steps at CFG 1.0 — no classifier-free guidance, so one forward a step
    /// rather than two, which is most of why a 6B model stays affordable.
    pub steps: usize,
    pub cfg: f32,
    pub flash_attn: bool,
    /// sd.cpp's sample_method_t, or -1 for the library's default. Qwen-Image
    /// is trained for euler and gives noise under the default sampler.
    pub sampler: i32,
    /// What the model was trained at. Bigger than anything this list offers for
    /// the newer ones, which is the honest situation: 1024 is minutes a step on
    /// a CPU, so the tab offers what is bearable and says what is native.
    pub native: usize,
    /// The smallest edge this model still behaves at. Diffusion models are
    /// trained at a resolution and drift off-distribution below it: SD 1.5 was
    /// trained at 512 and copes at 384, while the newer ones are trained
    /// around 1024 and answer a small canvas with a lattice of unresolved
    /// patches rather than a picture.
    pub min_size: usize,
    /// Whether the model can be asked for a transparent background. Only
    /// Qwen-Image 2.1 can, and it is asked in the prompt rather than through a
    /// switch — see `transparent_prompt`.
    pub transparency: bool,
    /// How to show the image forming, and how often. sd.cpp's cheap preview
    /// projects the latents through a small matrix, but it only knows certain
    /// latent spaces — asking for one it cannot do ("No latent to RGB
    /// projection known for this model") fails the whole generation rather
    /// than skipping the preview, which is what Qwen-Image 2.1's
    /// 64-dimensional latents do. Those decode through the model's own VAE
    /// instead: a real picture rather than an impression, at the price of a
    /// decode, so it runs every few steps rather than every one.
    pub preview: (i32, i32),
}

impl ImageModel {
    fn file(&self, role: Role) -> Option<&WeightFile> {
        self.files.iter().find(|f| f.role == role)
    }

    fn path_for(&self, role: Role) -> String {
        self.file(role)
            .map(|f| local_path(f).to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Everything this model needs on disk, whether or not it is there yet.
    /// Quants of the same model share their VAE and text encoder, so switching
    /// between them fetches only the part that differs.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Bytes still to fetch before this model can run.
    pub fn missing_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter(|f| !local_path(f).exists())
            .map(|f| f.size)
            .sum()
    }
}

pub const MODELS: &[ImageModel] = &[
    ImageModel {
        name: "Stable Diffusion 1.5",
        note: "2022, and it shows — but much the quickest of these, and the \
               smallest download",
        files: &[WeightFile {
            repo: "second-state/stable-diffusion-v1-5-GGUF",
            path: "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
            size: 1_763_578_176,
            role: Role::Checkpoint,
        }],
        steps: 10,
        cfg: 7.5,
        native: 512,
        min_size: 384,
        flash_attn: false,
        sampler: SAMPLER_DEFAULT,
        transparency: false,
        preview: (PREVIEW_PROJECTION, 1),
    },
    ImageModel {
        name: "Z-Image-Turbo",
        note: "2025, and far better pictures. Several times slower than SD 1.5, \
               and worth it.",
        files: &[
            WeightFile {
                repo: "leejet/Z-Image-Turbo-GGUF",
                path: "z_image_turbo-Q4_K.gguf",
                size: 3_864_250_304,
                role: Role::Diffusion,
            },
            WeightFile {
                // The same VAE as FLUX.1-schnell, but that repo is gated and a
                // first run would fail on a login prompt. This mirror is not.
                repo: "Comfy-Org/z_image_turbo",
                path: "split_files/vae/ae.safetensors",
                size: 335_304_388,
                role: Role::Vae,
            },
            WeightFile {
                repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                path: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                size: 2_497_281_120,
                role: Role::Llm,
            },
        ],
        steps: 8,
        cfg: 1.0,
        native: 1024,
        min_size: 512,
        flash_attn: true,
        sampler: SAMPLER_DEFAULT,
        transparency: false,
        preview: (PREVIEW_PROJECTION, 1),
    },
    ImageModel {
        name: "Z-Image-Turbo (Q3_K)",
        note: "the same model squeezed smaller: less to fetch and to hold, for \
               some loss of detail",
        files: &[
            WeightFile {
                repo: "leejet/Z-Image-Turbo-GGUF",
                path: "z_image_turbo-Q3_K.gguf",
                size: 3_143_559_104,
                role: Role::Diffusion,
            },
            WeightFile {
                repo: "Comfy-Org/z_image_turbo",
                path: "split_files/vae/ae.safetensors",
                size: 335_304_388,
                role: Role::Vae,
            },
            WeightFile {
                repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                path: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                size: 2_497_281_120,
                role: Role::Llm,
            },
        ],
        steps: 8,
        cfg: 1.0,
        native: 1024,
        min_size: 512,
        flash_attn: true,
        sampler: SAMPLER_DEFAULT,
        transparency: false,
        preview: (PREVIEW_PROJECTION, 1),
    },
    ImageModel {
        name: "Qwen-Image 2.1",
        note: "2026, the newest of these. An order of magnitude slower than \
               SD 1.5 on a CPU, and the largest download — meant for a GPU \
               build.",
        files: &[
            WeightFile {
                repo: "leejet/Qwen-Image-2.1-GGUF",
                path: "qwen_image_2.1-Q4_K.gguf",
                size: 4_197_494_816,
                role: Role::Diffusion,
            },
            WeightFile {
                // Qwen-Image 2.1 has its own VAE; the earlier Qwen-Image and
                // Wan 2.2 weights are not interchangeable with it.
                repo: "Comfy-Org/Qwen-Image-2.1",
                path: "vae/qwen_image_2.1_vae_bf16.safetensors",
                size: 675_509_688,
                role: Role::Vae,
            },
            WeightFile {
                repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                path: "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                size: 5_027_784_800,
                role: Role::Llm,
            },
        ],
        // Guidance at 6.0 means two forwards a step, where Z-Image needs one:
        // twenty steps of an 8B model is why this wants better hardware. The
        // settings come from sd.cpp's own documented example; the pictures
        // have not been judged here, only the plumbing.
        steps: 20,
        cfg: 6.0,
        native: 1024,
        // 512 was the first size anyone produced a good picture at, once the
        // step count was right: the lattice that looked like a resolution
        // problem was eight steps on a twenty-step model. Below 512 it does
        // degrade, so the floor stays — one step lower than it was.
        min_size: 512,
        flash_attn: true,
        sampler: SAMPLER_EULER,
        transparency: true,
        // Every fourth step: a VAE decode is not free, and at this model's
        // pace four steps is minutes of waiting to fill.
        preview: (PREVIEW_VAE, 4),
    },
    ImageModel {
        name: "Qwen-Image 2.1 (Q2_K)",
        note: "the smallest Qwen, and the quantisation shows. Shares its VAE \
               and text encoder with the larger one.",
        files: &[
            WeightFile {
                repo: "leejet/Qwen-Image-2.1-GGUF",
                path: "qwen_image_2.1-Q2_K.gguf",
                size: 2_561_716_256,
                role: Role::Diffusion,
            },
            WeightFile {
                repo: "Comfy-Org/Qwen-Image-2.1",
                path: "vae/qwen_image_2.1_vae_bf16.safetensors",
                size: 675_509_688,
                role: Role::Vae,
            },
            WeightFile {
                repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                path: "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                size: 5_027_784_800,
                role: Role::Llm,
            },
        ],
        steps: 20,
        cfg: 6.0,
        native: 1024,
        // 512 was the first size anyone produced a good picture at, once the
        // step count was right: the lattice that looked like a resolution
        // problem was eight steps on a twenty-step model. Below 512 it does
        // degrade, so the floor stays — one step lower than it was.
        min_size: 512,
        flash_attn: true,
        sampler: SAMPLER_EULER,
        transparency: true,
        // Every fourth step: a VAE decode is not free, and at this model's
        // pace four steps is minutes of waiting to fill.
        preview: (PREVIEW_VAE, 4),
    },
];

/// Ask for a transparent background, in the form Qwen-Image 2.1 expects.
///
/// There is no switch for this: the model decides between an opaque image and
/// an RGBA one from the prompt itself, and this is the wording its authors
/// recommend. The alpha then arrives as a fourth channel, which is carried
/// through to the PNG — cutouts for a button or a badge, without keying a
/// background out by hand.
pub fn transparent_prompt(prompt: &str) -> String {
    format!(
        "This is an RGBA image with transparency. {}. \
         The image has alpha channel and the background is transparent.",
        prompt.trim().trim_end_matches('.')
    )
}

/// Leave the sampler to sd.cpp.
const SAMPLER_DEFAULT: i32 = -1;
/// `EULER_SAMPLE_METHOD`, the first of sd.cpp's sample_method_t.
const SAMPLER_EULER: i32 = 0;

/// Where a weight file lands: `<models>/images/<basename>`.
fn local_path(file: &WeightFile) -> PathBuf {
    let base = Path::new(file.path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| file.path.to_string());
    crate::config::models_dir().join("images").join(base)
}

/// What produced an image, kept with it and written into the file.
#[derive(Clone)]
pub struct Recipe {
    pub model: &'static str,
    pub prompt: String,
    pub steps: usize,
    pub cfg: f32,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
}

impl Recipe {
    /// The one-string form the diffusion tools agreed on: prompt first, then
    /// comma-separated settings. Automatic1111 and ComfyUI both write and read
    /// it under the `parameters` key, so an image saved here can be dropped
    /// into one of those and still say where it came from.
    pub fn parameters(&self) -> String {
        format!(
            "{}\nSteps: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model: {}",
            self.prompt.trim(),
            self.steps,
            self.cfg,
            self.seed,
            self.width,
            self.height,
            self.model,
        )
    }
}

/// Write a PNG with the recipe in its text chunks. The `image` crate cannot
/// express those, and an image whose prompt is only in the UI that made it
/// loses the prompt the moment it is filed somewhere.
pub fn save_png(
    path: &Path,
    width: usize,
    height: usize,
    channels: usize,
    pixels: &[u8],
    recipe: &Recipe,
) -> Result<(), String> {
    let expected = width * height * channels;
    if pixels.len() != expected {
        return Err(format!(
            "image is {} bytes, expected {expected}",
            pixels.len()
        ));
    }
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(if channels == 4 {
        png::ColorType::Rgba
    } else {
        png::ColorType::Rgb
    });
    encoder.set_depth(png::BitDepth::Eight);
    let text = |encoder: &mut png::Encoder<_>, key: &str, value: String| {
        // A rejected chunk is not worth failing a save over: the pixels are
        // the point, the provenance is a bonus.
        let _ = encoder.add_text_chunk(key.to_string(), value);
    };
    text(&mut encoder, "parameters", recipe.parameters());
    text(&mut encoder, "prompt", recipe.prompt.trim().to_string());
    text(&mut encoder, "Software", "offgrid".to_string());
    encoder
        .write_header()
        .map_err(|e| format!("writing {}: {e}", path.display()))?
        .write_image_data(pixels)
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

pub enum ImageCmd {
    /// Download a model's weights without generating anything, so the list can
    /// offer it the way the model tab offers an LLM.
    Fetch { model: usize },
    /// Fetch the weights if needed, then generate.
    Generate {
        /// Index into `MODELS`.
        model: usize,
        /// Keep the weights in RAM and stream them to the accelerator. On a
        /// CPU build it changes nothing; on a GPU it is what lets a model
        /// larger than VRAM run at all.
        offload: bool,
        prompt: String,
        steps: usize,
        seed: u64,
        width: usize,
        height: usize,
    },
}

pub enum ImageEvent {
    /// Human-readable progress that is not a step count: downloads, loading.
    Note(String),
    /// The image part-way through: projected from the latents, or decoded
    /// through the VAE for models that have no projection.
    Preview {
        width: usize,
        height: usize,
        channels: usize,
        pixels: Vec<u8>,
    },
    Step {
        done: usize,
        total: usize,
    },
    Image {
        width: usize,
        height: usize,
        /// 3 for RGB, 4 where the model produces alpha — Qwen-Image does, the
        /// others do not, and none of them are trained to make it mean
        /// transparency, so it is carried rather than interpreted.
        channels: usize,
        pixels: Vec<u8>,
    },
    Done,
    Error(String),
}

pub struct ImageHandle {
    pub cmd_tx: Sender<ImageCmd>,
    pub event_rx: Receiver<ImageEvent>,
    pub stop: Arc<AtomicBool>,
}

pub fn spawn_worker() -> ImageHandle {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<ImageCmd>();
    let (event_tx, event_rx) = std::sync::mpsc::channel::<ImageEvent>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = stop.clone();
    std::thread::spawn(move || worker(cmd_rx, event_tx, stop_worker));
    ImageHandle {
        cmd_tx,
        event_rx,
        stop,
    }
}

/// What the C callbacks write through. Only ever borrowed for the duration of
/// one `generate`, on the thread that called it.
struct Callbacks {
    tx: Sender<ImageEvent>,
    /// The step count we asked for. sd.cpp reports tensor loading and VAE
    /// decoding through the same progress callback as sampling, so the only
    /// way to tell them apart is that their totals are not ours.
    steps: usize,
    /// The running context and the flag the UI sets: `generate_image` blocks
    /// for the whole run, so a stop can only be noticed from inside a
    /// callback, which is where it is turned into sd.cpp's own cancellation.
    ctx: *mut c_void,
    stop: *const AtomicBool,
}

extern "C" fn on_progress(step: c_int, steps: c_int, data: *mut c_void) {
    if data.is_null() {
        return;
    }
    // Safety: `data` is the &Callbacks handed to sd.cpp for this generation,
    // and sd.cpp calls back synchronously from inside `generate`.
    let cb = unsafe { &*(data as *const Callbacks) };
    // Safety: `stop` outlives the generation this callback belongs to.
    if unsafe { &*cb.stop }.load(Ordering::Relaxed) {
        // Checked once a step, so a stop lands within a step rather than at
        // once — which on these models is the difference between ten seconds
        // and a minute of waiting, but far better than the whole run.
        unsafe { ffi::offgrid_sd_cancel(cb.ctx, 0) };
    }
    let total = steps.max(0) as usize;
    if total != cb.steps {
        return; // loading tensors or decoding, not sampling
    }
    let _ = cb.tx.send(ImageEvent::Step {
        done: step.max(0) as usize,
        total,
    });
}

extern "C" fn on_preview(
    width: c_int,
    height: c_int,
    channels: c_int,
    pixels: *const c_uchar,
    data: *mut c_void,
) {
    if data.is_null() || pixels.is_null() || width <= 0 || height <= 0 {
        return;
    }
    let (width, height) = (width as usize, height as usize);
    let channels = (channels as usize).clamp(3, 4);
    // Safety: as above, and the shim only forwards 3- or 4-channel frames,
    // whose buffer is width*height*channels bytes borrowed for the call.
    let cb = unsafe { &*(data as *const Callbacks) };
    let borrowed = unsafe { std::slice::from_raw_parts(pixels, width * height * channels) };
    let _ = cb.tx.send(ImageEvent::Preview {
        width,
        height,
        channels,
        pixels: borrowed.to_vec(),
    });
}

/// The tail of sd.cpp's log. Its failures come out as log lines while the
/// function returns a bare false, and the interesting line is not always the
/// last one — a benign warning often follows the real complaint — so keep a
/// few and pick when asked.
static LOG_TAIL: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// `SD_LOG_INFO`: below this is per-tensor noise.
const LOG_INFO: c_int = 2;
const LOG_TAIL_LINES: usize = 8;

extern "C" fn on_log(level: c_int, text: *const c_char, _data: *mut c_void) {
    if level < LOG_INFO || text.is_null() {
        return;
    }
    // Safety: sd.cpp hands us a nul-terminated string it owns for the call.
    let text = unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .trim()
        .to_string();
    if text.is_empty() {
        return;
    }
    if let Ok(mut tail) = LOG_TAIL.lock() {
        if tail.len() == LOG_TAIL_LINES {
            tail.remove(0);
        }
        tail.push(text);
    }
}

/// The most recent line that sounds like a failure, or failing that the most
/// recent line at all. Clears what it looked at.
fn take_complaint() -> Option<String> {
    let mut tail = LOG_TAIL.lock().ok()?;
    let lines = std::mem::take(&mut *tail);
    lines
        .iter()
        .rev()
        .find(|line| {
            let line = line.to_lowercase();
            line.contains("fail") || line.contains("error") || line.contains("unsupported")
        })
        .or_else(|| lines.last())
        .cloned()
}

/// Owns the loaded model for as long as the worker lives.
struct Model(*mut c_void);

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ffi::offgrid_sd_free(self.0) };
    }
}

fn worker(cmd_rx: Receiver<ImageCmd>, tx: Sender<ImageEvent>, stop: Arc<AtomicBool>) {
    unsafe { ffi::offgrid_sd_set_log(Some(on_log), std::ptr::null_mut()) };

    // Loading costs the better part of a minute, so the model outlives a single
    // request: the second image in a session starts sampling immediately. The
    // index says which one it is, so switching models reloads — one at a time,
    // because two of these do not fit in RAM together.
    // The offload choice is baked into the context, so it is part of what
    // identifies the loaded model.
    let mut loaded: Option<(usize, bool, Model)> = None;

    for cmd in cmd_rx {
        match cmd {
            ImageCmd::Fetch { model } => {
                stop.store(false, Ordering::Relaxed);
                let result = MODELS
                    .get(model)
                    .ok_or_else(|| "no such model".to_string())
                    .and_then(|spec| fetch_weights(spec, &tx, &stop));
                if let Err(e) = result {
                    let _ = tx.send(ImageEvent::Error(e));
                }
                let _ = tx.send(ImageEvent::Done);
            }
            ImageCmd::Generate {
                model,
                offload,
                prompt,
                steps,
                seed,
                width,
                height,
            } => {
                stop.store(false, Ordering::Relaxed);
                let result = MODELS
                    .get(model)
                    .ok_or_else(|| "no such model".to_string())
                    .and_then(|spec| {
                        fetch_weights(spec, &tx, &stop)?;
                        let ctx = load(&mut loaded, model, spec, offload, &tx)?;
                        generate(ctx, spec, &prompt, steps, seed, width, height, &stop, &tx)
                    });
                if let Err(e) = result {
                    let _ = tx.send(ImageEvent::Error(e));
                }
                let _ = tx.send(ImageEvent::Done);
            }
        }
    }
}

/// Download whatever of the model is missing, through the same resuming
/// downloader the model tab uses.
fn fetch_weights(
    spec: &ImageModel,
    tx: &Sender<ImageEvent>,
    stop: &AtomicBool,
) -> Result<(), String> {
    for file in spec.files {
        let dest = local_path(file);
        if dest.exists() {
            continue;
        }
        let dir = dest.parent().ok_or("no models directory")?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = dest
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let _ = tx.send(ImageEvent::Note(format!(
            "downloading {name} ({})",
            crate::hardware::fmt_bytes(file.size)
        )));
        let dl = crate::hub::start_download(file.repo, file.path, file.size, dir);
        for event in dl.rx {
            if stop.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            match event {
                crate::hub::DownloadEvent::Progress { bytes, total } => {
                    let _ = tx.send(ImageEvent::Note(format!(
                        "downloading {name} — {} of {}",
                        crate::hardware::fmt_bytes(bytes),
                        crate::hardware::fmt_bytes(total)
                    )));
                }
                crate::hub::DownloadEvent::Done => break,
                crate::hub::DownloadEvent::Error(e) => return Err(format!("{name}: {e}")),
            }
        }
    }
    Ok(())
}

fn load<'a>(
    loaded: &'a mut Option<(usize, bool, Model)>,
    index: usize,
    spec: &ImageModel,
    offload: bool,
    tx: &Sender<ImageEvent>,
) -> Result<&'a Model, String> {
    if loaded.as_ref().map(|(i, o, _)| (*i, *o)) != Some((index, offload)) {
        // Free the old one before allocating the new: 4 GB each.
        *loaded = None;
        let _ = tx.send(ImageEvent::Note(format!("loading {}…", spec.name)));
        let cstr =
            |p: String| CString::new(p).map_err(|_| "a path contains a nul byte".to_string());
        let checkpoint = cstr(spec.path_for(Role::Checkpoint))?;
        let diffusion = cstr(spec.path_for(Role::Diffusion))?;
        let vae = cstr(spec.path_for(Role::Vae))?;
        let llm = cstr(spec.path_for(Role::Llm))?;
        // Every logical core, unlike the LLM worker. Token generation is
        // memory-bound, so SMT siblings buy it nothing; diffusion is
        // convolution and compute-bound, and using them all was worth nearly a
        // factor of two when measured. IMAGE_THREADS overrides it for a
        // machine that would rather keep some cores free.
        let threads = std::env::var("IMAGE_THREADS")
            .ok()
            .and_then(|t| t.parse::<i32>().ok())
            .filter(|t| *t > 0)
            .unwrap_or_else(|| crate::hardware::HardwareProfile::detect().cores as i32)
            as c_int;
        let ctx = unsafe {
            ffi::offgrid_sd_new(
                checkpoint.as_ptr(),
                diffusion.as_ptr(),
                vae.as_ptr(),
                llm.as_ptr(),
                threads,
                spec.flash_attn as c_int,
                offload as c_int,
            )
        };
        if ctx.is_null() {
            return Err(match take_complaint() {
                Some(why) => format!("could not load {}: {why}", spec.name),
                None => format!("could not load {}", spec.name),
            });
        }
        *loaded = Some((index, offload, Model(ctx)));
    }
    loaded
        .as_ref()
        .map(|(_, _, model)| model)
        .ok_or_else(|| "model vanished".to_string())
}

#[allow(clippy::too_many_arguments)]
fn generate(
    model: &Model,
    spec: &ImageModel,
    prompt: &str,
    steps: usize,
    seed: u64,
    width: usize,
    height: usize,
    stop: &AtomicBool,
    tx: &Sender<ImageEvent>,
) -> Result<(), String> {
    // A size the UNet cannot halve three times over produces garbage rather
    // than an error, so round rather than trust the caller.
    let round = |n: usize| (n.clamp(256, 1024) / 64 * 64) as c_int;
    let (width, height) = (round(width), round(height));
    let prompt_c =
        CString::new(prompt).map_err(|_| "the prompt contains a nul byte".to_string())?;
    let negative = CString::new("").unwrap();

    // A cancellation from a previous run would otherwise stop this one before
    // it starts.
    unsafe { ffi::offgrid_sd_cancel(model.0, 1) };
    let callbacks = Callbacks {
        tx: tx.clone(),
        steps,
        ctx: model.0,
        stop,
    };
    let data = &callbacks as *const Callbacks as *mut c_void;
    let (preview_mode, preview_interval) = spec.preview;
    unsafe {
        ffi::offgrid_sd_set_progress(Some(on_progress), data);
        ffi::offgrid_sd_set_preview(Some(on_preview), preview_mode, preview_interval, data);
    }

    let mut pixels: *mut c_uchar = std::ptr::null_mut();
    // Filled by the shim — kept distinct from the requested size, which they
    // would otherwise shadow.
    let (mut out_width, mut out_height, mut out_channels) = (0 as c_int, 0 as c_int, 0 as c_int);
    let ok = unsafe {
        ffi::offgrid_sd_generate(
            model.0,
            prompt_c.as_ptr(),
            negative.as_ptr(),
            steps as c_int,
            width,
            height,
            spec.cfg,
            seed as i64,
            spec.sampler as c_int,
            &mut pixels,
            &mut out_width,
            &mut out_height,
            &mut out_channels,
        )
    };
    // The callbacks borrow `callbacks`, which dies with this function: clear
    // them first, or a later generation would write through a dangling pointer.
    unsafe {
        ffi::offgrid_sd_set_progress(None, std::ptr::null_mut());
        ffi::offgrid_sd_set_preview(None, 0, 0, std::ptr::null_mut());
    }

    // A stop is not a failure: sd.cpp returns the same false for both.
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    if ok == 0 || pixels.is_null() {
        return Err(match take_complaint() {
            Some(why) => format!("generation failed: {why}"),
            None => "generation failed, and sd.cpp said nothing about why".into(),
        });
    }
    let (width, height) = (out_width as usize, out_height as usize);
    let channels = (out_channels as usize).clamp(3, 4);
    // Safety: on success the shim returns a malloc'd buffer of exactly
    // width*height*channels bytes, which is ours to copy out of and free.
    let copied = unsafe { std::slice::from_raw_parts(pixels, width * height * channels) }.to_vec();
    unsafe { ffi::offgrid_sd_free_buf(pixels) };

    let _ = tx.send(ImageEvent::Image {
        width,
        height,
        channels,
        pixels: copied,
    });
    Ok(())
}
