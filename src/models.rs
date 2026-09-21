use std::path::{Path, PathBuf};

use eframe::egui;

/// Working memory needed beyond the weights themselves: fixed compute and
/// activation buffers, plus the KV cache. The cache grows with the context
/// length, and — as a rough stand-in for a model's layer count and width,
/// which we can't read here — with the model's own size.
///
/// Calibrated against a real thrash: a 10 GB 30B model at the default 16k
/// context used ~14.7 GB on a 16 GB box (KV cache + buffers ≈ 4.5 GB over the
/// weights) and ground to a halt in swap — but the old fixed 1.5 GB overhead
/// called it merely "tight". The factors below put that case just over the
/// "too big" line, while a short context or a small model stays comfortable.
fn overhead(model_size: u64, n_ctx: u32) -> u64 {
    BASE_OVERHEAD + kv_size(model_size, None, n_ctx)
}

/// KV-cache bytes for an `n_ctx` context: exact when the file's header gave
/// us a per-token cost, otherwise the size-derived guess `overhead` is
/// calibrated around.
pub fn kv_size(model_size: u64, kv_per_token: Option<u64>, n_ctx: u32) -> u64 {
    match kv_per_token {
        Some(kv) => kv * n_ctx as u64,
        None => (model_size / 12) * (n_ctx as u64) / 4096,
    }
}

/// Compute/activation buffers beyond weights and KV cache, ~1 GB.
const BASE_OVERHEAD: u64 = 1024 * 1024 * 1024;

/// KV-cache bytes per token, read from the GGUF header: K and V, one f16
/// each, for every KV head of every layer. This is what actually decides
/// whether a long context fits — a 30B mixture-of-experts with 4 KV heads
/// needs ~100 KB/token, a dense model of the same file size several times
/// that — so the size-based guess in `overhead` is only the fallback.
pub fn kv_bytes_per_token(path: &Path) -> Option<u64> {
    gguf_dims(path).map(|d| d.kv_bytes_per_token)
}

/// What the GGUF header tells us about a model's shape.
#[derive(Clone, Copy, Debug)]
pub struct GgufDims {
    /// Transformer blocks — how many pieces the weights split into when only
    /// part of the model goes to the GPU.
    pub layers: u64,
    pub kv_bytes_per_token: u64,
}

/// Read `layers` and the KV cost per token out of a GGUF file's header.
pub fn gguf_dims(path: &Path) -> Option<GgufDims> {
    use std::io::Read as _;
    let mut r = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut buf = [0u8; 8];
    macro_rules! read {
        (u32) => {{
            r.read_exact(&mut buf[..4]).ok()?;
            u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
        }};
        (u64) => {{
            r.read_exact(&mut buf).ok()?;
            u64::from_le_bytes(buf)
        }};
    }
    if read!(u32) != 0x4655_4747 {
        return None; // not "GGUF"
    }
    let version = read!(u32);
    if !(2..=3).contains(&version) {
        return None;
    }
    let _tensors = read!(u64);
    let kv_count = read!(u64);
    // Scalar value sizes by GGUF type id; strings and arrays are variable.
    fn scalar_len(t: u32) -> Option<u64> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    let read_string = |r: &mut std::io::BufReader<std::fs::File>| -> Option<String> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b).ok()?;
        let len = u64::from_le_bytes(b);
        if len > 1 << 20 {
            return None;
        }
        let mut v = vec![0u8; len as usize];
        r.read_exact(&mut v).ok()?;
        Some(String::from_utf8_lossy(&v).into_owned())
    };
    let mut arch = String::new();
    let (mut layers, mut kv_heads, mut kv_heads_sum, mut key_len, mut val_len, mut embd, mut heads) =
        (None, None, None, None, None, None, None);
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let t = read!(u32);
        // Metadata that matters comes first; the tokenizer arrays after it
        // are megabytes we need not walk.
        if key.starts_with("tokenizer.") && layers.is_some() && kv_heads.or(kv_heads_sum).is_some()
        {
            break;
        }
        let want = !arch.is_empty() && key.starts_with(arch.as_str());
        match t {
            8 => {
                let v = read_string(&mut r)?;
                if key == "general.architecture" {
                    arch = format!("{v}.");
                }
            }
            9 => {
                let et = read!(u32);
                let n = read!(u64);
                if et == 8 {
                    for _ in 0..n {
                        read_string(&mut r)?;
                    }
                } else if et == 9 {
                    return None; // nested arrays: give up, use the fallback
                } else {
                    let elen = scalar_len(et)?;
                    let mut sum = 0u64;
                    // Per-layer KV head counts (some hybrid models).
                    let per_layer = want && key.ends_with(".attention.head_count_kv");
                    for _ in 0..n {
                        let mut v = [0u8; 8];
                        r.read_exact(&mut v[..elen as usize]).ok()?;
                        if per_layer {
                            sum += u64::from_le_bytes(v);
                        }
                    }
                    if per_layer {
                        kv_heads_sum = Some(sum);
                    }
                }
            }
            _ => {
                let len = scalar_len(t)?;
                let mut v = [0u8; 8];
                r.read_exact(&mut v[..len as usize]).ok()?;
                let n = u64::from_le_bytes(v);
                if want {
                    let sub = &key[arch.len()..];
                    match sub {
                        "block_count" => layers = Some(n),
                        "attention.head_count_kv" => kv_heads = Some(n),
                        "attention.head_count" => heads = Some(n),
                        "attention.key_length" => key_len = Some(n),
                        "attention.value_length" => val_len = Some(n),
                        "embedding_length" => embd = Some(n),
                        _ => {}
                    }
                }
            }
        }
    }
    let layers = layers?;
    let head_dim = key_len.or_else(|| Some(embd? / heads?))?;
    let v_dim = val_len.unwrap_or(head_dim);
    let kv_layers = kv_heads_sum.unwrap_or(kv_heads? * layers);
    Some(GgufDims {
        layers,
        kv_bytes_per_token: kv_layers * (head_dim + v_dim) * 2,
    })
}

/// Working memory a GPU needs on top of weights and KV cache: ggml's compute
/// buffers for the largest batch, plus what the driver and desktop already
/// hold beyond what it reported as free. Guessing low here costs an
/// out-of-memory abort during load, guessing high costs a layer or two, so
/// this leans high.
pub const GPU_OVERHEAD: u64 = 768 * 1024 * 1024;

/// How many layers to hand the GPU: as many as fit in `vram` alongside their
/// share of the KV cache. A result above `dims.layers` means everything, the
/// output layer included — that is llama.cpp's "all layers".
///
/// The per-layer cost divides the whole file by the layer count, so the
/// embedding and output tensors are spread over the layers instead of being
/// counted separately. That overstates a layer slightly, which is the safe
/// direction: we offload one fewer layer rather than one too many.
pub fn gpu_layers(model_size: u64, dims: &GgufDims, n_ctx: u32, vram: u64) -> u32 {
    let layers = dims.layers.max(1);
    let per_layer = (model_size + dims.kv_bytes_per_token * n_ctx as u64) / layers;
    if per_layer == 0 {
        return 0;
    }
    let budget = vram.saturating_sub(GPU_OVERHEAD);
    let n = u32::try_from((budget / per_layer).min(layers + 1)).unwrap_or(u32::MAX);
    // A handful of layers on the card is not worth the per-token round trip
    // between host and device it costs — keep the whole model in RAM.
    if n < MIN_OFFLOAD_LAYERS { 0 } else { n }
}

/// Below this many layers, offloading buys less than the transfers cost.
const MIN_OFFLOAD_LAYERS: u32 = 4;

/// `MIN_OFFLOAD_LAYERS` expressed as a share of the weights: models run 30 to
/// 50 layers, so four of them is roughly a tenth. Used where the layer count
/// is unknown (a catalog entry or a search result is a name and a size, not a
/// file we can read a header from).
const MIN_OFFLOAD_SHARE: f32 = 0.1;

/// Share of a model's bytes that ends up on the card, 0.0 to 1.0.
///
/// This is the byte-level view of what `gpu_layers` decides layer by layer:
/// the same VRAM budget, the same overhead, the same all-or-nothing floor.
/// Both the fit badges and the tok/s estimates read it, so what the UI
/// promises and what the loader actually does cannot drift apart.
pub fn gpu_share(model_size: u64, kv_per_token: Option<u64>, n_ctx: u32, vram: u64) -> f32 {
    let footprint = model_size + kv_size(model_size, kv_per_token, n_ctx);
    if footprint == 0 {
        return 0.0;
    }
    let budget = vram.saturating_sub(GPU_OVERHEAD);
    let share = (budget as f64 / footprint as f64).min(1.0) as f32;
    if share < MIN_OFFLOAD_SHARE {
        0.0
    } else {
        share
    }
}

/// The largest model file that still fits `vram` whole at `n_ctx` tokens of
/// context: `fits_vram` solved for the size, using the same size-derived KV
/// estimate. This is the number to put in front of someone asking what they
/// should download, instead of making them binary-search the model list.
pub fn largest_fitting_vram(vram: u64, n_ctx: u32) -> u64 {
    // fits_vram is  size + GPU_OVERHEAD + (size/12) * (n_ctx/4096) <= vram.
    let budget = vram.saturating_sub(GPU_OVERHEAD) as f64;
    (budget / (1.0 + n_ctx as f64 / (12.0 * 4096.0))) as u64
}

/// Does the whole model — weights, KV cache and compute buffers — fit in
/// `vram`? Everything on the GPU is the case worth aiming for: it runs an
/// order of magnitude faster than the same model split with system RAM.
pub fn fits_vram(model_size: u64, kv_per_token: Option<u64>, n_ctx: u32, vram: u64) -> bool {
    model_size + GPU_OVERHEAD + kv_size(model_size, kv_per_token, n_ctx) <= vram
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fit {
    Fits,
    Tight,
    TooBig,
}

impl Fit {
    /// Verdict for loading `model_size` bytes of weights with an `n_ctx`-token
    /// context on a machine with `total_ram` bytes. `n_ctx` matters: the same
    /// model can fit at a short context and thrash swap at a long one.
    pub fn of(model_size: u64, total_ram: u64, n_ctx: u32) -> Self {
        Self::of_model(model_size, None, total_ram, n_ctx)
    }

    /// Like `of`, but with the KV cost per token read from the file's own
    /// header when available (see `kv_bytes_per_token`) — the size-based
    /// guess called a 17 GB mixture-of-experts "too big" at 32k on a 32 GB
    /// box when it really needs ~22 GB.
    pub fn of_model(
        model_size: u64,
        kv_per_token: Option<u64>,
        total_ram: u64,
        n_ctx: u32,
    ) -> Self {
        let needed = match kv_per_token {
            Some(kv) => model_size + BASE_OVERHEAD + kv * n_ctx as u64,
            None => model_size + overhead(model_size, n_ctx),
        };
        if needed <= total_ram * 7 / 10 {
            Fit::Fits
        } else if needed <= total_ram * 9 / 10 {
            Fit::Tight
        } else {
            Fit::TooBig
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Fit::Fits => "fits",
            Fit::Tight => "tight",
            Fit::TooBig => "too big",
        }
    }
}

/// Where a model's bytes end up once loaded.
///
/// `Fit` alone answers "will this load at all", which is the whole story on a
/// CPU box and only half of it on a GPU box: a 19 GB model on a machine with
/// 64 GB of RAM and a 16 GB card fits comfortably and still runs ten times
/// slower than a model that fits the card, because the part that did not fit
/// is read over PCIe for every single token. The lists show this instead of a
/// bare "fits", which was true and useless.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Placement {
    /// The system-RAM verdict, unchanged.
    pub fit: Fit,
    /// Share of the weights on the card, or `None` without a discrete GPU.
    pub gpu: Option<f32>,
}

impl Placement {
    /// `vram` is the budget of a discrete card, `None` on a machine without
    /// one (or with unified memory, where there is no split to report).
    pub fn of(
        model_size: u64,
        kv_per_token: Option<u64>,
        total_ram: u64,
        vram: Option<u64>,
        n_ctx: u32,
    ) -> Self {
        Placement {
            fit: Fit::of_model(model_size, kv_per_token, total_ram, n_ctx),
            gpu: vram.map(|v| gpu_share(model_size, kv_per_token, n_ctx, v)),
        }
    }

    /// True when weights and KV cache fit the card whole — the fast case, and
    /// the one worth steering people towards.
    pub fn all_on_gpu(self) -> bool {
        self.gpu.is_some_and(|s| s >= 1.0)
    }

    /// Badge for the model lists. Everything on the card makes system RAM
    /// irrelevant; anything else runs from RAM in whole or in part, so the
    /// RAM verdict governs and the split is reported next to it.
    pub fn badge(self) -> (String, egui::Color32) {
        let skin = crate::theme::skin();
        if self.all_on_gpu() {
            return ("all on GPU".into(), skin.good);
        }
        match (self.fit, self.gpu) {
            (Fit::TooBig, _) => (self.fit.label().into(), skin.bad),
            (Fit::Tight, _) => (self.fit.label().into(), skin.warn),
            (Fit::Fits, Some(s)) if s <= 0.0 => ("CPU only".into(), skin.warn),
            (Fit::Fits, Some(s)) => (format!("{:.0}% on GPU", s * 100.0), skin.warn),
            (Fit::Fits, None) => (self.fit.label().into(), skin.good),
        }
    }

    /// Hover text for the badge: what the verdict means and what to do about
    /// it. The split cases are the ones people write bug reports about.
    pub fn tooltip(self) -> String {
        if self.all_on_gpu() {
            return "Weights and KV cache fit in the card's memory. \
                    Generation runs at VRAM speed, which is the fast case."
                .into();
        }
        match self.fit {
            Fit::TooBig => "Too large for this machine's RAM at the current context size. \
                            Loading it will abort or thrash swap — take a smaller \
                            quantisation or shorten the context."
                .into(),
            Fit::Tight => "Fits, but with little room to spare. Expect swapping if \
                           anything else needs memory."
                .into(),
            Fit::Fits => match self.gpu {
                Some(s) if s <= 0.0 => "Too little of this model would fit on the card to be \
                                        worth the transfers, so all of it runs on the CPU. \
                                        A smaller model or a shorter context may change that."
                    .into(),
                Some(s) => format!(
                    "Only about {:.0}% of the weights fit on the card. The rest is read \
                     from system RAM over PCIe on every token, so the whole model runs \
                     at close to CPU speed. A smaller model, a lower quantisation or a \
                     shorter context may get all of it onto the card.",
                    s * 100.0
                ),
                None => "Fits in system RAM with room to spare.".into(),
            },
        }
    }
}

pub struct QuantTag {
    pub label: &'static str,
    pub color: egui::Color32,
    /// Preference rank for picking a default (lower is better).
    pub pref: u8,
    pub desc: &'static str,
}

/// Hover text explaining what this quantisation means for the user.
pub fn quant_tooltip(name: &str) -> String {
    let mut text = quant_tag(name).desc.to_string();
    let n = name.to_ascii_uppercase();
    if n.contains("UD-") {
        text.push_str("\nUD = unsloth \"dynamic\": slightly better quality at the same size.");
    }
    if n.contains("IQ") {
        text.push_str("\nIQ = importance-quantised: smaller, but a bit slower on CPU.");
    }
    text
}

/// Files that are not standalone chat models (vision projectors etc.).
pub fn is_model_file(name: &str) -> bool {
    !name.to_ascii_lowercase().contains("mmproj")
}

/// Classify a GGUF quantisation from its file name.
pub fn quant_tag(name: &str) -> QuantTag {
    let n = name.to_ascii_uppercase();
    let has = |s: &str| n.contains(s);
    if has("IQ1") {
        QuantTag {
            label: "very low quality",
            color: crate::theme::skin().bad,
            pref: 40,
            desc: "Severely degraded — expect broken output. Avoid unless nothing else fits.",
        }
    } else if has("IQ2") || has("Q2_") || n.ends_with("Q2") {
        QuantTag {
            label: "low quality",
            color: crate::theme::skin().warn,
            pref: 30,
            desc: "Noticeably degraded. A last resort for RAM-starved machines.",
        }
    } else if has("IQ3") || has("Q3_") {
        QuantTag {
            label: "reduced quality",
            color: crate::theme::skin().warn,
            pref: 12,
            desc: "A compromise when Q4 doesn't fit: quality dips but stays usable.",
        }
    } else if has("Q4_K_M") {
        QuantTag {
            label: "recommended",
            color: crate::theme::skin().good,
            pref: 0,
            desc: "The sweet spot: ~95% of full quality at about a third of the size. \
                   Take this one if it fits.",
        }
    } else if has("IQ4") || has("Q4_") {
        QuantTag {
            label: "good",
            color: crate::theme::skin().good,
            pref: 2,
            desc: "Nearly as good as Q4_K_M — a fine choice if that variant is missing or too big.",
        }
    } else if has("Q5_") {
        QuantTag {
            label: "high quality",
            color: crate::theme::skin().good,
            pref: 5,
            desc: "Slightly better than Q4 for noticeably more RAM and slower generation. \
                   Only if you have room to spare.",
        }
    } else if has("Q6_") || has("Q6K") {
        QuantTag {
            label: "near-lossless",
            color: crate::theme::skin().accent,
            pref: 8,
            desc: "Practically indistinguishable from the original — big and slow on CPU, \
                   rarely worth it.",
        }
    } else if has("Q8_") {
        QuantTag {
            label: "near-lossless",
            color: crate::theme::skin().accent,
            pref: 10,
            desc: "Practically indistinguishable from the original — big and slow on CPU, \
                   rarely worth it.",
        }
    } else if has("F16") || has("BF16") || has("F32") {
        QuantTag {
            label: "unquantised",
            color: egui::Color32::GRAY,
            pref: 50,
            desc: "Original full-precision weights — huge and slow. Meant for conversion, \
                   not for running on a CPU.",
        }
    } else {
        QuantTag {
            label: "",
            color: egui::Color32::GRAY,
            pref: 20,
            desc: "Unrecognised quantisation scheme.",
        }
    }
}

/// What a catalog model is worth running for. Tiny models chat but can't
/// code usefully; the coder-tuned ones are the pick for the agent.
#[derive(Clone, Copy, PartialEq)]
pub enum Use {
    /// Good enough to chat with, too small to drive the coding agent.
    Chat,
    /// Instruct model that both chats and codes acceptably.
    ChatCode,
}

impl Use {
    fn codes(self) -> bool {
        self == Use::ChatCode
    }
}

#[derive(Clone)]
pub struct CatalogEntry {
    pub name: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub use_: Use,
}

/// Known-good chat models (file names and sizes verified against the HF API).
pub fn catalog() -> Vec<CatalogEntry> {
    use Use::{Chat, ChatCode};
    vec![
        CatalogEntry {
            name: "Qwen3 0.6B (Q4_K_M)",
            repo: "unsloth/Qwen3-0.6B-GGUF",
            file: "Qwen3-0.6B-Q4_K_M.gguf",
            size: 396_705_472,
            use_: Chat,
        },
        CatalogEntry {
            name: "Qwen3 1.7B (Q4_K_M)",
            repo: "unsloth/Qwen3-1.7B-GGUF",
            file: "Qwen3-1.7B-Q4_K_M.gguf",
            size: 1_107_409_472,
            use_: Chat,
        },
        CatalogEntry {
            name: "Llama 3.2 1B Instruct (Q4_K_M)",
            repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            size: 807_694_464,
            use_: Chat,
        },
        CatalogEntry {
            name: "Llama 3.2 3B Instruct (Q4_K_M)",
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size: 2_019_377_696,
            use_: ChatCode,
        },
        CatalogEntry {
            name: "Gemma 3 4B Instruct (Q4_K_M)",
            repo: "bartowski/google_gemma-3-4b-it-GGUF",
            file: "google_gemma-3-4b-it-Q4_K_M.gguf",
            size: 2_489_758_112,
            use_: ChatCode,
        },
        CatalogEntry {
            name: "Qwen3 4B Instruct 2507 (Q4_K_M)",
            repo: "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
            file: "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            size: 2_497_280_736,
            use_: ChatCode,
        },
        CatalogEntry {
            name: "Mistral 7B Instruct v0.3 (Q4_K_M)",
            repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            size: 4_372_812_000,
            use_: ChatCode,
        },
        // MoE: ~3.3B active params, so it generates at roughly 4B-dense speed
        // while coding far above anything else that fits in 32 GB of RAM.
        CatalogEntry {
            name: "Qwen3 Coder 30B-A3B (Q4_K_M)",
            repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            size: 18_556_689_568,
            use_: ChatCode,
        },
    ]
}

/// Whether the model at `path` can be loaded without blowing past RAM.
/// Used to guard auto-load on startup: loading a too-big model makes
/// llama.cpp abort and takes the whole process down with it (the dreaded
/// "Killed"), which on the last-model auto-load turns into a crash loop.
pub fn safe_to_load(path: &Path, total_ram: u64, n_ctx: u32) -> bool {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    size > 0 && Fit::of_model(size, kv_bytes_per_token(path), total_ram, n_ctx) != Fit::TooBig
}

/// The best chat and coding models this machine can run, by where they run.
///
/// Two answers rather than one, because on a machine with a card they are
/// different models and the trade between them belongs to the user: the
/// `gpu_*` picks fit the card whole and generate several times faster, the
/// `ram_*` picks are the largest that fit system memory and answer better for
/// it. Size stands in for quality within the curated list. On a machine
/// without a discrete GPU the `gpu_*` picks are None and the `ram_*` ones are
/// simply "the picks".
pub struct Proposals {
    pub gpu_chat: Option<CatalogEntry>,
    pub gpu_code: Option<CatalogEntry>,
    pub ram_chat: Option<CatalogEntry>,
    pub ram_code: Option<CatalogEntry>,
}

impl Proposals {
    /// The single pick to act on where only one is wanted (`/get chat`):
    /// the card's, when it has one.
    pub fn chat(&self) -> Option<&CatalogEntry> {
        self.gpu_chat.as_ref().or(self.ram_chat.as_ref())
    }

    pub fn code(&self) -> Option<&CatalogEntry> {
        self.gpu_code.as_ref().or(self.ram_code.as_ref())
    }

    /// Is there anything to show at all?
    pub fn is_empty(&self) -> bool {
        self.chat().is_none() && self.code().is_none()
    }

    /// Does the RAM pick offer something the GPU pick does not? When they are
    /// the same model there is no trade to present, only noise.
    pub fn ram_adds_anything(&self) -> bool {
        let differs = |gpu: &Option<CatalogEntry>, ram: &Option<CatalogEntry>| match (gpu, ram) {
            (Some(g), Some(r)) => r.size > g.size,
            (None, Some(_)) => false, // no card: the RAM pick *is* the pick
            _ => false,
        };
        differs(&self.gpu_chat, &self.ram_chat) || differs(&self.gpu_code, &self.ram_code)
    }
}

/// `vram` is the dedicated VRAM of a discrete GPU, when there is one.
pub fn propose(total_ram: u64, vram: Option<u64>, n_ctx: u32) -> Proposals {
    let fits_ram = |e: &CatalogEntry| Fit::of(e.size, total_ram, n_ctx) == Fit::Fits;
    let largest = |code_only: bool, on_card: bool| {
        catalog()
            .into_iter()
            .filter(|e| !code_only || e.use_.codes())
            .filter(fits_ram)
            .filter(|e| !on_card || vram.is_some_and(|v| fits_vram(e.size, None, n_ctx, v)))
            .max_by_key(|e| e.size)
    };
    Proposals {
        gpu_chat: vram.and_then(|_| largest(false, true)),
        gpu_code: vram.and_then(|_| largest(true, true)),
        ram_chat: largest(false, false),
        ram_code: largest(true, false),
    }
}

#[derive(Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    /// KV-cache bytes per token from the GGUF header, if it could be read.
    pub kv_per_token: Option<u64>,
}

/// Fraction of the model's weights read per token. Dense models read
/// everything; MoE models named like "30B-A3B" only read the active experts.
fn active_fraction(name: &str) -> f32 {
    let n = name.to_ascii_uppercase();
    // Look for "<total>B-A<active>B", e.g. "30B-A3B", "48B-A3B", "235B-A22B".
    let Some(pos) = n.find("B-A") else { return 1.0 };
    let total: f32 = n[..pos]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
        .parse()
        .unwrap_or(0.0);
    let active: f32 = n[pos + 3..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse()
        .unwrap_or(0.0);
    if total > 0.0 && active > 0.0 && active < total {
        // A couple of extra points for always-active layers and routing.
        (active / total + 0.02).min(1.0)
    } else {
        1.0
    }
}

/// Share of raw streaming bandwidth llama.cpp actually reaches. Both the
/// estimate and the measurement that calibrates it use it, so the two are
/// inverses of each other and a measured run reproduces its own estimate.
const EFFICIENCY: f32 = 0.8;

/// Bytes read per generated token: the weights, once, minus the experts a
/// mixture-of-experts model skips.
fn bytes_per_token(name: &str, size: u64) -> f32 {
    size as f32 * active_fraction(name)
}

/// Estimated generation speed on this machine: inference is memory-bound,
/// so tok/s ≈ effective bandwidth / bytes read per token.
pub fn est_tokens_per_sec(name: &str, size: u64, mem_bandwidth: u64) -> Option<f32> {
    if size == 0 || mem_bandwidth == 0 {
        return None;
    }
    Some(mem_bandwidth as f32 * EFFICIENCY / bytes_per_token(name, size))
}

/// The inverse: what streaming bandwidth a finished run implies.
///
/// Reading a model at `tok_per_sec` means moving its bytes that many times a
/// second, so the hardware underneath was at least that fast. Only meaningful
/// for a run whose weights all sat in one place — a split model measures the
/// slower half and the PCIe bus between them, not the card.
pub fn bandwidth_from_run(name: &str, size: u64, tok_per_sec: f32) -> Option<u64> {
    if size == 0 || !tok_per_sec.is_finite() || tok_per_sec <= 0.0 {
        return None;
    }
    Some((bytes_per_token(name, size) * tok_per_sec / EFFICIENCY) as u64)
}

pub fn fmt_tok_s(est: Option<f32>) -> String {
    match est {
        Some(t) if t >= 10.0 => format!("~{t:.0} t/s"),
        Some(t) if t >= 0.3 => format!("~{t:.1} t/s"),
        Some(_) => "<0.3 t/s".into(),
        None => "—".into(),
    }
}

pub fn scan_local(dir: &Path) -> Vec<LocalModel> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "gguf") {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let kv_per_token = kv_bytes_per_token(&path);
                models.push(LocalModel {
                    name,
                    path,
                    size,
                    kv_per_token,
                });
            }
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_preference_ordering() {
        // best-pick preference: Q4_K_M beats everything, Q5 beats Q3, IQ1 last
        let pref = |n: &str| quant_tag(n).pref;
        assert!(pref("m-Q4_K_M.gguf") < pref("m-Q4_K_S.gguf"));
        assert!(pref("m-Q4_K_S.gguf") < pref("m-Q5_K_M.gguf"));
        assert!(pref("m-Q5_K_M.gguf") < pref("m-Q3_K_M.gguf"));
        assert!(pref("m-UD-Q4_K_XL.gguf") < pref("m-UD-IQ2_XXS.gguf"));
        assert!(pref("m-IQ2_XXS.gguf") < pref("m-IQ1_M.gguf"));
        assert_eq!(quant_tag("m-Q4_K_M.gguf").label, "recommended");
    }

    #[test]
    fn moe_active_fraction_from_name() {
        assert!((active_fraction("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf") - 0.12).abs() < 0.03);
        assert!((active_fraction("Kimi-Linear-48B-A3B-Instruct") - 0.083).abs() < 0.03);
        assert_eq!(active_fraction("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), 1.0);
        // MoE model reads far fewer bytes per token -> much faster estimate
        let bw = 20_000_000_000u64;
        let dense = est_tokens_per_sec("x-7B", 4_400_000_000, bw).unwrap();
        let moe = est_tokens_per_sec("x-30B-A3B", 18_600_000_000, bw).unwrap();
        assert!(moe > dense * 1.5);
    }

    /// An 8 GB card (a 2070 Super, say) in a 32 GB box: the pick is the
    /// largest model that fits the *card*, not the largest that fits RAM —
    /// a 7B living entirely in VRAM beats a 19 GB model reading most of
    /// itself over PCIe every token. Nothing in the catalog codes and fits
    /// 8 GB, so the coding pick still falls back to the RAM rule.
    #[test]
    fn proposals_prefer_what_fits_in_vram() {
        let ctx = 16384;
        let ram = 64 * 1024 * 1024 * 1024;
        let vram = 8 * 1024 * 1024 * 1024;

        let with_gpu = propose(ram, Some(vram), ctx);
        for pick in [with_gpu.chat().unwrap(), with_gpu.code().unwrap()] {
            assert!(
                fits_vram(pick.size, None, ctx, vram),
                "{} does not fit an 8 GB card",
                pick.name
            );
        }

        // Same box without the card: the RAM rule, unchanged — and it reaches
        // for the 19 GB model the card could never hold.
        let cpu_only = propose(ram, None, ctx).chat().unwrap().clone();
        assert!(cpu_only.size > with_gpu.chat().unwrap().size);

        // That bigger model is still offered, as the explicit RAM option, so
        // the choice is presented rather than made for the user.
        assert!(with_gpu.ram_adds_anything());
        assert_eq!(with_gpu.ram_chat.as_ref().unwrap().name, cpu_only.name);

        // Without a card there is no second option to present.
        assert!(!propose(ram, None, ctx).ram_adds_anything());
    }

    /// Layer counting is what keeps a load from aborting with an
    /// out-of-memory error. A 4.4 GB 7B and its 2 GB of KV cache still fit an
    /// 8 GB card whole; a 19 GB MoE on the same card can only go part-way.
    #[test]
    fn gpu_layers_fill_the_card() {
        let ctx = 16384;
        let gb = 1024 * 1024 * 1024;
        let small = GgufDims {
            layers: 32,
            kv_bytes_per_token: 131_072, // 7B-class: 32 layers x 8 KV heads
        };
        let big = GgufDims {
            layers: 48,
            kv_bytes_per_token: 98_304, // 30B MoE: only 4 KV heads a layer
        };

        // Everything, output layer included — llama.cpp's "all".
        assert!(gpu_layers(4_372_812_000, &small, ctx, 8 * gb) > small.layers as u32);

        let partial = gpu_layers(18_556_689_568, &big, ctx, 8 * gb);
        assert!(
            (10..big.layers as u32).contains(&partial),
            "expected part of the 30B on an 8 GB card, got {partial}"
        );
        // Room for the compute buffers and little else: stay on the CPU.
        assert_eq!(gpu_layers(4_372_812_000, &small, ctx, gb), 0);
    }

    /// The badge and the tok/s estimate must agree with the loader, or the
    /// list promises one thing and llama.cpp does another. `gpu_share` is the
    /// shared answer, so it is pinned against `gpu_layers` at both ends.
    #[test]
    fn gpu_share_tracks_the_loader() {
        let ctx = 16384;
        let gb = 1024 * 1024 * 1024;
        let small = GgufDims {
            layers: 32,
            kv_bytes_per_token: 131_072,
        };
        let big = GgufDims {
            layers: 48,
            kv_bytes_per_token: 98_304,
        };

        // A 7B on an 8 GB card: all layers, and the share agrees.
        let (size, vram) = (4_372_812_000u64, 8 * gb);
        assert!(gpu_layers(size, &small, ctx, vram) > small.layers as u32);
        assert_eq!(gpu_share(size, None, ctx, vram), 1.0);

        // A 19 GB MoE on the same card: part of it, and the share says so
        // rather than reading as a comfortable "fits".
        let moe = 18_556_689_568u64;
        assert!(gpu_layers(moe, &big, ctx, vram) < big.layers as u32);
        let share = gpu_share(moe, None, ctx, vram);
        assert!(
            (0.2..0.6).contains(&share),
            "expected part of the 30B on an 8 GB card, got {share}"
        );

        // Below the offload floor the loader keeps everything in RAM, so the
        // estimate must not hand out partial GPU credit either.
        assert_eq!(gpu_layers(size, &small, ctx, gb), 0);
        assert_eq!(gpu_share(size, None, ctx, gb), 0.0);
    }

    /// The number the System panel puts in front of the user has to be one
    /// the fit rule actually accepts, at either edge.
    #[test]
    fn largest_fitting_vram_round_trips() {
        let gb = 1024 * 1024 * 1024;
        for vram in [6 * gb, 8 * gb, 12 * gb, 16 * gb, 24 * gb] {
            for ctx in [4096u32, 16384, 32768] {
                let largest = largest_fitting_vram(vram, ctx);
                assert!(
                    fits_vram(largest, None, ctx, vram),
                    "{largest} bytes should fit {vram} at {ctx}"
                );
                assert!(
                    !fits_vram(largest + 256 * 1024 * 1024, None, ctx, vram),
                    "{largest} should be the largest that fits {vram} at {ctx}"
                );
            }
        }
        // A card with nothing but overhead to give holds nothing.
        assert_eq!(largest_fitting_vram(GPU_OVERHEAD / 2, 4096), 0);
    }

    /// The speed estimate and the measurement that calibrates it are one
    /// formula read in two directions, so a run at the estimated rate has to
    /// report back the bandwidth it was estimated from.
    #[test]
    fn bandwidth_measurement_inverts_the_estimate() {
        let bw = 800_000_000_000u64;
        for name in ["Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "x-30B-A3B-Q4_K_M"] {
            let size = 4_372_812_000;
            let tok_s = est_tokens_per_sec(name, size, bw).unwrap();
            let back = bandwidth_from_run(name, size, tok_s).unwrap();
            assert!(
                (back as f64 - bw as f64).abs() / (bw as f64) < 0.01,
                "{name}: {back} should round-trip to {bw}"
            );
        }
        assert_eq!(bandwidth_from_run("x", 0, 10.0), None);
        assert_eq!(bandwidth_from_run("x", 100, 0.0), None);
    }

    /// The badge is the one line a user reads before downloading 18 GB. A
    /// model that fits RAM but not the card must not read like a green light.
    #[test]
    fn badge_separates_fitting_ram_from_fitting_the_card() {
        let gb = 1024 * 1024 * 1024;
        let (ram, vram, ctx) = (64 * gb, 16 * gb, 16384);
        let on_card = Placement::of(4_372_812_000, None, ram, Some(vram), ctx);
        assert!(on_card.all_on_gpu());
        assert_eq!(on_card.badge().0, "all on GPU");

        // The case that started this: comfortable in 64 GB of RAM, a split on
        // a 16 GB card, and ten times slower for it.
        let split = Placement::of(18_556_689_568, None, ram, Some(vram), ctx);
        assert_eq!(split.fit, Fit::Fits);
        assert!(!split.all_on_gpu());
        assert!(
            split.badge().0.ends_with("% on GPU"),
            "got {}",
            split.badge().0
        );

        // Same machine without a card: the RAM verdict, as before.
        assert_eq!(
            Placement::of(18_556_689_568, None, ram, None, ctx)
                .badge()
                .0,
            "fits"
        );
        // Not loadable at all outranks any placement.
        assert_eq!(
            Placement::of(60 * gb, None, ram, Some(vram), ctx).badge().0,
            "too big"
        );
    }

    #[test]
    fn proposals_scale_with_ram() {
        let ctx = 16384;
        // A tiny box: chat gets the largest fitting small model, and nothing
        // big enough to code comfortably may be available.
        let small = propose(4 * 1024 * 1024 * 1024, None, ctx);
        assert!(small.chat().is_some());
        assert!(
            small.chat().unwrap().size <= 3_000_000_000,
            "chat pick must actually fit 4 GB"
        );

        // A large box: chat and code both resolve, and each is the biggest
        // fitting entry of its kind — here the 30B coder for both.
        let big = propose(64 * 1024 * 1024 * 1024, None, ctx);
        assert_eq!(big.chat().unwrap().name, "Qwen3 Coder 30B-A3B (Q4_K_M)");
        assert_eq!(big.code().unwrap().name, "Qwen3 Coder 30B-A3B (Q4_K_M)");

        // The coding pick, when present, is always a code-capable entry.
        if let Some(code) = propose(8 * 1024 * 1024 * 1024, None, ctx).code() {
            assert!(code.use_.codes());
        }
    }

    #[test]
    fn fit_tightens_with_context() {
        // The run that started all this: an 11 GB model on a 16 GB box. At a
        // short context it loads; at the default 16k it should read too big,
        // because the KV cache no longer fits alongside the weights.
        let model = 11 * 1024 * 1024 * 1024;
        let ram = 16 * 1024 * 1024 * 1024;
        assert_eq!(Fit::of(model, ram, 4096), Fit::Tight);
        assert_eq!(Fit::of(model, ram, 16384), Fit::TooBig);
        // A small model is unaffected by context at this scale.
        let small = 4 * 1024 * 1024 * 1024;
        assert_eq!(Fit::of(small, ram, 16384), Fit::Fits);
        // With the header's real KV cost, a 17.3 GB MoE with 96 KB/token
        // (Qwen3-Coder-30B-A3B) is fine at 32k on 32 GB: ~21.4 GB needed.
        let moe = 17_300 * 1024 * 1024;
        let ram32 = 32 * 1024 * 1024 * 1024;
        assert_eq!(Fit::of(moe, ram32, 32768), Fit::TooBig); // the old guess
        assert_eq!(Fit::of_model(moe, Some(98_304), ram32, 32768), Fit::Fits);
        assert_eq!(Fit::of_model(moe, Some(98_304), ram32, 65_536), Fit::Tight);
        assert_eq!(
            Fit::of_model(moe, Some(98_304), ram32, 131_072),
            Fit::TooBig
        );
    }

    /// Reads the real header of a local model when one is on disk (the
    /// developer's box), and is a no-op elsewhere.
    #[test]
    fn kv_per_token_from_gguf_header() {
        let dir = crate::config::models_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "gguf") {
                continue;
            }
            let kv = kv_bytes_per_token(&p).expect("header parses");
            // Any real LLM: between 1 KB (tiny) and 2 MB (huge dense) per token.
            assert!((1024..=2 << 20).contains(&kv), "{}: {kv}", p.display());
            if p.to_string_lossy().contains("Qwen3-Coder-30B-A3B") {
                // 48 layers × 4 KV heads × 128 dims × (K+V) × f16
                assert_eq!(kv, 48 * 4 * 128 * 2 * 2);
            }
        }
    }

    #[test]
    fn mmproj_files_are_not_models() {
        assert!(!is_model_file("mmproj-BF16.gguf"));
        assert!(!is_model_file("mmproj-model-f16.gguf"));
        assert!(is_model_file("Qwen3-4B-Q4_K_M.gguf"));
    }
}
