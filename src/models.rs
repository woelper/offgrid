use std::path::{Path, PathBuf};

use eframe::egui;

/// Rough memory needed on top of the model file itself (KV cache, activations).
const OVERHEAD: u64 = 3 * 1024 * 1024 * 1024 / 2; // 1.5 GB

#[derive(Clone, Copy, PartialEq)]
pub enum Fit {
    Fits,
    Tight,
    TooBig,
}

impl Fit {
    pub fn of(model_size: u64, total_ram: u64) -> Self {
        let needed = model_size + OVERHEAD;
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

    pub fn badge(self) -> (&'static str, egui::Color32) {
        let color = match self {
            Fit::Fits => crate::theme::skin().good,
            Fit::Tight => crate::theme::skin().warn,
            Fit::TooBig => crate::theme::skin().bad,
        };
        (self.label(), color)
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
pub fn safe_to_load(path: &Path, total_ram: u64) -> bool {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    size > 0 && Fit::of(size, total_ram) != Fit::TooBig
}

/// The best chat and coding models this machine can comfortably run. Each is
/// the largest catalog entry of that kind that still fits — size stands in for
/// quality within the curated list. Either may be None on a very small box.
pub struct Proposals {
    pub chat: Option<CatalogEntry>,
    pub code: Option<CatalogEntry>,
}

pub fn propose(total_ram: u64) -> Proposals {
    let fits = |e: &CatalogEntry| Fit::of(e.size, total_ram) == Fit::Fits;
    Proposals {
        chat: catalog()
            .into_iter()
            .filter(fits)
            .max_by_key(|e| e.size),
        code: catalog()
            .into_iter()
            .filter(|e| e.use_.codes() && fits(e))
            .max_by_key(|e| e.size),
    }
}

#[derive(Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
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

/// Estimated generation speed on this machine: CPU inference is memory-bound,
/// so tok/s ≈ effective bandwidth / bytes read per token.
pub fn est_tokens_per_sec(name: &str, size: u64, mem_bandwidth: u64) -> Option<f32> {
    if size == 0 || mem_bandwidth == 0 {
        return None;
    }
    let bytes_per_token = size as f32 * active_fraction(name);
    // llama.cpp reaches roughly 80% of the raw streaming bandwidth.
    Some(mem_bandwidth as f32 * 0.8 / bytes_per_token)
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
                models.push(LocalModel { name, path, size });
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

    #[test]
    fn proposals_scale_with_ram() {
        // A tiny box: chat gets the largest fitting small model, and nothing
        // big enough to code comfortably may be available.
        let small = propose(4 * 1024 * 1024 * 1024);
        assert!(small.chat.is_some());
        assert!(
            small.chat.as_ref().unwrap().size <= 3_000_000_000,
            "chat pick must actually fit 4 GB"
        );

        // A large box: chat and code both resolve, and each is the biggest
        // fitting entry of its kind — here the 30B coder for both.
        let big = propose(64 * 1024 * 1024 * 1024);
        assert_eq!(big.chat.unwrap().name, "Qwen3 Coder 30B-A3B (Q4_K_M)");
        assert_eq!(big.code.unwrap().name, "Qwen3 Coder 30B-A3B (Q4_K_M)");

        // The coding pick, when present, is always a code-capable entry.
        if let Some(code) = propose(8 * 1024 * 1024 * 1024).code {
            assert!(code.use_.codes());
        }
    }

    #[test]
    fn mmproj_files_are_not_models() {
        assert!(!is_model_file("mmproj-BF16.gguf"));
        assert!(!is_model_file("mmproj-model-f16.gguf"));
        assert!(is_model_file("Qwen3-4B-Q4_K_M.gguf"));
    }
}
