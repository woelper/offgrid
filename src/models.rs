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

    pub fn badge(self) -> (&'static str, egui::Color32) {
        match self {
            Fit::Fits => ("fits", crate::theme::GOOD_GREEN),
            Fit::Tight => ("tight", crate::theme::WARN_AMBER),
            Fit::TooBig => ("too big", crate::theme::BAD_RED),
        }
    }
}

#[derive(Clone)]
pub struct CatalogEntry {
    pub name: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
}

/// Known-good chat models (file names and sizes verified against the HF API).
pub fn catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry {
            name: "Qwen3 0.6B (Q4_K_M)",
            repo: "unsloth/Qwen3-0.6B-GGUF",
            file: "Qwen3-0.6B-Q4_K_M.gguf",
            size: 396_705_472,
        },
        CatalogEntry {
            name: "Qwen3 1.7B (Q4_K_M)",
            repo: "unsloth/Qwen3-1.7B-GGUF",
            file: "Qwen3-1.7B-Q4_K_M.gguf",
            size: 1_107_409_472,
        },
        CatalogEntry {
            name: "Llama 3.2 1B Instruct (Q4_K_M)",
            repo: "bartowski/Llama-3.2-1B-Instruct-GGUF",
            file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            size: 807_694_464,
        },
        CatalogEntry {
            name: "Llama 3.2 3B Instruct (Q4_K_M)",
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size: 2_019_377_696,
        },
        CatalogEntry {
            name: "Gemma 3 4B Instruct (Q4_K_M)",
            repo: "bartowski/google_gemma-3-4b-it-GGUF",
            file: "google_gemma-3-4b-it-Q4_K_M.gguf",
            size: 2_489_758_112,
        },
        CatalogEntry {
            name: "Qwen3 4B Instruct 2507 (Q4_K_M)",
            repo: "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
            file: "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            size: 2_497_280_736,
        },
        CatalogEntry {
            name: "Mistral 7B Instruct v0.3 (Q4_K_M)",
            repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            size: 4_372_812_000,
        },
        // MoE: ~3.3B active params, so it generates at roughly 4B-dense speed
        // while coding far above anything else that fits in 32 GB of RAM.
        CatalogEntry {
            name: "Qwen3 Coder 30B-A3B (Q4_K_M)",
            repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            size: 18_556_689_568,
        },
    ]
}

/// The largest curated model that comfortably fits in RAM.
pub fn recommended(total_ram: u64) -> Option<CatalogEntry> {
    catalog()
        .into_iter()
        .filter(|e| Fit::of(e.size, total_ram) == Fit::Fits)
        .max_by_key(|e| e.size)
}

#[derive(Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
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
