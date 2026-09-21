use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Starting assumption for a discrete GPU's VRAM bandwidth, in bytes/s.
///
/// Neither Vulkan nor ggml reports it, and llama.cpp gives us no way to
/// benchmark VRAM the way `measure_read_bandwidth` benchmarks RAM, so the
/// estimates need a number before the machine has ever run anything.
/// 200 GB/s sits at or below nearly every discrete card of the last several
/// years (an RTX 2070 Super does 448, a modest RTX 4060 272), so a fresh
/// install errs low rather than promising speed the machine cannot deliver.
///
/// It is only the starting point: `calibrate_gpu` replaces it with what the
/// card actually did as soon as one model has run entirely on it, and on a
/// fast card the difference is large — a 5070 Ti streams over four times this.
const ASSUMED_GPU_BANDWIDTH: u64 = 200_000_000_000;

/// Bounds a measured VRAM bandwidth has to land in to be believed. Below the
/// floor something other than the card was the bottleneck; above the ceiling
/// (well past HBM3) the model's own size or its active-expert share must be
/// wrong. Either way a bad number would poison every estimate in the UI, so
/// it is dropped rather than stored.
const PLAUSIBLE_GPU_BANDWIDTH: std::ops::RangeInclusive<u64> = 20_000_000_000..=4_000_000_000_000;

/// Generated tokens a run needs before its rate says anything about hardware.
/// Shorter runs are mostly sampling and prompt-decode tail.
const MIN_CALIBRATION_TOKENS: usize = 64;

/// A VRAM bandwidth measured from a real run, remembered across sessions.
///
/// The card's name travels with it so that swapping GPUs re-measures instead
/// of silently inheriting the old one's number.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuBandwidth {
    pub gpu: String,
    pub bytes_per_sec: u64,
}

/// A GPU llama.cpp can offload layers to. Only ever `Some` in a build with a
/// GPU backend compiled in (`--features vulkan` or `--features cuda`, or
/// Metal on macOS) *and* on a machine that actually has one.
#[derive(Clone, Debug)]
pub struct Gpu {
    /// What the driver calls it, e.g. "NVIDIA GeForce RTX 2070 SUPER".
    pub name: String,
    /// The backend that found it: "Vulkan", "Metal", "CUDA".
    pub backend: String,
    pub vram_total: u64,
    /// VRAM the driver reports free — what we can actually take. The desktop
    /// compositor and every other open app already hold some of the total.
    pub vram_free: u64,
    /// Shares one memory pool with the CPU (integrated GPUs, Apple Silicon).
    /// There is no separate budget to size against and no transfer to pay
    /// for, so we leave the layer split to llama.cpp on these.
    pub unified: bool,
}

impl Gpu {
    /// VRAM we are willing to fill: what the driver says is free, but never
    /// more than the card has.
    pub fn budget(&self) -> u64 {
        self.vram_free.min(self.vram_total)
    }
}

/// The GPU this machine offers llama.cpp, detected once.
///
/// Enumerating devices spins up the GPU backend loader, so the result is
/// cached — which also means `vram_free` is a snapshot from startup. That is
/// the conservative direction: it was taken before we loaded anything ourselves.
pub fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(detect_gpu).as_ref()
}

/// Pick the best device ggml reports: a discrete GPU over an integrated one,
/// and the roomiest when there are several. `None` when the build has no GPU
/// backend (then ggml lists only the CPU) or the machine has no GPU.
fn detect_gpu() -> Option<Gpu> {
    use llama_cpp_2::{LlamaBackendDeviceType as Type, list_llama_ggml_backend_devices};

    list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| matches!(d.device_type, Type::Gpu | Type::IntegratedGpu))
        .max_by_key(|d| (d.device_type == Type::Gpu, d.memory_total))
        .map(|d| Gpu {
            name: if d.description.is_empty() {
                d.name.clone()
            } else {
                d.description.clone()
            },
            unified: d.device_type == Type::IntegratedGpu || d.backend == "Metal",
            backend: d.backend,
            vram_total: d.memory_total as u64,
            vram_free: d.memory_free as u64,
        })
}

pub struct HardwareProfile {
    pub total_ram: u64,
    pub cores: usize,
    /// Physical cores — llama.cpp runs fastest with one thread per physical
    /// core; using SMT threads slows generation down.
    pub physical_cores: usize,
    pub cpu_brand: String,
    /// Measured memory read bandwidth in bytes/s — CPU token generation is
    /// bound by it, so it drives the tok/s estimates.
    pub mem_bandwidth: u64,
    /// The GPU we can offload to, if this build and machine have one.
    pub gpu: Option<&'static Gpu>,
    /// VRAM streaming bandwidth in bytes/s: `ASSUMED_GPU_BANDWIDTH` until a
    /// run that lived entirely on the card measures the real thing.
    pub gpu_bandwidth: u64,
    /// Whether `gpu_bandwidth` is a measurement or still the assumption. The
    /// System panel says which, because the difference on a fast card is the
    /// difference between an estimate people trust and one they don't.
    pub gpu_bandwidth_measured: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        sys.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_else(|| "unknown CPU".to_string());
        let cores = sys.cpus().len().max(1);
        let physical_cores = System::physical_core_count().unwrap_or(cores / 2).max(1);
        Self {
            total_ram: sys.total_memory(),
            cores,
            physical_cores,
            cpu_brand,
            mem_bandwidth: measure_read_bandwidth(physical_cores.min(4)),
            gpu: gpu(),
            gpu_bandwidth: ASSUMED_GPU_BANDWIDTH,
            gpu_bandwidth_measured: false,
        }
    }

    /// Adopt a bandwidth measured in an earlier session.
    ///
    /// Ignored when it was taken on a different card or lands outside
    /// `PLAUSIBLE_GPU_BANDWIDTH`, so a config carried to another machine, or
    /// edited by hand, cannot quietly skew every estimate in the UI.
    pub fn adopt_gpu_bandwidth(&mut self, stored: &GpuBandwidth) -> bool {
        let Some(gpu) = self.gpu.filter(|g| !g.unified) else {
            return false;
        };
        if gpu.name != stored.gpu || !PLAUSIBLE_GPU_BANDWIDTH.contains(&stored.bytes_per_sec) {
            return false;
        }
        self.gpu_bandwidth = stored.bytes_per_sec;
        self.gpu_bandwidth_measured = true;
        true
    }

    /// Learn the card's real streaming bandwidth from a finished generation.
    ///
    /// Returns the value to persist when one was adopted, `None` otherwise.
    /// Three conditions have to hold for a run to say anything about the
    /// card: every layer was on it (a split run measures the PCIe bus and the
    /// CPU behind it), it generated enough tokens to be more than sampling
    /// overhead, and the result is physically plausible. Past that, only a
    /// faster result replaces a measurement we already have — background load
    /// can drag an observed rate below the hardware but never above it.
    pub fn calibrate_gpu(
        &mut self,
        model: &str,
        size: u64,
        gen_tokens: usize,
        gen_secs: f32,
        all_on_gpu: bool,
    ) -> Option<GpuBandwidth> {
        let gpu = self.gpu.filter(|g| !g.unified)?;
        if !all_on_gpu || gen_tokens < MIN_CALIBRATION_TOKENS || gen_secs <= 0.0 {
            return None;
        }
        let measured =
            crate::models::bandwidth_from_run(model, size, gen_tokens as f32 / gen_secs)?;
        if !PLAUSIBLE_GPU_BANDWIDTH.contains(&measured) {
            return None;
        }
        // The assumption is a floor, not a measurement, so the first real run
        // replaces it in either direction; later ones only raise it.
        if self.gpu_bandwidth_measured && measured <= self.gpu_bandwidth {
            return None;
        }
        self.gpu_bandwidth = measured;
        self.gpu_bandwidth_measured = true;
        Some(GpuBandwidth {
            gpu: gpu.name.clone(),
            bytes_per_sec: measured,
        })
    }

    /// Bandwidth that actually feeds generation for a model of `size` bytes.
    ///
    /// Every token reads the weights once, wherever they live: the share that
    /// fits in VRAM streams at GPU speed and the remainder at system-memory
    /// speed, so the two times add. Without a discrete GPU this is just the
    /// measured memory bandwidth, which is what drove the estimates before.
    ///
    /// The share is the one the loader will actually pick, so `n_ctx` counts:
    /// the KV cache sits in VRAM next to the layers it belongs to and takes
    /// room the weights would otherwise have had.
    pub fn bandwidth_for(&self, size: u64, n_ctx: u32) -> u64 {
        let Some(gpu) = self.gpu else {
            return self.mem_bandwidth;
        };
        // Unified memory: one pool, one bandwidth — already measured.
        if gpu.unified || size == 0 || self.mem_bandwidth == 0 {
            return self.mem_bandwidth;
        }
        // The same share the fit badge reports, so the speed the list
        // promises and the placement it shows come from one calculation.
        let on_gpu = crate::models::gpu_share(size, None, n_ctx, gpu.budget()) as f64;
        let secs_per_byte =
            on_gpu / self.gpu_bandwidth as f64 + (1.0 - on_gpu) / self.mem_bandwidth as f64;
        if secs_per_byte <= 0.0 {
            return self.mem_bandwidth;
        }
        (1.0 / secs_per_byte) as u64
    }

    /// VRAM to size model recommendations against: a dedicated pool only.
    /// On unified memory the "VRAM" a driver reports is system RAM under
    /// another name, and `total_ram` already covers it.
    pub fn dedicated_vram(&self) -> Option<u64> {
        self.gpu.filter(|g| !g.unified).map(|g| g.budget())
    }

    /// How fast this machine can stream a model's weights, and whether that
    /// is a measurement or still a guess. Every tok/s figure in the UI comes
    /// out of these two numbers, so the System panel shows both.
    pub fn bandwidth_summary(&self) -> String {
        let mut line = format!(
            "Memory bandwidth: {}/s measured",
            fmt_bytes(self.mem_bandwidth)
        );
        if self.gpu.is_some_and(|g| !g.unified) {
            line.push_str(&format!(
                " · VRAM: {}/s {}",
                fmt_bytes(self.gpu_bandwidth),
                if self.gpu_bandwidth_measured {
                    "measured"
                } else {
                    "assumed"
                }
            ));
        }
        line
    }

    /// What a discrete card can hold whole at `n_ctx`, and what that means —
    /// the answer to "which model should I download", stated once instead of
    /// left to be inferred from a column of badges.
    pub fn vram_summary(&self, n_ctx: u32) -> Option<String> {
        let gpu = self.gpu.filter(|g| !g.unified)?;
        let largest = crate::models::largest_fitting_vram(gpu.budget(), n_ctx);
        let mut line = format!(
            "Models up to {} run entirely on the card at {n_ctx} tokens of context \
             ({} free of {} when offgrid started). Anything larger is split with \
             system RAM and generates at a fraction of the speed — a shorter \
             context raises this limit.",
            fmt_bytes(largest),
            fmt_bytes(gpu.vram_free),
            fmt_bytes(gpu.vram_total),
        );
        if !self.gpu_bandwidth_measured {
            line.push_str(
                " The card's bandwidth is still an assumption, so its tok/s estimates \
                 read low; offgrid measures the real figure the first time a model \
                 runs entirely on it.",
            );
        }
        Some(line)
    }

    /// One line describing the accelerator, for the System panel.
    pub fn gpu_summary(&self) -> String {
        match self.gpu {
            Some(g) if g.unified => format!("{} ({}, shared memory)", g.name, g.backend),
            Some(g) => format!(
                "{} ({}) · {} VRAM, {} free",
                g.name,
                g.backend,
                fmt_bytes(g.vram_total),
                fmt_bytes(g.vram_free)
            ),
            None if cfg!(any(feature = "vulkan", feature = "cuda")) => {
                "none found — running on the CPU (no GPU device)".into()
            }
            None => {
                "not built in — running on the CPU (build with --features vulkan or cuda)".into()
            }
        }
    }
}

/// Free space on the filesystem holding `path` — the disk whose mount point
/// is the longest prefix of it (on Windows that is the drive, on unix it
/// picks `/home` over `/` when both match). None if no mount point matches
/// or the platform reports nothing.
pub fn free_space(path: &std::path::Path) -> Option<u64> {
    sysinfo::Disks::new_with_refreshed_list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// Rough aggregate memory read bandwidth: several threads stream through
/// their own buffers for ~120ms. Cheap, runs once at startup.
fn measure_read_bandwidth(threads: usize) -> u64 {
    const BUF: usize = 64 * 1024 * 1024;
    let handles: Vec<_> = (0..threads.max(1))
        .map(|_| {
            std::thread::spawn(|| {
                let buf = vec![1u8; BUF];
                let (words, _) = unsafe { buf.align_to::<u64>() }.1.split_at(BUF / 8 - 8);
                let mut sum = 0u64;
                let mut bytes = 0u64;
                let start = std::time::Instant::now();
                while start.elapsed() < std::time::Duration::from_millis(120) {
                    for w in words {
                        sum = sum.wrapping_add(*w);
                    }
                    bytes += words.len() as u64 * 8;
                }
                std::hint::black_box(sum);
                (bytes, start.elapsed().as_secs_f64())
            })
        })
        .collect();
    let mut total = 0u64;
    let mut slowest = 0.0f64;
    for h in handles {
        if let Ok((bytes, secs)) = h.join() {
            total += bytes;
            slowest = slowest.max(secs);
        }
    }
    if slowest > 0.0 {
        (total as f64 / slowest) as u64
    } else {
        20_000_000_000 // fall back to a modest dual-channel DDR4 guess
    }
}

/// Higher-precision variant for progress displays, where 0.1 GB steps are too
/// coarse to see movement on large downloads.
pub fn fmt_bytes_precise(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else {
        format!("{:.0} MB", b / MB)
    }
}

pub fn fmt_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else {
        format!("{:.0} MB", b / MB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn profile(gpu: Option<&'static Gpu>) -> HardwareProfile {
        HardwareProfile {
            total_ram: 32 * GB,
            cores: 16,
            physical_cores: 8,
            cpu_brand: "test".into(),
            mem_bandwidth: 20_000_000_000,
            gpu,
            gpu_bandwidth: ASSUMED_GPU_BANDWIDTH,
            gpu_bandwidth_measured: false,
        }
    }

    /// A model that fits the card reads at GPU speed. One several times its
    /// size spends most of every token on the CPU side of the split, so the
    /// estimate falls back towards RAM speed instead of promising the card's.
    #[test]
    fn bandwidth_follows_the_offloaded_share() {
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu {
            name: "test card".into(),
            backend: "Vulkan".into(),
            vram_total: 8 * GB,
            vram_free: 8 * GB,
            unified: false,
        }));
        let hw = profile(Some(gpu));

        assert!(hw.bandwidth_for(2 * GB, 4096) > hw.mem_bandwidth * 5);
        assert!(hw.bandwidth_for(30 * GB, 4096) < hw.mem_bandwidth * 2);
        // A longer context pushes weights out of VRAM, so the same model gets
        // slower — the KV cache is competing for the same bytes.
        assert!(hw.bandwidth_for(6 * GB, 32768) < hw.bandwidth_for(6 * GB, 4096));

        // No card: the measured memory bandwidth, exactly as before.
        let cpu = profile(None);
        assert_eq!(cpu.bandwidth_for(6 * GB, 4096), cpu.mem_bandwidth);
    }

    fn card(name: &str) -> &'static Gpu {
        Box::leak(Box::new(Gpu {
            name: name.into(),
            backend: "CUDA".into(),
            vram_total: 16 * GB,
            vram_free: 15 * GB,
            unified: false,
        }))
    }

    /// A 4.4 GB model generating 55 tok/s is a card doing roughly 300 GB/s.
    /// Nothing else on the machine can tell us that, so the run has to.
    #[test]
    fn calibration_learns_from_a_full_gpu_run() {
        let mut hw = profile(Some(card("test card")));
        assert!(!hw.gpu_bandwidth_measured);
        let before = hw.gpu_bandwidth;

        let measured = hw
            .calibrate_gpu(
                "Mistral-7B-Q4_K_M.gguf",
                4_372_812_000,
                200,
                200.0 / 55.0,
                true,
            )
            .expect("a full-GPU run should measure the card");
        assert!(hw.gpu_bandwidth_measured);
        assert!(hw.gpu_bandwidth > before);
        assert_eq!(measured.gpu, "test card");
        assert_eq!(measured.bytes_per_sec, hw.gpu_bandwidth);

        // A slower run afterwards is interference, not the hardware getting
        // worse, so the best measurement stands.
        let kept = hw.gpu_bandwidth;
        assert!(
            hw.calibrate_gpu(
                "Mistral-7B-Q4_K_M.gguf",
                4_372_812_000,
                200,
                200.0 / 20.0,
                true
            )
            .is_none()
        );
        assert_eq!(hw.gpu_bandwidth, kept);
    }

    /// Everything that would measure something other than the card.
    #[test]
    fn calibration_rejects_runs_that_prove_nothing() {
        let model = "Mistral-7B-Q4_K_M.gguf";
        let size = 4_372_812_000;

        // Split across card and CPU: measures the PCIe bus, not the VRAM.
        let mut hw = profile(Some(card("test card")));
        assert!(hw.calibrate_gpu(model, size, 200, 4.0, false).is_none());

        // Too short to be anything but sampling overhead.
        let mut hw = profile(Some(card("test card")));
        assert!(hw.calibrate_gpu(model, size, 8, 0.2, true).is_none());

        // Physically impossible — a mis-detected size or expert share.
        let mut hw = profile(Some(card("test card")));
        assert!(hw.calibrate_gpu(model, size, 10_000, 0.5, true).is_none());

        // No discrete card to measure.
        let mut hw = profile(None);
        assert!(hw.calibrate_gpu(model, size, 200, 4.0, true).is_none());

        for hw in [profile(Some(card("test card"))), profile(None)] {
            assert!(!hw.gpu_bandwidth_measured);
        }
    }

    /// A config that travelled to another machine must not quietly rewrite
    /// every estimate on it.
    #[test]
    fn stored_bandwidth_only_applies_to_the_same_card() {
        let mut hw = profile(Some(card("RTX 5070 Ti")));
        assert!(!hw.adopt_gpu_bandwidth(&GpuBandwidth {
            gpu: "RTX 3060".into(),
            bytes_per_sec: 360_000_000_000,
        }));
        assert!(!hw.adopt_gpu_bandwidth(&GpuBandwidth {
            gpu: "RTX 5070 Ti".into(),
            bytes_per_sec: 99_000_000_000_000,
        }));
        assert!(!hw.gpu_bandwidth_measured);

        assert!(hw.adopt_gpu_bandwidth(&GpuBandwidth {
            gpu: "RTX 5070 Ti".into(),
            bytes_per_sec: 870_000_000_000,
        }));
        assert_eq!(hw.gpu_bandwidth, 870_000_000_000);
        assert!(hw.gpu_bandwidth_measured);
        // And it shows up where the estimates read it.
        assert!(hw.bandwidth_for(4 * GB, 4096) > 400_000_000_000);
    }
}
