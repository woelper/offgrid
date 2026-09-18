use std::sync::OnceLock;

use sysinfo::System;

/// Assumed streaming bandwidth of a discrete GPU's VRAM, in bytes/s.
///
/// Neither Vulkan nor ggml reports it, and llama.cpp gives us no way to
/// benchmark VRAM the way `measure_read_bandwidth` benchmarks RAM, so the
/// tok/s estimates need one number. 200 GB/s sits at or below nearly every
/// discrete card of the last several years (an RTX 2070 Super does 448, a
/// modest RTX 4060 272), so the estimate errs low rather than promising
/// speed the machine cannot deliver.
const DISCRETE_GPU_BANDWIDTH: u64 = 200_000_000_000;

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
        }
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
        let vram = gpu.budget().saturating_sub(crate::models::GPU_OVERHEAD);
        let footprint = size + crate::models::kv_size(size, None, n_ctx);
        let on_gpu = (vram as f64 / footprint as f64).min(1.0);
        let secs_per_byte =
            on_gpu / DISCRETE_GPU_BANDWIDTH as f64 + (1.0 - on_gpu) / self.mem_bandwidth as f64;
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
}
