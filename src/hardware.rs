use sysinfo::System;

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
        }
    }
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
