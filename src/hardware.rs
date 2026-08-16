use sysinfo::System;

pub struct HardwareProfile {
    pub total_ram: u64,
    pub cores: usize,
    /// Physical cores — llama.cpp runs fastest with one thread per physical
    /// core; using SMT threads slows generation down.
    pub physical_cores: usize,
    pub cpu_brand: String,
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
        Self {
            total_ram: sys.total_memory(),
            cores,
            physical_cores: System::physical_core_count().unwrap_or(cores / 2).max(1),
            cpu_brand,
        }
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
