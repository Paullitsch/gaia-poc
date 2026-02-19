use anyhow::Result;
use serde::Serialize;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuInfo {
    pub available: bool,
    pub name: Option<String>,
    pub memory_mb: Option<u64>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
}

pub fn detect_gpu() -> GpuInfo {
    match run_nvidia_smi() {
        Ok(info) => info,
        Err(e) => {
            tracing::warn!("GPU detection failed: {e}");
            GpuInfo::default()
        }
    }
}

fn run_nvidia_smi() -> Result<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output()?;

    if !output.status.success() {
        anyhow::bail!("nvidia-smi failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().unwrap_or("");
    let parts: Vec<&str> = line.split(", ").collect();

    let mut info = GpuInfo { available: true, ..Default::default() };
    if let Some(name) = parts.first() { info.name = Some(name.trim().to_string()); }
    if let Some(mem) = parts.get(1) { info.memory_mb = mem.trim().parse().ok(); }
    if let Some(drv) = parts.get(2) { info.driver_version = Some(drv.trim().to_string()); }

    // Try to get CUDA version
    if let Ok(out) = Command::new("nvidia-smi").output() {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(pos) = s.find("CUDA Version:") {
            let rest = &s[pos + 14..];
            if let Some(ver) = rest.split_whitespace().next() {
                info.cuda_version = Some(ver.to_string());
            }
        }
    }

    Ok(info)
}
