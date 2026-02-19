use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub worker_name: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub vps: VpsConfig,
    #[serde(default)]
    pub auth_token: Option<String>,
    #[serde(default = "default_experiments_dir")]
    pub experiments_dir: String,
    #[serde(default = "default_python")]
    pub python_bin: String,
    #[serde(default = "default_timeout")]
    pub job_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VpsConfig {
    pub url: Option<String>,
    pub auth_token: Option<String>,
}

fn default_device() -> String { "auto".into() }
fn default_port() -> u16 { 8090 }
fn default_experiments_dir() -> String { "../experiments".into() }
fn default_python() -> String { "python3".into() }
fn default_timeout() -> u64 { 3600 }

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        serde_yaml::from_str(&contents).context("Failed to parse config YAML")
    }
}
