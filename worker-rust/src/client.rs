use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::Config;
use crate::gpu::GpuInfo;

pub struct ServerClient {
    http: reqwest::Client,
    base_url: String,
    token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub script: Option<String>,
    #[serde(default = "default_max_evals")]
    pub max_evals: u64,
}

fn default_max_evals() -> u64 { 100 }

impl ServerClient {
    pub fn new(cfg: &Config) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: cfg.server_url.trim_end_matches('/').to_string(),
            token: cfg.auth_token.clone(),
        }
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("Authorization", format!("Bearer {}", self.token))
    }

    pub async fn register(&self, name: &str, gpu: &GpuInfo) -> Result<String> {
        let url = format!("{}/api/workers/register", self.base_url);
        let body = serde_json::json!({
            "name": name,
            "gpu": gpu,
            "registered_at": chrono::Utc::now().to_rfc3339(),
        });
        let resp = self.auth(self.http.post(&url)).json(&body).send().await
            .context("Failed to connect to server")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("Registration failed: HTTP {status}");
        }
        let data: serde_json::Value = resp.json().await?;
        data["worker_id"].as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("No worker_id in response"))
    }

    pub async fn fetch_job(&self, worker_id: &str) -> Result<Option<Job>> {
        let url = format!("{}/api/jobs/next/{}", self.base_url, worker_id);
        let resp = self.auth(self.http.get(&url)).send().await
            .context("Failed to poll for jobs")?;
        if resp.status() == reqwest::StatusCode::NO_CONTENT
            || resp.status() == reqwest::StatusCode::NOT_FOUND
        {
            return Ok(None);
        }
        if !resp.status().is_success() {
            anyhow::bail!("Fetch job failed: HTTP {}", resp.status());
        }
        let job: Job = resp.json().await?;
        Ok(Some(job))
    }

    pub async fn stream_result(
        &self,
        job_id: &str,
        worker_id: &str,
        generation: u64,
        row: &HashMap<String, String>,
    ) -> Result<()> {
        let url = format!("{}/api/results/stream", self.base_url);
        let body = serde_json::json!({
            "job_id": job_id,
            "worker_id": worker_id,
            "generation": generation,
            "data": row,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        let resp = self.auth(self.http.post(&url)).json(&body).send().await
            .context("Failed to stream result")?;
        if !resp.status().is_success() {
            tracing::warn!("Stream result got HTTP {}", resp.status());
        }
        Ok(())
    }

    pub async fn complete_job(
        &self,
        job_id: &str,
        worker_id: &str,
        status: &str,
        error: Option<&str>,
    ) -> Result<()> {
        let url = format!("{}/api/results/complete", self.base_url);
        let body = serde_json::json!({
            "job_id": job_id,
            "worker_id": worker_id,
            "status": status,
            "error": error,
            "completed_at": chrono::Utc::now().to_rfc3339(),
        });
        let resp = self.auth(self.http.post(&url)).json(&body).send().await
            .context("Failed to report completion")?;
        if !resp.status().is_success() {
            tracing::warn!("Complete job got HTTP {}", resp.status());
        }
        Ok(())
    }
}
