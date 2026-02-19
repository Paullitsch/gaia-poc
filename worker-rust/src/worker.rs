use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock};

use crate::config::Config;

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

#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub job_id: String,
    pub status: JobStatus,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub stdout: Vec<String>,
    pub stderr: Vec<String>,
    pub csv_results: Vec<HashMap<String, String>>,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

pub struct WorkerState {
    pub config: Config,
    pub current_job: RwLock<Option<String>>,
    pub results: RwLock<HashMap<String, JobResult>>,
    pub cancel_tokens: Mutex<HashMap<String, tokio::sync::watch::Sender<bool>>>,
    pub started_at: chrono::DateTime<Utc>,
}

impl WorkerState {
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Self {
            config,
            current_job: RwLock::new(None),
            results: RwLock::new(HashMap::new()),
            cancel_tokens: Mutex::new(HashMap::new()),
            started_at: Utc::now(),
        })
    }
}

pub async fn execute_job(state: Arc<WorkerState>, job: Job) -> Result<()> {
    let job_id = job.id.clone();
    tracing::info!(job_id = %job_id, method = %job.method, "Starting job");

    // Set current job
    *state.current_job.write().await = Some(job_id.clone());

    // Create cancel token
    let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
    state.cancel_tokens.lock().await.insert(job_id.clone(), cancel_tx);

    // Initialize result
    let result = JobResult {
        job_id: job_id.clone(),
        status: JobStatus::Running,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        stdout: Vec::new(),
        stderr: Vec::new(),
        csv_results: Vec::new(),
        exit_code: None,
        error: None,
    };
    state.results.write().await.insert(job_id.clone(), result);

    // Resolve experiments directory
    let exp_dir = PathBuf::from(&state.config.experiments_dir).canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&state.config.experiments_dir));

    // Determine script
    let script = job.script.as_deref().unwrap_or("run_all.py");
    // Security: ensure script doesn't escape experiments dir
    if script.contains("..") {
        let mut r = state.results.write().await;
        if let Some(res) = r.get_mut(&job_id) {
            res.status = JobStatus::Failed;
            res.error = Some("Script path contains '..' — rejected".into());
            res.finished_at = Some(Utc::now().to_rfc3339());
        }
        *state.current_job.write().await = None;
        return Ok(());
    }

    let mut cmd = Command::new(&state.config.python_bin);
    cmd.arg(script)
        .arg("--method").arg(&job.method)
        .arg("--max-evals").arg(job.max_evals.to_string())
        .current_dir(&exp_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let mut r = state.results.write().await;
            if let Some(res) = r.get_mut(&job_id) {
                res.status = JobStatus::Failed;
                res.error = Some(format!("Failed to spawn process: {e}"));
                res.finished_at = Some(Utc::now().to_rfc3339());
            }
            *state.current_job.write().await = None;
            return Ok(());
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let stdout_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let stderr_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let csv_results: Arc<Mutex<Vec<HashMap<String, String>>>> = Arc::new(Mutex::new(Vec::new()));

    // Stdout reader — also parses CSV
    let sl = stdout_lines.clone();
    let cr = csv_results.clone();
    let vps_url = state.config.vps.url.clone();
    let vps_token = state.config.vps.auth_token.clone();
    let jid = job_id.clone();
    let stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut csv_headers: Option<Vec<String>> = None;
        while let Ok(Some(line)) = reader.next_line().await {
            tracing::debug!(job_id = %jid, "stdout: {line}");
            sl.lock().await.push(line.clone());

            // Try CSV parsing
            if csv_headers.is_none() && line.contains(',') {
                csv_headers = Some(line.split(',').map(|s| s.trim().to_string()).collect());
            } else if let Some(ref headers) = csv_headers {
                let vals: Vec<&str> = line.split(',').collect();
                if vals.len() == headers.len() {
                    let row: HashMap<String, String> = headers.iter()
                        .zip(vals.iter())
                        .map(|(k, v)| (k.clone(), v.trim().to_string()))
                        .collect();
                    cr.lock().await.push(row.clone());

                    // Stream to VPS
                    if let Some(ref url) = vps_url {
                        let _ = stream_result(url, vps_token.as_deref(), &jid, &row).await;
                    }
                }
            }
        }
    });

    let el = stderr_lines.clone();
    let jid2 = job_id.clone();
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            tracing::debug!(job_id = %jid2, "stderr: {line}");
            el.lock().await.push(line);
        }
    });

    let timeout = tokio::time::Duration::from_secs(state.config.job_timeout_secs);

    let status = tokio::select! {
        result = child.wait() => {
            match result {
                Ok(exit) => {
                    if exit.success() { JobStatus::Completed } else { JobStatus::Failed }
                }
                Err(_) => JobStatus::Failed,
            }
        }
        _ = tokio::time::sleep(timeout) => {
            let _ = child.kill().await;
            JobStatus::TimedOut
        }
        _ = async { loop {
            cancel_rx.changed().await.ok();
            if *cancel_rx.borrow() { break; }
        }} => {
            let _ = child.kill().await;
            JobStatus::Cancelled
        }
    };

    let exit_code = child.try_wait().ok().flatten().and_then(|s| s.code());

    // Wait for readers to finish
    let _ = tokio::join!(stdout_task, stderr_task);

    // Update result
    {
        let mut r = state.results.write().await;
        if let Some(res) = r.get_mut(&job_id) {
            res.status = status;
            res.finished_at = Some(Utc::now().to_rfc3339());
            res.stdout = stdout_lines.lock().await.clone();
            res.stderr = stderr_lines.lock().await.clone();
            res.csv_results = csv_results.lock().await.clone();
            res.exit_code = exit_code;
        }
    }

    // Clean up
    *state.current_job.write().await = None;
    state.cancel_tokens.lock().await.remove(&job_id);

    tracing::info!(job_id = %job_id, "Job finished");
    Ok(())
}

async fn stream_result(
    vps_url: &str,
    auth_token: Option<&str>,
    job_id: &str,
    row: &HashMap<String, String>,
) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/results", vps_url.trim_end_matches('/'));
    let payload = serde_json::json!({
        "job_id": job_id,
        "result": row,
        "timestamp": Utc::now().to_rfc3339(),
    });
    let mut req = client.post(&url).json(&payload);
    if let Some(token) = auth_token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    req.send().await.context("Failed to stream result to VPS")?;
    Ok(())
}
