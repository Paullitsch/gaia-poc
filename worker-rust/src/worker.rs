use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::client::{Job, ServerClient};
use crate::config::Config;

pub async fn execute_job(
    client: &ServerClient,
    cfg: &Config,
    worker_id: &str,
    job: Job,
) -> Result<()> {
    let job_id = job.id.clone();
    tracing::info!(job_id = %job_id, method = %job.method, "Starting job execution");

    let exp_dir = PathBuf::from(&cfg.experiments_dir)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&cfg.experiments_dir));

    let script = job.script.as_deref().unwrap_or("run_all.py");
    if script.contains("..") {
        client.complete_job(&job_id, worker_id, "failed", Some("Script path contains '..'")).await?;
        return Ok(());
    }

    let mut cmd = Command::new(&cfg.python_bin);
    cmd.arg(script)
        .arg("--method").arg(&job.method)
        .arg("--max-evals").arg(job.max_evals.to_string())
        .current_dir(&exp_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            client.complete_job(&job_id, worker_id, "failed", Some(&format!("Spawn failed: {e}"))).await?;
            return Ok(());
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Read stderr in background
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut lines = Vec::new();
        while let Ok(Some(line)) = reader.next_line().await {
            tracing::debug!("stderr: {line}");
            lines.push(line);
        }
        lines
    });

    // Read stdout, parse CSV, stream results
    let client_ref = client;
    let mut csv_headers: Option<Vec<String>> = None;
    let mut generation: u64 = 0;
    let mut reader = BufReader::new(stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        tracing::debug!("stdout: {line}");

        if csv_headers.is_none() && line.contains(',') {
            csv_headers = Some(line.split(',').map(|s| s.trim().to_string()).collect());
        } else if let Some(ref headers) = csv_headers {
            let vals: Vec<&str> = line.split(',').collect();
            if vals.len() == headers.len() {
                let row: HashMap<String, String> = headers
                    .iter()
                    .zip(vals.iter())
                    .map(|(k, v)| (k.clone(), v.trim().to_string()))
                    .collect();
                generation += 1;
                if let Err(e) = client_ref.stream_result(&job_id, worker_id, generation, &row).await {
                    tracing::warn!("Failed to stream result: {e}");
                }
            }
        }
    }

    let timeout = tokio::time::Duration::from_secs(cfg.job_timeout_secs);
    let status = tokio::select! {
        result = child.wait() => {
            match result {
                Ok(exit) if exit.success() => "completed",
                _ => "failed",
            }
        }
        _ = tokio::time::sleep(timeout) => {
            let _ = child.kill().await;
            "timed_out"
        }
    };

    let _ = stderr_task.await;

    client.complete_job(&job_id, worker_id, status, None).await?;
    tracing::info!(job_id = %job_id, status = %status, "Job finished");
    Ok(())
}
