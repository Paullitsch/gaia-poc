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

    // Extract max_evals from params (preferred) or fall back to job field
    let max_evals = job.params
        .get("max_evals")
        .and_then(|v| v.as_u64())
        .unwrap_or(job.max_evals as u64);

    let mut cmd = Command::new(&cfg.python_bin);
    cmd.arg(script)
        .arg("--method").arg(&job.method)
        .arg("--max-evals").arg(max_evals.to_string())
        .current_dir(&exp_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Pass additional params as --key value args
    if let Some(eps) = job.params.get("eval_episodes").and_then(|v| v.as_u64()) {
        cmd.arg("--eval-episodes").arg(eps.to_string());
    }
    if let Some(dir) = job.params.get("results_dir").and_then(|v| v.as_str()) {
        cmd.arg("--results-dir").arg(dir);
    }

    tracing::info!(job_id = %job_id, "Spawning: {} {} --method {} --max-evals {}", 
        cfg.python_bin, script, job.method, max_evals);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let err = format!("Spawn failed: {e}");
            tracing::error!(job_id = %job_id, "{}", err);
            client.complete_job(&job_id, worker_id, "failed", Some(&err)).await?;
            return Ok(());
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Read stderr in background
    let job_id_err = job_id.clone();
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        let mut lines = Vec::new();
        while let Ok(Some(line)) = reader.next_line().await {
            tracing::debug!(job_id = %job_id_err, "stderr: {line}");
            lines.push(line);
        }
        lines
    });

    // Read stdout, parse CSV-like output, stream results
    let mut csv_headers: Option<Vec<String>> = None;
    let mut generation: u64 = 0;
    let mut reader = BufReader::new(stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        // Log interesting lines
        if line.contains("Gen ") || line.contains("SOLVED") || line.contains("Result:") {
            tracing::info!(job_id = %job_id, "{line}");
        }

        // Try to detect CSV header
        if csv_headers.is_none() && line.contains(',') && !line.contains(' ') {
            csv_headers = Some(line.split(',').map(|s| s.trim().to_string()).collect());
            continue;
        }
        
        // Parse CSV data rows
        if let Some(ref headers) = csv_headers {
            let vals: Vec<&str> = line.split(',').collect();
            if vals.len() == headers.len() {
                let row: HashMap<String, String> = headers
                    .iter()
                    .zip(vals.iter())
                    .map(|(k, v)| (k.clone(), v.trim().to_string()))
                    .collect();
                generation += 1;
                if let Err(e) = client.stream_result(&job_id, worker_id, generation, &row).await {
                    tracing::warn!("Failed to stream result: {e}");
                }
            }
        }
    }

    // Wait for process to finish (stdout is closed, so process should be done or nearly done)
    let timeout = tokio::time::Duration::from_secs(cfg.job_timeout_secs);
    let exit_status = tokio::select! {
        result = child.wait() => result.ok(),
        _ = tokio::time::sleep(timeout) => {
            tracing::warn!(job_id = %job_id, "Job timed out, killing");
            let _ = child.kill().await;
            None
        }
    };

    let stderr_lines = stderr_task.await.unwrap_or_default();
    
    let status = match exit_status {
        Some(s) if s.success() => "completed",
        Some(_) => "failed",
        None => "timed_out",
    };

    let error_msg = if status == "failed" && !stderr_lines.is_empty() {
        let last_lines: Vec<_> = stderr_lines.iter().rev().take(5).rev().collect();
        Some(last_lines.into_iter().cloned().collect::<Vec<_>>().join("\n"))
    } else {
        None
    };

    client.complete_job(&job_id, worker_id, status, error_msg.as_deref()).await?;
    tracing::info!(job_id = %job_id, status = %status, generations = generation, "Job finished");
    Ok(())
}
