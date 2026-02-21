use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::client::{Job, ServerClient};
use crate::config::Config;

#[allow(unused_variables)]
pub async fn execute_job(
    client: &ServerClient,
    cfg: &Config,
    worker_id: &str,
    job: Job,
) -> Result<()> {
    let job_id = job.id.clone();
    tracing::info!(job_id = %job_id, method = %job.method, "Starting job execution");

    // Native Rust execution — no Python needed
    if crate::experiments::native_runner::can_run_native(&job.method, &job.environment) {
        tracing::info!(job_id = %job_id, "🦀 Running NATIVE Rust: {} on {}", job.method, job.environment);
        return execute_native(client, worker_id, job).await;
    }

    // Unsupported method/env combo — fail with clear message
    let err = format!("Unsupported native combo: {} on {}. Supported methods: cma_es, openai_es, scaling_test, curriculum, neuromod, neuromod_island, island_model, island_advanced, meta_learning, meta_learning_pure. Supported envs: CartPole-v1, LunarLander-v3, BipedalWalker-v3.",
        job.method, job.environment);
    tracing::warn!(job_id = %job_id, "{}", err);
    client.complete_job(&job_id, worker_id, "failed", Some(&err)).await?;
    return Ok(());

    // Legacy Python path (kept for reference, unreachable)
    #[allow(unreachable_code)]
    let exp_dir = PathBuf::from(&cfg.experiments_dir)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&cfg.experiments_dir));

    // Working directory: parent of experiments dir (run_all.py lives there)
    let work_dir = exp_dir.parent().unwrap_or(&exp_dir).to_path_buf();

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
    cmd.arg("-u") // Force unbuffered stdout
        .arg(script)
        .arg("--method").arg(&job.method)
        .arg("--environment").arg(&job.environment)
        .arg("--max-evals").arg(max_evals.to_string())
        .arg("--no-plots")
        .current_dir(&work_dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("GAIA_JOB_PARAMS", serde_json::to_string(&job.params).unwrap_or_default())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Pass additional params as --key value args
    if let Some(eps) = job.params.get("eval_episodes").and_then(|v| v.as_u64()) {
        cmd.arg("--eval-episodes").arg(eps.to_string());
    }
    if let Some(dir) = job.params.get("results_dir").and_then(|v| v.as_str()) {
        cmd.arg("--results-dir").arg(dir);
    }

    tracing::info!(job_id = %job_id, "Spawning: {} {} --method {} --environment {} --max-evals {}", 
        cfg.python_bin, script, job.method, job.environment, max_evals);

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

    // Early stopping config
    let early_stop_threshold: f64 = job.params
        .get("early_stop_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(200.0); // Default: solved = 200
    let early_stop_patience: u64 = job.params
        .get("early_stop_patience")
        .and_then(|v| v.as_u64())
        .unwrap_or(50); // Stop after 50 gens without improvement past threshold
    let plateau_patience: u64 = job.params
        .get("plateau_patience")
        .and_then(|v| v.as_u64())
        .unwrap_or(200); // Stop after 200 gens without ANY improvement

    let mut best_ever_score: f64 = f64::NEG_INFINITY;
    let mut gens_since_improvement: u64 = 0;
    let mut gens_since_solve: u64 = 0;
    let mut is_solved = false;

    // Read stdout, parse generation output, stream results
    let mut generation: u64 = 0;
    let mut reader = BufReader::new(stdout).lines();

    while let Ok(Some(line)) = reader.next_line().await {
        // Log interesting lines
        if line.contains("Gen ") || line.contains("SOLVED") || line.contains("Result:")
            || line.contains("Best:") || line.contains("SUMMARY") {
            tracing::info!(job_id = %job_id, "{line}");
        }

        // Parse formatted generation lines:
        // Gen    1 | Best:   -104.4 | Ever:   -104.4 | Mean:   -417.6 | σ: 0.4984 | Evals:    135 |    0.5s
        if line.trim_start().starts_with("Gen ") && line.contains('|') {
            let mut row: HashMap<String, String> = HashMap::new();
            for part in line.split('|') {
                let part = part.trim();
                if part.starts_with("Gen") {
                    if let Some(v) = part.split_whitespace().nth(1) {
                        row.insert("generation".into(), v.to_string());
                    }
                } else if let Some(rest) = part.strip_prefix("Best:") {
                    row.insert("best".into(), rest.trim().to_string());
                } else if let Some(rest) = part.strip_prefix("Ever:") {
                    row.insert("best_ever".into(), rest.trim().to_string());
                } else if let Some(rest) = part.strip_prefix("Mean:") {
                    row.insert("mean".into(), rest.trim().to_string());
                } else if let Some(rest) = part.strip_prefix("σ:") {
                    row.insert("sigma".into(), rest.trim().to_string());
                } else if let Some(rest) = part.strip_prefix("Evals:") {
                    row.insert("evals".into(), rest.trim().to_string());
                } else if part.ends_with('s') && part.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    row.insert("time".into(), part.trim_end_matches('s').trim().to_string());
                }
            }
            if !row.is_empty() {
                generation += 1;
                if let Err(e) = client.stream_result(&job_id, worker_id, generation, &row).await {
                    tracing::warn!("Failed to stream result: {e}");
                }
                // Track best_ever for early stopping
                if let Some(best_str) = row.get("best_ever") {
                    if let Ok(score) = best_str.parse::<f64>() {
                        if score > best_ever_score {
                            best_ever_score = score;
                            gens_since_improvement = 0;
                            if score >= early_stop_threshold && !is_solved {
                                is_solved = true;
                                gens_since_solve = 0;
                                tracing::info!(job_id = %job_id, score = score, "🎯 Threshold reached, starting patience countdown");
                            }
                        } else {
                            gens_since_improvement += 1;
                            if is_solved {
                                gens_since_solve += 1;
                            }
                        }
                    }
                }

                // Early stopping: solved + no improvement for patience gens
                if is_solved && gens_since_solve >= early_stop_patience {
                    tracing::info!(
                        job_id = %job_id,
                        best = best_ever_score,
                        patience = early_stop_patience,
                        "✅ Early stop: solved and converged (no improvement for {} gens)",
                        early_stop_patience
                    );
                    let _ = child.kill().await;
                    let _ = stderr_task.await;
                    client.complete_job(&job_id, worker_id, "completed", None).await?;
                    return Ok(());
                }

                // Plateau detection: no improvement at all for a long time
                if gens_since_improvement >= plateau_patience {
                    tracing::info!(
                        job_id = %job_id,
                        best = best_ever_score,
                        plateau = plateau_patience,
                        "📊 Early stop: plateau detected (no improvement for {} gens)",
                        plateau_patience
                    );
                    let _ = child.kill().await;
                    let _ = stderr_task.await;
                    client.complete_job(&job_id, worker_id, "completed", None).await?;
                    return Ok(());
                }

                // Check for cancellation every 10 generations
                if generation % 10 == 0 {
                    if client.check_job_cancelled(&job_id).await {
                        tracing::info!(job_id = %job_id, "Job cancelled by server, killing subprocess");
                        let _ = child.kill().await;
                        let _ = stderr_task.await;
                        return Ok(());
                    }
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
        let last_lines: Vec<_> = stderr_lines.iter().rev().take(10).rev().collect();
        Some(last_lines.into_iter().cloned().collect::<Vec<_>>().join("\n"))
    } else if status == "failed" {
        Some("Process failed with no stderr output".into())
    } else {
        None
    };

    client.complete_job(&job_id, worker_id, status, error_msg.as_deref()).await?;
    tracing::info!(job_id = %job_id, status = %status, generations = generation, "Job finished");
    Ok(())
}

/// Execute a job using native Rust experiments (no Python).
async fn execute_native(
    client: &ServerClient,
    worker_id: &str,
    job: Job,
) -> Result<()> {
    let job_id = job.id.clone();
    let env_name = job.environment.clone();
    let method = job.method.clone();
    // Inject job-level max_evals into params so methods see the actual budget
    let mut params = job.params.clone();
    if !params.get("max_evals").is_some() {
        params.as_object_mut().map(|m| m.insert("max_evals".into(), serde_json::json!(job.max_evals)));
    }

    // Stream results in real-time via channel
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<crate::experiments::native_runner::GenResult>();

    let handle = tokio::task::spawn_blocking(move || {
        crate::experiments::native_runner::run(
            &method, &env_name, &params,
            |gr| { let _ = tx.send(gr); },
        )
    });

    // Stream results as they arrive
    let job_id2 = job_id.clone();
    let wid2 = worker_id.to_string();
    let server_url = client.server_url().to_string();
    let token = client.token().to_string();
    let stream_client = crate::client::ServerClient::from_url_token(&server_url, &token);
    let stream_task = tokio::spawn(async move {
        while let Some(gr) = rx.recv().await {
            let mut data = std::collections::HashMap::new();
            data.insert("best".into(), format!("{:.1}", gr.best));
            data.insert("best_ever".into(), format!("{:.1}", gr.best_ever));
            data.insert("mean".into(), format!("{:.1}", gr.mean));
            data.insert("sigma".into(), format!("{:.4}", gr.sigma));
            data.insert("evals".into(), format!("{}", gr.evals));
            data.insert("generation".into(), format!("{}", gr.generation));
            data.insert("time".into(), format!("{:.1}", gr.time));
            let _ = stream_client.stream_result(&job_id2, &wid2, gr.generation as u64, &data).await;
        }
    });

    let _run_result = handle.await?;
    stream_task.await?;

    client.complete_job(&job_id, worker_id, "completed", None).await?;
    tracing::info!(job_id = %job_id, "Native job completed");
    Ok(())
}
