use anyhow::Result;
use std::path::Path;

use crate::state::AppState;

pub async fn save_state(state: &AppState) -> Result<()> {
    let dir = Path::new(&state.data_dir);
    tokio::fs::create_dir_all(dir).await?;

    let jobs = state.jobs.read().await;
    let jobs_json = serde_json::to_string_pretty(&*jobs)?;
    tokio::fs::write(dir.join("jobs.json"), jobs_json).await?;

    let workers = state.workers.read().await;
    let workers_json = serde_json::to_string_pretty(&*workers)?;
    tokio::fs::write(dir.join("workers.json"), workers_json).await?;

    let results = state.results.read().await;
    let results_json = serde_json::to_string_pretty(&*results)?;
    tokio::fs::write(dir.join("results.json"), results_json).await?;

    Ok(())
}

pub async fn load_state(state: &AppState) -> Result<()> {
    let dir = Path::new(&state.data_dir);

    if let Ok(data) = tokio::fs::read_to_string(dir.join("jobs.json")).await {
        if let Ok(jobs) = serde_json::from_str(&data) {
            *state.jobs.write().await = jobs;
        }
    }
    if let Ok(data) = tokio::fs::read_to_string(dir.join("workers.json")).await {
        if let Ok(workers) = serde_json::from_str(&data) {
            *state.workers.write().await = workers;
        }
    }
    if let Ok(data) = tokio::fs::read_to_string(dir.join("results.json")).await {
        if let Ok(results) = serde_json::from_str(&data) {
            *state.results.write().await = results;
        }
    }

    // Rebuild queue from loaded jobs
    let jobs = state.jobs.read().await;
    let mut queue = state.job_queue.write().await;
    for (id, job) in jobs.iter() {
        if job.status == crate::models::JobStatus::Queued {
            queue.push_back(id.clone());
        }
    }

    Ok(())
}
