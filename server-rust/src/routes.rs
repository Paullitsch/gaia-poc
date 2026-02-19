use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::models::*;
use crate::state::AppState;
use crate::storage;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/workers/register", post(register_worker))
        .route("/api/workers/heartbeat/{worker_id}", get(heartbeat))
        .route("/api/jobs/next/{worker_id}", get(next_job))
        .route("/api/jobs/submit", post(submit_job))
        .route("/api/results/stream", post(stream_result))
        .route("/api/results/complete", post(complete_job))
        .route("/api/results/{job_id}", get(get_results))
        .route("/api/results/{job_id}/csv", get(get_results_csv))
        .route("/api/status", get(status))
        .with_state(state)
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(token) if token == state.auth_token => Ok(()),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

async fn register_worker(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = Utc::now();
    let worker = Worker {
        id: id.clone(),
        name: req.name,
        gpu: req.gpu,
        registered_at: now,
        last_heartbeat: now,
        status: WorkerStatus::Idle,
        current_job: None,
    };
    state.workers.write().await.insert(id.clone(), worker);
    tracing::info!(worker_id = %id, "Worker registered");
    Ok(Json(json!({ "worker_id": id })))
}

async fn heartbeat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(worker_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let mut workers = state.workers.write().await;
    if let Some(w) = workers.get_mut(&worker_id) {
        w.last_heartbeat = Utc::now();
        Ok(Json(json!({ "status": "ok" })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn next_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(worker_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    // Update heartbeat
    {
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&worker_id) {
            w.last_heartbeat = Utc::now();
        }
    }

    // Pop next queued job
    let job_id = state.job_queue.write().await.pop_front();
    let job_id = match job_id {
        Some(id) => id,
        None => return Err(StatusCode::NO_CONTENT),
    };

    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&job_id) {
        job.status = JobStatus::Running;
        job.assigned_to = Some(worker_id.clone());
        job.started_at = Some(Utc::now());

        let resp = json!({
            "id": job.id,
            "method": job.method,
            "params": job.params,
            "script": job.script,
            "max_evals": job.max_evals,
        });

        // Mark worker busy
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&worker_id) {
            w.status = WorkerStatus::Busy;
            w.current_job = Some(job_id.clone());
        }

        let _ = storage::save_state(&state).await;
        Ok(Json(resp))
    } else {
        Err(StatusCode::NO_CONTENT)
    }
}

async fn submit_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<SubmitJobRequest>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let id = uuid::Uuid::new_v4().to_string();
    let job = Job {
        id: id.clone(),
        method: req.method,
        params: req.params,
        script: req.script,
        max_evals: req.max_evals,
        status: JobStatus::Queued,
        assigned_to: None,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    state.jobs.write().await.insert(id.clone(), job);
    state.job_queue.write().await.push_back(id.clone());
    tracing::info!(job_id = %id, "Job submitted");
    let _ = storage::save_state(&state).await;
    Ok(Json(json!({ "job_id": id, "status": "queued" })))
}

async fn stream_result(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<StreamResultRequest>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let row = ResultRow {
        job_id: req.job_id.clone(),
        worker_id: req.worker_id,
        generation: req.generation,
        data: req.data,
        timestamp: Utc::now(),
    };
    state.results.write().await.push(row);
    // Persist periodically (every 10 generations)
    if req.generation % 10 == 0 {
        let _ = storage::save_state(&state).await;
    }
    Ok(Json(json!({ "status": "ok" })))
}

async fn complete_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CompleteRequest>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&req.job_id) {
        job.status = match req.status.as_str() {
            "completed" => JobStatus::Completed,
            "timed_out" => JobStatus::TimedOut,
            _ => JobStatus::Failed,
        };
        job.completed_at = Some(Utc::now());
        job.error = req.error;
    }

    // Mark worker idle
    let mut workers = state.workers.write().await;
    if let Some(w) = workers.get_mut(&req.worker_id) {
        w.status = WorkerStatus::Idle;
        w.current_job = None;
    }

    let _ = storage::save_state(&state).await;
    tracing::info!(job_id = %req.job_id, status = %req.status, "Job completed");
    Ok(Json(json!({ "status": "ok" })))
}

async fn get_results(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let results = state.results.read().await;
    let rows: Vec<_> = results.iter().filter(|r| r.job_id == job_id).collect();
    let job = state.jobs.read().await.get(&job_id).cloned();
    Ok(Json(json!({
        "job": job,
        "results": rows,
        "count": rows.len(),
    })))
}

async fn get_results_csv(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> Result<String, StatusCode> {
    check_auth(&state, &headers)?;
    let results = state.results.read().await;
    let rows: Vec<_> = results.iter().filter(|r| r.job_id == job_id).collect();
    if rows.is_empty() {
        return Ok(String::new());
    }

    // Collect all keys
    let mut keys: Vec<String> = rows[0].data.keys().cloned().collect();
    keys.sort();

    let mut csv = format!("generation,{}\n", keys.join(","));
    for row in &rows {
        let vals: Vec<String> = keys.iter().map(|k| row.data.get(k).cloned().unwrap_or_default()).collect();
        csv.push_str(&format!("{},{}\n", row.generation, vals.join(",")));
    }
    Ok(csv)
}

async fn status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let workers = state.workers.read().await;
    let jobs = state.jobs.read().await;
    let results = state.results.read().await;
    let queue = state.job_queue.read().await;

    Ok(Json(json!({
        "workers": workers.values().collect::<Vec<_>>(),
        "jobs": jobs.values().collect::<Vec<_>>(),
        "queue_length": queue.len(),
        "total_results": results.len(),
    })))
}
