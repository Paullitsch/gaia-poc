use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;

use crate::gpu;
use crate::worker::{self, Job, WorkerState};

pub fn create_router(state: Arc<WorkerState>) -> Router {
    Router::new()
        .route("/api/status", get(status_handler))
        .route("/api/submit_job", post(submit_job_handler))
        .route("/api/results/{job_id}", get(results_handler))
        .route("/api/ping", post(ping_handler))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB
        .with_state(state)
}

fn check_auth(state: &WorkerState, headers: &HeaderMap) -> Result<(), StatusCode> {
    if let Some(ref expected) = state.config.auth_token {
        let provided = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match provided {
            Some(token) if token == expected => Ok(()),
            _ => Err(StatusCode::UNAUTHORIZED),
        }
    } else {
        Ok(())
    }
}

async fn status_handler(
    State(state): State<Arc<WorkerState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let current_job = state.current_job.read().await.clone();
    let gpu_info = gpu::detect_gpu();
    let uptime = chrono::Utc::now() - state.started_at;

    Ok(Json(json!({
        "worker_name": state.config.worker_name,
        "device": state.config.device,
        "uptime_secs": uptime.num_seconds(),
        "current_job": current_job,
        "gpu": gpu_info,
        "jobs_completed": state.results.read().await.len(),
    })))
}

async fn submit_job_handler(
    State(state): State<Arc<WorkerState>>,
    headers: HeaderMap,
    Json(mut job): Json<Job>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    // Check if already running a job
    if state.current_job.read().await.is_some() {
        return Ok(Json(json!({
            "error": "Worker is busy",
            "status": "rejected"
        })));
    }

    // Assign ID if empty
    if job.id.is_empty() {
        job.id = uuid::Uuid::new_v4().to_string();
    }

    let job_id = job.id.clone();
    let job_id2 = job_id.clone();
    let s = state.clone();
    tokio::spawn(async move {
        if let Err(e) = worker::execute_job(s, job).await {
            tracing::error!(job_id = %job_id2, "Job execution error: {e}");
        }
    });

    Ok(Json(json!({
        "status": "accepted",
        "job_id": job_id,
    })))
}

async fn results_handler(
    State(state): State<Arc<WorkerState>>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let results = state.results.read().await;
    match results.get(&job_id) {
        Some(r) => Ok(Json(serde_json::to_value(r).unwrap())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn ping_handler(
    State(state): State<Arc<WorkerState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    Ok(Json(json!({
        "status": "ok",
        "worker": state.config.worker_name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}
