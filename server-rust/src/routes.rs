use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_util::io::ReaderStream;

use crate::models::*;
use crate::state::AppState;
use crate::storage;
use gaia_protocol::{GossipMessage, gossip::GossipResponse};

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/api/workers/register", post(register_worker))
        .route("/api/workers/heartbeat/:worker_id", get(heartbeat))
        .route("/api/jobs/next/:worker_id", get(next_job))
        .route("/api/jobs/submit", post(submit_job))
        .route("/api/results/stream", post(stream_result))
        .route("/api/results/complete", post(complete_job))
        .route("/api/results/:job_id", get(get_results))
        .route("/api/results/:job_id/csv", get(get_results_csv))
        .route("/api/jobs/cancel/:job_id", post(cancel_job))
        .route("/api/workers/:worker_id/enable", post(toggle_worker))
        .route("/api/workers/:worker_id/force-update", post(force_update_worker))
        .route("/api/workers/force-update-all", post(force_update_all))
        .route("/api/status", get(status))
        // Release management (upload requires auth, download is public)
        .route("/api/releases/upload", post(upload_release))
        .route("/api/releases", get(list_releases))
        .route("/releases/latest", get(latest_release))
        .route("/releases/latest/:filename", get(download_latest))
        .route("/releases/:tag", get(get_release))
        .route("/releases/:tag/:filename", get(download_release))
        // P2P Gossip protocol
        .route("/gossip", post(gossip_endpoint))
        .route("/api/network", get(network_status))
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(100 * 1024 * 1024))
}

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
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
        enabled: true,
        version: req.version,
        force_update: false,
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
        let force = w.force_update;
        if force { w.force_update = false; } // consume the flag
        // Include latest release version for auto-update
        let releases = state.releases.read().await;
        let latest = releases.last().map(|r| r.tag.clone());
        drop(releases);
        let mut resp = json!({ "status": "ok", "force_update": force });
        if let Some(tag) = latest {
            resp["latest_version"] = json!(tag);
        }
        Ok(Json(resp))
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

    // Update heartbeat + check enabled (acquire and release immediately)
    let is_enabled = {
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&worker_id) {
            w.last_heartbeat = Utc::now();
            w.enabled
        } else {
            return Err(StatusCode::NOT_FOUND);
        }
    }; // lock released

    // Disabled workers don't get jobs
    if !is_enabled {
        return Err(StatusCode::NO_CONTENT);
    }

    // Pop next queued job (acquire and release immediately)
    let job_id = {
        state.job_queue.write().await.pop_front()
    }; // lock released
    let job_id = match job_id {
        Some(id) => id,
        None => return Err(StatusCode::NO_CONTENT),
    };

    // Update job status (acquire and release)
    let resp = {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running;
            job.assigned_to = Some(worker_id.clone());
            job.started_at = Some(Utc::now());
            Some(json!({
                "id": job.id,
                "method": job.method,
                "params": job.params,
                "script": job.script,
                "max_evals": job.max_evals,
            }))
        } else {
            None
        }
    }; // lock released

    let resp = match resp {
        Some(r) => r,
        None => return Err(StatusCode::NO_CONTENT),
    };

    // Mark worker busy (separate lock acquisition)
    {
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&worker_id) {
            w.status = WorkerStatus::Busy;
            w.current_job = Some(job_id.clone());
        }
    } // lock released

    let _ = storage::save_state(&state).await;
    Ok(Json(resp))
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

    // Update job (acquire and release)
    {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&req.job_id) {
            job.status = match req.status.as_str() {
                "completed" => JobStatus::Completed,
                "timed_out" => JobStatus::TimedOut,
                _ => JobStatus::Failed,
            };
            job.completed_at = Some(Utc::now());
            job.error = req.error.clone();
        }
    } // lock released

    // Mark worker idle (separate lock)
    {
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&req.worker_id) {
            w.status = WorkerStatus::Idle;
            w.current_job = None;
        }
    } // lock released

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

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    let (was_queued, assigned_worker) = {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            let was_queued = job.status == JobStatus::Queued;
            let worker = job.assigned_to.clone();
            job.status = JobStatus::Cancelled;
            job.completed_at = Some(Utc::now());
            job.error = Some("Cancelled by user".into());
            (was_queued, worker)
        } else {
            return Err(StatusCode::NOT_FOUND);
        }
    };

    // Remove from queue if queued
    if was_queued {
        let mut queue = state.job_queue.write().await;
        queue.retain(|id| id != &job_id);
    }

    // Free worker if running
    if let Some(wid) = assigned_worker {
        let mut workers = state.workers.write().await;
        if let Some(w) = workers.get_mut(&wid) {
            w.status = WorkerStatus::Idle;
            w.current_job = None;
        }
    }

    let _ = storage::save_state(&state).await;
    tracing::info!(job_id = %job_id, "Job cancelled");
    Ok(Json(json!({ "status": "cancelled" })))
}

async fn toggle_worker(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(worker_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let mut workers = state.workers.write().await;
    if let Some(w) = workers.get_mut(&worker_id) {
        w.enabled = !w.enabled;
        let enabled = w.enabled;
        tracing::info!(worker_id = %worker_id, enabled = enabled, "Worker toggled");
        Ok(Json(json!({ "enabled": enabled })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn force_update_worker(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(worker_id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let mut workers = state.workers.write().await;
    if let Some(w) = workers.get_mut(&worker_id) {
        w.force_update = true;
        tracing::info!(worker_id = %worker_id, "Force update flagged");
        Ok(Json(json!({ "force_update": true })))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn force_update_all(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let mut workers = state.workers.write().await;
    let mut count = 0;
    for w in workers.values_mut() {
        if w.enabled {
            w.force_update = true;
            count += 1;
        }
    }
    tracing::info!(count = count, "Force update flagged for all enabled workers");
    Ok(Json(json!({ "force_update_count": count })))
}

async fn status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    // Acquire each lock separately to avoid deadlocks
    let workers_val: Vec<_> = {
        state.workers.read().await.values().cloned().collect()
    };
    let jobs_val: Vec<_> = {
        state.jobs.read().await.values().cloned().collect()
    };
    let results_len = {
        state.results.read().await.len()
    };
    let queue_len = {
        state.job_queue.read().await.len()
    };

    Ok(Json(json!({
        "workers": workers_val,
        "jobs": jobs_val,
        "queue_length": queue_len,
        "total_results": results_len,
        "server_start_time": state.start_time,
    })))
}

// --- Release management ---

#[derive(Debug, Deserialize)]
struct UploadQuery {
    tag: String,
    filename: String,
    notes: Option<String>,
}

async fn upload_release(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(q): Query<UploadQuery>,
    body: axum::body::Bytes,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    let releases_dir = state.releases_dir().join(&q.tag);
    tokio::fs::create_dir_all(&releases_dir).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Validate filename (no path traversal)
    if q.filename.contains("..") || q.filename.contains('/') || q.filename.contains('\\') {
        return Err(StatusCode::BAD_REQUEST);
    }

    let file_path = releases_dir.join(&q.filename);
    tokio::fs::write(&file_path, &body).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Compute SHA-256
    let sha256 = {
        use sha2::Digest;
        let hash = sha2::Sha256::digest(&body);
        format!("{:x}", hash)
    };

    let release_file = ReleaseFile {
        filename: q.filename.clone(),
        size: body.len() as u64,
        sha256,
    };

    let mut releases = state.releases.write().await;
    if let Some(existing) = releases.iter_mut().find(|r| r.tag == q.tag) {
        // Update notes if provided
        if q.notes.is_some() {
            existing.notes = q.notes;
        }
        // Replace or add file
        existing.files.retain(|f| f.filename != q.filename);
        existing.files.push(release_file);
    } else {
        releases.push(Release {
            tag: q.tag.clone(),
            created_at: Utc::now(),
            notes: q.notes,
            files: vec![release_file],
        });
    }
    drop(releases);

    // Persist releases
    let _ = storage::save_releases(&state).await;

    tracing::info!(tag = %q.tag, filename = %q.filename, size = body.len(), "Release file uploaded");
    Ok(Json(json!({ "status": "ok", "tag": q.tag, "filename": q.filename })))
}

async fn list_releases(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;
    let releases = state.releases.read().await;
    Ok(Json(json!({ "releases": *releases })))
}

async fn latest_release(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, StatusCode> {
    let releases = state.releases.read().await;
    match releases.last() {
        Some(r) => Ok(Json(json!(r))),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_release(
    State(state): State<Arc<AppState>>,
    Path(tag): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let releases = state.releases.read().await;
    match releases.iter().find(|r| r.tag == tag) {
        Some(r) => Ok(Json(json!(r))),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn download_latest(
    State(state): State<Arc<AppState>>,
    Path(filename): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let releases = state.releases.read().await;
    let tag = releases.last().map(|r| r.tag.clone()).ok_or(StatusCode::NOT_FOUND)?;
    drop(releases);
    serve_release_file(&state, &tag, &filename).await
}

async fn download_release(
    State(state): State<Arc<AppState>>,
    Path((tag, filename)): Path<(String, String)>,
) -> Result<impl IntoResponse, StatusCode> {
    serve_release_file(&state, &tag, &filename).await
}

async fn serve_release_file(
    state: &AppState,
    tag: &str,
    filename: &str,
) -> Result<impl IntoResponse, StatusCode> {
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err(StatusCode::BAD_REQUEST);
    }
    let path = state.releases_dir().join(tag).join(filename);
    let file = tokio::fs::File::open(&path).await.map_err(|_| StatusCode::NOT_FOUND)?;
    let meta = file.metadata().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    Ok((
        [
            (header::CONTENT_TYPE, "application/octet-stream".to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\"")),
            (header::CONTENT_LENGTH, meta.len().to_string()),
        ],
        body,
    ))
}

// --- P2P Gossip Protocol ---

async fn gossip_endpoint(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(msg): Json<GossipMessage>,
) -> Result<Json<GossipResponse>, StatusCode> {
    check_auth(&state, &headers)?;

    if let Some(gossip) = &state.gossip_node {
        let response = gossip.handle_message(msg).await;
        Ok(Json(response))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn network_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, StatusCode> {
    check_auth(&state, &headers)?;

    if let Some(gossip) = &state.gossip_node {
        let status = gossip.network_status().await;
        Ok(Json(json!(status)))
    } else {
        Ok(Json(json!({
            "mode": "centralized",
            "message": "Gossip protocol not enabled. Start with --gossip to enable P2P mode."
        })))
    }
}
