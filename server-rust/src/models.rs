use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub id: String,
    pub name: String,
    pub gpu: serde_json::Value,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub status: WorkerStatus,
    pub current_job: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub version: Option<String>,
}

fn default_true() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStatus {
    Idle,
    Busy,
    Offline,
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
    pub status: JobStatus,
    pub assigned_to: Option<String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

fn default_max_evals() -> u64 { 100 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRow {
    pub job_id: String,
    pub worker_id: String,
    pub generation: u64,
    pub data: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

// --- Release types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub tag: String,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
    pub files: Vec<ReleaseFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseFile {
    pub filename: String,
    pub size: u64,
    pub sha256: String,
}

// --- Request/response types ---

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub name: String,
    pub gpu: serde_json::Value,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SubmitJobRequest {
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub script: Option<String>,
    #[serde(default = "default_max_evals")]
    pub max_evals: u64,
}

#[derive(Debug, Deserialize)]
pub struct StreamResultRequest {
    pub job_id: String,
    pub worker_id: String,
    pub generation: u64,
    pub data: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct CompleteRequest {
    pub job_id: String,
    pub worker_id: String,
    pub status: String,
    pub error: Option<String>,
}
