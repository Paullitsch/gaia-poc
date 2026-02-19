use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::models::*;

pub struct AppState {
    pub workers: RwLock<HashMap<String, Worker>>,
    pub job_queue: RwLock<VecDeque<String>>,  // queued job IDs
    pub jobs: RwLock<HashMap<String, Job>>,
    pub results: RwLock<Vec<ResultRow>>,
    pub auth_token: String,
    pub data_dir: String,
    pub start_time: DateTime<Utc>,
}

impl AppState {
    pub fn new(auth_token: String, data_dir: String) -> Arc<Self> {
        Arc::new(Self {
            workers: RwLock::new(HashMap::new()),
            job_queue: RwLock::new(VecDeque::new()),
            jobs: RwLock::new(HashMap::new()),
            results: RwLock::new(Vec::new()),
            auth_token,
            data_dir,
            start_time: Utc::now(),
        })
    }
}
