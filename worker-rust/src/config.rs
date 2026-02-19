#[derive(Debug, Clone)]
pub struct Config {
    pub server_url: String,
    pub auth_token: String,
    pub worker_name: String,
    pub poll_interval_secs: u64,
    pub experiments_dir: String,
    pub python_bin: String,
    pub job_timeout_secs: u64,
}
