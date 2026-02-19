mod client;
mod config;
mod gpu;
mod worker;

use anyhow::Result;
use clap::Parser;
use tokio::signal;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "gaia-worker", about = "GAIA distributed experiment worker")]
struct Cli {
    /// Server URL (e.g. https://your-vps:7434)
    #[arg(long, env = "GAIA_SERVER")]
    server: String,

    /// Auth token
    #[arg(long, env = "GAIA_TOKEN")]
    token: String,

    /// Worker name (e.g. paul-rtx5070)
    #[arg(long, env = "GAIA_WORKER_NAME")]
    name: String,

    /// Poll interval in seconds
    #[arg(long, default_value = "5")]
    poll_interval: u64,

    /// Path to experiments directory
    #[arg(long, default_value = "../experiments")]
    experiments_dir: String,

    /// Python binary
    #[arg(long, default_value = "python3")]
    python: String,

    /// Job timeout in seconds
    #[arg(long, default_value = "3600")]
    job_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = config::Config {
        server_url: cli.server,
        auth_token: cli.token,
        worker_name: cli.name,
        poll_interval_secs: cli.poll_interval,
        experiments_dir: cli.experiments_dir,
        python_bin: cli.python,
        job_timeout_secs: cli.job_timeout,
    };

    tracing::info!(worker = %cfg.worker_name, server = %cfg.server_url, "Starting GAIA worker");

    let gpu_info = gpu::detect_gpu();
    if gpu_info.available {
        tracing::info!(gpu = ?gpu_info.name, memory_mb = ?gpu_info.memory_mb, "GPU detected");
    } else {
        tracing::warn!("No GPU detected — experiments may run on CPU");
    }

    let client = client::ServerClient::new(&cfg);

    // Register with server (retry with backoff)
    let worker_id = loop {
        match client.register(&cfg.worker_name, &gpu_info).await {
            Ok(id) => {
                tracing::info!(worker_id = %id, "Registered with server");
                break id;
            }
            Err(e) => {
                tracing::warn!("Registration failed: {e} — retrying in 10s");
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            }
        }
    };

    // Main loop: poll for jobs, execute, repeat
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let poll_interval = std::time::Duration::from_secs(cfg.poll_interval_secs);

    // Spawn shutdown listener
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    loop {
        // Check shutdown
        if *shutdown_rx.borrow() {
            tracing::info!("Shutting down gracefully");
            break;
        }

        // Heartbeat + poll for job
        match client.fetch_job(&worker_id).await {
            Ok(Some(job)) => {
                tracing::info!(job_id = %job.id, method = %job.method, "Got job");
                if let Err(e) = worker::execute_job(&client, &cfg, &worker_id, job).await {
                    tracing::error!("Job execution error: {e}");
                }
            }
            Ok(None) => {
                // No job available, wait
            }
            Err(e) => {
                tracing::warn!("Failed to fetch job: {e}");
            }
        }

        // Wait before next poll, but check shutdown
        tokio::select! {
            _ = tokio::time::sleep(poll_interval) => {}
            _ = shutdown_rx.changed() => {}
        }
    }

    tracing::info!("Worker shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { signal::ctrl_c().await.ok(); };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT"),
        _ = terminate => tracing::info!("Received SIGTERM"),
    }
}
