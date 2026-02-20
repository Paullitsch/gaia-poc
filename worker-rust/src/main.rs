mod client;
mod config;
mod gpu;
mod updater;
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

    /// Enable auto-update from server
    #[arg(long, default_value = "false")]
    auto_update: bool,

    /// Sync experiment files from server on startup and updates
    #[arg(long, default_value = "false")]
    sync_experiments: bool,
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

    let auto_update = cli.auto_update;
    let sync_experiments = cli.sync_experiments;
    tracing::info!(
        worker = %cfg.worker_name, 
        server = %cfg.server_url, 
        version = %updater::VERSION,
        auto_update = auto_update,
        "Starting GAIA worker"
    );

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

    // Sync experiments if enabled (auto-enabled when auto_update is on)
    let sync_experiments = sync_experiments || auto_update;
    if sync_experiments {
        match updater::sync_experiments(client.http_client(), &cfg.server_url, &cfg.auth_token, &cfg.experiments_dir).await {
            Ok(true) => tracing::info!("Experiments synced from server"),
            Ok(false) => tracing::info!("No experiment bundle on server, using local files"),
            Err(e) => tracing::warn!("Experiment sync failed: {e}"),
        }
    }

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

        // Auto-update check via heartbeat
        if auto_update {
            match client.heartbeat(&worker_id).await {
                Ok((Some(latest_version), _force)) => {
                    if updater::is_newer(&latest_version, updater::VERSION) {
                        tracing::info!(
                            current = %updater::VERSION,
                            available = %latest_version,
                            "New version available — updating"
                        );
                        match updater::self_update(client.http_client(), &cfg.server_url, &cfg.auth_token, &latest_version).await {
                            Ok(true) => {
                                if sync_experiments {
                                    let _ = updater::sync_experiments(client.http_client(), &cfg.server_url, &cfg.auth_token, &cfg.experiments_dir).await;
                                }
                                updater::restart()
                            }
                            Ok(false) => {}
                            Err(e) => tracing::warn!("Auto-update failed: {e}"),
                        }
                    }
                }
                Ok((None, _)) => {}
                Err(e) => tracing::debug!("Heartbeat failed: {e}"),
            }
        }

        // Poll for job
        match client.fetch_job(&worker_id).await {
            Ok(Some(job)) => {
                // Sync experiments before each job (hot-reload without restart)
                if sync_experiments {
                    match updater::sync_experiments(client.http_client(), &cfg.server_url, &cfg.auth_token, &cfg.experiments_dir).await {
                        Ok(true) => tracing::info!("Experiments synced"),
                        Ok(false) | Err(_) => {}
                    }
                }
                tracing::info!(job_id = %job.id, method = %job.method, "Got job");
                // Send heartbeats during job execution (every 15s)
                // Also watch for force_update signal from server
                let hb_client = client::ServerClient::new(&cfg);
                let hb_wid = worker_id.clone();
                let (force_tx, mut force_rx) = tokio::sync::oneshot::channel::<String>();
                let hb_auto = auto_update;
                let hb_handle = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
                    let mut force_tx = Some(force_tx);
                    loop {
                        interval.tick().await;
                        match hb_client.heartbeat(&hb_wid).await {
                            Ok((version, force)) => {
                                if force && hb_auto {
                                    if let (Some(tx), Some(v)) = (force_tx.take(), version) {
                                        tracing::warn!("🔄 Server requested force update!");
                                        let _ = tx.send(v);
                                        return;
                                    }
                                }
                            }
                            Err(e) => tracing::debug!("Background heartbeat failed: {e}"),
                        }
                    }
                });
                // Run job, but abort if force update arrives
                tokio::select! {
                    result = worker::execute_job(&client, &cfg, &worker_id, job) => {
                        if let Err(e) = result {
                            tracing::error!("Job execution error: {e}");
                        }
                    }
                    Ok(version) = &mut force_rx => {
                        tracing::warn!("Force update received — aborting current job");
                        // Job subprocess will be killed when execute_job is dropped
                        // Now update
                        match updater::self_update(client.http_client(), &cfg.server_url, &cfg.auth_token, &version).await {
                            Ok(true) => {
                                if sync_experiments {
                                    let _ = updater::sync_experiments(client.http_client(), &cfg.server_url, &cfg.auth_token, &cfg.experiments_dir).await;
                                }
                                updater::restart()
                            }
                            Ok(false) => {}
                            Err(e) => tracing::warn!("Force update failed: {e}"),
                        }
                    }
                }
                hb_handle.abort();
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
