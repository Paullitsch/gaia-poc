mod config;
mod gpu;
mod server;
mod worker;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tokio::signal;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "gaia-worker", about = "GAIA distributed experiment worker")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = config::Config::load(&cli.config)?;
    tracing::info!(worker = %cfg.worker_name, port = cfg.port, "Starting GAIA worker");

    // Detect GPU
    let gpu_info = gpu::detect_gpu();
    if gpu_info.available {
        tracing::info!(gpu = ?gpu_info.name, memory_mb = ?gpu_info.memory_mb, "GPU detected");
    } else {
        tracing::warn!("No GPU detected — experiments may run on CPU");
    }

    let state = worker::WorkerState::new(cfg.clone());
    let app = server::create_router(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Worker shut down gracefully");
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
