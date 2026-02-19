mod config;
mod models;
mod routes;
mod state;
mod storage;

use anyhow::Result;
use clap::Parser;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "gaia-server", about = "GAIA central orchestration server")]
struct Cli {
    /// Port to listen on
    #[arg(long, default_value = "7434", env = "GAIA_PORT")]
    port: u16,

    /// Auth token (required for all endpoints)
    #[arg(long, env = "GAIA_TOKEN")]
    token: String,

    /// Data directory for persistence
    #[arg(long, default_value = "./server-data", env = "GAIA_DATA_DIR")]
    data_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    tracing::info!(port = cli.port, data_dir = %cli.data_dir, "Starting GAIA server");

    let state = state::AppState::new(cli.token, cli.data_dir);

    // Load persisted state
    if let Err(e) = storage::load_state(&state).await {
        tracing::warn!("Failed to load persisted state: {e}");
    }

    let app = routes::create_router(state).layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok(); };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
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
