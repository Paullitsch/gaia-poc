mod config;
mod models;
mod routes;
mod state;
mod storage;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
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

    /// Enable P2P gossip protocol
    #[arg(long, default_value = "false")]
    gossip: bool,

    /// Gossip listen port (separate from main API)
    #[arg(long, default_value = "7435", env = "GAIA_GOSSIP_PORT")]
    gossip_port: u16,

    /// Public address for gossip (how other nodes reach us)
    #[arg(long, env = "GAIA_PUBLIC_ADDR")]
    public_addr: Option<String>,

    /// Seed peers for gossip bootstrap (comma-separated host:port)
    #[arg(long, env = "GAIA_SEED_PEERS", value_delimiter = ',')]
    seed_peers: Vec<String>,

    /// Gossip interval in seconds
    #[arg(long, default_value = "30")]
    gossip_interval: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    tracing::info!(port = cli.port, data_dir = %cli.data_dir, gossip = cli.gossip, "Starting GAIA server");

    let state = if cli.gossip {
        let public_addr = cli.public_addr.unwrap_or_else(|| format!("127.0.0.1:{}", cli.gossip_port));
        let gossip_config = gaia_protocol::GossipConfig {
            node_id: uuid::Uuid::new_v4().to_string(),
            listen_addr: format!("0.0.0.0:{}", cli.gossip_port),
            public_addr: public_addr.clone(),
            seed_peers: cli.seed_peers,
            gossip_interval_secs: cli.gossip_interval,
            peer_timeout_secs: 120,
            capabilities: gaia_protocol::NodeCapabilities {
                gpu: None,
                gpu_memory_mb: None,
                cpu_cores: num_cpus(),
                ram_mb: total_ram_mb(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                tags: vec!["server".into()],
            },
            auth_token: cli.token.clone(),
        };

        let gossip_node = Arc::new(gaia_protocol::GossipNode::new(gossip_config));
        
        // Bootstrap gossip
        if let Err(e) = gossip_node.bootstrap().await {
            tracing::warn!("Gossip bootstrap error: {e}");
        }

        // Spawn gossip loop
        let gossip_loop = gossip_node.clone();
        tokio::spawn(async move {
            gossip_loop.run_gossip_loop().await;
        });

        tracing::info!(addr = %public_addr, "🌐 P2P gossip protocol enabled");
        state::AppState::with_gossip(cli.token, cli.data_dir, gossip_node)
    } else {
        state::AppState::new(cli.token, cli.data_dir)
    };

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

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn total_ram_mb() -> u64 {
    // Read from /proc/meminfo on Linux
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return kb / 1024;
                    }
                }
            }
        }
    }
    1024 // fallback
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
