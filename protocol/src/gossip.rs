use anyhow::{Context, Result};
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::peer::{PeerInfo, PeerRegistry};
use crate::types::*;

/// A GAIA gossip node — the core of the decentralized protocol.
///
/// Each GossipNode:
/// 1. Runs an HTTP server for receiving gossip messages
/// 2. Periodically sends its peer list to random peers (fan-out gossip)
/// 3. Handles job broadcasts, claims, and result routing
pub struct GossipNode {
    pub config: GossipConfig,
    pub peers: PeerRegistry,
    /// Jobs we've submitted that we're tracking
    pub submitted_jobs: Arc<RwLock<Vec<JobAnnouncement>>>,
    /// Jobs we've claimed and are working on
    pub claimed_jobs: Arc<RwLock<Vec<String>>>,
    http: reqwest::Client,
}

impl GossipNode {
    pub fn new(config: GossipConfig) -> Self {
        let peers = PeerRegistry::new(
            config.node_id.clone(),
            config.peer_timeout_secs,
        );

        Self {
            config,
            peers,
            submitted_jobs: Arc::new(RwLock::new(Vec::new())),
            claimed_jobs: Arc::new(RwLock::new(Vec::new())),
            http: reqwest::Client::new(),
        }
    }

    /// Bootstrap: connect to seed peers and exchange peer lists
    pub async fn bootstrap(&self) -> Result<()> {
        tracing::info!(
            node_id = %self.config.node_id,
            seeds = ?self.config.seed_peers,
            "🌱 Bootstrapping gossip node"
        );

        for seed_addr in &self.config.seed_peers {
            match self.sync_with_peer(seed_addr).await {
                Ok(count) => {
                    tracing::info!(seed = %seed_addr, new_peers = count, "Synced with seed");
                }
                Err(e) => {
                    tracing::warn!(seed = %seed_addr, error = %e, "Failed to sync with seed");
                }
            }
        }

        let peer_count = self.peers.count().await;
        tracing::info!(peers = peer_count, "Bootstrap complete");
        Ok(())
    }

    /// Send our peer list to a peer and receive theirs
    async fn sync_with_peer(&self, peer_addr: &str) -> Result<usize> {
        let our_announcements = self.peers.announcements().await;
        // Include ourselves in the announcement
        let mut all = our_announcements;
        all.push(PeerAnnouncement {
            node_id: self.config.node_id.clone(),
            address: self.config.public_addr.clone(),
            capabilities: self.config.capabilities.clone(),
            last_seen: Utc::now(),
        });

        let msg = GossipMessage::PeerSync {
            sender: self.config.node_id.clone(),
            peers: all,
        };

        let url = format!("http://{}/gossip", peer_addr);
        let resp = self.http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.auth_token))
            .json(&msg)
            .send()
            .await
            .context("Failed to reach peer")?;

        if !resp.status().is_success() {
            anyhow::bail!("Peer returned HTTP {}", resp.status());
        }

        // The peer responds with their peer list
        let response: GossipResponse = resp.json().await
            .context("Failed to parse peer response")?;

        let before = self.peers.count().await;
        self.peers.merge_peers(&response.peers).await;
        let after = self.peers.count().await;

        Ok(after - before)
    }

    /// Run the gossip loop — periodically sync with random peers
    pub async fn run_gossip_loop(&self) {
        let interval = std::time::Duration::from_secs(self.config.gossip_interval_secs);
        let mut timer = tokio::time::interval(interval);

        loop {
            timer.tick().await;

            // Prune dead peers
            self.peers.prune_dead().await;

            // Pick 2-3 random peers to gossip with (fan-out)
            let targets = self.peers.random_peers(3).await;
            if targets.is_empty() {
                // Re-try seeds if we lost all peers
                for seed in &self.config.seed_peers {
                    let _ = self.sync_with_peer(seed).await;
                }
                continue;
            }

            for peer in targets {
                if let Err(e) = self.sync_with_peer(&peer.address).await {
                    tracing::debug!(peer = %peer.node_id, error = %e, "Gossip sync failed");
                    self.peers.record_failure(&peer.node_id).await;
                } else {
                    self.peers.touch(&peer.node_id).await;
                }
            }

            let count = self.peers.count().await;
            tracing::debug!(peers = count, "Gossip round complete");
        }
    }

    /// Handle an incoming gossip message
    pub async fn handle_message(&self, msg: GossipMessage) -> GossipResponse {
        match msg {
            GossipMessage::PeerSync { sender, peers } => {
                // Merge their peers into ours
                self.peers.merge_peers(&peers).await;
                self.peers.touch(&sender).await;

                // Respond with our peer list
                let mut our_peers = self.peers.announcements().await;
                our_peers.push(PeerAnnouncement {
                    node_id: self.config.node_id.clone(),
                    address: self.config.public_addr.clone(),
                    capabilities: self.config.capabilities.clone(),
                    last_seen: Utc::now(),
                });

                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: our_peers,
                    data: None,
                }
            }

            GossipMessage::Ping { sender, nonce } => {
                self.peers.touch(&sender).await;
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: Some(serde_json::to_value(GossipMessage::Pong {
                        sender: self.config.node_id.clone(),
                        nonce,
                    }).unwrap()),
                }
            }

            GossipMessage::JobBroadcast { sender, job } => {
                self.peers.touch(&sender).await;
                tracing::info!(
                    job_id = %job.id,
                    method = %job.method,
                    from = %sender,
                    "📋 Received job broadcast"
                );

                // Calculate our capacity score for this job
                let score = self.capacity_score(&job);
                
                // If we can handle it, claim it
                if score > 0.0 {
                    // Send claim back to submitter
                    let claim = GossipMessage::JobClaim {
                        claimer: self.config.node_id.clone(),
                        job_id: job.id.clone(),
                        capacity_score: score,
                    };

                    if let Some(submitter) = self.peers.get(&sender).await {
                        let url = format!("http://{}/gossip", submitter.address);
                        let _ = self.http
                            .post(&url)
                            .header("Authorization", format!("Bearer {}", self.config.auth_token))
                            .json(&claim)
                            .send()
                            .await;
                    }
                }

                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }

            GossipMessage::JobClaim { claimer, job_id, capacity_score } => {
                tracing::info!(
                    job_id = %job_id,
                    claimer = %claimer,
                    score = capacity_score,
                    "🙋 Received job claim"
                );
                // TODO: Track claims, pick best claimer after timeout
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }

            GossipMessage::ResultStream { job_id, worker, generation, data } => {
                tracing::debug!(
                    job_id = %job_id,
                    worker = %worker,
                    gen = generation,
                    "📊 Result streamed"
                );
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }

            GossipMessage::JobComplete { job_id, worker, status, error } => {
                tracing::info!(
                    job_id = %job_id,
                    worker = %worker,
                    status = %status,
                    "✅ Job completed (P2P)"
                );
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }

            GossipMessage::ModelShare { sender, job_id, method, fitness, model_url } => {
                tracing::info!(
                    from = %sender,
                    method = %method,
                    fitness = fitness,
                    "🧬 Model shared from peer"
                );
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }

            GossipMessage::Pong { sender, nonce: _ } => {
                self.peers.touch(&sender).await;
                GossipResponse {
                    node_id: self.config.node_id.clone(),
                    peers: vec![],
                    data: None,
                }
            }
        }
    }

    /// Calculate this node's capacity score for a job (higher = more capable)
    fn capacity_score(&self, job: &JobAnnouncement) -> f64 {
        let caps = &self.config.capabilities;
        let mut score = 1.0;

        // GPU bonus
        if caps.gpu.is_some() {
            score += 5.0;
            if let Some(mem) = caps.gpu_memory_mb {
                score += mem as f64 / 1000.0;
            }
        }

        // CPU cores bonus
        score += caps.cpu_cores as f64 * 0.5;

        // RAM bonus
        score += caps.ram_mb as f64 / 4096.0;

        // Check requirements
        if let Some(req) = &job.requirements {
            if let Some(min_gpu) = req.min_gpu_memory_mb {
                if caps.gpu_memory_mb.unwrap_or(0) < min_gpu {
                    return 0.0; // Can't handle this job
                }
            }
            if let Some(min_cpu) = req.min_cpu_cores {
                if caps.cpu_cores < min_cpu {
                    return 0.0;
                }
            }
            if let Some(min_ram) = req.min_ram_mb {
                if caps.ram_mb < min_ram {
                    return 0.0;
                }
            }
            for tag in &req.required_tags {
                if !caps.tags.contains(tag) {
                    return 0.0;
                }
            }
        }

        score
    }

    /// Broadcast a job to the network
    pub async fn broadcast_job(&self, job: JobAnnouncement) -> Result<()> {
        let msg = GossipMessage::JobBroadcast {
            sender: self.config.node_id.clone(),
            job: job.clone(),
        };

        self.submitted_jobs.write().await.push(job);

        let peers = self.peers.alive_peers().await;
        tracing::info!(
            peers = peers.len(),
            "📡 Broadcasting job to network"
        );

        for peer in peers {
            let url = format!("http://{}/gossip", peer.address);
            let _ = self.http
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.auth_token))
                .json(&msg)
                .send()
                .await;
        }

        Ok(())
    }

    /// Get network status summary
    pub async fn network_status(&self) -> NetworkStatus {
        let peers = self.peers.alive_peers().await;
        let total_gpu_memory: u64 = peers.iter()
            .filter_map(|p| p.capabilities.gpu_memory_mb)
            .sum();
        let total_cpu_cores: u32 = peers.iter()
            .map(|p| p.capabilities.cpu_cores)
            .sum();
        let gpu_nodes = peers.iter()
            .filter(|p| p.capabilities.gpu.is_some())
            .count();

        NetworkStatus {
            node_id: self.config.node_id.clone(),
            peer_count: peers.len(),
            gpu_nodes,
            total_gpu_memory_mb: total_gpu_memory,
            total_cpu_cores,
            peers: peers.into_iter().map(|p| p.to_announcement()).collect(),
        }
    }
}

/// Response to a gossip message
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GossipResponse {
    pub node_id: NodeId,
    pub peers: Vec<PeerAnnouncement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Network status summary
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkStatus {
    pub node_id: NodeId,
    pub peer_count: usize,
    pub gpu_nodes: usize,
    pub total_gpu_memory_mb: u64,
    pub total_cpu_cores: u32,
    pub peers: Vec<PeerAnnouncement>,
}
