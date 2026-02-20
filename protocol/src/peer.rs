use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::types::*;

/// Information about a known peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub address: String,
    pub capabilities: NodeCapabilities,
    pub last_seen: DateTime<Utc>,
    pub last_ping_ms: Option<u64>,
    pub failed_pings: u32,
}

impl PeerInfo {
    pub fn is_alive(&self, timeout_secs: u64) -> bool {
        let age = Utc::now().signed_duration_since(self.last_seen);
        age.num_seconds() < timeout_secs as i64
    }

    pub fn to_announcement(&self) -> PeerAnnouncement {
        PeerAnnouncement {
            node_id: self.node_id.clone(),
            address: self.address.clone(),
            capabilities: self.capabilities.clone(),
            last_seen: self.last_seen,
        }
    }
}

/// The peer registry — thread-safe collection of known peers
#[derive(Clone)]
pub struct PeerRegistry {
    peers: Arc<RwLock<HashMap<NodeId, PeerInfo>>>,
    self_id: NodeId,
    peer_timeout_secs: u64,
}

impl PeerRegistry {
    pub fn new(self_id: NodeId, peer_timeout_secs: u64) -> Self {
        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
            self_id,
            peer_timeout_secs,
        }
    }

    /// Add or update a peer from a gossip announcement
    pub async fn merge_peer(&self, announcement: &PeerAnnouncement) {
        // Don't add ourselves
        if announcement.node_id == self.self_id {
            return;
        }

        let mut peers = self.peers.write().await;
        let entry = peers.entry(announcement.node_id.clone()).or_insert_with(|| {
            tracing::info!(
                peer = %announcement.node_id,
                addr = %announcement.address,
                "🌐 Discovered new peer"
            );
            PeerInfo {
                node_id: announcement.node_id.clone(),
                address: announcement.address.clone(),
                capabilities: announcement.capabilities.clone(),
                last_seen: announcement.last_seen,
                last_ping_ms: None,
                failed_pings: 0,
            }
        });

        // Update if newer
        if announcement.last_seen > entry.last_seen {
            entry.last_seen = announcement.last_seen;
            entry.address = announcement.address.clone();
            entry.capabilities = announcement.capabilities.clone();
        }
    }

    /// Merge a batch of peer announcements (from gossip sync)
    pub async fn merge_peers(&self, announcements: &[PeerAnnouncement]) {
        for ann in announcements {
            self.merge_peer(ann).await;
        }
    }

    /// Mark a peer as seen (update last_seen)
    pub async fn touch(&self, node_id: &str) {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(node_id) {
            peer.last_seen = Utc::now();
            peer.failed_pings = 0;
        }
    }

    /// Record a failed ping
    pub async fn record_failure(&self, node_id: &str) {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(node_id) {
            peer.failed_pings += 1;
        }
    }

    /// Remove dead peers (not seen within timeout)
    pub async fn prune_dead(&self) {
        let mut peers = self.peers.write().await;
        let before = peers.len();
        peers.retain(|id, peer| {
            let alive = peer.is_alive(self.peer_timeout_secs);
            if !alive {
                tracing::info!(peer = %id, "💀 Peer timed out, removing");
            }
            alive
        });
        let removed = before - peers.len();
        if removed > 0 {
            tracing::info!(removed = removed, remaining = peers.len(), "Pruned dead peers");
        }
    }

    /// Get all alive peers
    pub async fn alive_peers(&self) -> Vec<PeerInfo> {
        let peers = self.peers.read().await;
        peers.values()
            .filter(|p| p.is_alive(self.peer_timeout_secs))
            .cloned()
            .collect()
    }

    /// Get all peer announcements for gossip sync
    pub async fn announcements(&self) -> Vec<PeerAnnouncement> {
        self.alive_peers().await.iter().map(|p| p.to_announcement()).collect()
    }

    /// Pick N random alive peers for gossip fan-out
    pub async fn random_peers(&self, n: usize) -> Vec<PeerInfo> {
        use rand::seq::SliceRandom;
        let alive = self.alive_peers().await;
        if alive.len() <= n {
            return alive;
        }
        let mut rng = rand::thread_rng();
        let mut selected = alive;
        selected.shuffle(&mut rng);
        selected.truncate(n);
        selected
    }

    /// Number of known alive peers
    pub async fn count(&self) -> usize {
        self.alive_peers().await.len()
    }

    /// Get a specific peer by ID
    pub async fn get(&self, node_id: &str) -> Option<PeerInfo> {
        let peers = self.peers.read().await;
        peers.get(node_id).cloned()
    }
}
