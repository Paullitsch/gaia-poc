use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique node identifier
pub type NodeId = String;

/// Capabilities a node advertises to the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// GPU name (if available)
    pub gpu: Option<String>,
    /// GPU memory in MB
    pub gpu_memory_mb: Option<u64>,
    /// Number of CPU cores available for experiments
    pub cpu_cores: u32,
    /// Available RAM in MB
    pub ram_mb: u64,
    /// Worker version
    pub version: String,
    /// Custom tags (e.g. "high-memory", "arm64")
    #[serde(default)]
    pub tags: Vec<String>,
}

/// A gossip message exchanged between peers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GossipMessage {
    /// "Here are the peers I know about"
    PeerSync {
        sender: NodeId,
        peers: Vec<PeerAnnouncement>,
    },
    /// "I have a job that needs a worker"  
    JobBroadcast {
        sender: NodeId,
        job: JobAnnouncement,
    },
    /// "I'll take that job"
    JobClaim {
        claimer: NodeId,
        job_id: String,
        capacity_score: f64,
    },
    /// "Here's a result update"
    ResultStream {
        job_id: String,
        worker: NodeId,
        generation: u64,
        data: HashMap<String, String>,
    },
    /// "Job is done"
    JobComplete {
        job_id: String,
        worker: NodeId,
        status: String,
        error: Option<String>,
    },
    /// "Check out this model"
    ModelShare {
        sender: NodeId,
        job_id: String,
        method: String,
        fitness: f64,
        /// URL to download the model weights
        model_url: String,
    },
    /// Ping/Pong for liveness
    Ping { sender: NodeId, nonce: u64 },
    Pong { sender: NodeId, nonce: u64 },
}

/// Announcement of a peer's existence  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerAnnouncement {
    pub node_id: NodeId,
    /// The address where this peer's gossip HTTP endpoint lives
    pub address: String,
    pub capabilities: NodeCapabilities,
    pub last_seen: DateTime<Utc>,
}

/// A job being offered to the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobAnnouncement {
    pub id: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub script: Option<String>,
    pub max_evals: u64,
    /// Who submitted this job (results flow back here)
    pub submitter: NodeId,
    /// Minimum capabilities required
    pub requirements: Option<JobRequirements>,
}

/// Optional requirements for a job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequirements {
    pub min_gpu_memory_mb: Option<u64>,
    pub min_cpu_cores: Option<u32>,
    pub min_ram_mb: Option<u64>,
    pub required_tags: Vec<String>,
}

/// Configuration for the gossip protocol
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// This node's unique ID
    pub node_id: NodeId,
    /// Address this node listens on for gossip (e.g. "0.0.0.0:7435")
    pub listen_addr: String,
    /// Public address other nodes can reach us at (e.g. "myhost.com:7435")
    pub public_addr: String,
    /// Initial seed peers to bootstrap from
    pub seed_peers: Vec<String>,
    /// How often to gossip (seconds)
    pub gossip_interval_secs: u64,
    /// How long before a peer is considered dead (seconds)
    pub peer_timeout_secs: u64,
    /// This node's capabilities
    pub capabilities: NodeCapabilities,
    /// Auth token for the network
    pub auth_token: String,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            listen_addr: "0.0.0.0:7435".into(),
            public_addr: "127.0.0.1:7435".into(),
            seed_peers: vec![],
            gossip_interval_secs: 30,
            peer_timeout_secs: 120,
            capabilities: NodeCapabilities {
                gpu: None,
                gpu_memory_mb: None,
                cpu_cores: 1,
                ram_mb: 1024,
                version: "0.0.0".into(),
                tags: vec![],
            },
            auth_token: String::new(),
        }
    }
}
