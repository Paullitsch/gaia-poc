//! GAIA Decentralized Protocol
//!
//! Every node in the GAIA network is both a server and a worker.
//! Nodes discover each other via gossip and distribute work peer-to-peer.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────┐  gossip   ┌──────┐  gossip   ┌──────┐
//!  │Node A│◄──────────►│Node B│◄──────────►│Node C│
//!  │(GPU) │            │(CPU) │            │(GPU) │
//!  └──────┘            └──────┘            └──────┘
//!     ▲                                       ▲
//!     └───────────── job relay ────────────────┘
//! ```
//!
//! ## Gossip Protocol
//! - Each node maintains a peer list
//! - Periodically sends its peer list to random peers
//! - Merges received peer lists (union)
//! - Dead peers are removed after missed heartbeats
//!
//! ## Job Distribution
//! - Jobs are broadcast to all known peers
//! - Nodes claim jobs based on capacity (GPU, CPU cores, current load)
//! - Results stream back to the job submitter
//!
//! ## Model Sharing
//! - Best models from completed jobs can be shared across the network
//! - Enables population migration between nodes

pub mod gossip;
pub mod peer;
pub mod types;

pub use gossip::GossipNode;
pub use peer::PeerInfo;
pub use types::*;
