//! # cognis-graph
//!
//! v2-beta graph engine: typed `Graph<S>` + Pregel superstep executor.
//! `CompiledGraph<S>` implements `cognis_core::Runnable<S, S>`.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod analysis;
pub mod audit;
pub mod barrier;
pub mod builder;
pub mod channels;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod durability;
pub(crate) mod engine;
pub mod goto;
pub mod metrics;
pub mod node;
pub mod reducer;
pub mod snapshot;
pub mod state;
pub mod stream_mode;
pub mod subgraph;
pub(crate) mod validate;
pub mod viz;

pub use analysis::GraphAnalysis;
pub use audit::{AuditEntry, AuditKind, AuditLog, AuditLogObserver, InMemoryAuditLog};
pub use barrier::BarrierNode;
pub use builder::{Graph, LinearBuilder};
pub use channels::{
    AnyValue, BinaryOp, Broadcast, Channel, ChannelRef, CustomChannel, Topic, Untracked,
};
#[cfg(feature = "postgres")]
pub use checkpoint::PostgresCheckpointer;
#[cfg(feature = "sqlite")]
pub use checkpoint::SqliteCheckpointer;
pub use checkpoint::{ActiveSnapshot, Checkpointer, InMemoryCheckpointer};
pub use command::Command;
pub use compiled::CompiledGraph;
pub use durability::{Durability, DurabilityDecision, DurabilityHook};
pub use goto::Goto;
pub use metrics::{
    GraphMetrics, MetricsObserver, NodeTiming, ProfilingObserver, ThresholdCallback,
    ThresholdProfiler,
};
pub use node::{node_fn, Node, NodeCtx, NodeFn, NodeOut, NodeRetryPolicy};
pub use reducer::{Add, Append, Custom, LastValue, Merge, Reducer};
pub use snapshot::GraphSnapshot;
pub use state::GraphState;
pub use stream_mode::{StreamMode, StreamModes};
pub use subgraph::Subgraph;

/// Derive macro — generates `impl GraphState for <T>` with per-field reducers.
/// The derive name shadows the trait name; both are imported via
/// `use cognis_graph::GraphState;` because Rust allows a trait and a
/// derive macro to share the same identifier (they live in different namespaces).
pub use cognis_macros::GraphStateV2 as GraphState;

/// Re-export of [`cognis_core`] — graph users import from this crate
/// and `cognis_core::*` is implicitly in scope via re-export.
pub use cognis_core;

/// Re-export of the [`schemars`] crate (via cognis-core).
pub use cognis_core::schemars;
