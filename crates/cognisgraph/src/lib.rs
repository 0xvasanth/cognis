//! # cognisgraph
//!
//! Orchestration layer for building stateful, multi-actor agent workflows as
//! directed graphs. Inspired by Pregel and Apache Beam, cognisgraph provides a
//! [`StateGraph`](graph::state::StateGraph) builder that compiles into an
//! executable graph with support for checkpointing, streaming, human-in-the-loop
//! interrupts, and subgraph composition.
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use cognisgraph::graph::state::StateGraph;
//! use cognisgraph::{START, END};
//! use serde_json::{json, Value};
//!
//! let mut graph = StateGraph::new();
//! graph.add_node("greet", |state: Value| {
//!     Ok(json!({"response": "Hello!"}))
//! });
//! graph.add_edge(START, "greet");
//! graph.add_edge("greet", END);
//! let compiled = graph.compile(None).unwrap();
//! let result = compiled.invoke(json!({})).await.unwrap();
//! ```
//!
//! ## Key Modules
//!
//! - [`graph`] -- `StateGraph` builder, `CompiledStateGraph`, conditional branching, subgraphs.
//! - [`pregel`] -- Pregel-style execution engine with superstep processing.
//! - [`channels`] -- State channels: LastValue, BinaryOp, Topic, AnyValue, NamedBarrier.
//! - [`checkpoint`] -- `CheckpointSaver` trait with SQLite backend (behind `sqlite` feature).
//! - [`prebuilt`] -- `create_react_agent` for tool-calling ReAct loops.
//! - [`types`] -- Core types: `StreamMode`, `InterruptType`, `RetryPolicy`, `CachePolicy`.
//! - [`runtime`] -- Tokio-based async runtime utilities.

pub mod channels;
pub mod checkpoint;
pub mod config;
pub mod constants;
pub mod errors;
pub mod graph;
pub mod managed;
pub mod prebuilt;
pub mod pregel;
pub mod runtime;
pub mod types;
pub mod utils;

pub use constants::{END, START};
pub use errors::{LangGraphError, Result};
pub use graph::{GraphStateField, GraphStateSchema};
pub use runtime::Runtime;
pub use types::*;
