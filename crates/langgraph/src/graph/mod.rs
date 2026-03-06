//! Graph module — StateGraph builder API and related types.
//!
//! This module provides the core graph construction and execution primitives
//! for building stateful, multi-actor agent workflows.

pub mod branch;
pub mod mermaid;
pub mod message;
pub mod persistent;
pub mod serialize;
pub mod state;
pub mod subgraph;
pub mod ui;

pub use branch::{AsyncBranch, AsyncRouterFn, Branch, RouterFn, RouterResult};
pub use mermaid::{to_mermaid, to_mermaid_url};
pub use message::{add_messages, message_graph};
pub use serialize::{ConditionalEdgeDef, GraphDefinition, GraphRegistry};
pub use state::{AsyncNodeAction, CompiledStateGraph, NodeAction, NodeSpec, StateGraph};
pub use persistent::PersistentGraph;
pub use subgraph::SubgraphNode;
pub use ui::{AnyUIMessage, RemoveUIMessage, UIMessage, ui_message_reducer};
