//! Graph module — StateGraph builder API and related types.
//!
//! This module provides the core graph construction and execution primitives
//! for building stateful, multi-actor agent workflows.

pub mod branch;
pub mod message;
pub mod state;
pub mod subgraph;
pub mod ui;

pub use branch::{AsyncBranch, AsyncRouterFn, Branch, RouterFn, RouterResult};
pub use message::{add_messages, message_graph};
pub use state::{AsyncNodeAction, CompiledStateGraph, NodeAction, NodeSpec, StateGraph};
pub use subgraph::SubgraphNode;
pub use ui::{AnyUIMessage, RemoveUIMessage, UIMessage, ui_message_reducer};
