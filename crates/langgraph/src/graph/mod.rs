//! Graph module — StateGraph builder API and related types.
//!
//! This module provides the core graph construction and execution primitives
//! for building stateful, multi-actor agent workflows.

pub mod annotations;
pub mod branch;
pub mod human_in_loop;
pub mod mermaid;
pub mod message;
pub mod persistent;
pub mod send;
pub mod serialize;
pub mod state;
pub mod stream_events;
pub mod subgraph;
pub mod time_travel;
pub mod ui;

pub use annotations::{
    apply_annotations, AnnotatedState, AnnotatedStateBuilder, AnnotatedStateGraph,
    CompiledAnnotatedStateGraph, FieldAnnotation, JsonType,
};
pub use branch::{AsyncBranch, AsyncRouterFn, Branch, RouterFn, RouterResult};
pub use human_in_loop::{ApprovalRequest, HumanAction, HumanInTheLoop, HumanInTheLoopResult};
pub use mermaid::{to_mermaid, to_mermaid_url};
pub use message::{add_messages, message_graph};
pub use persistent::PersistentGraph;
pub use send::{deep_merge_values, fan_in, fan_out, send_to, MapReduceGraph, SendCommand};
pub use serialize::{ConditionalEdgeDef, GraphDefinition, GraphRegistry};
pub use state::{AsyncNodeAction, CompiledStateGraph, NodeAction, NodeSpec, StateGraph};
pub use stream_events::{stream_graph_events, GraphEventCollector, GraphStreamEvent};
pub use subgraph::SubgraphNode;
pub use time_travel::TimeTravelEngine;
pub use ui::{ui_message_reducer, AnyUIMessage, RemoveUIMessage, UIMessage};
