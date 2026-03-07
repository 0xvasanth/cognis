//! Graph module — StateGraph builder API and related types.
//!
//! This module provides the core graph construction and execution primitives
//! for building stateful, multi-actor agent workflows.
//!
//! Key submodules:
//! - [`state`] — `StateGraph` builder and `CompiledStateGraph` execution
//! - [`runner`] — `GraphRunner` with configurable step limits, timeouts, and lifecycle hooks (`StepHook`)
//! - [`stream_writer`] — `StreamWriter`/`StreamReader` for async chunk-based streaming, plus `StreamCollector` and `FilteredStream`
//! - [`persistent`] — `PersistentGraph` with automatic checkpoint save/restore
//! - [`subgraph`] — Subgraph composition for modular workflows
//! - [`human_in_loop`] — Human-in-the-loop interrupt and approval patterns
//! - [`snapshot`] — Graph state snapshots with pluggable storage
//! - [`audit`] — Execution audit log and trail tracing
//! - [`ascii`] / [`mermaid`] — Graph visualization (terminal and Mermaid diagrams)

pub mod annotations;
pub mod ascii;
pub mod audit;
pub mod branch;
pub mod breakpoint;
pub mod hooks;
pub mod human_in_loop;
pub mod mermaid;
pub mod message;
pub mod persistent;
pub mod runner;
pub mod send;
pub mod serialize;
pub mod snapshot;
pub mod state;
pub mod stream_events;
pub mod stream_writer;
pub mod subgraph;
pub mod time_travel;
pub mod ui;
pub mod validator;

pub use annotations::{
    apply_annotations, AnnotatedState, AnnotatedStateBuilder, AnnotatedStateGraph,
    CompiledAnnotatedStateGraph, FieldAnnotation, JsonType,
};
pub use ascii::{to_ascii, AsciiGraphRenderer, AsciiRenderOptions, NodeStyle};
pub use audit::{
    make_event, AuditEvent, AuditEventType, AuditLog, AuditLogConfig, AuditReport, AuditSeverity,
    AuditTrail,
};
pub use branch::{AsyncBranch, AsyncRouterFn, Branch, RouterFn, RouterResult};
pub use breakpoint::{
    AutoApproveHandler, BreakpointAction, BreakpointEvent, BreakpointHandler, BreakpointManager,
    BreakpointState, BreakpointType, LoggingBreakpointHandler,
};
pub use hooks::{
    ExecutionHook, HookAction, HookContext, HookPhase, HookRegistry,
    LoggingHook as HooksLoggingHook, StateSnapshot, StateSnapshotHook, StateValidationHook,
    TimingHook,
};
pub use human_in_loop::{ApprovalRequest, HumanAction, HumanInTheLoop, HumanInTheLoopResult};
pub use mermaid::{to_mermaid, to_mermaid_url};
pub use message::{add_messages, message_graph};
pub use persistent::PersistentGraph;
pub use runner::{GraphRunner, LoggingHook, MetricsHook, RunConfig, StepEvent, StepHook};
pub use send::{deep_merge_values, fan_in, fan_out, send_to, MapReduceGraph, SendCommand};
pub use serialize::{ConditionalEdgeDef, GraphDefinition, GraphRegistry};
pub use snapshot::{
    ExecutionStep, FileSnapshotStorage, GraphSnapshot, InMemorySnapshotStorage, SnapshotDiff,
    SnapshotManager, SnapshotStorage, SnapshotSummary,
};
pub use state::{AsyncNodeAction, CompiledStateGraph, NodeAction, NodeSpec, StateGraph};
pub use stream_events::{stream_graph_events, GraphEventCollector, GraphStreamEvent};
pub use stream_writer::{FilteredStream, StreamChunk, StreamCollector, StreamReader, StreamWriter};
pub use subgraph::SubgraphNode;
pub use time_travel::TimeTravelEngine;
pub use ui::{ui_message_reducer, AnyUIMessage, RemoveUIMessage, UIMessage};
pub use validator::{
    CycleDetector, GraphValidator, ReachabilityAnalyzer, ValidationIssue, ValidationResult,
    ValidationSeverity,
};
