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

pub mod analysis;
pub mod annotations;
pub mod ascii;
pub mod audit;
pub mod branch;
pub mod breakpoint;
pub mod compiled;
pub mod conditional;
pub mod debug;
pub mod hooks;
pub mod human_in_loop;
pub mod interrupt;
pub mod interrupts;
pub mod mermaid;
pub mod metrics;
pub mod message;
pub mod persistent;
pub mod runner;
pub mod send;
pub mod serialization;
pub mod serialize;
pub mod snapshot;
pub mod state;
pub mod state_machine;
pub mod stream_events;
pub mod stream_writer;
pub mod streaming;
pub mod subgraph;
pub mod time_travel;
pub mod ui;
pub mod validation;
pub mod validator;
pub mod versioning;

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
pub use compiled::{
    CompiledEdge, CompiledGraph, CompiledNode, ExecutionLog, GraphCompiler, GraphDefinitionBuilder,
    LogEntry,
};
pub use debug::{
    BreakCondition as DebugBreakCondition, Breakpoint as DebugBreakpoint, DebugSession, DebugStep,
    DeltaType, ExecutionProfiler, StateDelta, StateInspector, StateReport,
};
pub use hooks::{
    ExecutionHook, HookAction, HookContext, HookPhase, HookRegistry,
    LoggingHook as HooksLoggingHook, StateSnapshot as HooksStateSnapshot, StateSnapshotHook,
    StateValidationHook, TimingHook,
};
pub use human_in_loop::{
    ApprovalPolicy, ApprovalRequest, FeedbackLog, HumanAction, HumanApproval, HumanFeedback,
    HumanInLoopConfig, HumanInLoopConfigBuilder, HumanInTheLoop, HumanInTheLoopResult,
    HumanReviewPoint, ReviewApprovalRequest, ReviewApprovalRequestBuilder, ReviewQueue,
};
pub use interrupts::{
    InterruptAction, InterruptHandler as InterruptsHandler, InterruptLog as InterruptsLog,
    InterruptPolicy, InterruptQueue, InterruptRequest, InterruptResponse,
    InterruptType as InterruptsType, TimeoutPolicy,
};
pub use mermaid::{to_mermaid, to_mermaid_url};
pub use metrics::{
    EdgeMetrics, ExecutionProfile, GraphMetrics, InMemoryMetricsCollector, MetricsAggregator,
    MetricsCollector, MetricsExporter, MetricsReport, NodeMetrics, NodeSummary, ProfileEntry,
};
pub use message::{
    add_messages, message_graph, GraphMessage, MessageEdge, MessageGraph, MessageGraphBuilder,
    MessageNode, MessageRole, MessageState,
};
pub use persistent::PersistentGraph;
pub use runner::{GraphRunner, LoggingHook, MetricsHook, RunConfig, StepEvent, StepHook};
pub use send::{deep_merge_values, fan_in, fan_out, send_to, MapReduceGraph, SendCommand};
pub use serialization::{
    EdgeType as SerializationEdgeType, GraphChanges, GraphDefBuilder, GraphFormat, GraphTemplate,
    GraphVersioning, SerializableGraphDefinition, SerializedEdge, SerializedNode,
};
pub use serialize::{ConditionalEdgeDef, GraphDefinition, GraphRegistry};
pub use snapshot::{
    SnapshotComparator, SnapshotDiff, SnapshotId, SnapshotStore, StateSnapshot, TimeTravelDebugger,
};
pub use state::{AsyncNodeAction, CompiledStateGraph, NodeAction, NodeSpec, StateGraph};
pub use state_machine::{
    StateHistory, StateId, StateMachine, StateMachineBuilder, StateMachineValidator,
    StateTransitionRecord, Transition,
};
pub use stream_events::{stream_graph_events, GraphEventCollector, GraphStreamEvent};
pub use stream_writer::{FilteredStream, StreamChunk, StreamCollector, StreamReader, StreamWriter};
pub use streaming::{
    ChunkType, StreamAggregator as StreamingAggregator, StreamChunk as StreamingChunk,
    StreamCollector as StreamingCollector, StreamFilter, StreamMetrics, StreamMode, StreamReplay,
    StreamSubscription,
};
pub use subgraph::{
    GraphFn, NestedSubgraph, StateMapping, SubgraphBuilder, SubgraphConfig, SubgraphNode,
    SubgraphRegistry,
};
pub use time_travel::TimeTravelEngine;
pub use ui::{ui_message_reducer, AnyUIMessage, RemoveUIMessage, UIMessage};
pub use validation::{
    GraphValidator as StructuralGraphValidator, QuickCheck, SchemaValidator, StateRule,
    StateValidator, ValidationCache, ValidationIssue as ValidationFinding, ValidationLevel,
    ValidationReport,
};
pub use validator::{
    CycleDetector, GraphValidator, ReachabilityAnalyzer, ValidationIssue, ValidationResult,
    ValidationSeverity,
};
