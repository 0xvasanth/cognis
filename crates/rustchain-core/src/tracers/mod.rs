//! Tracer infrastructure for observability and run tracking.
//!
//! Mirrors Python `langchain_core.tracers`.

pub mod base;
pub mod event_stream;
pub mod log_stream;
pub mod otel;
pub mod run_collector;
pub mod schemas;
pub mod stdout;

pub use base::{AsyncBaseTracer, BaseTracer};
pub use event_stream::{
    EventData, EventStreamCallbackHandler, EventType, RootEventFilter, RunInfo, StreamEvent,
};
pub use log_stream::{JsonPatchOp, LogEntry, RunLogPatch, RunState};
pub use otel::{OtelTraceCallbackHandler, SpanEvent, SpanStatus, TraceSpan};
pub use run_collector::RunCollectorCallbackHandler;
pub use schemas::Run;
pub use stdout::ConsoleCallbackHandler;
