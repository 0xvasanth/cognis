//! Pluggable observability for Cognis.
//!
//! Bridges [`cognis_core::CallbackHandler`] events to external observability
//! backends (Langfuse natively, LangSmith and OpenTelemetry in later phases).
//! See `docs/superpowers/specs/2026-05-06-cognis-trace-design.md` for the
//! full design.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub use error::TraceError;

pub mod span;
pub use span::{
    CostDetails, Generation, ObservationLevel, ScoreRecord, ScoreValue, Span, SpanKind, TokenUsage,
};

pub mod meta;
pub use meta::TraceMeta;

pub mod cost;
pub use cost::{default_pricing_2026_05, ModelPrice, PriceTable};

pub mod parent;

pub mod batch;
pub use batch::{Batcher, BatcherConfig, BatcherStats};

pub mod exporter;
pub use exporter::TraceExporter;

pub mod handler;
pub use handler::{TracingHandler, TracingHandlerBuilder};

pub mod exporters;

#[cfg(feature = "stdout")]
pub use exporters::stdout::StdoutExporter;
