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
