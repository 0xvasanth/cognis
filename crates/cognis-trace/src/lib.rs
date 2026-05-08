//! Pluggable observability for Cognis.
//!
//! Bridges [`cognis_core::CallbackHandler`] events to external observability
//! backends. Phase 1 ships the bridge handler, three core types, two built-in
//! exporters (stdout + mock), and the Langfuse backend (traces, prompts,
//! scores). LangSmith and OpenTelemetry are Phase 2.
//!
//! # Quick start
//!
//! ```no_run
//! use cognis_trace::{MockExporter, TraceMeta, TracingHandler};
//! use cognis_trace::meta::merge_into;
//!
//! # async fn main_inner() {
//! let handler = TracingHandler::builder()
//!     .with_exporter(MockExporter::new())
//!     .with_default_pricing()
//!     .build();
//!
//! let metadata = merge_into(serde_json::Value::Null, TraceMeta::session("session-abc"));
//! # }
//! ```
//!
//! See `docs/superpowers/specs/2026-05-06-cognis-trace-design.md` for the
//! full design.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod batch;
pub mod cost;
pub mod error;
pub mod exporter;
pub mod exporters;
pub mod handler;
pub mod meta;
pub mod parent;
pub mod prompts;
pub mod scores;
pub mod span;

pub use batch::{Batcher, BatcherConfig, BatcherStats};
pub use cost::{default_pricing_2026_05, ModelPrice, PriceTable};
pub use error::TraceError;
pub use exporter::TraceExporter;
pub use exporters::mock::MockExporter;
pub use handler::{TracingHandler, TracingHandlerBuilder};
pub use meta::TraceMeta;
pub use prompts::{ChatMessageTemplate, Prompt, PromptBody, PromptStore};
pub use scores::ScoreSink;
pub use span::{
    CostDetails, Generation, ObservationLevel, ScoreRecord, ScoreValue, Span, SpanBuilder,
    SpanKind, TokenUsage,
};

#[cfg(feature = "stdout")]
pub use exporters::stdout::StdoutExporter;

#[cfg(feature = "langfuse")]
pub use exporters::langfuse::{
    LangfuseConfig, LangfuseExporter, LangfusePromptClient, LangfuseScorer,
};
