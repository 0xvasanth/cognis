//! `TraceExporter` trait — the contract for shipping spans to a backend.

use async_trait::async_trait;

use crate::error::TraceError;
use crate::span::{ScoreRecord, Span};

/// One backend's contract: receive batches of completed spans (and
/// optionally scores), flush on shutdown.
#[async_trait]
pub trait TraceExporter: Send + Sync {
    /// Ship a batch of completed spans. Called from the batcher's flush
    /// task — implementations should not block forever and should return
    /// errors for retry/logging by the bridge.
    async fn export_spans(&self, spans: Vec<Span>) -> Result<(), TraceError>;

    /// Ship a batch of scores. Default returns `Unsupported`.
    async fn export_scores(&self, _scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        Err(TraceError::Unsupported("scores"))
    }

    /// Graceful flush + close. Default is a no-op.
    async fn shutdown(&self) -> Result<(), TraceError> {
        Ok(())
    }

    /// Stable name for diagnostics (e.g. "langfuse", "stdout").
    fn name(&self) -> &str;
}
