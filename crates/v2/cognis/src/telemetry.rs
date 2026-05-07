//! Telemetry — typed counters / events emitted alongside agent runs.
//!
//! Distinct from `cognis2_core::Observer`: that's a generic event sink
//! tied to `RunnableConfig`. Telemetry is a strongly-typed sidecar that
//! tracks token usage, call counts, errors, and latencies — the kind of
//! metrics ops dashboards consume.
//!
//! Customization:
//! - Implement [`TelemetrySink`] to forward to Prometheus / OTel /
//!   Datadog / your in-house metrics system.
//! - The default [`InMemoryTelemetry`] keeps counts in a Mutex; great
//!   for tests and dev.
//! - [`TelemetryEvent::Custom`] is the open variant — emit anything via
//!   `kind: String, payload: serde_json::Value`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

/// Strongly-typed telemetry events.
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// One LLM call started.
    LlmCallStart {
        /// Model id.
        model: String,
        /// Optional session correlation.
        session_id: Option<String>,
    },
    /// One LLM call finished (success).
    LlmCallEnd {
        /// Model id.
        model: String,
        /// Latency.
        latency: Duration,
        /// Tokens consumed (in + out + total).
        prompt_tokens: u64,
        /// Tokens emitted.
        completion_tokens: u64,
        /// Optional session correlation.
        session_id: Option<String>,
    },
    /// One tool dispatch started.
    ToolCallStart {
        /// Tool name.
        tool: String,
        /// Optional session correlation.
        session_id: Option<String>,
    },
    /// One tool dispatch finished.
    ToolCallEnd {
        /// Tool name.
        tool: String,
        /// Latency.
        latency: Duration,
        /// Optional session correlation.
        session_id: Option<String>,
    },
    /// An error occurred.
    Error {
        /// Component that errored (e.g. `"openai"`, `"calculator-tool"`).
        source: String,
        /// Stringified error.
        message: String,
        /// Optional session correlation.
        session_id: Option<String>,
    },
    /// Open variant for custom counters / events. Backends should map to
    /// their native counter/gauge primitive.
    Custom {
        /// User-defined kind label.
        kind: String,
        /// Arbitrary payload.
        payload: serde_json::Value,
        /// Optional session correlation.
        session_id: Option<String>,
    },
}

/// Pluggable telemetry sink.
#[async_trait]
pub trait TelemetrySink: Send + Sync {
    /// Record one event. Implementations should be cheap and non-blocking
    /// — slow sinks slow agent execution.
    async fn record(&self, event: TelemetryEvent);
}

/// Closure-based sink.
#[async_trait]
impl<F> TelemetrySink for F
where
    F: Fn(TelemetryEvent) + Send + Sync,
{
    async fn record(&self, event: TelemetryEvent) {
        (self)(event)
    }
}

/// In-memory snapshot of cumulative counters.
#[derive(Debug, Default, Clone)]
pub struct TelemetrySnapshot {
    /// Total LLM calls observed (start events).
    pub llm_calls: u64,
    /// Total tool calls observed.
    pub tool_calls: u64,
    /// Total error events.
    pub errors: u64,
    /// Cumulative prompt tokens.
    pub prompt_tokens: u64,
    /// Cumulative completion tokens.
    pub completion_tokens: u64,
    /// Per-model call counts.
    pub by_model: HashMap<String, u64>,
    /// Per-tool call counts.
    pub by_tool: HashMap<String, u64>,
    /// Per-session call counts.
    pub by_session: HashMap<String, u64>,
}

/// In-memory sink. Tracks counters that callers can inspect via
/// [`InMemoryTelemetry::snapshot`].
#[derive(Default)]
pub struct InMemoryTelemetry {
    inner: Mutex<TelemetrySnapshot>,
}

impl InMemoryTelemetry {
    /// New sink with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current counters.
    pub async fn snapshot(&self) -> TelemetrySnapshot {
        self.inner.lock().await.clone()
    }

    /// Reset all counters to zero.
    pub async fn reset(&self) {
        *self.inner.lock().await = TelemetrySnapshot::default();
    }
}

#[async_trait]
impl TelemetrySink for InMemoryTelemetry {
    async fn record(&self, event: TelemetryEvent) {
        let mut g = self.inner.lock().await;
        match event {
            TelemetryEvent::LlmCallStart { model, session_id } => {
                g.llm_calls += 1;
                *g.by_model.entry(model).or_insert(0) += 1;
                if let Some(s) = session_id {
                    *g.by_session.entry(s).or_insert(0) += 1;
                }
            }
            TelemetryEvent::LlmCallEnd {
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                g.prompt_tokens += prompt_tokens;
                g.completion_tokens += completion_tokens;
            }
            TelemetryEvent::ToolCallStart { tool, session_id } => {
                g.tool_calls += 1;
                *g.by_tool.entry(tool).or_insert(0) += 1;
                if let Some(s) = session_id {
                    *g.by_session.entry(s).or_insert(0) += 1;
                }
            }
            TelemetryEvent::ToolCallEnd { .. } => {}
            TelemetryEvent::Error { .. } => {
                g.errors += 1;
            }
            TelemetryEvent::Custom { .. } => {
                // Custom events don't update built-in counters; backends
                // that care should override.
            }
        }
    }
}

/// Type alias for a shared sink handle.
pub type TelemetryHandle = Arc<dyn TelemetrySink>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn counters_increment() {
        let sink = InMemoryTelemetry::new();
        sink.record(TelemetryEvent::LlmCallStart {
            model: "gpt-4o".into(),
            session_id: Some("s1".into()),
        })
        .await;
        sink.record(TelemetryEvent::LlmCallStart {
            model: "gpt-4o".into(),
            session_id: Some("s2".into()),
        })
        .await;
        sink.record(TelemetryEvent::LlmCallEnd {
            model: "gpt-4o".into(),
            latency: Duration::from_millis(100),
            prompt_tokens: 50,
            completion_tokens: 25,
            session_id: None,
        })
        .await;
        sink.record(TelemetryEvent::ToolCallStart {
            tool: "search".into(),
            session_id: None,
        })
        .await;
        sink.record(TelemetryEvent::Error {
            source: "openai".into(),
            message: "rate limit".into(),
            session_id: None,
        })
        .await;
        let snap = sink.snapshot().await;
        assert_eq!(snap.llm_calls, 2);
        assert_eq!(snap.tool_calls, 1);
        assert_eq!(snap.errors, 1);
        assert_eq!(snap.prompt_tokens, 50);
        assert_eq!(snap.completion_tokens, 25);
        assert_eq!(snap.by_model.get("gpt-4o").copied().unwrap_or(0), 2);
        assert_eq!(snap.by_session.get("s1").copied().unwrap_or(0), 1);
    }

    #[tokio::test]
    async fn reset_clears_counters() {
        let sink = InMemoryTelemetry::new();
        sink.record(TelemetryEvent::LlmCallStart {
            model: "m".into(),
            session_id: None,
        })
        .await;
        sink.reset().await;
        let snap = sink.snapshot().await;
        assert_eq!(snap.llm_calls, 0);
    }

    #[tokio::test]
    async fn custom_event_does_not_break_counters() {
        let sink = InMemoryTelemetry::new();
        sink.record(TelemetryEvent::Custom {
            kind: "cache-hit".into(),
            payload: serde_json::json!({"size": 12}),
            session_id: None,
        })
        .await;
        let snap = sink.snapshot().await;
        assert_eq!(snap.llm_calls, 0);
    }

    #[tokio::test]
    async fn closure_sink_works() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let count = Arc::new(AtomicU64::new(0));
        let c2 = count.clone();
        let sink = move |_e: TelemetryEvent| {
            c2.fetch_add(1, Ordering::SeqCst);
        };
        sink.record(TelemetryEvent::Error {
            source: "x".into(),
            message: "y".into(),
            session_id: None,
        })
        .await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
