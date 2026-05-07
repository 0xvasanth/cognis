//! Bridge cognis2 [`Event`]s into the [`tracing`] crate.
//!
//! Construction is zero-config:
//!
//! ```ignore
//! use std::sync::Arc;
//! use cognis2::{RunnableConfig, Observer};
//! use cognis2::observers::TracingObserver;
//!
//! let cfg = RunnableConfig::default()
//!     .with_observer(Arc::new(TracingObserver::new()));
//! ```
//!
//! Wire any `tracing-subscriber` (OTel, JSON, ...) on top — that's the
//! Rust-native equivalent of a "LangSmith tracer."

use cognis2_core::{Event, Observer};

/// Forwards every `Event` into the `tracing` crate at the appropriate level.
///
/// - `OnStart`, `OnEnd`, `OnNodeStart`, `OnNodeEnd`, `OnToolStart`,
///   `OnToolEnd` → `tracing::info!`.
/// - `OnLlmToken` → `tracing::trace!` (high-volume, off by default).
/// - `OnError` → `tracing::error!`.
#[derive(Debug, Default, Clone, Copy)]
pub struct TracingObserver;

impl TracingObserver {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl Observer for TracingObserver {
    fn on_event(&self, event: &Event) {
        match event {
            Event::OnStart {
                runnable,
                run_id,
                input,
            } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    runnable = runnable.as_str(),
                    input = %input,
                    "runnable.start"
                );
            }
            Event::OnEnd {
                runnable,
                run_id,
                output,
            } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    runnable = runnable.as_str(),
                    output = %output,
                    "runnable.end"
                );
            }
            Event::OnNodeStart { node, step, run_id } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    node = node.as_str(),
                    step,
                    "node.start"
                );
            }
            Event::OnNodeEnd {
                node,
                step,
                output,
                run_id,
            } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    node = node.as_str(),
                    step,
                    output = %output,
                    "node.end"
                );
            }
            Event::OnToolStart { tool, args, run_id } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    tool = tool.as_str(),
                    args = %args,
                    "tool.start"
                );
            }
            Event::OnToolEnd {
                tool,
                result,
                run_id,
            } => {
                tracing::info!(
                    target: "cognis2",
                    %run_id,
                    tool = tool.as_str(),
                    result = %result,
                    "tool.end"
                );
            }
            Event::OnLlmToken { token, run_id } => {
                tracing::trace!(
                    target: "cognis2",
                    %run_id,
                    token = token.as_str(),
                    "llm.token"
                );
            }
            Event::OnError { error, run_id } => {
                tracing::error!(
                    target: "cognis2",
                    %run_id,
                    error = error.as_str(),
                    "runnable.error"
                );
            }
            Event::OnCheckpoint { step, run_id } => {
                tracing::debug!(
                    target: "cognis2",
                    %run_id,
                    step,
                    "graph.checkpoint"
                );
            }
            Event::Custom {
                kind,
                payload,
                run_id,
            } => {
                tracing::debug!(
                    target: "cognis2",
                    %run_id,
                    kind = kind.as_str(),
                    payload = %payload,
                    "graph.custom"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn forwards_without_panicking() {
        let obs = TracingObserver::new();
        obs.on_event(&Event::OnStart {
            runnable: "x".into(),
            run_id: Uuid::nil(),
            input: serde_json::json!({}),
        });
        obs.on_event(&Event::OnEnd {
            runnable: "x".into(),
            run_id: Uuid::nil(),
            output: serde_json::json!(null),
        });
        obs.on_event(&Event::OnError {
            error: "boom".into(),
            run_id: Uuid::nil(),
        });
    }
}
