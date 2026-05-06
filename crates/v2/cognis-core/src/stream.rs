//! Streaming primitives for cognis2: token-output streams and structured
//! event streams.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A structured event emitted by `stream_events()` — exposes per-step
/// graph activity, tool calls, token deltas, and errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    /// A `Runnable` started.
    OnStart {
        /// Name of the runnable that started.
        runnable: String,
        /// Correlation ID for this run.
        run_id: Uuid,
        /// Serialized input value.
        input: serde_json::Value,
    },
    /// A graph node started.
    OnNodeStart {
        /// Node name.
        node: String,
        /// Superstep number.
        step: u64,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// A graph node finished.
    OnNodeEnd {
        /// Node name.
        node: String,
        /// Superstep number.
        step: u64,
        /// Serialized output value.
        output: serde_json::Value,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// LLM emitted a token.
    OnLlmToken {
        /// The token text.
        token: String,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// Tool execution started.
    OnToolStart {
        /// Tool name.
        tool: String,
        /// Serialized arguments.
        args: serde_json::Value,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// Tool execution finished.
    OnToolEnd {
        /// Tool name.
        tool: String,
        /// Serialized result.
        result: serde_json::Value,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// A `Runnable` errored.
    OnError {
        /// Error description.
        error: String,
        /// Correlation ID for this run.
        run_id: Uuid,
    },
    /// A `Runnable` finished successfully.
    OnEnd {
        /// Name of the runnable that finished.
        runnable: String,
        /// Correlation ID for this run.
        run_id: Uuid,
        /// Serialized output value.
        output: serde_json::Value,
    },
}

/// Pluggable event sink. Multiple observers can subscribe to a single run.
pub trait Observer: Send + Sync {
    /// Called for every event emitted during execution. Implementations
    /// should be cheap and non-blocking — a slow observer slows execution.
    fn on_event(&self, event: &Event);
}

/// Convenience: any `Fn(&Event) + Send + Sync` is an `Observer`.
impl<F> Observer for F
where
    F: Fn(&Event) + Send + Sync,
{
    fn on_event(&self, event: &Event) {
        self(event)
    }
}

/// A stream of structured events. Same shape as `RunnableStream<Event>`,
/// but named separately to make stream-of-events vs stream-of-output
/// distinguishable at the type level.
pub struct EventStream(Pin<Box<dyn Stream<Item = Event> + Send>>);

impl EventStream {
    /// Wrap an arbitrary `Stream<Item = Event>`.
    pub fn new(s: impl Stream<Item = Event> + Send + 'static) -> Self {
        Self(Box::pin(s))
    }
}

impl Stream for EventStream {
    type Item = Event;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.as_mut().poll_next(cx)
    }
}

/// A stream of `Result<O>` items — the canonical output stream type for
/// `Runnable::stream`. Wraps `Pin<Box<dyn Stream>>` for trait-object
/// flexibility, with helper combinators on the wrapper.
pub struct RunnableStream<O> {
    inner: Pin<Box<dyn Stream<Item = crate::Result<O>> + Send>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fn_observer_works() {
        let count = Arc::new(AtomicUsize::new(0));
        let count2 = count.clone();
        let observer: Arc<dyn Observer> = Arc::new(move |_e: &Event| {
            count2.fetch_add(1, Ordering::SeqCst);
        });

        let e = Event::OnStart {
            runnable: "x".into(),
            run_id: Uuid::nil(),
            input: serde_json::json!({}),
        };
        observer.on_event(&e);
        observer.on_event(&e);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn event_serialization_tagged() {
        let e = Event::OnLlmToken { token: "hi".into(), run_id: Uuid::nil() };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"type\":\"OnLlmToken\""));
        assert!(s.contains("\"token\":\"hi\""));
    }
}
