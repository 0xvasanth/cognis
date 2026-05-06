//! Streaming primitives for cognis2: token-output streams and structured
//! event streams.

use std::pin::Pin;

use futures::{Stream, StreamExt};
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

impl<O> RunnableStream<O>
where
    O: Send + 'static,
{
    /// Wrap any `Stream<Item = Result<O>>`.
    pub fn new(s: impl Stream<Item = crate::Result<O>> + Send + 'static) -> Self {
        Self { inner: Box::pin(s) }
    }

    /// Build from a single value (one-shot stream).
    pub fn once(value: crate::Result<O>) -> Self {
        Self::new(futures::stream::once(async move { value }))
    }

    /// Collect all items into a `Vec`. Stops at the first `Err`.
    pub async fn collect_into_vec(mut self) -> crate::Result<Vec<O>> {
        let mut out = Vec::new();
        while let Some(item) = self.inner.next().await {
            out.push(item?);
        }
        Ok(out)
    }

    /// Apply a side-effect callback to each item (errors pass through unchanged).
    pub fn with_callback<F>(self, f: F) -> Self
    where
        F: Fn(&O) + Send + Sync + 'static,
    {
        let inner = self.inner.map(move |item| {
            if let Ok(ref v) = item {
                f(v);
            }
            item
        });
        Self::new(inner)
    }
}

impl<O> Stream for RunnableStream<O> {
    type Item = crate::Result<O>;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
        let e = Event::OnLlmToken {
            token: "hi".into(),
            run_id: Uuid::nil(),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"type\":\"OnLlmToken\""));
        assert!(s.contains("\"token\":\"hi\""));
    }

    #[tokio::test]
    async fn runnable_stream_collect() {
        let s = RunnableStream::new(futures::stream::iter(vec![Ok(1u32), Ok(2), Ok(3)]));
        let v = s.collect_into_vec().await.unwrap();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn runnable_stream_callback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter2 = counter.clone();
        let s = RunnableStream::new(futures::stream::iter(vec![Ok(10u32), Ok(20)])).with_callback(
            move |v| {
                counter2.fetch_add(*v as usize, Ordering::SeqCst);
            },
        );
        let _ = s.collect_into_vec().await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 30);
    }

    #[tokio::test]
    async fn runnable_stream_short_circuits_on_error() {
        let s: RunnableStream<u32> = RunnableStream::new(futures::stream::iter(vec![
            Ok(1),
            Err(crate::CognisError::Internal("stop".into())),
            Ok(3),
        ]));
        let result = s.collect_into_vec().await;
        assert!(result.is_err());
    }
}
