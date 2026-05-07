//! The unified `Runnable<I, O>` trait + per-call configuration.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use uuid::Uuid;

use crate::extensions::Extensions;
use crate::stream::Observer;

/// Per-invocation configuration. Defaults are sensible; override only
/// what you need.
#[derive(Clone)]
pub struct RunnableConfig {
    /// Maximum number of graph supersteps / chain depth before erroring with
    /// `CognisError::RecursionLimit`.
    pub recursion_limit: u32,
    /// Maximum concurrent in-flight tasks (used by `batch` and parallel nodes).
    pub max_concurrency: usize,
    /// Free-form telemetry tags (e.g. ["production", "feature/foo"]).
    pub tags: Vec<String>,
    /// User-supplied metadata, attached to every emitted Event.
    pub metadata: serde_json::Value,
    /// Event subscribers. Multiple are allowed; each receives every event.
    pub observers: Vec<Arc<dyn Observer>>,
    /// Correlation ID. Defaults to a fresh UUID per `Default::default()`.
    pub run_id: Uuid,
    /// Cooperative cancellation token.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// Hard deadline (if set, framework checks it at every superstep boundary).
    pub deadline: Option<Instant>,
    /// Plugin-supplied typed payloads.
    pub extras: Extensions,
}

impl Default for RunnableConfig {
    fn default() -> Self {
        Self {
            recursion_limit: 25,
            max_concurrency: num_cpus::get().max(1),
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            observers: Vec::new(),
            run_id: Uuid::new_v4(),
            cancel_token: None,
            deadline: None,
            extras: Extensions::new(),
        }
    }
}

impl RunnableConfig {
    /// Create with defaults. Equivalent to `RunnableConfig::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the recursion limit (builder-style).
    pub fn with_recursion_limit(mut self, n: u32) -> Self {
        self.recursion_limit = n;
        self
    }

    /// Set the max concurrency (builder-style).
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    /// Add a single observer (builder-style).
    pub fn with_observer(mut self, o: Arc<dyn Observer>) -> Self {
        self.observers.push(o);
        self
    }

    /// Add a tag (builder-style).
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set the cancellation token (builder-style).
    pub fn with_cancel_token(mut self, t: tokio_util::sync::CancellationToken) -> Self {
        self.cancel_token = Some(t);
        self
    }

    /// Notify every registered observer of an event.
    pub fn emit(&self, event: &crate::stream::Event) {
        for o in &self.observers {
            o.on_event(event);
        }
    }

    /// True if the cancel token has been triggered.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| t.is_cancelled())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_sane() {
        let c = RunnableConfig::default();
        assert_eq!(c.recursion_limit, 25);
        assert!(c.max_concurrency >= 1);
        assert!(c.observers.is_empty());
    }

    #[test]
    fn builder_chains() {
        let c = RunnableConfig::new()
            .with_recursion_limit(10)
            .with_max_concurrency(4)
            .with_tag("prod");
        assert_eq!(c.recursion_limit, 10);
        assert_eq!(c.max_concurrency, 4);
        assert_eq!(c.tags, vec!["prod"]);
    }

    #[test]
    fn cancel_default_false() {
        let c = RunnableConfig::default();
        assert!(!c.is_cancelled());
    }

    #[test]
    fn config_clones_with_extras_emptied() {
        let mut c = RunnableConfig::default()
            .with_recursion_limit(50)
            .with_max_concurrency(8)
            .with_tag("test");
        c.extras.insert(42u32);
        assert!(c.extras.contains::<u32>());

        let cloned = c.clone();
        assert_eq!(cloned.recursion_limit, 50);
        assert_eq!(cloned.max_concurrency, 8);
        assert_eq!(cloned.tags, vec!["test"]);
        // Per the Extensions::clone contract (Plan #2), extras don't deep-clone.
        assert!(cloned.extras.is_empty());
    }
}

/// The unified contract every cognis primitive implements.
///
/// Generic over `I` (input) and `O` (output). One required method (`invoke`);
/// `batch`, `stream`, and `stream_events` have sensible defaults that
/// implementations override only when they can do better.
#[async_trait]
pub trait Runnable<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// One-shot invocation. The hot path.
    async fn invoke(&self, input: I, config: RunnableConfig) -> crate::Result<O>;

    /// Run multiple inputs in parallel. Defaults to `buffer_unordered`
    /// honouring `config.max_concurrency`.
    async fn batch(&self, inputs: Vec<I>, config: RunnableConfig) -> crate::Result<Vec<O>>
    where
        I: 'static,
        O: 'static,
        Self: Sized + Sync,
    {
        let concurrency = config.max_concurrency.max(1);
        let cfg = Arc::new(config);
        stream::iter(inputs)
            .map(|input| {
                let cfg = cfg.clone();
                async move {
                    self.invoke(input, RunnableConfig::clone_for_subcall(&cfg))
                        .await
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
    }

    /// Stream the final output (chunks of `O`). Default emits one item via
    /// `invoke` — non-streaming runnables are correct without override.
    async fn stream(&self, input: I, config: RunnableConfig) -> crate::Result<RunnableStream<O>>
    where
        Self: Sized + Sync,
    {
        let result = self.invoke(input, config).await;
        Ok(RunnableStream::once(result))
    }

    /// Stream structured events. Default emits OnStart + OnEnd around an
    /// `invoke` call. Graph engines override to surface per-node events.
    async fn stream_events(&self, input: I, config: RunnableConfig) -> crate::Result<EventStream>
    where
        I: serde::Serialize,
        O: serde::Serialize,
        Self: Sized + Sync,
    {
        let runnable = self.name().to_string();
        let run_id = config.run_id;
        let input_json = serde_json::to_value(&input).unwrap_or(serde_json::Value::Null);

        let on_start = Event::OnStart {
            runnable: runnable.clone(),
            run_id,
            input: input_json,
        };
        let result = self.invoke(input, config).await;
        let on_end_or_err = match &result {
            Ok(o) => Event::OnEnd {
                runnable,
                run_id,
                output: serde_json::to_value(o).unwrap_or(serde_json::Value::Null),
            },
            Err(e) => Event::OnError {
                error: e.to_string(),
                run_id,
            },
        };

        Ok(EventStream::new(stream::iter(vec![
            on_start,
            on_end_or_err,
        ])))
    }

    /// Friendly name for telemetry / introspection.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// JSON Schema for the input type, if known.
    fn input_schema(&self) -> Option<serde_json::Value> {
        None
    }

    /// JSON Schema for the output type, if known.
    fn output_schema(&self) -> Option<serde_json::Value> {
        None
    }
}

use crate::stream::{Event, EventStream, RunnableStream};

impl RunnableConfig {
    /// Build a child config for a sub-call (batch / fan-out).
    /// Reuses `tags`, `metadata`, `observers`, `cancel_token`, `deadline`
    /// — everything except a fresh `run_id` and an empty `extras`.
    pub fn clone_for_subcall(parent: &Arc<RunnableConfig>) -> RunnableConfig {
        RunnableConfig {
            recursion_limit: parent.recursion_limit,
            max_concurrency: parent.max_concurrency,
            tags: parent.tags.clone(),
            metadata: parent.metadata.clone(),
            observers: parent.observers.clone(),
            run_id: Uuid::new_v4(),
            cancel_token: parent.cancel_token.clone(),
            deadline: parent.deadline,
            extras: Extensions::new(),
        }
    }
}

#[cfg(test)]
mod runnable_tests {
    use super::*;
    use async_trait::async_trait;

    struct Doubler;

    #[async_trait]
    impl Runnable<u32, u32> for Doubler {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> crate::Result<u32> {
            Ok(input * 2)
        }
    }

    #[tokio::test]
    async fn invoke_works() {
        let d = Doubler;
        let out = d.invoke(5, RunnableConfig::default()).await.unwrap();
        assert_eq!(out, 10);
    }

    #[tokio::test]
    async fn default_batch_runs_each() {
        let d = Doubler;
        let out = d
            .batch(vec![1, 2, 3, 4], RunnableConfig::default())
            .await
            .unwrap();
        let mut sorted = out;
        sorted.sort();
        assert_eq!(sorted, vec![2, 4, 6, 8]);
    }

    #[tokio::test]
    async fn default_stream_emits_one_item() {
        let d = Doubler;
        let s = d.stream(7, RunnableConfig::default()).await.unwrap();
        let v = s.collect_into_vec().await.unwrap();
        assert_eq!(v, vec![14]);
    }

    #[tokio::test]
    async fn default_stream_events_emits_start_end() {
        use futures::StreamExt;
        let d = Doubler;
        let mut s = d.stream_events(3, RunnableConfig::default()).await.unwrap();
        let mut events = Vec::new();
        while let Some(e) = s.next().await {
            events.push(e);
        }
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Event::OnStart { .. }));
        assert!(matches!(events[1], Event::OnEnd { .. }));
    }

    #[tokio::test]
    async fn batch_respects_max_concurrency() {
        let d = Doubler;
        let cfg = RunnableConfig::default().with_max_concurrency(1);
        let out = d.batch(vec![1, 2, 3], cfg).await.unwrap();
        let mut sorted = out;
        sorted.sort();
        assert_eq!(sorted, vec![2, 4, 6]);
    }
}
