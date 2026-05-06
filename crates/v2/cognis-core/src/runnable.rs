//! The unified `Runnable<I, O>` trait + per-call configuration.

use std::sync::Arc;
use std::time::Instant;

use uuid::Uuid;

use crate::extensions::Extensions;
use crate::stream::Observer;

/// Per-invocation configuration. Defaults are sensible; override only
/// what you need.
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
}
