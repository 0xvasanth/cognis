use serde_json::Value;
use std::fmt;
use std::sync::Arc;

/// A stream writer function that accepts a Value.
pub type StreamWriter = Arc<dyn Fn(Value) + Send + Sync>;

/// No-op stream writer that discards values.
pub fn no_op_stream_writer() -> StreamWriter {
    Arc::new(|_| {})
}

/// Returns a default `Runtime` instance.
pub fn default_runtime() -> Runtime {
    Runtime::new()
}

/// Runtime context for graph execution.
///
/// Provides runtime information to nodes: context, store reference,
/// stream writer, and previous return value.
#[derive(Clone)]
pub struct Runtime {
    /// Static context for the graph run (user_id, db_conn, etc.)
    pub context: Option<Value>,
    /// Stream writer function for custom streaming output.
    pub stream_writer: StreamWriter,
    /// Previous return value for the thread (functional API with checkpointer).
    pub previous: Option<Value>,
}

impl Runtime {
    /// Creates a new `Runtime` with default values: no context, no-op writer, no previous.
    pub fn new() -> Self {
        Self {
            context: None,
            stream_writer: no_op_stream_writer(),
            previous: None,
        }
    }

    /// Builder method to set the context.
    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }

    /// Builder method to set the stream writer.
    pub fn with_stream_writer(mut self, writer: StreamWriter) -> Self {
        self.stream_writer = writer;
        self
    }

    /// Builder method to set the previous return value.
    pub fn with_previous(mut self, previous: Value) -> Self {
        self.previous = Some(previous);
        self
    }

    /// Merges two runtimes. Fields from `other` take priority when they are `Some`.
    /// The stream writer from `other` is always used.
    pub fn merge(&self, other: &Runtime) -> Runtime {
        Runtime {
            context: other.context.clone().or_else(|| self.context.clone()),
            stream_writer: Arc::clone(&other.stream_writer),
            previous: other.previous.clone().or_else(|| self.previous.clone()),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runtime")
            .field("context", &self.context)
            .field("stream_writer", &"<StreamWriter>")
            .field("previous", &self.previous)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_runtime_default() {
        let rt = Runtime::new();
        assert!(rt.context.is_none());
        assert!(rt.previous.is_none());
    }

    #[test]
    fn test_runtime_builders() {
        let rt = Runtime::new()
            .with_context(json!({"user_id": "abc"}))
            .with_previous(json!(42));

        assert_eq!(rt.context, Some(json!({"user_id": "abc"})));
        assert_eq!(rt.previous, Some(json!(42)));
    }

    #[test]
    fn test_runtime_merge() {
        let base = Runtime::new();
        let other = Runtime::new()
            .with_context(json!("from_other"))
            .with_previous(json!("prev_other"));

        let merged = base.merge(&other);
        assert_eq!(merged.context, Some(json!("from_other")));
        assert_eq!(merged.previous, Some(json!("prev_other")));
    }

    #[test]
    fn test_runtime_merge_preserves_existing() {
        let base = Runtime::new().with_context(json!("from_base"));
        let other = Runtime::new().with_previous(json!("prev_other"));

        let merged = base.merge(&other);
        // other.context is None, so base's context is preserved
        assert_eq!(merged.context, Some(json!("from_base")));
        assert_eq!(merged.previous, Some(json!("prev_other")));
    }
}
