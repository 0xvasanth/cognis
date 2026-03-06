use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::chat_history::BaseChatMessageHistory;
use crate::error::{Result, RustChainError};
use crate::messages::Message;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;
use crate::runnables::RunnableStream;

/// Specification for a configurable field used to resolve session identity.
#[derive(Debug, Clone)]
pub struct ConfigurableFieldSpec {
    /// Unique identifier for this field (e.g. `"session_id"`).
    pub id: String,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Optional description of the field's purpose.
    pub description: Option<String>,
    /// Default value if none is provided in the config.
    pub default: Option<String>,
    /// Whether this field is shared across runnables in a sequence.
    pub is_shared: bool,
}

impl ConfigurableFieldSpec {
    /// Create a new spec with only an id; all other fields default.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            description: None,
            default: None,
            is_shared: false,
        }
    }
}

/// Type alias for the session history factory function.
type SessionHistoryFactory = Box<dyn Fn(&str) -> Arc<dyn BaseChatMessageHistory> + Send + Sync>;

/// Wraps a runnable to automatically manage message history.
///
/// Before invoking the inner runnable, loads history from a session store
/// and injects it into the input. After invocation, saves the new input
/// and output messages to the store.
///
/// # Session resolution
///
/// The session ID is read from `RunnableConfig::configurable` using the key
/// specified in `history_factory_config` (defaults to `"session_id"`).
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use rustchain_core::runnables::history::RunnableWithMessageHistory;
/// use rustchain_core::chat_history::InMemoryChatMessageHistory;
///
/// let wrapped = RunnableWithMessageHistory::new(
///     my_runnable,
///     |_session_id| Arc::new(InMemoryChatMessageHistory::new()),
/// );
/// ```
pub struct RunnableWithMessageHistory {
    /// The inner runnable to wrap.
    runnable: Arc<dyn Runnable>,
    /// Factory that creates a history store for a given session ID.
    get_session_history: SessionHistoryFactory,
    /// Key in the input dict that contains the user's new message(s).
    input_messages_key: Option<String>,
    /// Key in the output dict that contains the response message(s).
    output_messages_key: Option<String>,
    /// Key in the input dict where loaded history should be placed.
    history_messages_key: Option<String>,
    /// Configurable field specs used to resolve the session ID.
    history_factory_config: Vec<ConfigurableFieldSpec>,
}

impl RunnableWithMessageHistory {
    /// Create a new `RunnableWithMessageHistory`.
    ///
    /// The `get_session_history` closure receives a session ID string and must
    /// return a `BaseChatMessageHistory` implementation for that session.
    pub fn new(
        runnable: Arc<dyn Runnable>,
        get_session_history: impl Fn(&str) -> Arc<dyn BaseChatMessageHistory> + Send + Sync + 'static,
    ) -> Self {
        Self {
            runnable,
            get_session_history: Box::new(get_session_history),
            input_messages_key: None,
            output_messages_key: None,
            history_messages_key: None,
            history_factory_config: vec![ConfigurableFieldSpec::new("session_id")],
        }
    }

    /// Set the key in the input dict that holds the user's new message(s).
    pub fn with_input_messages_key(mut self, key: impl Into<String>) -> Self {
        self.input_messages_key = Some(key.into());
        self
    }

    /// Set the key in the output dict that holds the response message(s).
    pub fn with_output_messages_key(mut self, key: impl Into<String>) -> Self {
        self.output_messages_key = Some(key.into());
        self
    }

    /// Set the key in the input dict where loaded history will be injected.
    pub fn with_history_messages_key(mut self, key: impl Into<String>) -> Self {
        self.history_messages_key = Some(key.into());
        self
    }

    /// Override the configurable field specs used to resolve the session ID.
    pub fn with_history_factory_config(mut self, config: Vec<ConfigurableFieldSpec>) -> Self {
        self.history_factory_config = config;
        self
    }

    /// Extract the session ID from a `RunnableConfig`.
    fn get_session_id(&self, config: Option<&RunnableConfig>) -> Result<String> {
        let config = config.ok_or_else(|| {
            RustChainError::Other(
                "RunnableConfig is required with session_id in configurable".into(),
            )
        })?;

        for spec in &self.history_factory_config {
            if let Some(val) = config.configurable.get(&spec.id) {
                if let Some(s) = val.as_str() {
                    return Ok(s.to_string());
                }
            }
            // Fall back to the spec's default value if present.
            if let Some(ref default) = spec.default {
                return Ok(default.clone());
            }
        }

        Err(RustChainError::Other(
            "session_id not found in RunnableConfig configurable".into(),
        ))
    }
}

#[async_trait]
impl Runnable for RunnableWithMessageHistory {
    fn name(&self) -> &str {
        "RunnableWithMessageHistory"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let session_id = self.get_session_id(config)?;
        let history = (self.get_session_history)(&session_id);

        // Load existing messages from the history store.
        let existing_messages = history.messages().await?;

        // Build an enriched input that includes the loaded history.
        let mut enriched_input = input.clone();
        if let Value::Object(ref mut map) = enriched_input {
            let history_key = self.history_messages_key.as_deref().unwrap_or("history");
            let history_value = serde_json::to_value(&existing_messages)?;
            map.insert(history_key.to_string(), history_value);
        }

        // Invoke the inner runnable with the enriched input.
        let output = self.runnable.invoke(enriched_input, config).await?;

        // Persist new input messages to history.
        if let Value::Object(ref input_map) = input {
            let input_key = self.input_messages_key.as_deref().unwrap_or("input");
            if let Some(input_val) = input_map.get(input_key) {
                if let Ok(msgs) = serde_json::from_value::<Vec<Message>>(input_val.clone()) {
                    if !msgs.is_empty() {
                        history.add_messages(msgs).await?;
                    }
                }
            }
        }

        // Persist new output messages to history.
        if let Value::Object(ref output_map) = output {
            let output_key = self.output_messages_key.as_deref().unwrap_or("output");
            if let Some(output_val) = output_map.get(output_key) {
                if let Ok(msgs) = serde_json::from_value::<Vec<Message>>(output_val.clone()) {
                    if !msgs.is_empty() {
                        history.add_messages(msgs).await?;
                    }
                }
            }
        }

        Ok(output)
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        // Delegate to the default stream implementation (single-value from invoke).
        let result = self.invoke(input, config).await;
        Ok(Box::pin(futures::stream::once(async { result })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_history::InMemoryChatMessageHistory;
    use serde_json::json;
    use std::collections::HashMap;

    /// A trivial runnable that echoes its input for testing purposes.
    struct EchoRunnable;

    #[async_trait]
    impl Runnable for EchoRunnable {
        fn name(&self) -> &str {
            "EchoRunnable"
        }

        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn test_session_id_required() {
        let runnable = RunnableWithMessageHistory::new(Arc::new(EchoRunnable), |_| {
            Arc::new(InMemoryChatMessageHistory::new())
        });

        let result = runnable.invoke(json!({"input": "hello"}), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invoke_with_session() {
        let runnable = RunnableWithMessageHistory::new(Arc::new(EchoRunnable), |_| {
            Arc::new(InMemoryChatMessageHistory::new())
        });

        let mut configurable = HashMap::new();
        configurable.insert(
            "session_id".to_string(),
            Value::String("test-session".to_string()),
        );

        let config = RunnableConfig {
            configurable,
            ..RunnableConfig::default()
        };

        let result = runnable
            .invoke(json!({"input": "hello"}), Some(&config))
            .await;
        assert!(result.is_ok());
    }
}
