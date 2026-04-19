use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use super::base::CallbackHandler;
use super::events::{ToolEndEvent, ToolErrorEvent, ToolStartEvent};
use crate::agents::{AgentAction, AgentFinish};
use crate::error::Result;
use crate::outputs::LLMResult;
use crate::runnables::config::RunnableConfig;

/// Manager that dispatches callback events to multiple handlers.
///
/// Supports inheritable handlers, tags, and metadata that propagate
/// to child managers created via `get_child()`.
pub struct CallbackManager {
    handlers: Vec<Arc<dyn CallbackHandler>>,
    inheritable_handlers: Vec<Arc<dyn CallbackHandler>>,
    parent_run_id: Option<Uuid>,
    tags: Vec<String>,
    inheritable_tags: Vec<String>,
    metadata: HashMap<String, Value>,
    inheritable_metadata: HashMap<String, Value>,
}

impl CallbackManager {
    /// Create a new CallbackManager with the given handlers and optional parent run ID.
    ///
    /// All provided handlers are also set as inheritable by default.
    pub fn new(handlers: Vec<Arc<dyn CallbackHandler>>, parent_run_id: Option<Uuid>) -> Self {
        Self {
            inheritable_handlers: handlers.clone(),
            handlers,
            parent_run_id,
            tags: Vec::new(),
            inheritable_tags: Vec::new(),
            metadata: HashMap::new(),
            inheritable_metadata: HashMap::new(),
        }
    }

    /// Returns a reference to all handlers.
    pub fn handlers(&self) -> &[Arc<dyn CallbackHandler>] {
        &self.handlers
    }

    /// Returns a reference to inheritable handlers.
    pub fn inheritable_handlers(&self) -> &[Arc<dyn CallbackHandler>] {
        &self.inheritable_handlers
    }

    /// Returns the parent run ID, if set.
    pub fn parent_run_id(&self) -> Option<Uuid> {
        self.parent_run_id
    }

    /// Returns a reference to the tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Returns a reference to the inheritable tags.
    pub fn inheritable_tags(&self) -> &[String] {
        &self.inheritable_tags
    }

    /// Returns a reference to the metadata.
    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }

    /// Returns a reference to the inheritable metadata.
    pub fn inheritable_metadata(&self) -> &HashMap<String, Value> {
        &self.inheritable_metadata
    }

    /// Builder method to set the parent run ID.
    pub fn with_parent_run_id(mut self, id: Uuid) -> Self {
        self.parent_run_id = Some(id);
        self
    }

    /// Add a handler. If `inherit` is true, it will also be added to inheritable handlers.
    pub fn add_handler(&mut self, handler: Arc<dyn CallbackHandler>, inherit: bool) {
        self.handlers.push(handler.clone());
        if inherit {
            self.inheritable_handlers.push(handler);
        }
    }

    /// Remove a handler by index.
    pub fn remove_handler(&mut self, index: usize) {
        if index < self.handlers.len() {
            self.handlers.remove(index);
        }
    }

    /// Remove all handlers whose `name()` matches the given name.
    ///
    /// Removes from both the handlers and inheritable handlers lists.
    /// Returns the number of handlers removed.
    pub fn remove_handler_by_name(&mut self, name: &str) -> usize {
        let before = self.handlers.len() + self.inheritable_handlers.len();
        self.handlers.retain(|h| h.name() != name);
        self.inheritable_handlers.retain(|h| h.name() != name);
        let after = self.handlers.len() + self.inheritable_handlers.len();
        before - after
    }

    /// Populate this manager from a `RunnableConfig`.
    ///
    /// Adds all callbacks, tags, and metadata from the config.
    /// Tags and metadata are added as inheritable.
    pub fn configure(&mut self, config: &RunnableConfig) {
        for handler in &config.callbacks {
            self.add_handler(handler.clone(), true);
        }
        if !config.tags.is_empty() {
            self.add_tags(config.tags.clone(), true);
        }
        if !config.metadata.is_empty() {
            self.add_metadata(config.metadata.clone(), true);
        }
        if let Some(run_id) = config.run_id {
            self.parent_run_id = Some(run_id);
        }
    }

    /// Add tags. If `inherit` is true, they will also be added to inheritable tags.
    pub fn add_tags(&mut self, tags: Vec<String>, inherit: bool) {
        for tag in tags {
            self.tags.push(tag.clone());
            if inherit {
                self.inheritable_tags.push(tag);
            }
        }
    }

    /// Add metadata. If `inherit` is true, entries will also be added to inheritable metadata.
    pub fn add_metadata(&mut self, metadata: HashMap<String, Value>, inherit: bool) {
        for (k, v) in metadata {
            self.metadata.insert(k.clone(), v.clone());
            if inherit {
                self.inheritable_metadata.insert(k, v);
            }
        }
    }

    /// Create a child CallbackManager that inherits handlers, tags, and metadata.
    pub fn get_child(&self, parent_run_id: Uuid) -> Self {
        Self {
            handlers: self.inheritable_handlers.clone(),
            inheritable_handlers: self.inheritable_handlers.clone(),
            parent_run_id: Some(parent_run_id),
            tags: self.inheritable_tags.clone(),
            inheritable_tags: self.inheritable_tags.clone(),
            metadata: self.inheritable_metadata.clone(),
            inheritable_metadata: self.inheritable_metadata.clone(),
        }
    }

    // --- Dispatch methods ---

    pub async fn on_llm_start(
        &self,
        serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
    ) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_start(serialized, prompts, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_llm_new_token(&self, token: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_new_token(token, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_llm_end(&self, response: &LLMResult, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_end(response, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_llm_error(&self, error: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_error(error, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_chain_start(
        &self,
        serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
    ) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chain() {
                handler
                    .on_chain_start(serialized, inputs, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_chain_end(&self, outputs: &Value, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chain() {
                handler
                    .on_chain_end(outputs, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_chain_error(&self, error: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chain() {
                handler
                    .on_chain_error(error, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_tool_start(&self, event: ToolStartEvent) -> Result<()> {
        for handler in &self.handlers {
            handler.on_tool_start(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn on_tool_end(&self, event: ToolEndEvent) -> Result<()> {
        for handler in &self.handlers {
            handler.on_tool_end(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn on_tool_error(&self, event: ToolErrorEvent) -> Result<()> {
        for handler in &self.handlers {
            handler.on_tool_error(event.clone()).await?;
        }
        Ok(())
    }

    pub async fn on_agent_action(&self, action: &AgentAction, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_agent() {
                handler
                    .on_agent_action(action, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_agent_finish(&self, finish: &AgentFinish, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_agent() {
                handler
                    .on_agent_finish(finish, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_text(&self, text: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            handler.on_text(text, run_id, self.parent_run_id).await?;
        }
        Ok(())
    }

    pub async fn on_retry(&self, retry_state: &Value, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_retry() {
                handler
                    .on_retry(retry_state, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_custom_event(&self, name: &str, data: &Value, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_custom_event() {
                handler
                    .on_custom_event(name, data, run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new(vec![], None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callbacks::handlers::{LogLevel, LoggingCallbackHandler, MetricsCallbackHandler};
    use crate::outputs::{Generation, LLMResult};
    use serde_json::json;

    fn make_llm_result() -> LLMResult {
        LLMResult {
            generations: vec![vec![Generation::new("hello")]],
            llm_output: None,
            run: None,
        }
    }

    fn start_event(run_id: Uuid, input: &str) -> ToolStartEvent {
        ToolStartEvent {
            tool: "test".into(),
            serialized: json!({}),
            input_str: input.into(),
            inputs: json!({}),
            tool_call_id: None,
            run_id,
            parent_run_id: None,
            tags: vec![],
            metadata: HashMap::new(),
        }
    }

    fn end_event(run_id: Uuid, out: &str) -> ToolEndEvent {
        ToolEndEvent {
            tool: "test".into(),
            output_str: out.into(),
            output_value: Value::String(out.into()),
            artifact: None,
            tool_call_id: None,
            run_id,
            parent_run_id: None,
        }
    }

    fn error_event(run_id: Uuid, err: &str) -> ToolErrorEvent {
        ToolErrorEvent {
            tool: "test".into(),
            error: err.into(),
            error_kind: crate::callbacks::ToolErrorKind::Execution,
            tool_call_id: None,
            run_id,
            parent_run_id: None,
        }
    }

    #[tokio::test]
    async fn test_dispatch_to_multiple_handlers() {
        let logging = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
        let metrics = Arc::new(MetricsCallbackHandler::new());

        let manager = CallbackManager::new(vec![logging.clone(), metrics.clone()], None);

        let run_id = Uuid::new_v4();
        manager
            .on_llm_start(&json!({}), &["prompt1".to_string()], run_id)
            .await
            .unwrap();

        // Logging handler should have captured the event
        assert_eq!(logging.get_logs().len(), 1);
        assert!(logging.get_logs()[0].contains("llm/start"));

        // Metrics handler should have counted the call
        let m = metrics.get_metrics();
        assert_eq!(m.total_llm_calls, 1);
    }

    #[tokio::test]
    async fn test_add_and_remove_handlers() {
        let mut manager = CallbackManager::default();
        assert_eq!(manager.handlers().len(), 0);

        let logging = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
        manager.add_handler(logging.clone(), true);
        assert_eq!(manager.handlers().len(), 1);

        let removed = manager.remove_handler_by_name("LoggingCallbackHandler");
        assert_eq!(removed, 2); // removed from both handlers and inheritable_handlers
        assert_eq!(manager.handlers().len(), 0);
        assert_eq!(manager.inheritable_handlers().len(), 0);
    }

    #[tokio::test]
    async fn test_logging_handler_captures_events() {
        let handler = Arc::new(LoggingCallbackHandler::new(LogLevel::Debug));
        let manager = CallbackManager::new(vec![handler.clone()], None);
        let run_id = Uuid::new_v4();

        manager
            .on_chain_start(&json!({}), &json!({"key": "value"}), run_id)
            .await
            .unwrap();
        manager
            .on_chain_end(&json!({"result": 42}), run_id)
            .await
            .unwrap();
        manager
            .on_tool_start(start_event(run_id, "search query"))
            .await
            .unwrap();

        let logs = handler.get_logs();
        assert_eq!(logs.len(), 3);
        assert!(logs[0].contains("chain/start"));
        assert!(logs[1].contains("chain/end"));
        assert!(logs[2].contains("tool/start"));
        // Verify log level prefix
        assert!(logs[0].contains("[DEBUG]"));
    }

    #[tokio::test]
    async fn test_metrics_handler_tracks_counts() {
        let handler = Arc::new(MetricsCallbackHandler::new());
        let manager = CallbackManager::new(vec![handler.clone()], None);
        let run_id = Uuid::new_v4();

        // Two LLM calls
        manager
            .on_llm_start(&json!({}), &["p1".into()], run_id)
            .await
            .unwrap();
        manager
            .on_llm_start(&json!({}), &["p2".into()], run_id)
            .await
            .unwrap();

        // One chain call
        manager
            .on_chain_start(&json!({}), &json!({}), run_id)
            .await
            .unwrap();

        // One tool call
        manager
            .on_tool_start(start_event(run_id, "input"))
            .await
            .unwrap();

        // One error
        manager.on_llm_error("oops", run_id).await.unwrap();

        let m = handler.get_metrics();
        assert_eq!(m.total_llm_calls, 2);
        assert_eq!(m.total_chain_calls, 1);
        assert_eq!(m.total_tool_calls, 1);
        assert_eq!(m.total_errors, 1);
    }

    #[tokio::test]
    async fn test_metrics_reset() {
        let handler = Arc::new(MetricsCallbackHandler::new());
        let manager = CallbackManager::new(vec![handler.clone()], None);
        let run_id = Uuid::new_v4();

        manager
            .on_llm_start(&json!({}), &["p".into()], run_id)
            .await
            .unwrap();
        assert_eq!(handler.get_metrics().total_llm_calls, 1);

        handler.reset();
        let m = handler.get_metrics();
        assert_eq!(m.total_llm_calls, 0);
        assert_eq!(m.total_tool_calls, 0);
        assert_eq!(m.total_chain_calls, 0);
        assert_eq!(m.total_errors, 0);
        assert_eq!(m.total_tokens, 0);
    }

    #[tokio::test]
    async fn test_child_manager_inherits_handlers() {
        let metrics = Arc::new(MetricsCallbackHandler::new());
        let manager = CallbackManager::new(vec![metrics.clone()], None);

        let parent_run_id = Uuid::new_v4();
        let child = manager.get_child(parent_run_id);

        assert_eq!(child.handlers().len(), 1);
        assert_eq!(child.parent_run_id(), Some(parent_run_id));

        // Events dispatched through child should still reach the shared handler
        let run_id = Uuid::new_v4();
        child
            .on_llm_start(&json!({}), &["p".into()], run_id)
            .await
            .unwrap();
        assert_eq!(metrics.get_metrics().total_llm_calls, 1);
    }

    #[tokio::test]
    async fn test_empty_manager_is_noop() {
        let manager = CallbackManager::default();
        let run_id = Uuid::new_v4();

        // All dispatch methods should succeed without handlers
        manager
            .on_llm_start(&json!({}), &["p".into()], run_id)
            .await
            .unwrap();
        manager
            .on_llm_end(&make_llm_result(), run_id)
            .await
            .unwrap();
        manager.on_llm_error("err", run_id).await.unwrap();
        manager
            .on_chain_start(&json!({}), &json!({}), run_id)
            .await
            .unwrap();
        manager.on_chain_end(&json!({}), run_id).await.unwrap();
        manager.on_chain_error("err", run_id).await.unwrap();
        manager
            .on_tool_start(start_event(run_id, "in"))
            .await
            .unwrap();
        manager.on_tool_end(end_event(run_id, "out")).await.unwrap();
        manager
            .on_tool_error(error_event(run_id, "err"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_tags_and_metadata_propagation() {
        let mut manager = CallbackManager::default();
        manager.add_tags(vec!["tag1".into(), "tag2".into()], true);
        let mut meta = HashMap::new();
        meta.insert("key".into(), json!("value"));
        manager.add_metadata(meta, true);

        let child = manager.get_child(Uuid::new_v4());
        assert_eq!(child.tags(), &["tag1".to_string(), "tag2".to_string()]);
        assert_eq!(child.metadata().get("key"), Some(&json!("value")));

        // Grandchild should also inherit
        let grandchild = child.get_child(Uuid::new_v4());
        assert_eq!(grandchild.tags(), &["tag1".to_string(), "tag2".to_string()]);
        assert_eq!(grandchild.metadata().get("key"), Some(&json!("value")));
    }

    #[tokio::test]
    async fn test_all_event_types_dispatched() {
        let logging = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
        let manager = CallbackManager::new(vec![logging.clone()], None);
        let run_id = Uuid::new_v4();

        manager
            .on_llm_start(&json!({}), &["p".into()], run_id)
            .await
            .unwrap();
        manager
            .on_llm_end(&make_llm_result(), run_id)
            .await
            .unwrap();
        manager.on_llm_error("err", run_id).await.unwrap();
        manager
            .on_chain_start(&json!({}), &json!({}), run_id)
            .await
            .unwrap();
        manager.on_chain_end(&json!({}), run_id).await.unwrap();
        manager.on_chain_error("err", run_id).await.unwrap();
        manager
            .on_tool_start(start_event(run_id, "in"))
            .await
            .unwrap();
        manager.on_tool_end(end_event(run_id, "out")).await.unwrap();
        manager
            .on_tool_error(error_event(run_id, "err"))
            .await
            .unwrap();

        let logs = logging.get_logs();
        assert_eq!(logs.len(), 9);
        assert!(logs[0].contains("llm/start"));
        assert!(logs[1].contains("llm/end"));
        assert!(logs[2].contains("llm/error"));
        assert!(logs[3].contains("chain/start"));
        assert!(logs[4].contains("chain/end"));
        assert!(logs[5].contains("chain/error"));
        assert!(logs[6].contains("tool/start"));
        assert!(logs[7].contains("tool/end"));
        assert!(logs[8].contains("tool/error"));
    }

    #[tokio::test]
    async fn test_configure_from_runnable_config() {
        let metrics = Arc::new(MetricsCallbackHandler::new());
        let run_id = Uuid::new_v4();

        let config = RunnableConfig {
            tags: vec!["config_tag".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("source".into(), json!("test"));
                m
            },
            callbacks: vec![metrics.clone() as Arc<dyn CallbackHandler>],
            run_id: Some(run_id),
            ..RunnableConfig::default()
        };

        let mut manager = CallbackManager::default();
        manager.configure(&config);

        assert_eq!(manager.handlers().len(), 1);
        assert_eq!(manager.tags(), &["config_tag".to_string()]);
        assert_eq!(manager.metadata().get("source"), Some(&json!("test")));
        assert_eq!(manager.parent_run_id(), Some(run_id));

        // Dispatch should work
        let id = Uuid::new_v4();
        manager
            .on_llm_start(&json!({}), &["p".into()], id)
            .await
            .unwrap();
        assert_eq!(metrics.get_metrics().total_llm_calls, 1);
    }

    #[tokio::test]
    async fn test_metrics_token_estimation() {
        let handler = Arc::new(MetricsCallbackHandler::new());
        let manager = CallbackManager::new(vec![handler.clone()], None);
        let run_id = Uuid::new_v4();

        // "hello world" = 11 chars, estimated ~2 tokens (11/4 = 2)
        manager
            .on_llm_start(&json!({}), &["hello world".into()], run_id)
            .await
            .unwrap();

        let m = handler.get_metrics();
        assert!(m.total_tokens > 0, "should estimate some tokens");
        assert_eq!(m.total_tokens, 2); // 11/4 = 2
    }

    #[tokio::test]
    async fn test_remove_nonexistent_handler() {
        let mut manager = CallbackManager::default();
        let logging = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
        manager.add_handler(logging, true);

        let removed = manager.remove_handler_by_name("NonExistentHandler");
        assert_eq!(removed, 0);
        assert_eq!(manager.handlers().len(), 1);
    }
}
