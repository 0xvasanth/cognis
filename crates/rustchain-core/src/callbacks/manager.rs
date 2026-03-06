use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use super::base::CallbackHandler;
use crate::agents::{AgentAction, AgentFinish};
use crate::error::Result;
use crate::outputs::LLMResult;

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

    pub async fn on_tool_start(
        &self,
        serialized: &Value,
        input_str: &str,
        run_id: Uuid,
    ) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_tool_start(serialized, input_str, run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }

    pub async fn on_tool_end(&self, output: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_tool_end(output, run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }

    pub async fn on_tool_error(&self, error: &str, run_id: Uuid) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_tool_error(error, run_id, self.parent_run_id)
                .await?;
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

    pub async fn on_agent_finish(
        &self,
        finish: &AgentFinish,
        run_id: Uuid,
    ) -> Result<()> {
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
            handler
                .on_text(text, run_id, self.parent_run_id)
                .await?;
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

    pub async fn on_custom_event(
        &self,
        name: &str,
        data: &Value,
        run_id: Uuid,
    ) -> Result<()> {
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
