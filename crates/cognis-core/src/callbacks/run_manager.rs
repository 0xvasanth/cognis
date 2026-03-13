use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use super::base::CallbackHandler;
use super::manager::CallbackManager;
use crate::agents::{AgentAction, AgentFinish};
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

/// Run manager for chain executions.
///
/// Dispatches chain-related events to handlers (respecting ignore flags)
/// and can produce a child `CallbackManager` for nested runs.
pub struct RunManagerForChain {
    run_id: Uuid,
    handlers: Vec<Arc<dyn CallbackHandler>>,
    inheritable_handlers: Vec<Arc<dyn CallbackHandler>>,
    parent_run_id: Option<Uuid>,
    tags: Vec<String>,
    inheritable_tags: Vec<String>,
    metadata: HashMap<String, Value>,
    inheritable_metadata: HashMap<String, Value>,
}

impl RunManagerForChain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: Uuid,
        handlers: Vec<Arc<dyn CallbackHandler>>,
        inheritable_handlers: Vec<Arc<dyn CallbackHandler>>,
        parent_run_id: Option<Uuid>,
        tags: Vec<String>,
        inheritable_tags: Vec<String>,
        metadata: HashMap<String, Value>,
        inheritable_metadata: HashMap<String, Value>,
    ) -> Self {
        Self {
            run_id,
            handlers,
            inheritable_handlers,
            parent_run_id,
            tags,
            inheritable_tags,
            metadata,
            inheritable_metadata,
        }
    }

    pub fn run_id(&self) -> Uuid {
        self.run_id
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }

    /// Returns a reference to inheritable tags.
    pub fn inheritable_tags(&self) -> &[String] {
        &self.inheritable_tags
    }

    /// Returns a reference to inheritable metadata.
    pub fn inheritable_metadata(&self) -> &HashMap<String, Value> {
        &self.inheritable_metadata
    }

    /// Create a child CallbackManager for nested runs.
    pub fn get_child(&self) -> CallbackManager {
        let mut child = CallbackManager::new(self.inheritable_handlers.clone(), Some(self.run_id));
        child.add_tags(self.inheritable_tags.clone(), true);
        child.add_metadata(self.inheritable_metadata.clone(), true);
        child
    }

    pub async fn on_chain_end(&self, outputs: &Value) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chain() {
                handler
                    .on_chain_end(outputs, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_chain_error(&self, error: &str) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chain() {
                handler
                    .on_chain_error(error, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_agent_action(&self, action: &AgentAction) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_agent() {
                handler
                    .on_agent_action(action, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_agent_finish(&self, finish: &AgentFinish) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_agent() {
                handler
                    .on_agent_finish(finish, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_text(&self, text: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_text(text, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }
}

/// Run manager for LLM executions.
///
/// Dispatches LLM-related events to handlers that don't have `ignore_llm` set.
pub struct RunManagerForLlm {
    run_id: Uuid,
    handlers: Vec<Arc<dyn CallbackHandler>>,
    parent_run_id: Option<Uuid>,
}

impl RunManagerForLlm {
    pub fn new(
        run_id: Uuid,
        handlers: Vec<Arc<dyn CallbackHandler>>,
        parent_run_id: Option<Uuid>,
    ) -> Self {
        Self {
            run_id,
            handlers,
            parent_run_id,
        }
    }

    pub fn run_id(&self) -> Uuid {
        self.run_id
    }

    pub async fn on_llm_new_token(&self, token: &str) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_new_token(token, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_llm_end(&self, response: &LLMResult) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_end(response, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_llm_error(&self, error: &str) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_llm() {
                handler
                    .on_llm_error(error, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_chat_model_start(
        &self,
        serialized: &Value,
        messages: &[Vec<Message>],
    ) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_chat_model() {
                handler
                    .on_chat_model_start(serialized, messages, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_text(&self, text: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_text(text, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }
}

/// Run manager for tool executions.
///
/// Dispatches tool-related events and can produce a child `CallbackManager`.
pub struct RunManagerForTool {
    run_id: Uuid,
    handlers: Vec<Arc<dyn CallbackHandler>>,
    inheritable_handlers: Vec<Arc<dyn CallbackHandler>>,
    parent_run_id: Option<Uuid>,
}

impl RunManagerForTool {
    pub fn new(
        run_id: Uuid,
        handlers: Vec<Arc<dyn CallbackHandler>>,
        inheritable_handlers: Vec<Arc<dyn CallbackHandler>>,
        parent_run_id: Option<Uuid>,
    ) -> Self {
        Self {
            run_id,
            handlers,
            inheritable_handlers,
            parent_run_id,
        }
    }

    pub fn run_id(&self) -> Uuid {
        self.run_id
    }

    /// Create a child CallbackManager for nested runs.
    pub fn get_child(&self) -> CallbackManager {
        CallbackManager::new(self.inheritable_handlers.clone(), Some(self.run_id))
    }

    pub async fn on_tool_end(&self, output: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_tool_end(output, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }

    pub async fn on_tool_error(&self, error: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_tool_error(error, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }

    pub async fn on_text(&self, text: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_text(text, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }
}

/// Run manager for retriever executions.
///
/// Dispatches retriever-related events to handlers that don't have `ignore_retriever` set.
pub struct RunManagerForRetriever {
    run_id: Uuid,
    handlers: Vec<Arc<dyn CallbackHandler>>,
    parent_run_id: Option<Uuid>,
}

impl RunManagerForRetriever {
    pub fn new(
        run_id: Uuid,
        handlers: Vec<Arc<dyn CallbackHandler>>,
        parent_run_id: Option<Uuid>,
    ) -> Self {
        Self {
            run_id,
            handlers,
            parent_run_id,
        }
    }

    pub fn run_id(&self) -> Uuid {
        self.run_id
    }

    pub async fn on_retriever_end(&self, documents: &[Document]) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_retriever() {
                handler
                    .on_retriever_end(documents, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_retriever_error(&self, error: &str) -> Result<()> {
        for handler in &self.handlers {
            if !handler.ignore_retriever() {
                handler
                    .on_retriever_error(error, self.run_id, self.parent_run_id)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn on_text(&self, text: &str) -> Result<()> {
        for handler in &self.handlers {
            handler
                .on_text(text, self.run_id, self.parent_run_id)
                .await?;
        }
        Ok(())
    }
}
