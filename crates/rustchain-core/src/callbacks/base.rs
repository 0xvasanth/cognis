use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::agents::{AgentAction, AgentFinish};
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

/// Callback handler for LLM events.
///
/// All methods have default no-op implementations.
/// Filter flag methods control which categories of events a handler receives.
#[async_trait]
pub trait CallbackHandler: Send + Sync {
    // --- Filter flags ---

    /// If true, this handler will not receive LLM events.
    fn ignore_llm(&self) -> bool {
        false
    }

    /// If true, this handler will not receive chain events.
    fn ignore_chain(&self) -> bool {
        false
    }

    /// If true, this handler will not receive agent events.
    fn ignore_agent(&self) -> bool {
        false
    }

    /// If true, this handler will not receive retriever events.
    fn ignore_retriever(&self) -> bool {
        false
    }

    /// If true, this handler will not receive chat model events.
    fn ignore_chat_model(&self) -> bool {
        false
    }

    /// If true, this handler will not receive retry events.
    fn ignore_retry(&self) -> bool {
        false
    }

    /// If true, this handler will not receive custom events.
    fn ignore_custom_event(&self) -> bool {
        false
    }

    /// If true, errors in this handler will be raised instead of logged.
    fn raise_error(&self) -> bool {
        false
    }

    // --- Event methods ---

    async fn on_llm_start(
        &self,
        _serialized: &Value,
        _prompts: &[String],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_llm_new_token(
        &self,
        _token: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_llm_end(
        &self,
        _response: &LLMResult,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_llm_error(
        &self,
        _error: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_chain_start(
        &self,
        _serialized: &Value,
        _inputs: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_chain_end(
        &self,
        _outputs: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_chain_error(
        &self,
        _error: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_tool_start(
        &self,
        _serialized: &Value,
        _input_str: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_tool_end(
        &self,
        _output: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_tool_error(
        &self,
        _error: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_retriever_start(
        &self,
        _serialized: &Value,
        _query: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        _documents: &[Document],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_retriever_error(
        &self,
        _error: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_chat_model_start(
        &self,
        _serialized: &Value,
        _messages: &[Vec<Message>],
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_agent_action(
        &self,
        _action: &AgentAction,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_agent_finish(
        &self,
        _finish: &AgentFinish,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_text(
        &self,
        _text: &str,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_retry(
        &self,
        _retry_state: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_custom_event(
        &self,
        _name: &str,
        _data: &Value,
        _run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        Ok(())
    }
}
