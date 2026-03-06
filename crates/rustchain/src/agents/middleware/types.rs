//! Core types for the agent middleware system.
//!
//! Mirrors Python `langchain.agents.middleware.types`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::Message;
use rustchain_core::tools::base::BaseTool;

/// Destination for middleware-driven control flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JumpTo {
    /// Route to tool execution.
    Tools,
    /// Route back to model invocation.
    Model,
    /// End the agent loop.
    End,
}

/// State schema for the agent.
///
/// The base state tracks messages and optional structured output.
/// Middleware can extend this with additional fields via `extra`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    /// Conversation message history.
    pub messages: Vec<Message>,
    /// Optional structured response from the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_response: Option<Value>,
    /// Control flow override — where to route next.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jump_to: Option<JumpTo>,
    /// Extra state fields added by middleware.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
}

impl AgentState {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// Get an extra state field.
    pub fn get_extra(&self, key: &str) -> Option<&Value> {
        self.extra.get(key)
    }

    /// Set an extra state field.
    pub fn set_extra(&mut self, key: impl Into<String>, value: Value) {
        self.extra.insert(key.into(), value);
    }

    /// Merge state updates into this state.
    pub fn apply_updates(&mut self, updates: HashMap<String, Value>) {
        for (key, value) in updates {
            match key.as_str() {
                "jump_to" => {
                    self.jump_to = serde_json::from_value(value).ok();
                }
                "structured_response" => {
                    self.structured_response = Some(value);
                }
                _ => {
                    self.extra.insert(key, value);
                }
            }
        }
    }
}

/// Model request information passed to middleware and the model.
pub struct ModelRequest {
    /// The chat model to invoke.
    pub model: Arc<dyn BaseChatModel>,
    /// Conversation messages (excluding system message).
    pub messages: Vec<Message>,
    /// Optional system message prepended to the conversation.
    pub system_message: Option<Message>,
    /// Tool choice configuration.
    pub tool_choice: Option<Value>,
    /// Available tools for the model.
    pub tools: Vec<Arc<dyn BaseTool>>,
    /// Optional response format specification (for structured output).
    pub response_format: Option<Value>,
    /// Current agent state.
    pub state: AgentState,
    /// Additional model settings (temperature, max_tokens, etc.).
    pub model_settings: HashMap<String, Value>,
}

impl ModelRequest {
    /// Create a new ModelRequest with the given model and messages.
    pub fn new(model: Arc<dyn BaseChatModel>, messages: Vec<Message>) -> Self {
        Self {
            model,
            messages,
            system_message: None,
            tool_choice: None,
            tools: Vec::new(),
            response_format: None,
            state: AgentState::default(),
            model_settings: HashMap::new(),
        }
    }

    /// Get the system prompt text, if set.
    pub fn system_prompt(&self) -> Option<String> {
        self.system_message
            .as_ref()
            .map(|m| m.content().text())
    }

    /// Create a new request with overridden fields.
    pub fn with_system_message(mut self, msg: Message) -> Self {
        self.system_message = Some(msg);
        self
    }

    pub fn with_tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_state(mut self, state: AgentState) -> Self {
        self.state = state;
        self
    }
}

/// Response from model execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Messages returned by the model (usually one AIMessage).
    pub result: Vec<Message>,
    /// Parsed structured output if response_format was specified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_response: Option<Value>,
}

impl ModelResponse {
    pub fn new(result: Vec<Message>) -> Self {
        Self {
            result,
            structured_response: None,
        }
    }

    pub fn with_structured_response(mut self, response: Value) -> Self {
        self.structured_response = Some(response);
        self
    }
}

/// Extended model response with an optional state update command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedModelResponse {
    /// The underlying model response.
    pub model_response: ModelResponse,
    /// Optional state updates to apply after the model node completes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_update: Option<HashMap<String, Value>>,
}

/// Return type for wrap_model_call handlers.
pub enum ModelCallResult {
    /// Full response with messages and optional structured output.
    Response(ModelResponse),
    /// Simplified return — a single AI message.
    Message(Box<Message>),
    /// Extended response with state updates.
    Extended(ExtendedModelResponse),
}

impl From<ModelResponse> for ModelCallResult {
    fn from(r: ModelResponse) -> Self {
        ModelCallResult::Response(r)
    }
}

impl From<Message> for ModelCallResult {
    fn from(m: Message) -> Self {
        ModelCallResult::Message(Box::new(m))
    }
}

impl From<ExtendedModelResponse> for ModelCallResult {
    fn from(e: ExtendedModelResponse) -> Self {
        ModelCallResult::Extended(e)
    }
}

/// Normalize a ModelCallResult into a ModelResponse.
pub fn normalize_model_call_result(result: ModelCallResult) -> ModelResponse {
    match result {
        ModelCallResult::Response(r) => r,
        ModelCallResult::Message(m) => ModelResponse::new(vec![*m]),
        ModelCallResult::Extended(e) => e.model_response,
    }
}

/// Handler function type for model call wrapping.
pub type ModelHandler =
    Box<dyn Fn(&ModelRequest) -> Result<ModelResponse> + Send + Sync>;

/// Async handler function type for model call wrapping.
pub type AsyncModelHandler = Box<
    dyn Fn(
            &ModelRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ModelResponse>> + Send + '_>,
        > + Send
        + Sync,
>;

/// Base trait for agent middleware.
///
/// Middleware can hook into various points in the agent execution loop:
/// - `before_agent` / `after_agent` — run once at start/end of agent execution
/// - `before_model` / `after_model` — run before/after each model call
/// - `wrap_model_call` — intercept model execution (for retries, fallbacks, etc.)
/// - `wrap_tool_call` — intercept tool execution
///
/// All methods have default no-op implementations.
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// The name of this middleware.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Additional tools registered by this middleware.
    fn tools(&self) -> Vec<Arc<dyn BaseTool>> {
        Vec::new()
    }

    /// Logic to run before the agent execution starts.
    async fn before_agent(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        Ok(None)
    }

    /// Logic to run after the agent execution completes.
    async fn after_agent(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        Ok(None)
    }

    /// Logic to run before each model call.
    async fn before_model(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        Ok(None)
    }

    /// Logic to run after each model call.
    async fn after_model(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        Ok(None)
    }

    /// Intercept and control model execution.
    ///
    /// The handler callback executes the model request. Middleware can:
    /// - Call handler normally for pass-through
    /// - Modify the request before calling handler
    /// - Call handler multiple times for retry logic
    /// - Skip handler to short-circuit with a cached/mock response
    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        let response = handler(request).await?;
        Ok(ModelCallResult::Response(response))
    }

    /// Intercept tool execution.
    async fn wrap_tool_call(
        &self,
        tool: &dyn BaseTool,
        input: &Value,
        handler: &(dyn for<'a, 'b> Fn(&'a dyn BaseTool, &'b Value) -> Result<Value> + Send + Sync),
    ) -> Result<Value> {
        handler(tool, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert!(state.messages.is_empty());
        assert!(state.structured_response.is_none());
        assert!(state.jump_to.is_none());
        assert!(state.extra.is_empty());
    }

    #[test]
    fn test_agent_state_new() {
        let msg = Message::human("hello");
        let state = AgentState::new(vec![msg]);
        assert_eq!(state.messages.len(), 1);
    }

    #[test]
    fn test_agent_state_extra() {
        let mut state = AgentState::default();
        state.set_extra("count", serde_json::json!(42));
        assert_eq!(state.get_extra("count"), Some(&serde_json::json!(42)));
        assert_eq!(state.get_extra("missing"), None);
    }

    #[test]
    fn test_agent_state_apply_updates() {
        let mut state = AgentState::default();
        let mut updates = HashMap::new();
        updates.insert("jump_to".into(), serde_json::json!("end"));
        updates.insert("my_field".into(), serde_json::json!("value"));
        state.apply_updates(updates);
        assert_eq!(state.jump_to, Some(JumpTo::End));
        assert_eq!(state.extra.get("my_field"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_model_response_new() {
        let msg = Message::ai("hello");
        let resp = ModelResponse::new(vec![msg]);
        assert_eq!(resp.result.len(), 1);
        assert!(resp.structured_response.is_none());
    }

    #[test]
    fn test_model_response_with_structured() {
        let resp = ModelResponse::new(vec![])
            .with_structured_response(serde_json::json!({"name": "test"}));
        assert!(resp.structured_response.is_some());
    }

    #[test]
    fn test_normalize_model_call_result_response() {
        let resp = ModelResponse::new(vec![Message::ai("hi")]);
        let result = normalize_model_call_result(ModelCallResult::Response(resp));
        assert_eq!(result.result.len(), 1);
    }

    #[test]
    fn test_normalize_model_call_result_message() {
        let result = normalize_model_call_result(ModelCallResult::Message(Box::new(Message::ai("hi"))));
        assert_eq!(result.result.len(), 1);
    }

    #[test]
    fn test_jump_to_serialize() {
        assert_eq!(
            serde_json::to_string(&JumpTo::End).unwrap(),
            "\"end\""
        );
        assert_eq!(
            serde_json::to_string(&JumpTo::Tools).unwrap(),
            "\"tools\""
        );
    }

    #[test]
    fn test_agent_state_serialization() {
        let state = AgentState::new(vec![Message::human("test")]);
        let json = serde_json::to_value(&state).unwrap();
        assert!(json.get("messages").unwrap().is_array());
        // jump_to should be skipped when None
        assert!(json.get("jump_to").is_none());
    }
}
