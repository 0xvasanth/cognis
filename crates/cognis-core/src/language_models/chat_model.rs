use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::messages::{AIMessage, HumanMessage, Message};
use crate::outputs::{ChatGenerationChunk, ChatResult};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;
use crate::tools::ToolSchema;

/// Configuration for structured output from a chat model.
///
/// Returned by [`BaseChatModel::with_structured_output`] to describe how the
/// model should parse its output according to a JSON schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutputModel {
    /// The JSON schema describing the expected output structure.
    pub schema: Value,
    /// The method to use: `"tool_calling"` or `"json_mode"`.
    pub method: String,
    /// Whether to include the raw AI message alongside the parsed output.
    pub include_raw: bool,
}

/// Controls whether the model should use tools.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// Model decides whether to use tools.
    Auto,
    /// Model must use at least one tool.
    Any,
    /// Model must use the named tool.
    Tool(String),
    /// Model must not use tools.
    None,
}

/// Streaming mode control for chat models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Always stream.
    Always,
    /// Never stream (fall back to invoke).
    Never,
    /// Skip streaming when tools are bound.
    SkipToolCalling,
}

/// Capability metadata for a chat model.
///
/// Mirrors the Python `langchain_core.language_models.model_profile.ModelProfile`
/// TypedDict. All fields are optional to allow partial specification.
///
/// This is a beta feature. The format of model profiles is subject to change.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelProfile {
    // --- Input constraints ---
    /// Maximum context window (tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<usize>,

    /// Whether text inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_inputs: Option<bool>,

    /// Whether image inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_inputs: Option<bool>,

    /// Whether image URL inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url_inputs: Option<bool>,

    /// Whether PDF inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_inputs: Option<bool>,

    /// Whether audio inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_inputs: Option<bool>,

    /// Whether video inputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_inputs: Option<bool>,

    /// Whether images can be included in tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tool_message: Option<bool>,

    /// Whether PDFs can be included in tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_tool_message: Option<bool>,

    // --- Output constraints ---
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,

    /// Whether the model supports reasoning / chain-of-thought output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_output: Option<bool>,

    /// Whether text outputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_outputs: Option<bool>,

    /// Whether image outputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_outputs: Option<bool>,

    /// Whether audio outputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_outputs: Option<bool>,

    /// Whether video outputs are supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_outputs: Option<bool>,

    // --- Tool calling ---
    /// Whether the model supports tool calling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calling: Option<bool>,

    /// Whether the model supports tool choice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<bool>,

    // --- Structured output ---
    /// Whether the model supports a native structured output feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<bool>,
}

/// Registry mapping model identifiers or names to their `ModelProfile`.
pub type ModelProfileRegistry = HashMap<String, ModelProfile>;

/// Type alias for a chat model stream.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatGenerationChunk>> + Send>>;

/// Trait for chat-based language models (messages in, AIMessage out).
///
/// Implementors must provide `_generate` and `llm_type`. Optionally
/// override `_stream` for streaming and `bind_tools` for tool calling.
#[async_trait]
pub trait BaseChatModel: Send + Sync {
    /// Core generation method. Implementors must override this.
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult>;

    /// The model type identifier (e.g., "openai", "anthropic").
    fn llm_type(&self) -> &str;

    /// Optional streaming support.
    async fn _stream(&self, _messages: &[Message], _stop: Option<&[String]>) -> Result<ChatStream> {
        Err(CognisError::NotImplemented(
            "Streaming not supported for this chat model".into(),
        ))
    }

    /// Bind tools to the model for function calling.
    ///
    /// Default returns an error. Providers should override this.
    fn bind_tools(
        &self,
        _tools: &[ToolSchema],
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        Err(CognisError::NotImplemented(format!(
            "{} does not support tool binding",
            self.llm_type()
        )))
    }

    /// Get model capability profile.
    fn profile(&self) -> ModelProfile {
        ModelProfile::default()
    }

    /// Get the number of tokens for the given messages.
    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        // Default rough estimate
        messages.iter().map(|m| m.content().text().len() / 4).sum()
    }

    /// Bind a JSON schema for structured output.
    ///
    /// Returns a model wrapper that will parse the output according to the schema.
    /// The default implementation uses tool calling: it creates a tool from the schema
    /// and parses the tool call result.
    ///
    /// # Arguments
    /// * `schema` - JSON Schema describing the expected output structure.
    /// * `method` - `"tool_calling"` (default) or `"json_mode"`.
    /// * `include_raw` - Whether to include the raw AI message alongside the parsed output.
    async fn with_structured_output(
        &self,
        schema: Value,
        method: Option<&str>,
        include_raw: bool,
    ) -> Result<StructuredOutputModel> {
        Ok(StructuredOutputModel {
            schema,
            method: method.unwrap_or("tool_calling").to_string(),
            include_raw,
        })
    }

    /// Generate for a batch of message sets.
    async fn generate(
        &self,
        message_batches: &[Vec<Message>],
        stop: Option<&[String]>,
    ) -> Result<Vec<ChatResult>> {
        let mut results = Vec::with_capacity(message_batches.len());
        for messages in message_batches {
            results.push(self._generate(messages, stop).await?);
        }
        Ok(results)
    }

    /// Invoke with messages, returning the first AIMessage from the result.
    async fn invoke_messages(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<AIMessage> {
        let result = self._generate(messages, stop).await?;
        let gen = result
            .generations
            .into_iter()
            .next()
            .ok_or_else(|| CognisError::Other("No generations returned".into()))?;
        match gen.message {
            Message::Ai(ai_msg) => Ok(ai_msg),
            _ => Err(CognisError::Other(
                "Expected AIMessage in ChatGeneration, got a different message type".into(),
            )),
        }
    }
}

/// Runnable wrapper for `BaseChatModel`, bridging chat models into LCEL chains.
///
/// Accepts `Value::String` (converted to a single human message) or
/// `Value::Array` (deserialized as `Vec<Message>`). Returns the AI message as JSON.
pub struct ChatModelRunnable {
    model: Arc<dyn BaseChatModel>,
    name: String,
}

impl ChatModelRunnable {
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        let name = format!("ChatModelRunnable({})", model.llm_type());
        Self { model, name }
    }
}

#[async_trait]
impl Runnable for ChatModelRunnable {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let messages = parse_chat_input(input)?;
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        serde_json::to_value(&ai_msg).map_err(|e| CognisError::Other(e.to_string()))
    }

    async fn stream(
        &self,
        input: Value,
        _config: Option<&RunnableConfig>,
    ) -> Result<crate::runnables::RunnableStream> {
        let messages = parse_chat_input(input)?;
        let chat_stream = self.model._stream(&messages, None).await?;

        use futures::StreamExt;
        let mapped = chat_stream.map(|chunk_result| {
            chunk_result.and_then(|chunk| {
                serde_json::to_value(&chunk).map_err(|e| CognisError::Other(e.to_string()))
            })
        });
        Ok(Box::pin(mapped))
    }
}

/// Parse a `Value` input into a `Vec<Message>` for chat model invocation.
fn parse_chat_input(input: Value) -> Result<Vec<Message>> {
    match input {
        Value::String(s) => Ok(vec![Message::Human(HumanMessage::new(&s))]),
        Value::Array(_) => serde_json::from_value(input)
            .map_err(|e| CognisError::Other(format!("Failed to deserialize messages: {e}"))),
        Value::Object(ref map) => {
            // If has "messages" key, use that
            if let Some(msgs) = map.get("messages") {
                serde_json::from_value(msgs.clone()).map_err(|e| {
                    CognisError::Other(format!("Failed to deserialize messages: {e}"))
                })
            } else if let Some(text) = map.get("input").and_then(|v| v.as_str()) {
                Ok(vec![Message::Human(HumanMessage::new(text))])
            } else {
                Err(CognisError::TypeMismatch {
                    expected: "String, Array, or Object with 'messages'/'input'".into(),
                    got: "Object without recognized keys".into(),
                })
            }
        }
        _ => Err(CognisError::TypeMismatch {
            expected: "String or Array of Messages".into(),
            got: format!("{}", input),
        }),
    }
}
