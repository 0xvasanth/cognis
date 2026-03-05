//! Anthropic (Claude) chat model implementation.
//!
//! Provides [`ChatAnthropic`], a production-ready implementation of
//! [`BaseChatModel`] for the Anthropic Messages API.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::{
    AIMessage, AIMessageChunk, Message, ToolCall, ToolCallChunk, UsageMetadata,
};
use rustchain_core::outputs::{ChatGeneration, ChatGenerationChunk, ChatResult};
use rustchain_core::tools::ToolSchema;

/// Builder for constructing a [`ChatAnthropic`] instance.
#[derive(Debug)]
pub struct ChatAnthropicBuilder {
    model: Option<String>,
    api_key: Option<SecretString>,
    api_url: Option<String>,
    api_version: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    max_retries: Option<u32>,
    streaming: Option<bool>,
    default_headers: HashMap<String, String>,
}

impl ChatAnthropicBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            api_url: None,
            api_version: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            max_retries: None,
            streaming: None,
            default_headers: HashMap::new(),
        }
    }

    /// Set the model name (required).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the API key. Falls back to `ANTHROPIC_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the base API URL.
    pub fn api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = Some(url.into());
        self
    }

    /// Set the API version header.
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Set the max tokens for generation.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top_p sampling parameter.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top_k sampling parameter.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set stop sequences.
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(sequences);
        self
    }

    /// Set maximum retry count.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Enable or disable streaming by default.
    pub fn streaming(mut self, streaming: bool) -> Self {
        self.streaming = Some(streaming);
        self
    }

    /// Add a default header.
    pub fn default_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }

    /// Build the [`ChatAnthropic`] instance.
    ///
    /// Returns an error if `model` is not set or if the API key cannot be
    /// resolved from the builder or environment.
    pub fn build(self) -> Result<ChatAnthropic> {
        let model = self.model.ok_or_else(|| {
            RustChainError::Other("model is required for ChatAnthropic".into())
        })?;

        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                    RustChainError::Other(
                        "api_key not provided and ANTHROPIC_API_KEY env var not set".into(),
                    )
                })?;
                SecretString::from(key)
            }
        };

        Ok(ChatAnthropic {
            model,
            api_key,
            api_url: self
                .api_url
                .unwrap_or_else(|| "https://api.anthropic.com".into()),
            api_version: self
                .api_version
                .unwrap_or_else(|| "2023-06-01".into()),
            max_tokens: self.max_tokens.unwrap_or(1024),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            max_retries: self.max_retries.unwrap_or(2),
            streaming: self.streaming.unwrap_or(false),
            client: Client::new(),
            default_headers: self.default_headers,
            bound_tools: Vec::new(),
            tool_choice: None,
        })
    }
}

impl Default for ChatAnthropicBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Anthropic (Claude) chat model.
///
/// Implements the Anthropic Messages API for chat completion, including
/// tool calling and SSE streaming.
///
/// # Example
///
/// ```no_run
/// use rustchain::chat_models::anthropic::ChatAnthropic;
///
/// let model = ChatAnthropic::builder()
///     .model("claude-sonnet-4-20250514")
///     .api_key("sk-ant-...")
///     .build()
///     .unwrap();
/// ```
pub struct ChatAnthropic {
    /// The model identifier (e.g. "claude-sonnet-4-20250514").
    pub model: String,
    /// Secret API key.
    api_key: SecretString,
    /// Base URL for the Anthropic API.
    pub api_url: String,
    /// API version header value.
    pub api_version: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Maximum number of retries on transient errors.
    pub max_retries: u32,
    /// Whether streaming is the default mode.
    pub streaming: bool,
    /// HTTP client.
    client: Client,
    /// Default headers added to every request.
    pub default_headers: HashMap<String, String>,
    /// Tools bound via `bind_tools`.
    bound_tools: Vec<Value>,
    /// Tool choice configuration.
    tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for ChatAnthropic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatAnthropic")
            .field("model", &self.model)
            .field("api_url", &self.api_url)
            .field("api_version", &self.api_version)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ChatAnthropic {
    /// Returns a new builder for `ChatAnthropic`.
    pub fn builder() -> ChatAnthropicBuilder {
        ChatAnthropicBuilder::new()
    }

    /// Format messages for the Anthropic Messages API.
    ///
    /// Returns `(system_prompt, api_messages)` where system messages are
    /// extracted into a separate string as required by the Anthropic API.
    pub fn format_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut api_messages: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                Message::System(sys) => {
                    system_parts.push(sys.base.content.text());
                }
                Message::Human(human) => {
                    api_messages.push(json!({
                        "role": "user",
                        "content": human.base.content.text()
                    }));
                }
                Message::Ai(ai) => {
                    let mut content_blocks: Vec<Value> = Vec::new();
                    let text = ai.base.content.text();
                    if !text.is_empty() {
                        content_blocks.push(json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    for tc in &ai.tool_calls {
                        content_blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.id.clone().unwrap_or_default(),
                            "name": tc.name,
                            "input": tc.args
                        }));
                    }
                    if content_blocks.is_empty() {
                        content_blocks.push(json!({
                            "type": "text",
                            "text": ""
                        }));
                    }
                    api_messages.push(json!({
                        "role": "assistant",
                        "content": content_blocks
                    }));
                }
                Message::Tool(tool) => {
                    api_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool.tool_call_id,
                            "content": tool.base.content.text()
                        }]
                    }));
                }
                _ => {
                    // Best-effort: treat as user message
                    api_messages.push(json!({
                        "role": "user",
                        "content": msg.content().text()
                    }));
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n"))
        };

        (system, api_messages)
    }

    /// Build the JSON payload for the Anthropic Messages API.
    pub fn build_payload(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        tools: &[Value],
        stream: bool,
    ) -> Value {
        let (system, api_messages) = Self::format_messages(messages);

        let mut payload = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        });

        if let Some(sys) = system {
            payload["system"] = json!(sys);
        }

        if let Some(temp) = self.temperature {
            payload["temperature"] = json!(temp);
        }

        if let Some(tp) = self.top_p {
            payload["top_p"] = json!(tp);
        }

        if let Some(tk) = self.top_k {
            payload["top_k"] = json!(tk);
        }

        // Stop sequences: merge explicit stop param with configured sequences
        let mut all_stop = Vec::new();
        if let Some(configured) = &self.stop_sequences {
            all_stop.extend(configured.iter().cloned());
        }
        if let Some(stop_param) = stop {
            all_stop.extend(stop_param.iter().cloned());
        }
        if !all_stop.is_empty() {
            payload["stop_sequences"] = json!(all_stop);
        }

        if !tools.is_empty() {
            payload["tools"] = json!(tools);
        }

        if let Some(ref choice) = self.tool_choice {
            payload["tool_choice"] = match choice {
                ToolChoice::Auto => json!({"type": "auto"}),
                ToolChoice::Any => json!({"type": "any"}),
                ToolChoice::Tool(name) => json!({"type": "tool", "name": name}),
                ToolChoice::None => json!({"type": "none"}),
            };
        }

        if stream {
            payload["stream"] = json!(true);
        }

        payload
    }

    /// Parse an Anthropic Messages API response into a [`ChatResult`].
    pub fn parse_response(response: &Value) -> Result<ChatResult> {
        let content = response
            .get("content")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RustChainError::Other("Missing 'content' array in Anthropic response".into())
            })?;

        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in content {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                Some("tool_use") => {
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let args: HashMap<String, Value> = block
                        .get("input")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall { name, args, id });
                }
                _ => {}
            }
        }

        let full_text = text_parts.join("");

        // Parse usage metadata
        let usage_metadata = response.get("usage").map(|u| {
            let input_tokens = u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = u
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            UsageMetadata::new(input_tokens, output_tokens, input_tokens + output_tokens)
        });

        let mut ai_message = AIMessage::new(&full_text);
        ai_message.tool_calls = tool_calls;
        ai_message.usage_metadata = usage_metadata;

        // Store response id
        if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
            ai_message.base.id = Some(id.to_string());
        }

        let generation = ChatGeneration::new(ai_message);

        Ok(ChatResult {
            generations: vec![generation],
            llm_output: None,
        })
    }

    /// Parse an SSE stream event into a [`ChatGenerationChunk`].
    pub fn parse_stream_event(event: &Value) -> Option<ChatGenerationChunk> {
        let event_type = event.get("type").and_then(|v| v.as_str())?;

        match event_type {
            "content_block_start" => {
                let content_block = event.get("content_block")?;
                let block_type = content_block.get("type").and_then(|v| v.as_str())?;
                let index = event.get("index").and_then(|v| v.as_u64()).map(|n| n as usize);

                if block_type == "tool_use" {
                    let name = content_block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let id = content_block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let mut chunk = AIMessageChunk::new("");
                    chunk.tool_call_chunks.push(ToolCallChunk {
                        name,
                        args: Some(String::new()),
                        id,
                        index,
                    });
                    Some(ChatGenerationChunk::new(chunk))
                } else {
                    // text block start — no content yet
                    None
                }
            }
            "content_block_delta" => {
                let delta = event.get("delta")?;
                let delta_type = delta.get("type").and_then(|v| v.as_str())?;
                let index = event.get("index").and_then(|v| v.as_u64()).map(|n| n as usize);

                match delta_type {
                    "text_delta" => {
                        let text = delta
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let chunk = AIMessageChunk::new(text);
                        Some(ChatGenerationChunk::new(chunk))
                    }
                    "input_json_delta" => {
                        let partial_json = delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let mut chunk = AIMessageChunk::new("");
                        chunk.tool_call_chunks.push(ToolCallChunk {
                            name: None,
                            args: Some(partial_json.to_string()),
                            id: None,
                            index,
                        });
                        Some(ChatGenerationChunk::new(chunk))
                    }
                    _ => None,
                }
            }
            "message_delta" => {
                let delta = event.get("delta")?;
                let usage = event.get("usage");

                let mut chunk = AIMessageChunk::new("");
                chunk.chunk_position = Some("last".to_string());

                if let Some(u) = usage {
                    let output_tokens = u
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    chunk.usage_metadata =
                        Some(UsageMetadata::new(0, output_tokens, output_tokens));
                }

                // Check for stop_reason
                if let Some(stop_reason) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                    chunk
                        .base
                        .response_metadata
                        .insert("stop_reason".to_string(), json!(stop_reason));
                }

                Some(ChatGenerationChunk::new(chunk))
            }
            "message_start" => {
                // Extract usage from the initial message
                let message = event.get("message")?;
                let usage = message.get("usage")?;
                let input_tokens = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let mut chunk = AIMessageChunk::new("");
                chunk.usage_metadata = Some(UsageMetadata::new(input_tokens, 0, input_tokens));
                Some(ChatGenerationChunk::new(chunk))
            }
            _ => None,
        }
    }

    /// Call the Anthropic Messages API with retry logic.
    async fn call_api(&self, payload: &Value) -> Result<Value> {
        let url = format!("{}/v1/messages", self.api_url);

        let mut last_error = RustChainError::Other("No attempts made".into());

        for attempt in 0..=self.max_retries {
            let mut req = self
                .client
                .post(&url)
                .header("x-api-key", self.api_key.expose_secret())
                .header("anthropic-version", &self.api_version)
                .header("content-type", "application/json");

            for (key, value) in &self.default_headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let response = req.json(payload).send().await.map_err(|e| {
                RustChainError::Other(format!("HTTP request failed: {}", e))
            })?;

            let status = response.status().as_u16();

            if status >= 200 && status < 300 {
                let body: Value = response.json().await.map_err(|e| {
                    RustChainError::Other(format!("Failed to parse response JSON: {}", e))
                })?;
                return Ok(body);
            }

            let body = response.text().await.unwrap_or_default();

            // Retry on 429 (rate limit) and 5xx (server errors)
            if (status == 429 || status >= 500) && attempt < self.max_retries {
                let delay_ms = 500 * 2u64.pow(attempt);
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                last_error = RustChainError::HttpError { status, body };
                continue;
            }

            return Err(RustChainError::HttpError { status, body });
        }

        Err(last_error)
    }

    /// Call the Anthropic Messages API with streaming, returning an async stream of SSE events.
    async fn call_api_stream(
        &self,
        payload: &Value,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value>> + Send>>> {
        let url = format!("{}/v1/messages", self.api_url);

        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json");

        for (key, value) in &self.default_headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let response = req.json(payload).send().await.map_err(|e| {
            RustChainError::Other(format!("HTTP request failed: {}", e))
        })?;

        let status = response.status().as_u16();
        if status < 200 || status >= 300 {
            let body = response.text().await.unwrap_or_default();
            return Err(RustChainError::HttpError { status, body });
        }

        let byte_stream = response.bytes_stream();

        let event_stream = byte_stream
            .map(|chunk_result| match chunk_result {
                Ok(bytes) => Ok(bytes),
                Err(e) => Err(RustChainError::Other(format!("Stream error: {}", e))),
            })
            .scan(String::new(), |buffer, chunk_result| {
                let result = match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        let mut events = Vec::new();

                        // Parse SSE events from buffer
                        while let Some(pos) = buffer.find("\n\n") {
                            let event_str = buffer[..pos].to_string();
                            *buffer = buffer[pos + 2..].to_string();

                            for line in event_str.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    let trimmed = data.trim();
                                    if trimmed == "[DONE]" {
                                        continue;
                                    }
                                    match serde_json::from_str::<Value>(trimmed) {
                                        Ok(val) => events.push(Ok(val)),
                                        Err(e) => events.push(Err(
                                            RustChainError::Other(format!(
                                                "Failed to parse SSE event: {}",
                                                e
                                            )),
                                        )),
                                    }
                                }
                            }
                        }
                        events
                    }
                    Err(e) => vec![Err(e)],
                };
                futures::future::ready(Some(futures::stream::iter(result)))
            })
            .flatten();

        Ok(Box::pin(event_stream))
    }

    /// Convert a [`ToolSchema`] to the Anthropic tool format.
    fn tool_schema_to_anthropic(schema: &ToolSchema) -> Value {
        let mut tool = json!({
            "name": schema.name,
            "description": schema.description,
        });
        if let Some(ref params) = schema.parameters {
            tool["input_schema"] = params.clone();
        } else {
            tool["input_schema"] = json!({
                "type": "object",
                "properties": {},
            });
        }
        tool
    }
}

#[async_trait]
impl BaseChatModel for ChatAnthropic {
    async fn _generate(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let payload = self.build_payload(messages, stop, &self.bound_tools, false);
        let response = self.call_api(&payload).await?;
        Self::parse_response(&response)
    }

    fn llm_type(&self) -> &str {
        "anthropic"
    }

    async fn _stream(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        let payload = self.build_payload(messages, stop, &self.bound_tools, true);
        let event_stream = self.call_api_stream(&payload).await?;

        let chunk_stream = event_stream.filter_map(|event_result| async move {
            match event_result {
                Ok(event) => ChatAnthropic::parse_stream_event(&event)
                    .map(Ok),
                Err(e) => Some(Err(e)),
            }
        });

        Ok(Box::pin(chunk_stream))
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        let bound_tools: Vec<Value> = tools
            .iter()
            .map(Self::tool_schema_to_anthropic)
            .collect();

        Ok(Box::new(ChatAnthropic {
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            api_url: self.api_url.clone(),
            api_version: self.api_version.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences.clone(),
            max_retries: self.max_retries,
            streaming: self.streaming,
            client: self.client.clone(),
            default_headers: self.default_headers.clone(),
            bound_tools,
            tool_choice,
        }))
    }

    fn profile(&self) -> ModelProfile {
        ModelProfile {
            tool_calling: Some(true),
            tool_choice: Some(true),
            structured_output: Some(true),
            text_inputs: Some(true),
            text_outputs: Some(true),
            image_inputs: Some(true),
            pdf_inputs: Some(true),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{HumanMessage, SystemMessage, ToolMessage};

    #[test]
    fn test_anthropic_config_builder() {
        let model = ChatAnthropic::builder()
            .model("claude-sonnet-4-20250514")
            .api_key("test-key")
            .api_url("https://custom.api.com")
            .api_version("2024-01-01")
            .max_tokens(2048)
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .stop_sequences(vec!["STOP".to_string()])
            .max_retries(3)
            .streaming(true)
            .default_header("X-Custom", "value")
            .build()
            .unwrap();

        assert_eq!(model.model, "claude-sonnet-4-20250514");
        assert_eq!(model.api_url, "https://custom.api.com");
        assert_eq!(model.api_version, "2024-01-01");
        assert_eq!(model.max_tokens, 2048);
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.top_p, Some(0.9));
        assert_eq!(model.top_k, Some(40));
        assert_eq!(
            model.stop_sequences,
            Some(vec!["STOP".to_string()])
        );
        assert_eq!(model.max_retries, 3);
        assert!(model.streaming);
        assert_eq!(
            model.default_headers.get("X-Custom"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_builder_requires_model() {
        let result = ChatAnthropic::builder()
            .api_key("test-key")
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("model is required"));
    }

    #[test]
    fn test_format_messages_simple() {
        let messages = vec![Message::Human(HumanMessage::new("Hello"))];
        let (system, api_messages) = ChatAnthropic::format_messages(&messages);

        assert!(system.is_none());
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "user");
        assert_eq!(api_messages[0]["content"], "Hello");
    }

    #[test]
    fn test_format_messages_with_system() {
        let messages = vec![
            Message::System(SystemMessage::new("You are helpful")),
            Message::Human(HumanMessage::new("Hi")),
        ];
        let (system, api_messages) = ChatAnthropic::format_messages(&messages);

        assert_eq!(system, Some("You are helpful".to_string()));
        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "user");
    }

    #[test]
    fn test_format_messages_with_tool_calls() {
        let mut args = HashMap::new();
        args.insert("query".to_string(), json!("rust"));
        let ai = AIMessage::new("Let me search").with_tool_calls(vec![ToolCall {
            name: "search".to_string(),
            args,
            id: Some("call_123".to_string()),
        }]);
        let messages = vec![Message::Ai(ai)];
        let (_, api_messages) = ChatAnthropic::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "assistant");
        let content = api_messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me search");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "search");
        assert_eq!(content[1]["id"], "call_123");
        assert_eq!(content[1]["input"]["query"], "rust");
    }

    #[test]
    fn test_format_messages_with_tool_result() {
        let messages = vec![Message::Tool(ToolMessage::new(
            "Search results here",
            "call_123",
        ))];
        let (_, api_messages) = ChatAnthropic::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "user");
        let content = api_messages[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_123");
        assert_eq!(content[0]["content"], "Search results here");
    }

    #[test]
    fn test_parse_response_text() {
        let response = json!({
            "id": "msg_123",
            "type": "message",
            "content": [
                {"type": "text", "text": "Hello, world!"}
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });

        let result = ChatAnthropic::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);
        assert_eq!(result.generations[0].text, "Hello, world!");

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert!(ai.tool_calls.is_empty());
            let usage = ai.usage_metadata.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
            assert_eq!(ai.base.id, Some("msg_123".to_string()));
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_response_tool_use() {
        let response = json!({
            "id": "msg_456",
            "type": "message",
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "web_search",
                    "input": {"query": "rust programming"}
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15
            }
        });

        let result = ChatAnthropic::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "web_search");
            assert_eq!(
                ai.tool_calls[0].id,
                Some("toolu_abc".to_string())
            );
            assert_eq!(
                ai.tool_calls[0].args.get("query"),
                Some(&json!("rust programming"))
            );
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_stream_event_text_delta() {
        let event = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        });

        let chunk = ChatAnthropic::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.text, "Hello");
        assert_eq!(chunk.message.base.content.text(), "Hello");
    }

    #[test]
    fn test_parse_stream_event_tool_use_start() {
        let event = json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_xyz",
                "name": "calculator"
            }
        });

        let chunk = ChatAnthropic::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.message.tool_call_chunks.len(), 1);
        assert_eq!(
            chunk.message.tool_call_chunks[0].name,
            Some("calculator".to_string())
        );
        assert_eq!(
            chunk.message.tool_call_chunks[0].id,
            Some("toolu_xyz".to_string())
        );
        assert_eq!(chunk.message.tool_call_chunks[0].index, Some(1));
    }

    #[test]
    fn test_build_payload_basic() {
        let model = ChatAnthropic::builder()
            .model("claude-sonnet-4-20250514")
            .api_key("test-key")
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[], false);

        assert_eq!(payload["model"], "claude-sonnet-4-20250514");
        assert_eq!(payload["max_tokens"], 1024);
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
        assert!(payload.get("stream").is_none());
        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn test_build_payload_with_tools() {
        let model = ChatAnthropic::builder()
            .model("claude-sonnet-4-20250514")
            .api_key("test-key")
            .build()
            .unwrap();

        let tools = vec![json!({
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        })];

        let messages = vec![Message::Human(HumanMessage::new("Search for rust"))];
        let payload = model.build_payload(&messages, None, &tools, false);

        assert!(payload.get("tools").is_some());
        let payload_tools = payload["tools"].as_array().unwrap();
        assert_eq!(payload_tools.len(), 1);
        assert_eq!(payload_tools[0]["name"], "search");
    }
}
