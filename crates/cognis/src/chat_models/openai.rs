//! OpenAI chat model implementation.
//!
//! Provides [`ChatOpenAI`], a production-ready implementation of
//! [`BaseChatModel`] for the OpenAI Chat Completions API.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::{
    AIMessage, AIMessageChunk, Message, ToolCall, ToolCallChunk, UsageMetadata,
};
use cognis_core::outputs::{ChatGeneration, ChatGenerationChunk, ChatResult};
use cognis_core::tools::ToolSchema;

/// Builder for constructing a [`ChatOpenAI`] instance.
#[derive(Debug)]
pub struct ChatOpenAIBuilder {
    model: Option<String>,
    api_key: Option<SecretString>,
    base_url: Option<String>,
    organization: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
    max_retries: Option<u32>,
    streaming: Option<bool>,
    extra_headers: HashMap<String, String>,
}

impl ChatOpenAIBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            organization: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            max_retries: None,
            streaming: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Set the model name (required, e.g. "gpt-4o").
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the API key. Falls back to `OPENAI_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the base API URL (default: `https://api.openai.com`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the OpenAI organization ID.
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
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

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Set the random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
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

    /// Add a custom HTTP header that will be sent on every request.
    ///
    /// Useful for OpenAI-compatible gateways that require attribution or
    /// routing headers — for example OpenRouter's `HTTP-Referer` and
    /// `X-Title`, or Together.ai's custom routing tags. Calling this method
    /// multiple times with the same key overwrites the prior value.
    ///
    /// The standard `Authorization`, `Content-Type`, and `OpenAI-Organization`
    /// headers are applied after extra headers, so they cannot be
    /// accidentally overridden.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cognis::chat_models::openai::ChatOpenAI;
    ///
    /// let model = ChatOpenAI::builder()
    ///     .model("meta-llama/llama-3.3-70b-instruct:free")
    ///     .api_key("sk-or-...")
    ///     .base_url("https://openrouter.ai/api")
    ///     .extra_header("HTTP-Referer", "https://mysite.com")
    ///     .extra_header("X-Title", "my-app")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn extra_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(key.into(), value.into());
        self
    }

    /// Replace the entire set of extra HTTP headers.
    ///
    /// Prefer [`extra_header`](Self::extra_header) for additive use. This
    /// method is intended for cases where the caller already has a header
    /// map (e.g. loaded from config).
    pub fn extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Build the [`ChatOpenAI`] instance.
    ///
    /// Returns an error if `model` is not set or if the API key cannot be
    /// resolved from the builder or environment.
    pub fn build(self) -> Result<ChatOpenAI> {
        let model = self
            .model
            .ok_or_else(|| CognisError::Other("model is required for ChatOpenAI".into()))?;

        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    CognisError::Other(
                        "api_key not provided and OPENAI_API_KEY env var not set".into(),
                    )
                })?;
                SecretString::from(key)
            }
        };

        Ok(ChatOpenAI {
            model,
            api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| "https://api.openai.com".into()),
            organization: self.organization,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stop: self.stop,
            max_retries: self.max_retries.unwrap_or(2),
            streaming: self.streaming.unwrap_or(false),
            extra_headers: self.extra_headers,
            client: Client::new(),
            bound_tools: Vec::new(),
            tool_choice: None,
        })
    }
}

impl Default for ChatOpenAIBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenAI chat model.
///
/// Implements the OpenAI Chat Completions API for chat completion, including
/// tool calling and SSE streaming.
///
/// # Example
///
/// ```no_run
/// use cognis::chat_models::openai::ChatOpenAI;
///
/// let model = ChatOpenAI::builder()
///     .model("gpt-4o")
///     .api_key("sk-...")
///     .build()
///     .unwrap();
/// ```
pub struct ChatOpenAI {
    /// The model identifier (e.g. "gpt-4o").
    pub model: String,
    /// Secret API key.
    api_key: SecretString,
    /// Base URL for the OpenAI API.
    pub base_url: String,
    /// Optional organization ID.
    pub organization: Option<String>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    pub presence_penalty: Option<f64>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    /// Maximum number of retries on transient errors.
    pub max_retries: u32,
    /// Whether streaming is the default mode.
    pub streaming: bool,
    /// Custom HTTP headers applied to every request, used for attribution or
    /// routing on OpenAI-compatible gateways (OpenRouter, Together.ai, Groq,
    /// Fireworks, etc.). Applied before the standard auth/content-type
    /// headers so those cannot be overridden.
    pub extra_headers: HashMap<String, String>,
    /// HTTP client.
    client: Client,
    /// Tools bound via `bind_tools`.
    bound_tools: Vec<Value>,
    /// Tool choice configuration.
    tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for ChatOpenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatOpenAI")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("organization", &self.organization)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ChatOpenAI {
    /// Returns a new builder for `ChatOpenAI`.
    pub fn builder() -> ChatOpenAIBuilder {
        ChatOpenAIBuilder::new()
    }

    /// Format messages for the OpenAI Chat Completions API.
    ///
    /// OpenAI keeps system messages inline (no separate extraction needed).
    pub fn format_messages(messages: &[Message]) -> Vec<Value> {
        let mut api_messages: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                Message::System(sys) => {
                    api_messages.push(json!({
                        "role": "system",
                        "content": sys.base.content.text()
                    }));
                }
                Message::Human(human) => {
                    api_messages.push(json!({
                        "role": "user",
                        "content": human.base.content.text()
                    }));
                }
                Message::Ai(ai) => {
                    let text = ai.base.content.text();
                    let mut msg_obj = json!({
                        "role": "assistant",
                    });

                    if !text.is_empty() {
                        msg_obj["content"] = json!(text);
                    } else if ai.tool_calls.is_empty() {
                        // Only set content to empty string if no tool_calls
                        msg_obj["content"] = json!("");
                    }

                    if !ai.tool_calls.is_empty() {
                        let tool_calls: Vec<Value> = ai
                            .tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id.clone().unwrap_or_default(),
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": serde_json::to_string(&tc.args)
                                            .unwrap_or_else(|_| "{}".to_string())
                                    }
                                })
                            })
                            .collect();
                        msg_obj["tool_calls"] = json!(tool_calls);
                    }

                    api_messages.push(msg_obj);
                }
                Message::Tool(tool) => {
                    api_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool.tool_call_id,
                        "content": tool.base.content.text()
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

        api_messages
    }

    /// Build the JSON payload for the OpenAI Chat Completions API.
    pub fn build_payload(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        tools: &[Value],
        stream: bool,
    ) -> Value {
        let api_messages = Self::format_messages(messages);

        let mut payload = json!({
            "model": self.model,
            "messages": api_messages,
        });

        if let Some(max_tokens) = self.max_tokens {
            payload["max_tokens"] = json!(max_tokens);
        }

        if let Some(temp) = self.temperature {
            payload["temperature"] = json!(temp);
        }

        if let Some(tp) = self.top_p {
            payload["top_p"] = json!(tp);
        }

        if let Some(fp) = self.frequency_penalty {
            payload["frequency_penalty"] = json!(fp);
        }

        if let Some(pp) = self.presence_penalty {
            payload["presence_penalty"] = json!(pp);
        }

        if let Some(seed) = self.seed {
            payload["seed"] = json!(seed);
        }

        // Stop sequences: merge explicit stop param with configured sequences
        let mut all_stop = Vec::new();
        if let Some(configured) = &self.stop {
            all_stop.extend(configured.iter().cloned());
        }
        if let Some(stop_param) = stop {
            all_stop.extend(stop_param.iter().cloned());
        }
        if !all_stop.is_empty() {
            payload["stop"] = json!(all_stop);
        }

        if !tools.is_empty() {
            payload["tools"] = json!(tools);
        }

        if let Some(ref choice) = self.tool_choice {
            payload["tool_choice"] = match choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::Any => json!("required"),
                ToolChoice::Tool(name) => json!({"type": "function", "function": {"name": name}}),
                ToolChoice::None => json!("none"),
            };
        }

        if stream {
            payload["stream"] = json!(true);
        }

        payload
    }

    /// Parse an OpenAI Chat Completions API response into a [`ChatResult`].
    pub fn parse_response(response: &Value) -> Result<ChatResult> {
        let choices = response
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CognisError::Other("Missing 'choices' array in OpenAI response".into())
            })?;

        let choice = choices
            .first()
            .ok_or_else(|| CognisError::Other("Empty 'choices' array in OpenAI response".into()))?;

        let message = choice
            .get("message")
            .ok_or_else(|| CognisError::Other("Missing 'message' in choice".into()))?;

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut tool_calls: Vec<ToolCall> = Vec::new();
        if let Some(tcs) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let function = tc.get("function").unwrap_or(&Value::Null);
                let name = function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                // arguments is a JSON string in OpenAI responses
                let args_str = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let args: HashMap<String, Value> =
                    serde_json::from_str(args_str).unwrap_or_default();
                tool_calls.push(ToolCall { name, args, id });
            }
        }

        // Parse usage metadata
        let usage_metadata = response.get("usage").map(|u| {
            let prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let completion_tokens = u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total_tokens = u
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(prompt_tokens + completion_tokens);
            UsageMetadata::new(prompt_tokens, completion_tokens, total_tokens)
        });

        let mut ai_message = AIMessage::new(&content);
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
        let choices = event.get("choices").and_then(|v| v.as_array())?;
        let choice = choices.first()?;

        let delta = choice.get("delta")?;
        let finish_reason = choice.get("finish_reason");

        let content = delta.get("content").and_then(|v| v.as_str()).unwrap_or("");

        let mut chunk = AIMessageChunk::new(content);

        // Parse tool call deltas
        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let index = tc.get("index").and_then(|v| v.as_u64()).map(|n| n as usize);
                let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                let function = tc.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let args = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                chunk.tool_call_chunks.push(ToolCallChunk {
                    name,
                    args,
                    id,
                    index,
                });
            }
        }

        // Mark last chunk based on finish_reason
        if let Some(fr) = finish_reason {
            if !fr.is_null() {
                chunk.chunk_position = Some("last".to_string());
                if let Some(reason) = fr.as_str() {
                    chunk
                        .base
                        .response_metadata
                        .insert("finish_reason".to_string(), json!(reason));
                }
            }
        }

        // Parse usage from final chunk (OpenAI includes it when stream_options.include_usage is set)
        if let Some(usage) = event.get("usage") {
            if !usage.is_null() {
                let prompt_tokens = usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let completion_tokens = usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let total_tokens = usage
                    .get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(prompt_tokens + completion_tokens);
                chunk.usage_metadata = Some(UsageMetadata::new(
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ));
            }
        }

        // Skip empty deltas with no content, no tool calls, and no finish reason
        if content.is_empty()
            && chunk.tool_call_chunks.is_empty()
            && chunk.chunk_position.is_none()
            && chunk.usage_metadata.is_none()
        {
            return None;
        }

        Some(ChatGenerationChunk::new(chunk))
    }

    /// Call the OpenAI Chat Completions API with retry logic.
    async fn call_api(&self, payload: &Value) -> Result<Value> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut last_error = CognisError::Other("No attempts made".into());

        for attempt in 0..=self.max_retries {
            let mut req = self.client.post(&url);

            for (k, v) in &self.extra_headers {
                req = req.header(k.as_str(), v.as_str());
            }

            req = req
                .header(
                    "Authorization",
                    format!("Bearer {}", self.api_key.expose_secret()),
                )
                .header("Content-Type", "application/json");

            if let Some(ref org) = self.organization {
                req = req.header("OpenAI-Organization", org.as_str());
            }

            let response = req
                .json(payload)
                .send()
                .await
                .map_err(|e| CognisError::Other(format!("HTTP request failed: {}", e)))?;

            let status = response.status().as_u16();

            if (200..300).contains(&status) {
                let body: Value = response.json().await.map_err(|e| {
                    CognisError::Other(format!("Failed to parse response JSON: {}", e))
                })?;
                return Ok(body);
            }

            let body = response.text().await.unwrap_or_default();

            // Retry on 429 (rate limit) and 5xx (server errors)
            if (status == 429 || status >= 500) && attempt < self.max_retries {
                let delay_ms = 500 * 2u64.pow(attempt);
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                last_error = CognisError::HttpError { status, body };
                continue;
            }

            return Err(CognisError::HttpError { status, body });
        }

        Err(last_error)
    }

    /// Call the OpenAI Chat Completions API with streaming, returning an async stream of SSE events.
    async fn call_api_stream(
        &self,
        payload: &Value,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value>> + Send>>> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = self.client.post(&url);

        for (k, v) in &self.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        req = req
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json");

        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org.as_str());
        }

        let response = req
            .json(payload)
            .send()
            .await
            .map_err(|e| CognisError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(CognisError::HttpError { status, body });
        }

        let byte_stream = response.bytes_stream();

        let event_stream = byte_stream
            .map(|chunk_result| match chunk_result {
                Ok(bytes) => Ok(bytes),
                Err(e) => Err(CognisError::Other(format!("Stream error: {}", e))),
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
                                        Err(e) => events.push(Err(CognisError::Other(format!(
                                            "Failed to parse SSE event: {}",
                                            e
                                        )))),
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

    /// Convert a [`ToolSchema`] to the OpenAI tool format.
    fn tool_schema_to_openai(schema: &ToolSchema) -> Value {
        let mut function = json!({
            "name": schema.name,
            "description": schema.description,
        });
        if let Some(ref params) = schema.parameters {
            function["parameters"] = params.clone();
        } else {
            function["parameters"] = json!({
                "type": "object",
                "properties": {},
            });
        }
        json!({
            "type": "function",
            "function": function
        })
    }
}

#[async_trait]
impl BaseChatModel for ChatOpenAI {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        let payload = self.build_payload(messages, stop, &self.bound_tools, false);
        let response = self.call_api(&payload).await?;
        Self::parse_response(&response)
    }

    fn llm_type(&self) -> &str {
        "openai"
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        let payload = self.build_payload(messages, stop, &self.bound_tools, true);
        let event_stream = self.call_api_stream(&payload).await?;

        let chunk_stream = event_stream.filter_map(|event_result| async move {
            match event_result {
                Ok(event) => ChatOpenAI::parse_stream_event(&event).map(Ok),
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
        let bound_tools: Vec<Value> = tools.iter().map(Self::tool_schema_to_openai).collect();

        Ok(Box::new(ChatOpenAI {
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stop: self.stop.clone(),
            max_retries: self.max_retries,
            streaming: self.streaming,
            extra_headers: self.extra_headers.clone(),
            client: self.client.clone(),
            bound_tools,
            tool_choice,
        }))
    }

    fn profile(&self) -> ModelProfile {
        ModelProfile {
            tool_calling: Some(true),
            structured_output: Some(true),
            text_inputs: Some(true),
            text_outputs: Some(true),
            image_inputs: Some(true),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::{HumanMessage, SystemMessage, ToolMessage};

    #[test]
    fn test_openai_config_builder() {
        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .base_url("https://custom.api.com")
            .organization("org-123")
            .max_tokens(2048)
            .temperature(0.7)
            .top_p(0.9)
            .frequency_penalty(0.5)
            .presence_penalty(0.3)
            .seed(42)
            .stop(vec!["STOP".to_string()])
            .max_retries(3)
            .streaming(true)
            .build()
            .unwrap();

        assert_eq!(model.model, "gpt-4o");
        assert_eq!(model.base_url, "https://custom.api.com");
        assert_eq!(model.organization, Some("org-123".to_string()));
        assert_eq!(model.max_tokens, Some(2048));
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.top_p, Some(0.9));
        assert_eq!(model.frequency_penalty, Some(0.5));
        assert_eq!(model.presence_penalty, Some(0.3));
        assert_eq!(model.seed, Some(42));
        assert_eq!(model.stop, Some(vec!["STOP".to_string()]));
        assert_eq!(model.max_retries, 3);
        assert!(model.streaming);
    }

    #[test]
    fn test_builder_requires_model() {
        let result = ChatOpenAI::builder().api_key("test-key").build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("model is required"));
    }

    #[test]
    fn test_format_messages() {
        let messages = vec![
            Message::System(SystemMessage::new("You are helpful")),
            Message::Human(HumanMessage::new("Hello")),
        ];
        let api_messages = ChatOpenAI::format_messages(&messages);

        assert_eq!(api_messages.len(), 2);
        assert_eq!(api_messages[0]["role"], "system");
        assert_eq!(api_messages[0]["content"], "You are helpful");
        assert_eq!(api_messages[1]["role"], "user");
        assert_eq!(api_messages[1]["content"], "Hello");
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
        let api_messages = ChatOpenAI::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "assistant");
        assert_eq!(api_messages[0]["content"], "Let me search");

        let tool_calls = api_messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_123");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "search");
        // arguments must be a JSON string, not an object
        let args_str = tool_calls[0]["function"]["arguments"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["query"], "rust");
    }

    #[test]
    fn test_format_messages_with_tool_result() {
        let messages = vec![Message::Tool(ToolMessage::new(
            "Search results here",
            "call_123",
        ))];
        let api_messages = ChatOpenAI::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "tool");
        assert_eq!(api_messages[0]["tool_call_id"], "call_123");
        assert_eq!(api_messages[0]["content"], "Search results here");
    }

    #[test]
    fn test_parse_response_text() {
        let response = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let result = ChatOpenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);
        assert_eq!(result.generations[0].text, "Hello, world!");

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert!(ai.tool_calls.is_empty());
            let usage = ai.usage_metadata.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
            assert_eq!(ai.base.id, Some("chatcmpl-123".to_string()));
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_response_tool_calls() {
        let response = json!({
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": "{\"query\":\"rust programming\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        });

        let result = ChatOpenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "web_search");
            assert_eq!(ai.tool_calls[0].id, Some("call_abc".to_string()));
            assert_eq!(
                ai.tool_calls[0].args.get("query"),
                Some(&json!("rust programming"))
            );
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_stream_event_content() {
        let event = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        });

        let chunk = ChatOpenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.text, "Hello");
        assert_eq!(chunk.message.base.content.text(), "Hello");
    }

    #[test]
    fn test_parse_stream_event_tool_call() {
        let event = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"expr\":"
                        }
                    }]
                },
                "finish_reason": null
            }]
        });

        let chunk = ChatOpenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.message.tool_call_chunks.len(), 1);
        assert_eq!(
            chunk.message.tool_call_chunks[0].name,
            Some("calculator".to_string())
        );
        assert_eq!(
            chunk.message.tool_call_chunks[0].id,
            Some("call_xyz".to_string())
        );
        assert_eq!(chunk.message.tool_call_chunks[0].index, Some(0));
        assert_eq!(
            chunk.message.tool_call_chunks[0].args,
            Some("{\"expr\":".to_string())
        );
    }

    #[test]
    fn test_build_payload() {
        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .temperature(0.5)
            .max_tokens(1024)
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[], false);

        assert_eq!(payload["model"], "gpt-4o");
        assert_eq!(payload["temperature"], 0.5);
        assert_eq!(payload["max_tokens"], 1024);
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
        assert!(payload.get("stream").is_none());
        assert!(payload.get("tools").is_none());
    }

    #[test]
    fn test_extra_headers_on_builder() {
        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .extra_header("HTTP-Referer", "https://mysite.com")
            .extra_header("X-Title", "assistant")
            .build()
            .unwrap();

        assert_eq!(model.extra_headers.len(), 2);
        assert_eq!(
            model.extra_headers.get("HTTP-Referer").map(String::as_str),
            Some("https://mysite.com"),
        );
        assert_eq!(
            model.extra_headers.get("X-Title").map(String::as_str),
            Some("assistant"),
        );
    }

    #[test]
    fn test_extra_header_overwrites_same_key() {
        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .extra_header("X-Title", "first")
            .extra_header("X-Title", "second")
            .build()
            .unwrap();

        assert_eq!(model.extra_headers.len(), 1);
        assert_eq!(
            model.extra_headers.get("X-Title").map(String::as_str),
            Some("second"),
        );
    }

    #[test]
    fn test_extra_headers_map_replaces_all() {
        let mut headers = HashMap::new();
        headers.insert("X-Custom".to_string(), "value".to_string());

        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .extra_header("HTTP-Referer", "https://old.com")
            .extra_headers(headers)
            .build()
            .unwrap();

        assert_eq!(model.extra_headers.len(), 1);
        assert!(model.extra_headers.contains_key("X-Custom"));
        assert!(!model.extra_headers.contains_key("HTTP-Referer"));
    }

    #[test]
    fn test_build_payload_with_tools() {
        let model = ChatOpenAI::builder()
            .model("gpt-4o")
            .api_key("test-key")
            .build()
            .unwrap();

        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        })];

        let messages = vec![Message::Human(HumanMessage::new("Search for rust"))];
        let payload = model.build_payload(&messages, None, &tools, false);

        assert!(payload.get("tools").is_some());
        let payload_tools = payload["tools"].as_array().unwrap();
        assert_eq!(payload_tools.len(), 1);
        assert_eq!(payload_tools[0]["type"], "function");
        assert_eq!(payload_tools[0]["function"]["name"], "search");
    }
}
