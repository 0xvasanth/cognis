//! Google Gemini chat model implementation.
//!
//! Provides [`ChatGoogleGenAI`], a production-ready implementation of
//! [`BaseChatModel`] for the Google Gemini (Generative AI) REST API.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::{
    AIMessage, AIMessageChunk, Message, ToolCall, ToolCallChunk, UsageMetadata,
};
use cognis_core::outputs::{ChatGeneration, ChatGenerationChunk, ChatResult};
use cognis_core::tools::ToolSchema;

/// Builder for constructing a [`ChatGoogleGenAI`] instance.
#[derive(Debug)]
pub struct ChatGoogleGenAIBuilder {
    model: Option<String>,
    api_key: Option<SecretString>,
    base_url: Option<String>,
    temperature: Option<f64>,
    max_output_tokens: Option<u32>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    max_retries: Option<u32>,
    streaming: Option<bool>,
}

impl ChatGoogleGenAIBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: None,
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            max_retries: None,
            streaming: None,
        }
    }

    /// Set the model name (e.g. "gemini-2.0-flash").
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the API key. Falls back to `GOOGLE_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the base API URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the max output tokens for generation.
    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
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
    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
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

    /// Build the [`ChatGoogleGenAI`] instance.
    ///
    /// Returns an error if the API key cannot be resolved from the builder
    /// or environment.
    pub fn build(self) -> Result<ChatGoogleGenAI> {
        let model = self.model.unwrap_or_else(|| "gemini-2.0-flash".to_string());

        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("GOOGLE_API_KEY").map_err(|_| {
                    CognisError::Other(
                        "api_key not provided and GOOGLE_API_KEY env var not set".into(),
                    )
                })?;
                SecretString::from(key)
            }
        };

        Ok(ChatGoogleGenAI {
            model,
            api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".into()),
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            max_retries: self.max_retries.unwrap_or(2),
            streaming: self.streaming.unwrap_or(false),
            client: Client::new(),
            bound_tools: Vec::new(),
            tool_choice: None,
        })
    }
}

impl Default for ChatGoogleGenAIBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Google Gemini chat model.
///
/// Implements the Google Generative AI REST API for chat completion, including
/// tool calling and SSE streaming.
///
/// # Example
///
/// ```no_run
/// use cognis::chat_models::google::ChatGoogleGenAI;
///
/// let model = ChatGoogleGenAI::builder()
///     .model("gemini-2.0-flash")
///     .api_key("AIza...")
///     .build()
///     .unwrap();
/// ```
pub struct ChatGoogleGenAI {
    /// The model identifier (e.g. "gemini-2.0-flash").
    pub model: String,
    /// Secret API key.
    api_key: SecretString,
    /// Base URL for the Gemini API.
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum output tokens to generate.
    pub max_output_tokens: Option<u32>,
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
    /// Tools bound via `bind_tools`.
    bound_tools: Vec<Value>,
    /// Tool choice configuration.
    tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for ChatGoogleGenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatGoogleGenAI")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ChatGoogleGenAI {
    /// Returns a new builder for `ChatGoogleGenAI`.
    pub fn builder() -> ChatGoogleGenAIBuilder {
        ChatGoogleGenAIBuilder::new()
    }

    /// Format messages for the Google Gemini API.
    ///
    /// Returns `(system_instruction, contents)` where system messages are
    /// extracted into a separate `systemInstruction` object, similar to Anthropic.
    pub fn format_messages(messages: &[Message]) -> (Option<Value>, Vec<Value>) {
        let mut system_parts: Vec<String> = Vec::new();
        let mut contents: Vec<Value> = Vec::new();

        for msg in messages {
            match msg {
                Message::System(sys) => {
                    system_parts.push(sys.base.content.text());
                }
                Message::Human(human) => {
                    contents.push(json!({
                        "role": "user",
                        "parts": [{"text": human.base.content.text()}]
                    }));
                }
                Message::Ai(ai) => {
                    let mut parts: Vec<Value> = Vec::new();
                    let text = ai.base.content.text();
                    if !text.is_empty() {
                        parts.push(json!({"text": text}));
                    }
                    for tc in &ai.tool_calls {
                        let mut args_value = json!({});
                        for (k, v) in &tc.args {
                            args_value[k] = v.clone();
                        }
                        parts.push(json!({
                            "functionCall": {
                                "name": tc.name,
                                "args": args_value
                            }
                        }));
                    }
                    if parts.is_empty() {
                        parts.push(json!({"text": ""}));
                    }
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
                Message::Tool(tool) => {
                    // Parse the tool result content as JSON if possible, otherwise wrap as string
                    let response_value: Value = serde_json::from_str(&tool.base.content.text())
                        .unwrap_or_else(|_| json!({"result": tool.base.content.text()}));
                    contents.push(json!({
                        "role": "function",
                        "parts": [{
                            "functionResponse": {
                                "name": tool.tool_call_id,
                                "response": response_value
                            }
                        }]
                    }));
                }
                _ => {
                    // Best-effort: treat as user message
                    contents.push(json!({
                        "role": "user",
                        "parts": [{"text": msg.content().text()}]
                    }));
                }
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(json!({
                "parts": system_parts.iter().map(|s| json!({"text": s})).collect::<Vec<_>>()
            }))
        };

        (system_instruction, contents)
    }

    /// Build the JSON payload for the Google Gemini API.
    pub fn build_payload(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        tools: &[Value],
    ) -> Value {
        let (system_instruction, contents) = Self::format_messages(messages);

        let mut payload = json!({
            "contents": contents,
        });

        if let Some(sys) = system_instruction {
            payload["systemInstruction"] = sys;
        }

        // Build generationConfig
        let mut gen_config = json!({});
        let mut has_gen_config = false;

        if let Some(temp) = self.temperature {
            gen_config["temperature"] = json!(temp);
            has_gen_config = true;
        }
        if let Some(max_tokens) = self.max_output_tokens {
            gen_config["maxOutputTokens"] = json!(max_tokens);
            has_gen_config = true;
        }
        if let Some(tp) = self.top_p {
            gen_config["topP"] = json!(tp);
            has_gen_config = true;
        }
        if let Some(tk) = self.top_k {
            gen_config["topK"] = json!(tk);
            has_gen_config = true;
        }
        // Stop sequences: merge configured sequences with explicit stop param
        let mut all_stop = Vec::new();
        if let Some(configured) = &self.stop_sequences {
            all_stop.extend(configured.iter().cloned());
        }
        if let Some(stop_param) = stop {
            all_stop.extend(stop_param.iter().cloned());
        }
        if !all_stop.is_empty() {
            gen_config["stopSequences"] = json!(all_stop);
            has_gen_config = true;
        }
        if has_gen_config {
            payload["generationConfig"] = gen_config;
        }

        if !tools.is_empty() {
            payload["tools"] = json!([{
                "functionDeclarations": tools
            }]);
        }

        if let Some(ref choice) = self.tool_choice {
            payload["toolConfig"] = match choice {
                ToolChoice::Auto => json!({"functionCallingConfig": {"mode": "AUTO"}}),
                ToolChoice::Any => json!({"functionCallingConfig": {"mode": "ANY"}}),
                ToolChoice::None => json!({"functionCallingConfig": {"mode": "NONE"}}),
                ToolChoice::Tool(name) => json!({
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [name]
                    }
                }),
            };
        }

        payload
    }

    /// Parse a Google Gemini API response into a [`ChatResult`].
    pub fn parse_response(response: &Value) -> Result<ChatResult> {
        let candidates = response
            .get("candidates")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                CognisError::Other("Missing 'candidates' array in Gemini response".into())
            })?;

        let candidate = candidates.first().ok_or_else(|| {
            CognisError::Other("Empty 'candidates' array in Gemini response".into())
        })?;

        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .ok_or_else(|| {
                CognisError::Other("Missing 'content.parts' in Gemini response candidate".into())
            })?;

        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for part in parts {
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                text_parts.push(text.to_string());
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args: HashMap<String, Value> = fc
                    .get("args")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                // Gemini does not provide tool call IDs; generate a placeholder
                let id = Some(format!("call_{}", tool_calls.len()));
                tool_calls.push(ToolCall { name, args, id });
            }
        }

        let full_text = text_parts.join("");

        // Parse usage metadata
        let usage_metadata = response.get("usageMetadata").map(|u| {
            let input_tokens = u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total_tokens = u
                .get("totalTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(input_tokens + output_tokens);
            UsageMetadata::new(input_tokens, output_tokens, total_tokens)
        });

        let mut ai_message = AIMessage::new(&full_text);
        ai_message.tool_calls = tool_calls;
        ai_message.usage_metadata = usage_metadata;

        let generation = ChatGeneration::new(ai_message);

        Ok(ChatResult {
            generations: vec![generation],
            llm_output: None,
        })
    }

    /// Parse an SSE stream event into a [`ChatGenerationChunk`].
    pub fn parse_stream_event(event: &Value) -> Option<ChatGenerationChunk> {
        let candidates = event.get("candidates").and_then(|v| v.as_array())?;
        let candidate = candidates.first()?;

        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())?;

        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_call_chunks: Vec<ToolCallChunk> = Vec::new();

        for (idx, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                text_parts.push(text.to_string());
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let args = fc
                    .get("args")
                    .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()));
                tool_call_chunks.push(ToolCallChunk {
                    name,
                    args,
                    id: Some(format!("call_{}", idx)),
                    index: Some(idx),
                });
            }
        }

        let combined_text = text_parts.join("");

        let mut chunk = AIMessageChunk::new(&combined_text);
        chunk.tool_call_chunks = tool_call_chunks;

        // Check finish reason
        let finish_reason = candidate.get("finishReason").and_then(|v| v.as_str());
        if let Some(reason) = finish_reason {
            if reason == "STOP" || reason == "MAX_TOKENS" {
                chunk.chunk_position = Some("last".to_string());
                chunk
                    .base
                    .response_metadata
                    .insert("finish_reason".to_string(), json!(reason));
            }
        }

        // Parse usage from stream event
        if let Some(usage) = event.get("usageMetadata") {
            let input_tokens = usage
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = usage
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total_tokens = usage
                .get("totalTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(input_tokens + output_tokens);
            chunk.usage_metadata = Some(UsageMetadata::new(
                input_tokens,
                output_tokens,
                total_tokens,
            ));
        }

        // Skip empty chunks with no meaningful content
        if combined_text.is_empty()
            && chunk.tool_call_chunks.is_empty()
            && chunk.chunk_position.is_none()
            && chunk.usage_metadata.is_none()
        {
            return None;
        }

        Some(ChatGenerationChunk::new(chunk))
    }

    /// Call the Google Gemini API with retry logic.
    async fn call_api(&self, payload: &Value) -> Result<Value> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url,
            self.model,
            self.api_key.expose_secret()
        );

        let mut last_error = CognisError::Other("No attempts made".into());

        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
                .header("Content-Type", "application/json");

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

    /// Call the Google Gemini API with streaming, returning an async stream of SSE events.
    async fn call_api_stream(
        &self,
        payload: &Value,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value>> + Send>>> {
        let url = format!(
            "{}/models/{}:streamGenerateContent?key={}&alt=sse",
            self.base_url,
            self.model,
            self.api_key.expose_secret()
        );

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

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

    /// Convert a [`ToolSchema`] to the Google Gemini function declaration format.
    fn tool_schema_to_google(schema: &ToolSchema) -> Value {
        let mut func = json!({
            "name": schema.name,
            "description": schema.description,
        });
        if let Some(ref params) = schema.parameters {
            func["parameters"] = params.clone();
        } else {
            func["parameters"] = json!({
                "type": "object",
                "properties": {},
            });
        }
        func
    }
}

#[async_trait]
impl BaseChatModel for ChatGoogleGenAI {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        let payload = self.build_payload(messages, stop, &self.bound_tools);
        let response = self.call_api(&payload).await?;
        Self::parse_response(&response)
    }

    fn llm_type(&self) -> &str {
        "google_gemini"
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        let payload = self.build_payload(messages, stop, &self.bound_tools);
        let event_stream = self.call_api_stream(&payload).await?;

        let chunk_stream = event_stream.filter_map(|event_result| async move {
            match event_result {
                Ok(event) => ChatGoogleGenAI::parse_stream_event(&event).map(Ok),
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
        let bound_tools: Vec<Value> = tools.iter().map(Self::tool_schema_to_google).collect();

        Ok(Box::new(ChatGoogleGenAI {
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences.clone(),
            max_retries: self.max_retries,
            streaming: self.streaming,
            client: self.client.clone(),
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
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::{HumanMessage, SystemMessage, ToolMessage};

    #[test]
    fn test_google_config_builder_defaults() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(model.model, "gemini-2.0-flash");
        assert_eq!(
            model.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(model.temperature, None);
        assert_eq!(model.max_output_tokens, None);
        assert_eq!(model.top_p, None);
        assert_eq!(model.top_k, None);
        assert_eq!(model.stop_sequences, None);
        assert_eq!(model.max_retries, 2);
        assert!(!model.streaming);
    }

    #[test]
    fn test_google_config_builder_custom() {
        let model = ChatGoogleGenAI::builder()
            .model("gemini-1.5-pro")
            .api_key("test-key")
            .base_url("https://custom.api.com")
            .temperature(0.7)
            .max_output_tokens(2048)
            .top_p(0.9)
            .top_k(40)
            .stop_sequences(vec!["STOP".to_string(), "END".to_string()])
            .max_retries(3)
            .streaming(true)
            .build()
            .unwrap();

        assert_eq!(model.model, "gemini-1.5-pro");
        assert_eq!(model.base_url, "https://custom.api.com");
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.max_output_tokens, Some(2048));
        assert_eq!(model.top_p, Some(0.9));
        assert_eq!(model.top_k, Some(40));
        assert_eq!(
            model.stop_sequences,
            Some(vec!["STOP".to_string(), "END".to_string()])
        );
        assert_eq!(model.max_retries, 3);
        assert!(model.streaming);
    }

    #[test]
    fn test_url_construction_with_model_name() {
        let model = ChatGoogleGenAI::builder()
            .model("gemini-2.0-flash")
            .api_key("test-api-key")
            .build()
            .unwrap();

        // Verify the URL is constructed correctly by checking base_url and model
        let expected_base = "https://generativelanguage.googleapis.com/v1beta";
        assert_eq!(model.base_url, expected_base);
        assert_eq!(model.model, "gemini-2.0-flash");
        // The full URL would be: {base_url}/models/{model}:generateContent?key={api_key}
        // For streaming: {base_url}/models/{model}:streamGenerateContent?key={api_key}&alt=sse
    }

    #[test]
    fn test_url_construction_custom_model() {
        let model = ChatGoogleGenAI::builder()
            .model("gemini-1.5-pro")
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(model.model, "gemini-1.5-pro");
        assert_eq!(
            model.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn test_format_messages_simple() {
        let messages = vec![Message::Human(HumanMessage::new("Hello"))];
        let (system, contents) = ChatGoogleGenAI::format_messages(&messages);

        assert!(system.is_none());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello");
    }

    #[test]
    fn test_format_messages_system_extracted_to_system_instruction() {
        let messages = vec![
            Message::System(SystemMessage::new("You are helpful")),
            Message::Human(HumanMessage::new("Hi")),
        ];
        let (system, contents) = ChatGoogleGenAI::format_messages(&messages);

        // System messages become systemInstruction, not part of contents
        assert!(system.is_some());
        let sys = system.unwrap();
        assert_eq!(sys["parts"][0]["text"], "You are helpful");
        // Only the human message should be in contents
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hi");
    }

    #[test]
    fn test_format_messages_multiple_system_messages() {
        let messages = vec![
            Message::System(SystemMessage::new("Be helpful")),
            Message::System(SystemMessage::new("Be concise")),
            Message::Human(HumanMessage::new("Hello")),
        ];
        let (system, contents) = ChatGoogleGenAI::format_messages(&messages);

        let sys = system.unwrap();
        let parts = sys["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "Be helpful");
        assert_eq!(parts[1]["text"], "Be concise");
        assert_eq!(contents.len(), 1);
    }

    #[test]
    fn test_format_messages_multi_turn_conversation() {
        let messages = vec![
            Message::System(SystemMessage::new("You are a helpful assistant")),
            Message::Human(HumanMessage::new("What is Rust?")),
            Message::Ai(AIMessage::new("Rust is a systems programming language.")),
            Message::Human(HumanMessage::new("What about its memory model?")),
        ];
        let (system, contents) = ChatGoogleGenAI::format_messages(&messages);

        // System extracted
        assert!(system.is_some());
        assert_eq!(
            system.unwrap()["parts"][0]["text"],
            "You are a helpful assistant"
        );

        // Three turns in contents
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "What is Rust?");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(
            contents[1]["parts"][0]["text"],
            "Rust is a systems programming language."
        );
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(
            contents[2]["parts"][0]["text"],
            "What about its memory model?"
        );
    }

    #[test]
    fn test_message_role_mapping() {
        // Human -> user, AI -> model, System -> systemInstruction
        let messages = vec![
            Message::System(SystemMessage::new("System prompt")),
            Message::Human(HumanMessage::new("User message")),
            Message::Ai(AIMessage::new("Model response")),
        ];
        let (system, contents) = ChatGoogleGenAI::format_messages(&messages);

        // System goes to systemInstruction
        assert!(system.is_some());

        // Human maps to "user"
        assert_eq!(contents[0]["role"], "user");

        // AI maps to "model"
        assert_eq!(contents[1]["role"], "model");
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
        let (_, contents) = ChatGoogleGenAI::format_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "model");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "Let me search");
        assert_eq!(parts[1]["functionCall"]["name"], "search");
        assert_eq!(parts[1]["functionCall"]["args"]["query"], "rust");
    }

    #[test]
    fn test_format_messages_with_tool_result() {
        let messages = vec![Message::Tool(ToolMessage::new(
            "Search results here",
            "search",
        ))];
        let (_, contents) = ChatGoogleGenAI::format_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "function");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionResponse"]["name"], "search");
    }

    #[test]
    fn test_parse_response_text() {
        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello, world!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });

        let result = ChatGoogleGenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);
        assert_eq!(result.generations[0].text, "Hello, world!");

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert!(ai.tool_calls.is_empty());
            let usage = ai.usage_metadata.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 10);
            assert_eq!(usage.output_tokens, 5);
            assert_eq!(usage.total_tokens, 15);
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_response_tool_call() {
        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "I'll search for that."},
                        {
                            "functionCall": {
                                "name": "web_search",
                                "args": {"query": "rust programming"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 15,
                "totalTokenCount": 35
            }
        });

        let result = ChatGoogleGenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "web_search");
            assert_eq!(
                ai.tool_calls[0].args.get("query"),
                Some(&json!("rust programming"))
            );
            assert!(ai.tool_calls[0].id.is_some());
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_response_multiple_tool_calls() {
        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "Tokyo"}
                            }
                        },
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "London"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });

        let result = ChatGoogleGenAI::parse_response(&response).unwrap();
        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 2);
            assert_eq!(ai.tool_calls[0].name, "get_weather");
            assert_eq!(ai.tool_calls[0].args["city"], "Tokyo");
            assert_eq!(ai.tool_calls[1].name, "get_weather");
            assert_eq!(ai.tool_calls[1].args["city"], "London");
            // Each should have a unique generated ID
            assert_ne!(ai.tool_calls[0].id, ai.tool_calls[1].id);
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_stream_event_text() {
        let event = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}],
                    "role": "model"
                }
            }]
        });

        let chunk = ChatGoogleGenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.text, "Hello");
        assert_eq!(chunk.message.base.content.text(), "Hello");
    }

    #[test]
    fn test_parse_stream_event_with_finish_reason() {
        let event = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Done"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 1,
                "totalTokenCount": 6
            }
        });

        let chunk = ChatGoogleGenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.text, "Done");
        assert_eq!(chunk.message.chunk_position, Some("last".to_string()));
        let usage = chunk.message.usage_metadata.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 1);
        assert_eq!(usage.total_tokens, 6);
    }

    #[test]
    fn test_parse_stream_event_max_tokens_finish() {
        let event = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "truncated"}],
                    "role": "model"
                },
                "finishReason": "MAX_TOKENS"
            }]
        });

        let chunk = ChatGoogleGenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.message.chunk_position, Some("last".to_string()));
        assert_eq!(
            chunk.message.base.response_metadata.get("finish_reason"),
            Some(&json!("MAX_TOKENS"))
        );
    }

    #[test]
    fn test_parse_stream_event_tool_call() {
        let event = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "calculator",
                            "args": {"expr": "2+2"}
                        }
                    }],
                    "role": "model"
                }
            }]
        });

        let chunk = ChatGoogleGenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.message.tool_call_chunks.len(), 1);
        assert_eq!(
            chunk.message.tool_call_chunks[0].name,
            Some("calculator".to_string())
        );
        assert!(chunk.message.tool_call_chunks[0].args.is_some());
    }

    #[test]
    fn test_parse_stream_event_empty_skipped() {
        let event = json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": ""}],
                    "role": "model"
                }
            }]
        });

        let chunk = ChatGoogleGenAI::parse_stream_event(&event);
        assert!(
            chunk.is_none(),
            "Empty chunks with no meaningful content should be skipped"
        );
    }

    #[test]
    fn test_build_payload_basic() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .temperature(0.7)
            .max_output_tokens(1024)
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[]);

        assert!(payload.get("contents").is_some());
        assert!(payload.get("tools").is_none());
        let gen_config = &payload["generationConfig"];
        assert_eq!(gen_config["temperature"], 0.7);
        assert_eq!(gen_config["maxOutputTokens"], 1024);
    }

    #[test]
    fn test_build_payload_with_all_gen_config() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .temperature(0.5)
            .max_output_tokens(512)
            .top_p(0.95)
            .top_k(50)
            .stop_sequences(vec!["END".to_string()])
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hello"))];
        let payload = model.build_payload(&messages, None, &[]);

        let gen_config = &payload["generationConfig"];
        assert_eq!(gen_config["temperature"], 0.5);
        assert_eq!(gen_config["maxOutputTokens"], 512);
        assert_eq!(gen_config["topP"], 0.95);
        assert_eq!(gen_config["topK"], 50);
        let stop_seqs = gen_config["stopSequences"].as_array().unwrap();
        assert_eq!(stop_seqs.len(), 1);
        assert_eq!(stop_seqs[0], "END");
    }

    #[test]
    fn test_build_payload_stop_sequences_merged() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .stop_sequences(vec!["STOP1".to_string()])
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hello"))];
        let extra_stop = vec!["STOP2".to_string()];
        let payload = model.build_payload(&messages, Some(&extra_stop), &[]);

        let gen_config = &payload["generationConfig"];
        let stop_seqs = gen_config["stopSequences"].as_array().unwrap();
        assert_eq!(stop_seqs.len(), 2);
        assert!(stop_seqs.contains(&json!("STOP1")));
        assert!(stop_seqs.contains(&json!("STOP2")));
    }

    #[test]
    fn test_build_payload_with_system_instruction() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let messages = vec![
            Message::System(SystemMessage::new("Be helpful")),
            Message::Human(HumanMessage::new("Hi")),
        ];
        let payload = model.build_payload(&messages, None, &[]);

        assert!(payload.get("systemInstruction").is_some());
        assert_eq!(
            payload["systemInstruction"]["parts"][0]["text"],
            "Be helpful"
        );
        let contents = payload["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn test_build_payload_with_tools() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let tools = vec![json!({
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        })];

        let messages = vec![Message::Human(HumanMessage::new("Search for rust"))];
        let payload = model.build_payload(&messages, None, &tools);

        assert!(payload.get("tools").is_some());
        let tool_decls = payload["tools"][0]["functionDeclarations"]
            .as_array()
            .unwrap();
        assert_eq!(tool_decls.len(), 1);
        assert_eq!(tool_decls[0]["name"], "search");
    }

    #[test]
    fn test_tool_schema_conversion() {
        let schema = ToolSchema {
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            })),
            extras: None,
        };

        let google_tool = ChatGoogleGenAI::tool_schema_to_google(&schema);
        assert_eq!(google_tool["name"], "get_weather");
        assert_eq!(google_tool["description"], "Get weather for a city");
        assert_eq!(google_tool["parameters"]["type"], "object");
        assert_eq!(
            google_tool["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn test_tool_schema_conversion_no_params() {
        let schema = ToolSchema {
            name: "get_time".to_string(),
            description: "Get the current time".to_string(),
            parameters: None,
            extras: None,
        };

        let google_tool = ChatGoogleGenAI::tool_schema_to_google(&schema);
        assert_eq!(google_tool["name"], "get_time");
        assert_eq!(google_tool["parameters"]["type"], "object");
    }

    #[test]
    fn test_bind_tools_creates_new_model() {
        let model = ChatGoogleGenAI::builder()
            .model("gemini-2.0-flash")
            .api_key("test-key")
            .temperature(0.5)
            .build()
            .unwrap();

        let tools = vec![ToolSchema {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: Some(json!({"type": "object", "properties": {"q": {"type": "string"}}})),
            extras: None,
        }];

        let bound = model.bind_tools(&tools, Some(ToolChoice::Auto)).unwrap();
        assert_eq!(bound.llm_type(), "google_gemini");
    }

    #[test]
    fn test_tool_choice_auto() {
        let mut model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        model.tool_choice = Some(ToolChoice::Auto);

        let tools = vec![json!({"name": "test", "description": "test"})];
        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &tools);

        assert_eq!(
            payload["toolConfig"]["functionCallingConfig"]["mode"],
            "AUTO"
        );
    }

    #[test]
    fn test_tool_choice_none() {
        let mut model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        model.tool_choice = Some(ToolChoice::None);

        let tools = vec![json!({"name": "test", "description": "test"})];
        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &tools);

        assert_eq!(
            payload["toolConfig"]["functionCallingConfig"]["mode"],
            "NONE"
        );
    }

    #[test]
    fn test_tool_choice_specific_tool() {
        let mut model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        model.tool_choice = Some(ToolChoice::Tool("search".to_string()));

        let tools = vec![json!({"name": "search", "description": "search"})];
        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &tools);

        assert_eq!(
            payload["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY"
        );
        let allowed = payload["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"]
            .as_array()
            .unwrap();
        assert_eq!(allowed[0], "search");
    }

    #[test]
    fn test_llm_type() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        assert_eq!(model.llm_type(), "google_gemini");
    }

    #[test]
    fn test_profile_capabilities() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();
        let profile = model.profile();
        assert_eq!(profile.tool_calling, Some(true));
        assert_eq!(profile.tool_choice, Some(true));
        assert_eq!(profile.structured_output, Some(true));
        assert_eq!(profile.text_inputs, Some(true));
        assert_eq!(profile.text_outputs, Some(true));
        assert_eq!(profile.image_inputs, Some(true));
    }

    #[test]
    fn test_build_payload_no_gen_config_when_empty() {
        let model = ChatGoogleGenAI::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[]);

        // No generationConfig when no parameters set
        assert!(payload.get("generationConfig").is_none());
    }
}
