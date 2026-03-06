//! Azure OpenAI chat model implementation.
//!
//! Provides [`ChatAzureOpenAI`], a production-ready implementation of
//! [`BaseChatModel`] for the Azure OpenAI Chat Completions API.
//!
//! Azure OpenAI uses the same request/response format as OpenAI, but with
//! different endpoint URLs and authentication headers. Two authentication
//! methods are supported: API key (`api-key` header) and Azure AD token
//! (`Authorization: Bearer` header).

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

/// Default Azure OpenAI API version.
const DEFAULT_API_VERSION: &str = "2024-08-01-preview";

/// Builder for constructing a [`ChatAzureOpenAI`] instance.
#[derive(Debug)]
pub struct ChatAzureOpenAIBuilder {
    azure_endpoint: Option<String>,
    deployment_name: Option<String>,
    api_version: Option<String>,
    api_key: Option<SecretString>,
    azure_ad_token: Option<SecretString>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    top_p: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
    max_retries: Option<u32>,
    streaming: Option<bool>,
}

impl ChatAzureOpenAIBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            azure_endpoint: None,
            deployment_name: None,
            api_version: None,
            api_key: None,
            azure_ad_token: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            max_retries: None,
            streaming: None,
        }
    }

    /// Set the Azure endpoint URL (e.g., `"https://myinstance.openai.azure.com"`).
    ///
    /// Falls back to the `AZURE_OPENAI_ENDPOINT` environment variable.
    pub fn azure_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.azure_endpoint = Some(endpoint.into());
        self
    }

    /// Set the deployment name (required).
    pub fn deployment_name(mut self, name: impl Into<String>) -> Self {
        self.deployment_name = Some(name.into());
        self
    }

    /// Set the API version (default: `"2024-08-01-preview"`).
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Set the API key for key-based authentication.
    ///
    /// Falls back to the `AZURE_OPENAI_API_KEY` environment variable.
    /// Either `api_key` or `azure_ad_token` must be provided.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the Azure AD token for token-based authentication.
    ///
    /// When set, the `Authorization: Bearer <token>` header is used instead
    /// of the `api-key` header. Takes precedence over `api_key`.
    pub fn azure_ad_token(mut self, token: impl Into<String>) -> Self {
        self.azure_ad_token = Some(SecretString::from(token.into()));
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the max tokens for generation.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
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

    /// Build the [`ChatAzureOpenAI`] instance.
    ///
    /// Returns an error if required fields are missing or if no authentication
    /// method is configured (neither API key nor Azure AD token).
    pub fn build(self) -> Result<ChatAzureOpenAI> {
        let azure_endpoint = self
            .azure_endpoint
            .or_else(|| std::env::var("AZURE_OPENAI_ENDPOINT").ok())
            .ok_or_else(|| {
                RustChainError::Other(
                    "azure_endpoint not provided and AZURE_OPENAI_ENDPOINT env var not set".into(),
                )
            })?;

        let deployment_name = self.deployment_name.ok_or_else(|| {
            RustChainError::Other("deployment_name is required for ChatAzureOpenAI".into())
        })?;

        let api_version = self
            .api_version
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_string());

        // Resolve API key from builder or environment
        let api_key = match self.api_key {
            Some(key) => Some(key),
            None => std::env::var("AZURE_OPENAI_API_KEY")
                .ok()
                .map(SecretString::from),
        };

        let azure_ad_token = self.azure_ad_token;

        if api_key.is_none() && azure_ad_token.is_none() {
            return Err(RustChainError::Other(
                "Either api_key or azure_ad_token must be provided for ChatAzureOpenAI".into(),
            ));
        }

        Ok(ChatAzureOpenAI {
            azure_endpoint,
            deployment_name,
            api_version,
            api_key,
            azure_ad_token,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stop: self.stop,
            max_retries: self.max_retries.unwrap_or(2),
            streaming: self.streaming.unwrap_or(false),
            client: Client::new(),
            bound_tools: Vec::new(),
            tool_choice: None,
        })
    }
}

impl Default for ChatAzureOpenAIBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Azure OpenAI chat model.
///
/// Implements the Azure OpenAI Chat Completions API for chat completion,
/// including tool calling and SSE streaming. Uses the same request/response
/// format as OpenAI but with Azure-specific endpoint URLs and authentication.
///
/// # Authentication
///
/// Two authentication methods are supported:
/// - **API key**: Uses the `api-key` HTTP header.
/// - **Azure AD token**: Uses the `Authorization: Bearer <token>` header.
///   When both are provided, the Azure AD token takes precedence.
///
/// # Example
///
/// ```no_run
/// use rustchain::chat_models::azure::ChatAzureOpenAI;
///
/// let model = ChatAzureOpenAI::builder()
///     .azure_endpoint("https://myinstance.openai.azure.com")
///     .deployment_name("gpt-4o")
///     .api_key("my-azure-key")
///     .build()
///     .unwrap();
/// ```
pub struct ChatAzureOpenAI {
    /// Azure endpoint URL (e.g., `"https://myinstance.openai.azure.com"`).
    pub azure_endpoint: String,
    /// Deployment name (used in URL instead of model in body).
    pub deployment_name: String,
    /// Azure OpenAI API version.
    pub api_version: String,
    /// Optional API key for key-based authentication.
    api_key: Option<SecretString>,
    /// Optional Azure AD token for token-based authentication.
    azure_ad_token: Option<SecretString>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
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
    /// HTTP client.
    client: Client,
    /// Tools bound via `bind_tools`.
    bound_tools: Vec<Value>,
    /// Tool choice configuration.
    tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for ChatAzureOpenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatAzureOpenAI")
            .field("azure_endpoint", &self.azure_endpoint)
            .field("deployment_name", &self.deployment_name)
            .field("api_version", &self.api_version)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ChatAzureOpenAI {
    /// Returns a new builder for `ChatAzureOpenAI`.
    pub fn builder() -> ChatAzureOpenAIBuilder {
        ChatAzureOpenAIBuilder::new()
    }

    /// Build the Azure OpenAI endpoint URL.
    ///
    /// Format: `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}`
    pub fn build_url(&self) -> String {
        let endpoint = self.azure_endpoint.trim_end_matches('/');
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            endpoint, self.deployment_name, self.api_version
        )
    }

    /// Apply authentication headers to the request builder.
    ///
    /// When an Azure AD token is present it takes precedence and the
    /// `Authorization: Bearer` header is used. Otherwise the `api-key`
    /// header is used.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref ad_token) = self.azure_ad_token {
            req.header(
                "Authorization",
                format!("Bearer {}", ad_token.expose_secret()),
            )
        } else if let Some(ref api_key) = self.api_key {
            req.header("api-key", api_key.expose_secret())
        } else {
            req
        }
    }

    /// Format messages for the Azure OpenAI Chat Completions API.
    ///
    /// The message format is identical to OpenAI.
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
                    api_messages.push(json!({
                        "role": "user",
                        "content": msg.content().text()
                    }));
                }
            }
        }

        api_messages
    }

    /// Build the JSON payload for the Azure OpenAI Chat Completions API.
    ///
    /// Unlike standard OpenAI, the `model` field is not included in the body
    /// because it is implicit from the deployment name in the URL.
    pub fn build_payload(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        tools: &[Value],
        stream: bool,
    ) -> Value {
        let api_messages = Self::format_messages(messages);

        let mut payload = json!({
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

        // Stop sequences
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

    /// Parse an Azure OpenAI Chat Completions API response into a [`ChatResult`].
    ///
    /// The response format is identical to standard OpenAI.
    pub fn parse_response(response: &Value) -> Result<ChatResult> {
        let choices = response
            .get("choices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RustChainError::Other("Missing 'choices' array in Azure OpenAI response".into())
            })?;

        let choice = choices.first().ok_or_else(|| {
            RustChainError::Other("Empty 'choices' array in Azure OpenAI response".into())
        })?;

        let message = choice.get("message").ok_or_else(|| {
            RustChainError::Other("Missing 'message' in choice".into())
        })?;

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
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let args_str = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let args: HashMap<String, Value> =
                    serde_json::from_str(args_str).unwrap_or_default();
                tool_calls.push(ToolCall { name, args, id });
            }
        }

        let usage_metadata = response.get("usage").map(|u| {
            let prompt_tokens = u
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
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
    ///
    /// The streaming format is identical to standard OpenAI.
    pub fn parse_stream_event(event: &Value) -> Option<ChatGenerationChunk> {
        let choices = event.get("choices").and_then(|v| v.as_array())?;
        let choice = choices.first()?;

        let delta = choice.get("delta")?;
        let finish_reason = choice.get("finish_reason");

        let content = delta
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut chunk = AIMessageChunk::new(content);

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
                chunk.usage_metadata =
                    Some(UsageMetadata::new(prompt_tokens, completion_tokens, total_tokens));
            }
        }

        if content.is_empty()
            && chunk.tool_call_chunks.is_empty()
            && chunk.chunk_position.is_none()
            && chunk.usage_metadata.is_none()
        {
            return None;
        }

        Some(ChatGenerationChunk::new(chunk))
    }

    /// Call the Azure OpenAI Chat Completions API with retry logic.
    async fn call_api(&self, payload: &Value) -> Result<Value> {
        let url = self.build_url();

        let mut last_error = RustChainError::Other("No attempts made".into());

        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
                .header("Content-Type", "application/json");

            let req = self.apply_auth(req);

            let response = req.json(payload).send().await.map_err(|e| {
                RustChainError::Other(format!("HTTP request failed: {}", e))
            })?;

            let status = response.status().as_u16();

            if (200..300).contains(&status) {
                let body: Value = response.json().await.map_err(|e| {
                    RustChainError::Other(format!("Failed to parse response JSON: {}", e))
                })?;
                return Ok(body);
            }

            let body = response.text().await.unwrap_or_default();

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

    /// Call the Azure OpenAI Chat Completions API with streaming.
    async fn call_api_stream(
        &self,
        payload: &Value,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value>> + Send>>> {
        let url = self.build_url();

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        let req = self.apply_auth(req);

        let response = req.json(payload).send().await.map_err(|e| {
            RustChainError::Other(format!("HTTP request failed: {}", e))
        })?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
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
impl BaseChatModel for ChatAzureOpenAI {
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
        "azure_openai"
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
                Ok(event) => ChatAzureOpenAI::parse_stream_event(&event).map(Ok),
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

        Ok(Box::new(ChatAzureOpenAI {
            azure_endpoint: self.azure_endpoint.clone(),
            deployment_name: self.deployment_name.clone(),
            api_version: self.api_version.clone(),
            api_key: self.api_key.clone(),
            azure_ad_token: self.azure_ad_token.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
            stop: self.stop.clone(),
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
    use rustchain_core::messages::{HumanMessage, SystemMessage, ToolMessage};

    #[test]
    fn test_url_construction() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(
            model.build_url(),
            format!(
                "https://myinstance.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version={}",
                DEFAULT_API_VERSION
            )
        );
    }

    #[test]
    fn test_url_construction_trailing_slash() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com/")
            .deployment_name("gpt-35-turbo")
            .api_version("2024-02-01")
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(
            model.build_url(),
            "https://myinstance.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-01"
        );
    }

    #[test]
    fn test_api_key_auth_header() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("my-secret-key")
            .build()
            .unwrap();

        assert!(model.api_key.is_some());
        assert!(model.azure_ad_token.is_none());

        let client = Client::new();
        let req = client.post("https://example.com");
        let req = model.apply_auth(req);
        let built = req.build().unwrap();
        assert_eq!(
            built.headers().get("api-key").unwrap().to_str().unwrap(),
            "my-secret-key"
        );
    }

    #[test]
    fn test_azure_ad_token_auth_header() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .azure_ad_token("my-ad-token-value")
            .build()
            .unwrap();

        assert!(model.azure_ad_token.is_some());

        let client = Client::new();
        let req = client.post("https://example.com");
        let req = model.apply_auth(req);
        let built = req.build().unwrap();
        assert_eq!(
            built.headers().get("Authorization").unwrap().to_str().unwrap(),
            "Bearer my-ad-token-value"
        );
    }

    #[test]
    fn test_ad_token_takes_precedence_over_api_key() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("my-api-key")
            .azure_ad_token("my-ad-token")
            .build()
            .unwrap();

        let client = Client::new();
        let req = client.post("https://example.com");
        let req = model.apply_auth(req);
        let built = req.build().unwrap();
        // AD token should take precedence
        assert_eq!(
            built.headers().get("Authorization").unwrap().to_str().unwrap(),
            "Bearer my-ad-token"
        );
        assert!(built.headers().get("api-key").is_none());
    }

    #[test]
    fn test_request_body_formatting() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .temperature(0.5)
            .max_tokens(1024)
            .top_p(0.9)
            .frequency_penalty(0.3)
            .presence_penalty(0.1)
            .seed(42)
            .build()
            .unwrap();

        let messages = vec![
            Message::System(SystemMessage::new("You are helpful")),
            Message::Human(HumanMessage::new("Hello")),
        ];
        let payload = model.build_payload(&messages, None, &[], false);

        // Azure does NOT include the model field in the body
        assert!(payload.get("model").is_none());
        assert_eq!(payload["temperature"], 0.5);
        assert_eq!(payload["max_tokens"], 1024);
        assert_eq!(payload["top_p"], 0.9);
        assert_eq!(payload["frequency_penalty"], 0.3);
        assert_eq!(payload["presence_penalty"], 0.1);
        assert_eq!(payload["seed"], 42);

        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hello");

        assert!(payload.get("stream").is_none());
    }

    #[test]
    fn test_response_parsing() {
        let response = json!({
            "id": "chatcmpl-azure-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Azure!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        });

        let result = ChatAzureOpenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);
        assert_eq!(result.generations[0].text, "Hello from Azure!");

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert!(ai.tool_calls.is_empty());
            let usage = ai.usage_metadata.as_ref().unwrap();
            assert_eq!(usage.input_tokens, 12);
            assert_eq!(usage.output_tokens, 8);
            assert_eq!(usage.total_tokens, 20);
            assert_eq!(ai.base.id, Some("chatcmpl-azure-123".to_string()));
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_tool_call_response_parsing() {
        let response = json!({
            "id": "chatcmpl-azure-456",
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
                            "arguments": "{\"query\":\"azure openai\"}"
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

        let result = ChatAzureOpenAI::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "web_search");
            assert_eq!(ai.tool_calls[0].id, Some("call_abc".to_string()));
            assert_eq!(
                ai.tool_calls[0].args.get("query"),
                Some(&json!("azure openai"))
            );
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_builder_defaults() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(model.azure_endpoint, "https://myinstance.openai.azure.com");
        assert_eq!(model.deployment_name, "gpt-4o");
        assert_eq!(model.api_version, DEFAULT_API_VERSION);
        assert_eq!(model.temperature, None);
        assert_eq!(model.max_tokens, None);
        assert_eq!(model.top_p, None);
        assert_eq!(model.frequency_penalty, None);
        assert_eq!(model.presence_penalty, None);
        assert_eq!(model.seed, None);
        assert_eq!(model.max_retries, 2);
        assert!(!model.streaming);
    }

    #[test]
    fn test_builder_all_fields() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://custom.openai.azure.com")
            .deployment_name("my-deployment")
            .api_key("test-key")
            .api_version("2024-06-01")
            .temperature(0.7)
            .max_tokens(2048)
            .top_p(0.9)
            .frequency_penalty(0.4)
            .presence_penalty(0.2)
            .seed(99)
            .stop(vec!["STOP".to_string()])
            .max_retries(5)
            .streaming(true)
            .build()
            .unwrap();

        assert_eq!(model.azure_endpoint, "https://custom.openai.azure.com");
        assert_eq!(model.api_version, "2024-06-01");
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.max_tokens, Some(2048));
        assert_eq!(model.top_p, Some(0.9));
        assert_eq!(model.frequency_penalty, Some(0.4));
        assert_eq!(model.presence_penalty, Some(0.2));
        assert_eq!(model.seed, Some(99));
        assert_eq!(model.stop, Some(vec!["STOP".to_string()]));
        assert_eq!(model.max_retries, 5);
        assert!(model.streaming);
    }

    #[test]
    fn test_builder_requires_endpoint() {
        let result = ChatAzureOpenAI::builder()
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("azure_endpoint") || err.contains("AZURE_OPENAI_ENDPOINT"));
    }

    #[test]
    fn test_builder_requires_deployment_name() {
        let result = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .api_key("test-key")
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("deployment_name"));
    }

    #[test]
    fn test_builder_requires_auth() {
        let result = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("api_key") || err.contains("azure_ad_token"));
    }

    #[test]
    fn test_llm_type() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(model.llm_type(), "azure_openai");
    }

    #[test]
    fn test_parse_stream_event_content() {
        let event = json!({
            "id": "chatcmpl-azure-789",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        });

        let chunk = ChatAzureOpenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.text, "Hello");
        assert_eq!(chunk.message.base.content.text(), "Hello");
    }

    #[test]
    fn test_parse_stream_event_tool_call() {
        let event = json!({
            "id": "chatcmpl-azure-stream",
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

        let chunk = ChatAzureOpenAI::parse_stream_event(&event).unwrap();
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
    }

    #[test]
    fn test_parse_stream_event_finish() {
        let event = json!({
            "id": "chatcmpl-azure-fin",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });

        let chunk = ChatAzureOpenAI::parse_stream_event(&event).unwrap();
        assert_eq!(chunk.message.chunk_position, Some("last".to_string()));
        assert_eq!(
            chunk.message.base.response_metadata.get("finish_reason"),
            Some(&json!("stop"))
        );
    }

    #[test]
    fn test_format_messages() {
        let messages = vec![
            Message::System(SystemMessage::new("You are helpful")),
            Message::Human(HumanMessage::new("Hello")),
        ];
        let api_messages = ChatAzureOpenAI::format_messages(&messages);

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
        let api_messages = ChatAzureOpenAI::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "assistant");
        assert_eq!(api_messages[0]["content"], "Let me search");

        let tool_calls = api_messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_123");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "search");
    }

    #[test]
    fn test_format_messages_with_tool_result() {
        let messages = vec![Message::Tool(ToolMessage::new(
            "Search results here",
            "call_123",
        ))];
        let api_messages = ChatAzureOpenAI::format_messages(&messages);

        assert_eq!(api_messages.len(), 1);
        assert_eq!(api_messages[0]["role"], "tool");
        assert_eq!(api_messages[0]["tool_call_id"], "call_123");
        assert_eq!(api_messages[0]["content"], "Search results here");
    }

    #[test]
    fn test_payload_with_tools_and_stream() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
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
        let payload = model.build_payload(&messages, None, &tools, true);

        assert!(payload.get("tools").is_some());
        assert_eq!(payload["tools"].as_array().unwrap().len(), 1);
        assert_eq!(payload["stream"], true);
    }

    #[test]
    fn test_profile() {
        let model = ChatAzureOpenAI::builder()
            .azure_endpoint("https://myinstance.openai.azure.com")
            .deployment_name("gpt-4o")
            .api_key("test-key")
            .build()
            .unwrap();

        let profile = model.profile();
        assert_eq!(profile.tool_calling, Some(true));
        assert_eq!(profile.structured_output, Some(true));
        assert_eq!(profile.text_inputs, Some(true));
        assert_eq!(profile.text_outputs, Some(true));
        assert_eq!(profile.image_inputs, Some(true));
    }
}
