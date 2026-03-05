//! Ollama chat model implementation.
//!
//! Provides [`ChatOllama`], a production-ready implementation of
//! [`BaseChatModel`] for the Ollama local LLM server API.
//!
//! Ollama runs locally and requires no authentication.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
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

/// Builder for constructing a [`ChatOllama`] instance.
#[derive(Debug)]
pub struct ChatOllamaBuilder {
    model: Option<String>,
    base_url: Option<String>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    num_ctx: Option<u32>,
    num_predict: Option<u32>,
    repeat_penalty: Option<f64>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
    format: Option<Value>,
    keep_alive: Option<String>,
    streaming: Option<bool>,
}

impl ChatOllamaBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            base_url: None,
            temperature: None,
            top_p: None,
            top_k: None,
            num_ctx: None,
            num_predict: None,
            repeat_penalty: None,
            seed: None,
            stop: None,
            format: None,
            keep_alive: None,
            streaming: None,
        }
    }

    /// Set the model name (required, e.g. "llama3.2").
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the base URL (default: `http://localhost:11434`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
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

    /// Set the context window size.
    pub fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    /// Set the maximum number of tokens to predict.
    pub fn num_predict(mut self, num_predict: u32) -> Self {
        self.num_predict = Some(num_predict);
        self
    }

    /// Set the repeat penalty.
    pub fn repeat_penalty(mut self, repeat_penalty: f64) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
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

    /// Set the output format (e.g. `"json"` or a JSON schema value).
    pub fn format(mut self, format: Value) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the keep-alive duration (e.g. "5m", "10s").
    pub fn keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Enable or disable streaming by default.
    pub fn streaming(mut self, streaming: bool) -> Self {
        self.streaming = Some(streaming);
        self
    }

    /// Build the [`ChatOllama`] instance.
    ///
    /// Returns an error if `model` is not set.
    pub fn build(self) -> Result<ChatOllama> {
        let model = self.model.ok_or_else(|| {
            RustChainError::Other("model is required for ChatOllama".into())
        })?;

        Ok(ChatOllama {
            model,
            base_url: self
                .base_url
                .unwrap_or_else(|| "http://localhost:11434".into()),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            num_ctx: self.num_ctx,
            num_predict: self.num_predict,
            repeat_penalty: self.repeat_penalty,
            seed: self.seed,
            stop: self.stop,
            format: self.format,
            keep_alive: self.keep_alive,
            streaming: self.streaming.unwrap_or(false),
            client: Client::new(),
            bound_tools: Vec::new(),
            tool_choice: None,
        })
    }
}

impl Default for ChatOllamaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Ollama chat model.
///
/// Implements the Ollama `/api/chat` endpoint for local LLM inference,
/// including tool calling and NDJSON streaming.
///
/// No authentication is required since Ollama runs locally.
///
/// # Example
///
/// ```no_run
/// use rustchain::chat_models::ollama::ChatOllama;
///
/// let model = ChatOllama::builder()
///     .model("llama3.2")
///     .build()
///     .unwrap();
/// ```
pub struct ChatOllama {
    /// The model identifier (e.g. "llama3.2").
    pub model: String,
    /// Base URL for the Ollama server.
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Context window size.
    pub num_ctx: Option<u32>,
    /// Maximum tokens to predict.
    pub num_predict: Option<u32>,
    /// Repeat penalty.
    pub repeat_penalty: Option<f64>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    /// Output format (e.g. `"json"` or a JSON schema).
    pub format: Option<Value>,
    /// Keep-alive duration for the model in memory.
    pub keep_alive: Option<String>,
    /// Whether streaming is the default mode.
    pub streaming: bool,
    /// HTTP client.
    client: Client,
    /// Tools bound via `bind_tools`.
    bound_tools: Vec<Value>,
    /// Tool choice configuration (reserved for future use).
    #[allow(dead_code)]
    tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for ChatOllama {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatOllama")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.temperature)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl ChatOllama {
    /// Returns a new builder for `ChatOllama`.
    pub fn builder() -> ChatOllamaBuilder {
        ChatOllamaBuilder::new()
    }

    /// Format messages for the Ollama chat API.
    ///
    /// Uses the same role format as OpenAI: system, user, assistant, tool.
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
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.args
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

    /// Build the JSON payload for the Ollama `/api/chat` endpoint.
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
            "stream": stream,
        });

        // Build options object for model parameters
        let mut options = json!({});
        let mut has_options = false;

        if let Some(temp) = self.temperature {
            options["temperature"] = json!(temp);
            has_options = true;
        }
        if let Some(tp) = self.top_p {
            options["top_p"] = json!(tp);
            has_options = true;
        }
        if let Some(tk) = self.top_k {
            options["top_k"] = json!(tk);
            has_options = true;
        }
        if let Some(nc) = self.num_ctx {
            options["num_ctx"] = json!(nc);
            has_options = true;
        }
        if let Some(np) = self.num_predict {
            options["num_predict"] = json!(np);
            has_options = true;
        }
        if let Some(rp) = self.repeat_penalty {
            options["repeat_penalty"] = json!(rp);
            has_options = true;
        }
        if let Some(s) = self.seed {
            options["seed"] = json!(s);
            has_options = true;
        }

        // Merge stop sequences
        let mut all_stop = Vec::new();
        if let Some(configured) = &self.stop {
            all_stop.extend(configured.iter().cloned());
        }
        if let Some(stop_param) = stop {
            all_stop.extend(stop_param.iter().cloned());
        }
        if !all_stop.is_empty() {
            options["stop"] = json!(all_stop);
            has_options = true;
        }

        if has_options {
            payload["options"] = options;
        }

        if let Some(ref fmt) = self.format {
            payload["format"] = fmt.clone();
        }

        if let Some(ref ka) = self.keep_alive {
            payload["keep_alive"] = json!(ka);
        }

        if !tools.is_empty() {
            payload["tools"] = json!(tools);
        }

        payload
    }

    /// Parse an Ollama chat API response into a [`ChatResult`].
    pub fn parse_response(response: &Value) -> Result<ChatResult> {
        let message = response.get("message").ok_or_else(|| {
            RustChainError::Other("Missing 'message' in Ollama response".into())
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
                let args: HashMap<String, Value> = function
                    .get("arguments")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                tool_calls.push(ToolCall {
                    name,
                    args,
                    id: None,
                });
            }
        }

        // Parse usage metadata from Ollama fields
        let eval_count = response
            .get("eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let prompt_eval_count = response
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let usage_metadata = if eval_count > 0 || prompt_eval_count > 0 {
            Some(UsageMetadata::new(
                prompt_eval_count,
                eval_count,
                prompt_eval_count + eval_count,
            ))
        } else {
            None
        };

        let mut ai_message = AIMessage::new(&content);
        ai_message.tool_calls = tool_calls;
        ai_message.usage_metadata = usage_metadata;

        let generation = ChatGeneration::new(ai_message);

        Ok(ChatResult {
            generations: vec![generation],
            llm_output: None,
        })
    }

    /// Parse a single NDJSON stream line into a [`ChatGenerationChunk`].
    ///
    /// Ollama streams newline-delimited JSON. Each line is a complete JSON
    /// object. The final line has `"done": true` with usage statistics.
    pub fn parse_stream_line(line: &Value) -> Option<ChatGenerationChunk> {
        let done = line.get("done").and_then(|v| v.as_bool()).unwrap_or(false);

        if done {
            // Final line with usage stats
            let eval_count = line
                .get("eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let prompt_eval_count = line
                .get("prompt_eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            let mut chunk = AIMessageChunk::new("");
            chunk.chunk_position = Some("last".to_string());

            if eval_count > 0 || prompt_eval_count > 0 {
                chunk.usage_metadata = Some(UsageMetadata::new(
                    prompt_eval_count,
                    eval_count,
                    prompt_eval_count + eval_count,
                ));
            }

            return Some(ChatGenerationChunk::new(chunk));
        }

        let message = line.get("message")?;

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let mut chunk = AIMessageChunk::new(content);

        // Parse tool call chunks if present
        if let Some(tcs) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for (i, tc) in tcs.iter().enumerate() {
                let function = tc.get("function");
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let args = function
                    .and_then(|f| f.get("arguments"))
                    .map(|v| v.to_string());
                chunk.tool_call_chunks.push(ToolCallChunk {
                    name,
                    args,
                    id: None,
                    index: Some(i),
                });
            }
        }

        // Skip empty content with no tool calls
        if content.is_empty() && chunk.tool_call_chunks.is_empty() {
            return None;
        }

        Some(ChatGenerationChunk::new(chunk))
    }

    /// Call the Ollama chat API with simple retry logic.
    ///
    /// Retries on connection errors since the local server might be starting up.
    async fn call_api(&self, payload: &Value) -> Result<Value> {
        let url = format!("{}/api/chat", self.base_url);
        let max_retries = 2u32;

        let mut last_error = RustChainError::Other("No attempts made".into());

        for attempt in 0..=max_retries {
            let result = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(payload)
                .send()
                .await;

            match result {
                Ok(response) => {
                    let status = response.status().as_u16();
                    if (200..300).contains(&status) {
                        let body: Value = response.json().await.map_err(|e| {
                            RustChainError::Other(format!(
                                "Failed to parse response JSON: {}",
                                e
                            ))
                        })?;
                        return Ok(body);
                    }

                    let body = response.text().await.unwrap_or_default();
                    return Err(RustChainError::HttpError { status, body });
                }
                Err(e) => {
                    // Retry on connection errors (server might be starting up)
                    if attempt < max_retries {
                        let delay_ms = 500 * 2u64.pow(attempt);
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        last_error =
                            RustChainError::Other(format!("HTTP request failed: {}", e));
                        continue;
                    }
                    return Err(RustChainError::Other(format!(
                        "HTTP request failed: {}",
                        e
                    )));
                }
            }
        }

        Err(last_error)
    }

    /// Call the Ollama chat API with streaming, returning an async stream
    /// of NDJSON objects.
    ///
    /// Ollama streams newline-delimited JSON (not SSE). Each line is a
    /// complete JSON object terminated by a newline.
    async fn call_api_stream(
        &self,
        payload: &Value,
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<Value>> + Send>>> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await
            .map_err(|e| {
                RustChainError::Other(format!("HTTP request failed: {}", e))
            })?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(RustChainError::HttpError { status, body });
        }

        let byte_stream = response.bytes_stream();

        // Parse NDJSON: each line is a complete JSON object
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

                        // Split by newlines — each line is a JSON object
                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].trim().to_string();
                            *buffer = buffer[pos + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            match serde_json::from_str::<Value>(&line) {
                                Ok(val) => events.push(Ok(val)),
                                Err(e) => events.push(Err(RustChainError::Other(
                                    format!("Failed to parse NDJSON line: {}", e),
                                ))),
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

    /// Convert a [`ToolSchema`] to the OpenAI-compatible tool format used by Ollama.
    fn tool_schema_to_ollama(schema: &ToolSchema) -> Value {
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
impl BaseChatModel for ChatOllama {
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
        "ollama"
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
                Ok(event) => ChatOllama::parse_stream_line(&event).map(Ok),
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
        let bound_tools: Vec<Value> = tools.iter().map(Self::tool_schema_to_ollama).collect();

        Ok(Box::new(ChatOllama {
            model: self.model.clone(),
            base_url: self.base_url.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            num_ctx: self.num_ctx,
            num_predict: self.num_predict,
            repeat_penalty: self.repeat_penalty,
            seed: self.seed,
            stop: self.stop.clone(),
            format: self.format.clone(),
            keep_alive: self.keep_alive.clone(),
            streaming: self.streaming,
            client: self.client.clone(),
            bound_tools,
            tool_choice,
        }))
    }

    fn profile(&self) -> ModelProfile {
        ModelProfile {
            tool_calling: Some(true),
            text_inputs: Some(true),
            text_outputs: Some(true),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{HumanMessage, SystemMessage};

    #[test]
    fn test_ollama_config_builder() {
        let model = ChatOllama::builder()
            .model("llama3.2")
            .base_url("http://localhost:11434")
            .temperature(0.7)
            .top_p(0.9)
            .top_k(40)
            .num_ctx(4096)
            .num_predict(512)
            .repeat_penalty(1.1)
            .seed(42)
            .stop(vec!["STOP".to_string()])
            .format(json!("json"))
            .keep_alive("5m")
            .streaming(true)
            .build()
            .unwrap();

        assert_eq!(model.model, "llama3.2");
        assert_eq!(model.base_url, "http://localhost:11434");
        assert_eq!(model.temperature, Some(0.7));
        assert_eq!(model.top_p, Some(0.9));
        assert_eq!(model.top_k, Some(40));
        assert_eq!(model.num_ctx, Some(4096));
        assert_eq!(model.num_predict, Some(512));
        assert_eq!(model.repeat_penalty, Some(1.1));
        assert_eq!(model.seed, Some(42));
        assert_eq!(model.stop, Some(vec!["STOP".to_string()]));
        assert_eq!(model.format, Some(json!("json")));
        assert_eq!(model.keep_alive, Some("5m".to_string()));
        assert!(model.streaming);
    }

    #[test]
    fn test_builder_requires_model() {
        let result = ChatOllama::builder().build();
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
        let api_messages = ChatOllama::format_messages(&messages);

        assert_eq!(api_messages.len(), 2);
        assert_eq!(api_messages[0]["role"], "system");
        assert_eq!(api_messages[0]["content"], "You are helpful");
        assert_eq!(api_messages[1]["role"], "user");
        assert_eq!(api_messages[1]["content"], "Hello");
    }

    #[test]
    fn test_parse_response_text() {
        let response = json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello, world!"
            },
            "done": true,
            "eval_count": 5,
            "prompt_eval_count": 10
        });

        let result = ChatOllama::parse_response(&response).unwrap();
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
    fn test_parse_response_with_tool_calls() {
        let response = json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "rust programming"}
                    }
                }]
            },
            "done": true,
            "eval_count": 15,
            "prompt_eval_count": 20
        });

        let result = ChatOllama::parse_response(&response).unwrap();
        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "web_search");
            assert_eq!(
                ai.tool_calls[0].args.get("query"),
                Some(&json!("rust programming"))
            );
            assert!(ai.tool_calls[0].id.is_none());
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[test]
    fn test_parse_stream_line() {
        let line = json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "Hello"
            },
            "done": false
        });

        let chunk = ChatOllama::parse_stream_line(&line).unwrap();
        assert_eq!(chunk.text, "Hello");
        assert_eq!(chunk.message.base.content.text(), "Hello");
    }

    #[test]
    fn test_parse_stream_line_done() {
        let line = json!({
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": true,
            "eval_count": 42,
            "prompt_eval_count": 10
        });

        let chunk = ChatOllama::parse_stream_line(&line).unwrap();
        assert_eq!(chunk.text, "");
        assert_eq!(chunk.message.chunk_position, Some("last".to_string()));
        let usage = chunk.message.usage_metadata.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.total_tokens, 52);
    }

    #[test]
    fn test_build_payload() {
        let model = ChatOllama::builder()
            .model("llama3.2")
            .temperature(0.5)
            .top_p(0.9)
            .top_k(40)
            .num_ctx(4096)
            .num_predict(512)
            .repeat_penalty(1.1)
            .seed(42)
            .stop(vec!["END".to_string()])
            .keep_alive("10m")
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[], false);

        assert_eq!(payload["model"], "llama3.2");
        assert_eq!(payload["stream"], false);
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
        assert_eq!(payload["keep_alive"], "10m");

        let options = &payload["options"];
        assert_eq!(options["temperature"], 0.5);
        assert_eq!(options["top_p"], 0.9);
        assert_eq!(options["top_k"], 40);
        assert_eq!(options["num_ctx"], 4096);
        assert_eq!(options["num_predict"], 512);
        assert_eq!(options["repeat_penalty"], 1.1);
        assert_eq!(options["seed"], 42);
        assert_eq!(options["stop"], json!(["END"]));

        assert!(payload.get("tools").is_none());
        assert!(payload.get("format").is_none());
    }

    #[test]
    fn test_build_payload_with_format() {
        let model = ChatOllama::builder()
            .model("llama3.2")
            .format(json!("json"))
            .build()
            .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Hi"))];
        let payload = model.build_payload(&messages, None, &[], false);

        assert_eq!(payload["format"], "json");
    }
}
