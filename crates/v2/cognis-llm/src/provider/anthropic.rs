//! Anthropic Claude provider (Messages API).
//!
//! Notable differences from OpenAI's wire format:
//! - No `system` role; the system prompt is a top-level field.
//! - Tool use surfaces as content blocks of type `tool_use` / `tool_result`,
//!   not as a separate `tool_calls` field.
//! - Auth header is `x-api-key`, plus a required `anthropic-version` header.
//! - Streaming is SSE with named events (`content_block_delta`, etc.)
//!   instead of the `[DONE]` sentinel.

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta, Usage};
use crate::tools::ToolDefinition;
use crate::{AiMessage, Message, ToolCall};

use super::{LLMProvider, Provider};

const DEFAULT_BASE: &str = "https://api.anthropic.com/v1/";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic provider configuration.
#[derive(Debug)]
pub struct AnthropicProvider {
    base_url: String,
    api_key: SecretString,
    model: String,
    anthropic_version: String,
    extra_headers: Vec<(String, String)>,
    http: reqwest::Client,
}

impl AnthropicProvider {
    /// Build with explicit config.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder()
            .api_key(api_key)
            .build()
            .expect("default Anthropic build")
    }

    /// Fluent builder.
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    fn endpoint(&self, path: &str) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str(path);
        s
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(self.api_key.expose_secret())
                .map_err(|e| CognisError::Configuration(format!("invalid api key: {e}")))?,
        );
        h.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_str(&self.anthropic_version)
                .map_err(|e| CognisError::Configuration(format!("invalid version: {e}")))?,
        );
        for (k, v) in &self.extra_headers {
            let name = HeaderName::from_bytes(k.as_bytes())
                .map_err(|e| CognisError::Configuration(format!("bad header `{k}`: {e}")))?;
            let val = HeaderValue::from_str(v)
                .map_err(|e| CognisError::Configuration(format!("bad header value: {e}")))?;
            h.insert(name, val);
        }
        Ok(h)
    }

    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        opts: &ChatOptions,
        stream: bool,
    ) -> serde_json::Value {
        let model = opts.model.as_deref().unwrap_or(&self.model);
        let (system, anthropic_messages) = split_system_and_messages(messages);
        // Anthropic requires max_tokens; default to a reasonable cap.
        let max_tokens = opts.max_tokens.unwrap_or(4096);

        let mut body = serde_json::json!({
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": stream,
        });
        if let Some(s) = system {
            body["system"] = serde_json::Value::String(s);
        }
        if !tools.is_empty() {
            body["tools"] = tools_to_anthropic_format(tools);
        }
        if let Some(t) = opts.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = opts.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if !opts.stop.is_empty() {
            body["stop_sequences"] = serde_json::json!(opts.stop);
        }
        body
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }
    fn provider_type(&self) -> Provider {
        Provider::Anthropic
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.chat_completion_with_tools(messages, Vec::new(), opts)
            .await
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_request(&messages, &tools, &opts, false);
        let resp = self
            .http
            .post(self.endpoint("messages"))
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Network {
                status_code: None,
                message: e.to_string(),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(CognisError::Network {
                status_code: Some(status.as_u16()),
                message: txt,
            });
        }

        let raw: AnthropicResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "anthropic".into(),
            message: format!("response decode: {e}"),
        })?;

        let message = anthropic_response_to_cognis(&raw);
        Ok(ChatResponse {
            message,
            usage: Some(Usage {
                prompt_tokens: raw.usage.input_tokens,
                completion_tokens: raw.usage.output_tokens,
                total_tokens: raw.usage.input_tokens + raw.usage.output_tokens,
            }),
            finish_reason: raw.stop_reason.unwrap_or_else(|| "end_turn".into()),
            model: raw.model,
        })
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        let body = self.build_request(&messages, &[], &opts, true);
        let resp = self
            .http
            .post(self.endpoint("messages"))
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Network {
                status_code: None,
                message: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(CognisError::Network {
                status_code: Some(resp.status().as_u16()),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let byte_stream = resp.bytes_stream();
        let chunk_stream = byte_stream
            .filter_map(|res| async move {
                match res {
                    Ok(bytes) => Some(parse_anthropic_sse(&bytes)),
                    Err(e) => Some(Err(CognisError::Network {
                        status_code: None,
                        message: e.to_string(),
                    })),
                }
            })
            .filter_map(|res| async move {
                match res {
                    Ok(Some(chunk)) => Some(Ok(chunk)),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(RunnableStream::new(chunk_stream))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        // Anthropic has no public health endpoint; do a minimal-cost
        // round-trip via the Messages API with 1 max_tokens.
        let start = Instant::now();
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}],
        });
        let resp = self
            .http
            .post(self.endpoint("messages"))
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::Healthy {
                latency_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(r) => Ok(HealthStatus::Degraded {
                reason: format!("messages endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(HealthStatus::Unhealthy {
                reason: e.to_string(),
            }),
        }
    }
}

/// Fluent builder for `AnthropicProvider`.
#[derive(Default)]
pub struct AnthropicBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    anthropic_version: Option<String>,
    extra_headers: Vec<(String, String)>,
    timeout_secs: Option<u64>,
}

impl AnthropicBuilder {
    /// Set the API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Override the base URL.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Set the default model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    /// Override the `anthropic-version` header (default: `2023-06-01`).
    pub fn anthropic_version(mut self, v: impl Into<String>) -> Self {
        self.anthropic_version = Some(v.into());
        self
    }
    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
    /// Set HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Construct.
    pub fn build(self) -> Result<AnthropicProvider> {
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("Anthropic: API key required".into()))?;
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(std::time::Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(AnthropicProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into_boxed_str()),
            model: self
                .model
                .unwrap_or_else(|| Provider::Anthropic.default_model().to_string()),
            anthropic_version: self
                .anthropic_version
                .unwrap_or_else(|| ANTHROPIC_VERSION.to_string()),
            extra_headers: self.extra_headers,
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AnthropicResponse {
    model: String,
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Catch-all for forward-compat (image, document, etc.).
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn split_system_and_messages(messages: &[Message]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system_parts = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m {
            Message::System(s) => system_parts.push(s.content.clone()),
            Message::Human(h) => {
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if !h.content.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": h.content}));
                }
                for p in &h.parts {
                    blocks.push(p.to_anthropic());
                }
                out.push(serde_json::json!({
                    "role": "user",
                    "content": blocks,
                }));
            }
            Message::Ai(a) => {
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if !a.content.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": a.content}));
                }
                for p in &a.parts {
                    blocks.push(p.to_anthropic());
                }
                for tc in &a.tool_calls {
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                out.push(serde_json::json!({
                    "role": "assistant",
                    "content": blocks,
                }));
            }
            Message::Tool(t) => out.push(serde_json::json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": t.tool_call_id,
                    "content": t.content,
                }],
            })),
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, out)
}

fn anthropic_response_to_cognis(resp: &AnthropicResponse) -> Message {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &resp.content {
        match block {
            AnthropicContentBlock::Text { text } => text_parts.push(text.clone()),
            AnthropicContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: input.clone(),
            }),
            AnthropicContentBlock::Other => {}
        }
    }
    Message::Ai(AiMessage {
        content: text_parts.join(""),
        tool_calls,
        parts: Vec::new(),
    })
}

fn tools_to_anthropic_format(tools: &[ToolDefinition]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters.clone().unwrap_or_else(
                    || serde_json::json!({"type": "object", "properties": {}})
                ),
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn parse_anthropic_sse(bytes: &[u8]) -> Result<Option<StreamChunk>> {
    let s = std::str::from_utf8(bytes).map_err(|e| CognisError::Provider {
        provider: "anthropic".into(),
        message: format!("invalid UTF-8 in stream: {e}"),
    })?;

    let mut event: Option<&str> = None;
    let mut data: Option<&str> = None;
    for line in s.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("event: ") {
            event = Some(rest);
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data = Some(rest);
        }
    }
    let (Some(event), Some(data)) = (event, data) else {
        return Ok(Some(StreamChunk::default()));
    };

    let v: serde_json::Value = serde_json::from_str(data).map_err(|e| CognisError::Provider {
        provider: "anthropic".into(),
        message: format!("stream parse: {e}"),
    })?;

    match event {
        "content_block_delta" => {
            let delta = &v["delta"];
            let kind = delta["type"].as_str().unwrap_or("");
            let mut chunk = StreamChunk {
                is_delta: true,
                ..Default::default()
            };
            match kind {
                "text_delta" => {
                    chunk.content = delta["text"].as_str().unwrap_or("").to_string();
                }
                "input_json_delta" => {
                    let idx = v["index"].as_u64().unwrap_or(0) as u32;
                    chunk.tool_calls_delta.push(ToolCallDelta {
                        index: idx,
                        id: None,
                        name: None,
                        arguments_delta: delta["partial_json"].as_str().map(|s| s.to_string()),
                    });
                }
                _ => {}
            }
            Ok(Some(chunk))
        }
        "content_block_start" => {
            let block = &v["content_block"];
            if block["type"] == "tool_use" {
                let idx = v["index"].as_u64().unwrap_or(0) as u32;
                let mut chunk = StreamChunk {
                    is_delta: true,
                    ..Default::default()
                };
                chunk.tool_calls_delta.push(ToolCallDelta {
                    index: idx,
                    id: block["id"].as_str().map(|s| s.to_string()),
                    name: block["name"].as_str().map(|s| s.to_string()),
                    arguments_delta: None,
                });
                return Ok(Some(chunk));
            }
            Ok(Some(StreamChunk::default()))
        }
        "message_delta" => {
            let chunk = StreamChunk {
                is_delta: false,
                is_done: v["delta"]["stop_reason"].is_string(),
                finish_reason: v["delta"]["stop_reason"].as_str().map(|s| s.to_string()),
                usage: v["usage"].as_object().map(|u| Usage {
                    prompt_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                        as u32,
                    completion_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                        as u32,
                    total_tokens: u
                        .get("input_tokens")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0)
                        .saturating_add(
                            u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0),
                        ) as u32,
                }),
                ..Default::default()
            };
            Ok(Some(chunk))
        }
        "message_stop" => Ok(None),
        _ => Ok(Some(StreamChunk::default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_extracts_top_level() {
        let msgs = vec![
            Message::system("be terse"),
            Message::human("hi"),
            Message::system("also be polite"),
            Message::ai("ok"),
        ];
        let (sys, body) = split_system_and_messages(&msgs);
        assert_eq!(sys.as_deref(), Some("be terse\n\nalso be polite"));
        assert_eq!(body.len(), 2);
        assert_eq!(body[0]["role"], "user");
        assert_eq!(body[1]["role"], "assistant");
    }

    #[test]
    fn ai_with_tool_calls_emits_tool_use_blocks() {
        let m = Message::Ai(AiMessage {
            content: "calling".into(),
            tool_calls: vec![ToolCall {
                id: "tu_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
            parts: Vec::new(),
        });
        let (_sys, body) = split_system_and_messages(&[m]);
        let blocks = body[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "tu_1");
    }

    #[test]
    fn tool_message_renders_as_user_tool_result() {
        let m = Message::tool("tu_1", "result-text");
        let (_sys, body) = split_system_and_messages(&[m]);
        assert_eq!(body[0]["role"], "user");
        assert_eq!(body[0]["content"][0]["type"], "tool_result");
        assert_eq!(body[0]["content"][0]["tool_use_id"], "tu_1");
    }

    #[test]
    fn anthropic_response_combines_text_and_tool_use() {
        let resp = AnthropicResponse {
            model: "claude".into(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "thinking…".into(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "tu_a".into(),
                    name: "search".into(),
                    input: serde_json::json!({"q": "rust"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: AnthropicUsage {
                input_tokens: 7,
                output_tokens: 3,
            },
        };
        let m = anthropic_response_to_cognis(&resp);
        if let Message::Ai(a) = m {
            assert_eq!(a.content, "thinking…");
            assert_eq!(a.tool_calls.len(), 1);
            assert_eq!(a.tool_calls[0].name, "search");
        } else {
            panic!("expected Ai");
        }
    }

    #[test]
    fn parse_sse_text_delta() {
        let bytes = b"event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
        let chunk = parse_anthropic_sse(bytes).unwrap().unwrap();
        assert_eq!(chunk.content, "hi");
        assert!(!chunk.is_done);
    }

    #[test]
    fn parse_sse_message_stop_returns_none() {
        let bytes = b"event: message_stop\ndata: {}\n\n";
        let r = parse_anthropic_sse(bytes).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn builder_requires_api_key() {
        let err = AnthropicBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let p = AnthropicBuilder::default()
            .api_key("sk-ant-test")
            .build()
            .unwrap();
        assert_eq!(p.name(), "anthropic");
        assert_eq!(p.provider_type(), Provider::Anthropic);
    }
}
