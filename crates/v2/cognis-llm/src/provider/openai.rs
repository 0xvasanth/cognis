//! OpenAI Chat Completions provider.

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta, Usage};
use crate::tools::ToolDefinition;
use crate::{AiMessage, Message, ToolCall};

use super::{LLMProvider, Provider};

const DEFAULT_BASE: &str = "https://api.openai.com/v1/";

/// OpenAI provider configuration.
#[derive(Debug)]
pub struct OpenAIProvider {
    base_url: String,
    api_key: SecretString,
    model: String,
    organization: Option<String>,
    extra_headers: Vec<(String, String)>,
    http: reqwest::Client,
}

impl OpenAIProvider {
    /// Build with explicit config.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder().api_key(api_key).build().expect("default OpenAI build")
    }

    /// Fluent builder.
    pub fn builder() -> OpenAIBuilder {
        OpenAIBuilder::default()
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .map_err(|e| CognisError::Configuration(format!("invalid api key: {e}")))?,
        );
        if let Some(org) = &self.organization {
            h.insert(
                HeaderName::from_static("openai-organization"),
                HeaderValue::from_str(org)
                    .map_err(|e| CognisError::Configuration(format!("invalid org: {e}")))?,
            );
        }
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
        let openai_messages: Vec<serde_json::Value> =
            messages.iter().map(message_to_openai).collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": openai_messages,
            "stream": stream,
        });

        if !tools.is_empty() {
            body["tools"] = tools_to_openai_format(tools);
        }
        if let Some(t) = opts.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = opts.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(m) = opts.max_tokens {
            body["max_tokens"] = serde_json::json!(m);
        }
        if !opts.stop.is_empty() {
            body["stop"] = serde_json::json!(opts.stop);
        }
        if let Some(p) = opts.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(p);
        }
        if let Some(p) = opts.presence_penalty {
            body["presence_penalty"] = serde_json::json!(p);
        }

        body
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }
    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.chat_completion_with_tools(messages, Vec::new(), opts).await
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
            .post(self.endpoint("chat/completions"))
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

        let raw: OpenAIChatResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "openai".into(),
            message: format!("response decode: {e}"),
        })?;

        let choice = raw.choices.into_iter().next().ok_or_else(|| CognisError::Provider {
            provider: "openai".into(),
            message: "no choices in response".into(),
        })?;

        let message = openai_message_to_cognis(choice.message);
        Ok(ChatResponse {
            message,
            usage: raw.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            finish_reason: choice.finish_reason.unwrap_or_else(|| "stop".into()),
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
            .post(self.endpoint("chat/completions"))
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
                    Ok(bytes) => Some(parse_sse_chunk(&bytes)),
                    Err(e) => Some(Err(CognisError::Network {
                        status_code: None,
                        message: e.to_string(),
                    })),
                }
            })
            .filter_map(|res| async move {
                match res {
                    Ok(Some(chunk)) => Some(Ok(chunk)),
                    Ok(None) => None, // [DONE] marker
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(RunnableStream::new(chunk_stream))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();
        let resp = self
            .http
            .get(self.endpoint("models"))
            .headers(self.headers()?)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::Healthy {
                latency_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(r) => Ok(HealthStatus::Degraded {
                reason: format!("models endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(HealthStatus::Unhealthy { reason: e.to_string() }),
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for `OpenAIProvider`.
#[derive(Default)]
pub struct OpenAIBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    organization: Option<String>,
    extra_headers: Vec<(String, String)>,
    timeout_secs: Option<u64>,
}

impl OpenAIBuilder {
    /// Set the API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }

    /// Override the base URL (e.g. for OpenAI-compatible proxies).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the default model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }

    /// Set the OpenAI-Organization header.
    pub fn organization(mut self, o: impl Into<String>) -> Self {
        self.organization = Some(o.into());
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
    pub fn build(self) -> Result<OpenAIProvider> {
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("OpenAI: API key required".into()))?;
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(std::time::Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(OpenAIProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into_boxed_str()),
            model: self
                .model
                .unwrap_or_else(|| Provider::OpenAI.default_model().to_string()),
            organization: self.organization,
            extra_headers: self.extra_headers,
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OpenAIChatResponse {
    model: String,
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct OpenAIMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCall>,
}

#[derive(Deserialize, Serialize, Clone)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type", default)]
    kind: String,
    function: OpenAIFunctionCall,
}

#[derive(Deserialize, Serialize, Clone)]
struct OpenAIFunctionCall {
    name: String,
    /// OpenAI returns this as a JSON-encoded *string*.
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

fn message_to_openai(m: &Message) -> serde_json::Value {
    match m {
        Message::Human(h) => serde_json::json!({"role": "user", "content": h.content}),
        Message::Ai(a) => {
            let mut v = serde_json::json!({"role": "assistant", "content": a.content});
            if !a.tool_calls.is_empty() {
                v["tool_calls"] = serde_json::json!(a
                    .tool_calls
                    .iter()
                    .map(|tc| serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string(),
                        }
                    }))
                    .collect::<Vec<_>>());
            }
            v
        }
        Message::System(s) => serde_json::json!({"role": "system", "content": s.content}),
        Message::Tool(t) => serde_json::json!({
            "role": "tool",
            "tool_call_id": t.tool_call_id,
            "content": t.content,
        }),
    }
}

fn openai_message_to_cognis(m: OpenAIMessage) -> Message {
    let content = m.content.unwrap_or_default();
    let tool_calls = m
        .tool_calls
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Null),
        })
        .collect();
    Message::Ai(AiMessage { content, tool_calls })
}

fn tools_to_openai_format(tools: &[ToolDefinition]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters.clone().unwrap_or(serde_json::json!({"type": "object"})),
                }
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn parse_sse_chunk(bytes: &[u8]) -> Result<Option<StreamChunk>> {
    // OpenAI streams SSE lines like "data: {json}\n\n" plus "data: [DONE]\n\n".
    let s = std::str::from_utf8(bytes).map_err(|e| CognisError::Provider {
        provider: "openai".into(),
        message: format!("invalid UTF-8 in stream: {e}"),
    })?;
    for line in s.lines() {
        let line = line.trim();
        if let Some(payload) = line.strip_prefix("data: ") {
            if payload == "[DONE]" {
                return Ok(None);
            }
            let v: serde_json::Value =
                serde_json::from_str(payload).map_err(|e| CognisError::Provider {
                    provider: "openai".into(),
                    message: format!("stream parse: {e}"),
                })?;
            // Pull delta content + tool-call deltas
            let delta = &v["choices"][0]["delta"];
            let content = delta["content"].as_str().unwrap_or("").to_string();
            let mut tool_calls_delta = Vec::new();
            if let Some(arr) = delta["tool_calls"].as_array() {
                for (i, t) in arr.iter().enumerate() {
                    tool_calls_delta.push(ToolCallDelta {
                        index: t["index"].as_u64().unwrap_or(i as u64) as u32,
                        id: t["id"].as_str().map(|s| s.to_string()),
                        name: t["function"]["name"].as_str().map(|s| s.to_string()),
                        arguments_delta: t["function"]["arguments"]
                            .as_str()
                            .map(|s| s.to_string()),
                    });
                }
            }
            let finish_reason =
                v["choices"][0]["finish_reason"].as_str().map(|s| s.to_string());
            let is_done = finish_reason.is_some();
            return Ok(Some(StreamChunk {
                content,
                is_delta: true,
                is_done,
                finish_reason,
                usage: None,
                tool_calls_delta,
            }));
        }
    }
    // No data line in this chunk — return an empty no-op chunk.
    Ok(Some(StreamChunk::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_to_openai_human() {
        let m = Message::human("hi");
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hi");
    }

    #[test]
    fn message_to_openai_ai_with_tool_calls() {
        let m = Message::Ai(AiMessage {
            content: "calling tool".into(),
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
        });
        let v = message_to_openai(&m);
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["tool_calls"][0]["id"], "c1");
        assert_eq!(v["tool_calls"][0]["function"]["name"], "search");
    }

    #[test]
    fn openai_message_to_cognis_parses_args_string() {
        let m = OpenAIMessage {
            role: "assistant".into(),
            content: Some("ok".into()),
            tool_calls: vec![OpenAIToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: OpenAIFunctionCall {
                    name: "search".into(),
                    arguments: "{\"q\":\"rust\"}".into(),
                },
            }],
        };
        let cognis = openai_message_to_cognis(m);
        if let Message::Ai(a) = cognis {
            assert_eq!(a.content, "ok");
            assert_eq!(a.tool_calls.len(), 1);
            assert_eq!(a.tool_calls[0].arguments["q"], "rust");
        } else {
            panic!("expected Ai");
        }
    }

    #[test]
    fn parse_sse_done_sentinel() {
        let bytes = b"data: [DONE]\n\n";
        let r = parse_sse_chunk(bytes).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn parse_sse_content_chunk() {
        let bytes = br#"data: {"choices":[{"delta":{"content":"hello"}}]}
"#;
        let r = parse_sse_chunk(bytes).unwrap().unwrap();
        assert_eq!(r.content, "hello");
        assert!(!r.is_done);
    }

    #[test]
    fn builder_requires_api_key() {
        let err = OpenAIBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let p = OpenAIBuilder::default().api_key("sk-test").model("gpt-4o").build().unwrap();
        assert_eq!(p.name(), "openai");
        assert_eq!(p.provider_type(), Provider::OpenAI);
    }
}
