//! Azure OpenAI provider.
//!
//! Wire format is identical to OpenAI's chat completions API, but:
//! - URLs are deployment-scoped:
//!   `https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=...`
//! - Auth uses an `api-key` header (not `Authorization: Bearer`).
//! - The "model" is the deployment name; we send it in the body for
//!   request-correlation but the URL path is what selects the deployment.

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta, Usage};
use crate::tools::ToolDefinition;
use crate::{AiMessage, Message, ToolCall};

use super::{LLMProvider, Provider};

const DEFAULT_API_VERSION: &str = "2024-08-01-preview";

/// Azure OpenAI provider configuration.
#[derive(Debug)]
pub struct AzureProvider {
    /// e.g. `https://my-resource.openai.azure.com/`
    endpoint: String,
    deployment: String,
    api_version: String,
    api_key: SecretString,
    extra_headers: Vec<(String, String)>,
    http: reqwest::Client,
}

impl AzureProvider {
    /// Fluent builder.
    pub fn builder() -> AzureBuilder {
        AzureBuilder::default()
    }

    fn url(&self, action: &str) -> String {
        let endpoint = self.endpoint.trim_end_matches('/');
        format!(
            "{endpoint}/openai/deployments/{deployment}/{action}?api-version={api_version}",
            deployment = self.deployment,
            api_version = self.api_version,
        )
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            HeaderName::from_static("api-key"),
            HeaderValue::from_str(self.api_key.expose_secret())
                .map_err(|e| CognisError::Configuration(format!("invalid api key: {e}")))?,
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
        let openai_messages: Vec<serde_json::Value> =
            messages.iter().map(message_to_openai).collect();
        let mut body = serde_json::json!({
            "messages": openai_messages,
            "stream": stream,
        });
        // Some Azure deployments allow a `model` override in body for routing;
        // include it only if the caller set one.
        if let Some(m) = &opts.model {
            body["model"] = serde_json::json!(m);
        }
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
impl LLMProvider for AzureProvider {
    fn name(&self) -> &str {
        "azure"
    }
    fn provider_type(&self) -> Provider {
        Provider::Azure
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
            .post(self.url("chat/completions"))
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
            provider: "azure".into(),
            message: format!("response decode: {e}"),
        })?;

        let choice = raw
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| CognisError::Provider {
                provider: "azure".into(),
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
            .post(self.url("chat/completions"))
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
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(RunnableStream::new(chunk_stream))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();
        // Azure has no list-models on the deployment URL; do a 1-token chat probe.
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
        });
        let resp = self
            .http
            .post(self.url("chat/completions"))
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::Healthy {
                latency_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(r) => Ok(HealthStatus::Degraded {
                reason: format!("chat probe returned {}", r.status()),
            }),
            Err(e) => Ok(HealthStatus::Unhealthy {
                reason: e.to_string(),
            }),
        }
    }
}

/// Fluent builder for `AzureProvider`.
#[derive(Default)]
pub struct AzureBuilder {
    endpoint: Option<String>,
    deployment: Option<String>,
    api_version: Option<String>,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    timeout_secs: Option<u64>,
}

impl AzureBuilder {
    /// Resource endpoint, e.g. `https://my-resource.openai.azure.com/`.
    pub fn endpoint(mut self, e: impl Into<String>) -> Self {
        self.endpoint = Some(e.into());
        self
    }
    /// Deployment name within the resource.
    pub fn deployment(mut self, d: impl Into<String>) -> Self {
        self.deployment = Some(d.into());
        self
    }
    /// API version (e.g. `2024-08-01-preview`).
    pub fn api_version(mut self, v: impl Into<String>) -> Self {
        self.api_version = Some(v.into());
        self
    }
    /// API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Construct.
    pub fn build(self) -> Result<AzureProvider> {
        let endpoint = self
            .endpoint
            .ok_or_else(|| CognisError::Configuration("Azure: endpoint required".into()))?;
        let deployment = self
            .deployment
            .ok_or_else(|| CognisError::Configuration("Azure: deployment required".into()))?;
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("Azure: API key required".into()))?;
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(std::time::Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(AzureProvider {
            endpoint,
            deployment,
            api_version: self
                .api_version
                .unwrap_or_else(|| DEFAULT_API_VERSION.to_string()),
            api_key: SecretString::new(api_key.into_boxed_str()),
            extra_headers: self.extra_headers,
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire-format helpers (shared shape with OpenAI; duplicated locally to keep
// each provider self-contained).
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
        Message::Human(h) => {
            if h.parts.is_empty() {
                serde_json::json!({"role": "user", "content": h.content})
            } else {
                let mut content = Vec::with_capacity(h.parts.len() + 1);
                if !h.content.is_empty() {
                    content.push(serde_json::json!({"type": "text", "text": h.content}));
                }
                for p in &h.parts {
                    content.push(p.to_openai());
                }
                serde_json::json!({"role": "user", "content": content})
            }
        }
        Message::Ai(a) => {
            let mut v = if a.parts.is_empty() {
                serde_json::json!({"role": "assistant", "content": a.content})
            } else {
                let mut content = Vec::with_capacity(a.parts.len() + 1);
                if !a.content.is_empty() {
                    content.push(serde_json::json!({"type": "text", "text": a.content}));
                }
                for p in &a.parts {
                    content.push(p.to_openai());
                }
                serde_json::json!({"role": "assistant", "content": content})
            };
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
    Message::Ai(AiMessage {
        content,
        tool_calls,
        parts: Vec::new(),
    })
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
    let s = std::str::from_utf8(bytes).map_err(|e| CognisError::Provider {
        provider: "azure".into(),
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
                    provider: "azure".into(),
                    message: format!("stream parse: {e}"),
                })?;
            let delta = &v["choices"][0]["delta"];
            let content = delta["content"].as_str().unwrap_or("").to_string();
            let mut tool_calls_delta = Vec::new();
            if let Some(arr) = delta["tool_calls"].as_array() {
                for (i, t) in arr.iter().enumerate() {
                    tool_calls_delta.push(ToolCallDelta {
                        index: t["index"].as_u64().unwrap_or(i as u64) as u32,
                        id: t["id"].as_str().map(|s| s.to_string()),
                        name: t["function"]["name"].as_str().map(|s| s.to_string()),
                        arguments_delta: t["function"]["arguments"].as_str().map(|s| s.to_string()),
                    });
                }
            }
            let finish_reason = v["choices"][0]["finish_reason"]
                .as_str()
                .map(|s| s.to_string());
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
    Ok(Some(StreamChunk::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_construction() {
        let p = AzureBuilder::default()
            .endpoint("https://r.openai.azure.com/")
            .deployment("gpt-4o")
            .api_key("k")
            .build()
            .unwrap();
        let u = p.url("chat/completions");
        assert!(
            u.starts_with("https://r.openai.azure.com/openai/deployments/gpt-4o/chat/completions")
        );
        assert!(u.contains("api-version="));
    }

    #[test]
    fn builder_requires_endpoint_deployment_key() {
        assert!(AzureBuilder::default().build().is_err());
        assert!(AzureBuilder::default()
            .endpoint("https://r")
            .build()
            .is_err());
        assert!(AzureBuilder::default()
            .endpoint("https://r")
            .deployment("d")
            .build()
            .is_err());
    }

    #[test]
    fn provider_metadata() {
        let p = AzureBuilder::default()
            .endpoint("https://r.openai.azure.com/")
            .deployment("d")
            .api_key("k")
            .build()
            .unwrap();
        assert_eq!(p.name(), "azure");
        assert_eq!(p.provider_type(), Provider::Azure);
    }
}
