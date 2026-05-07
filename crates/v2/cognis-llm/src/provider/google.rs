//! Google Gemini provider (`generateContent` / `streamGenerateContent`).
//!
//! Wire-format notes:
//! - Roles are `user` and `model` (not `assistant`).
//! - System instruction is a top-level field, not a message.
//! - Tool calls surface as `parts[].functionCall = {name, args}`.
//! - Tool results are `parts[].functionResponse = {name, response}`.
//! - Auth via `x-goog-api-key` header.

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use crate::tools::ToolDefinition;
use crate::{AiMessage, Message, ToolCall};

use super::{LLMProvider, Provider};

const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/";

/// Google Gemini provider configuration.
#[derive(Debug)]
pub struct GoogleProvider {
    base_url: String,
    api_key: SecretString,
    model: String,
    extra_headers: Vec<(String, String)>,
    http: reqwest::Client,
}

impl GoogleProvider {
    /// Build with explicit config.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder()
            .api_key(api_key)
            .build()
            .expect("default Google build")
    }

    /// Fluent builder.
    pub fn builder() -> GoogleBuilder {
        GoogleBuilder::default()
    }

    fn endpoint(&self, model: &str, action: &str) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        // models/{model}:{action}
        s.push_str("models/");
        s.push_str(model);
        s.push(':');
        s.push_str(action);
        s
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            HeaderName::from_static("x-goog-api-key"),
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
    ) -> serde_json::Value {
        let (system, contents) = split_system_and_contents(messages);
        let mut body = serde_json::json!({ "contents": contents });
        if let Some(s) = system {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{"text": s}],
            });
        }
        if !tools.is_empty() {
            body["tools"] = serde_json::json!([{
                "functionDeclarations": tools_to_gemini_format(tools),
            }]);
        }
        let mut gen_config = serde_json::Map::new();
        if let Some(t) = opts.temperature {
            gen_config.insert("temperature".into(), serde_json::json!(t));
        }
        if let Some(p) = opts.top_p {
            gen_config.insert("topP".into(), serde_json::json!(p));
        }
        if let Some(m) = opts.max_tokens {
            gen_config.insert("maxOutputTokens".into(), serde_json::json!(m));
        }
        if !opts.stop.is_empty() {
            gen_config.insert("stopSequences".into(), serde_json::json!(opts.stop));
        }
        if !gen_config.is_empty() {
            body["generationConfig"] = serde_json::Value::Object(gen_config);
        }
        body
    }
}

#[async_trait]
impl LLMProvider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }
    fn provider_type(&self) -> Provider {
        Provider::Google
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
        let model = opts.model.clone().unwrap_or_else(|| self.model.clone());
        let body = self.build_request(&messages, &tools, &opts);
        let resp = self
            .http
            .post(self.endpoint(&model, "generateContent"))
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

        let raw: GeminiResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "google".into(),
            message: format!("response decode: {e}"),
        })?;

        let candidate = raw
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| CognisError::Provider {
                provider: "google".into(),
                message: "no candidates in response".into(),
            })?;

        let message = gemini_content_to_cognis(&candidate.content);
        Ok(ChatResponse {
            message,
            usage: raw.usage_metadata.map(|u| Usage {
                prompt_tokens: u.prompt_token_count,
                completion_tokens: u.candidates_token_count,
                total_tokens: u.total_token_count,
            }),
            finish_reason: candidate.finish_reason.unwrap_or_else(|| "STOP".into()),
            model,
        })
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        let model = opts.model.clone().unwrap_or_else(|| self.model.clone());
        let body = self.build_request(&messages, &[], &opts);
        let resp = self
            .http
            .post(format!(
                "{}?alt=sse",
                self.endpoint(&model, "streamGenerateContent")
            ))
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
                    Ok(bytes) => Some(parse_gemini_sse(&bytes)),
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
        // Lightweight: list models.
        let url = format!("{}models", trim_trailing_slash(&self.base_url));
        let resp = self.http.get(url).headers(self.headers()?).send().await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::Healthy {
                latency_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(r) => Ok(HealthStatus::Degraded {
                reason: format!("models endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(HealthStatus::Unhealthy {
                reason: e.to_string(),
            }),
        }
    }
}

fn trim_trailing_slash(s: &str) -> String {
    if let Some(stripped) = s.strip_suffix('/') {
        stripped.to_string()
    } else {
        s.to_string()
    }
    .trim_end_matches('/')
    .to_string()
        + "/"
}

/// Fluent builder for `GoogleProvider`.
#[derive(Default)]
pub struct GoogleBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    extra_headers: Vec<(String, String)>,
    timeout_secs: Option<u64>,
}

impl GoogleBuilder {
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
    pub fn build(self) -> Result<GoogleProvider> {
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("Google: API key required".into()))?;
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(std::time::Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(GoogleProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into_boxed_str()),
            model: self
                .model
                .unwrap_or_else(|| Provider::Google.default_model().to_string()),
            extra_headers: self.extra_headers,
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct GeminiContent {
    #[serde(default)]
    role: String,
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Deserialize, Serialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Deserialize, Serialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
}

fn split_system_and_contents(messages: &[Message]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system_parts = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m {
            Message::System(s) => system_parts.push(s.content.clone()),
            Message::Human(h) => {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                if !h.content.is_empty() {
                    parts.push(serde_json::json!({"text": h.content}));
                }
                for p in &h.parts {
                    parts.push(p.to_gemini());
                }
                out.push(serde_json::json!({
                    "role": "user",
                    "parts": parts,
                }));
            }
            Message::Ai(a) => {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                if !a.content.is_empty() {
                    parts.push(serde_json::json!({"text": a.content}));
                }
                for p in &a.parts {
                    parts.push(p.to_gemini());
                }
                for tc in &a.tool_calls {
                    parts.push(serde_json::json!({
                        "functionCall": {"name": tc.name, "args": tc.arguments},
                    }));
                }
                out.push(serde_json::json!({
                    "role": "model",
                    "parts": parts,
                }));
            }
            Message::Tool(t) => {
                let response: serde_json::Value = serde_json::from_str(&t.content)
                    .unwrap_or_else(|_| serde_json::json!({"result": t.content}));
                out.push(serde_json::json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            // Gemini ties responses by name; preserve the call id
                            // as the name when the upstream `tool_call_id` is opaque.
                            "name": t.tool_call_id,
                            "response": response,
                        }
                    }],
                }));
            }
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, out)
}

fn gemini_content_to_cognis(content: &GeminiContent) -> Message {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for (i, p) in content.parts.iter().enumerate() {
        match p {
            GeminiPart::Text { text } => text_parts.push(text.clone()),
            GeminiPart::FunctionCall { function_call } => tool_calls.push(ToolCall {
                id: format!("call_{i}"),
                name: function_call.name.clone(),
                arguments: function_call.args.clone(),
            }),
            GeminiPart::FunctionResponse { .. } => {}
        }
    }
    Message::Ai(AiMessage {
        content: text_parts.join(""),
        tool_calls,
        parts: Vec::new(),
    })
}

fn tools_to_gemini_format(tools: &[ToolDefinition]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.clone().unwrap_or_else(
                    || serde_json::json!({"type": "object", "properties": {}})
                ),
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn parse_gemini_sse(bytes: &[u8]) -> Result<Option<StreamChunk>> {
    let s = std::str::from_utf8(bytes).map_err(|e| CognisError::Provider {
        provider: "google".into(),
        message: format!("invalid UTF-8 in stream: {e}"),
    })?;
    for line in s.lines() {
        let line = line.trim();
        if let Some(payload) = line.strip_prefix("data: ") {
            if payload.is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(payload).map_err(|e| CognisError::Provider {
                    provider: "google".into(),
                    message: format!("stream parse: {e}"),
                })?;
            let cand = &v["candidates"][0];
            let parts = cand["content"]["parts"].as_array();
            let mut content = String::new();
            if let Some(parts) = parts {
                for p in parts {
                    if let Some(t) = p["text"].as_str() {
                        content.push_str(t);
                    }
                }
            }
            let finish_reason = cand["finishReason"].as_str().map(|s| s.to_string());
            let is_done = finish_reason.is_some();
            let usage = v["usageMetadata"].as_object().map(|u| Usage {
                prompt_tokens: u
                    .get("promptTokenCount")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as u32,
                completion_tokens: u
                    .get("candidatesTokenCount")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as u32,
                total_tokens: u
                    .get("totalTokenCount")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as u32,
            });
            return Ok(Some(StreamChunk {
                content,
                is_delta: true,
                is_done,
                finish_reason,
                usage,
                tool_calls_delta: Vec::new(),
            }));
        }
    }
    Ok(Some(StreamChunk::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_extracts_top_level() {
        let msgs = vec![
            Message::system("be helpful"),
            Message::human("hi"),
            Message::ai("hello"),
        ];
        let (sys, contents) = split_system_and_contents(&msgs);
        assert_eq!(sys.as_deref(), Some("be helpful"));
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
    }

    #[test]
    fn ai_with_tool_call_emits_function_call_part() {
        let m = Message::Ai(AiMessage {
            content: "calling".into(),
            tool_calls: vec![ToolCall {
                id: "ignored".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
            parts: Vec::new(),
        });
        let (_sys, contents) = split_system_and_contents(&[m]);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[1]["functionCall"]["name"], "search");
        assert_eq!(parts[1]["functionCall"]["args"]["q"], "rust");
    }

    #[test]
    fn gemini_response_text_only() {
        let content = GeminiContent {
            role: "model".into(),
            parts: vec![GeminiPart::Text { text: "ok".into() }],
        };
        let m = gemini_content_to_cognis(&content);
        assert_eq!(m.content(), "ok");
        assert!(m.tool_calls().is_empty());
    }

    #[test]
    fn gemini_response_function_call() {
        let content = GeminiContent {
            role: "model".into(),
            parts: vec![GeminiPart::FunctionCall {
                function_call: GeminiFunctionCall {
                    name: "search".into(),
                    args: serde_json::json!({"q": "rust"}),
                },
            }],
        };
        let m = gemini_content_to_cognis(&content);
        if let Message::Ai(a) = m {
            assert_eq!(a.tool_calls.len(), 1);
            assert_eq!(a.tool_calls[0].name, "search");
        } else {
            panic!("expected Ai");
        }
    }

    #[test]
    fn parse_sse_text_chunk() {
        let bytes = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n";
        let chunk = parse_gemini_sse(bytes).unwrap().unwrap();
        assert_eq!(chunk.content, "hi");
    }

    #[test]
    fn builder_requires_api_key() {
        let err = GoogleBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let p = GoogleBuilder::default()
            .api_key("AIza-test")
            .build()
            .unwrap();
        assert_eq!(p.name(), "google");
        assert_eq!(p.provider_type(), Provider::Google);
    }
}
