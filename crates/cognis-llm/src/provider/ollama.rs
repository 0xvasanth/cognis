//! Ollama provider (local models).

use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
use crate::tools::ToolDefinition;
use crate::{AiMessage, Message, ToolCall};

use super::{LLMProvider, Provider};

const DEFAULT_BASE: &str = "http://localhost:11434/api/";

/// Ollama provider.
pub struct OllamaProvider {
    base_url: String,
    model: String,
    http: reqwest::Client,
}

impl OllamaProvider {
    /// Build with explicit config.
    pub fn new(model: impl Into<String>) -> Self {
        Self::builder()
            .model(model)
            .build()
            .expect("default Ollama build")
    }

    /// Fluent builder.
    pub fn builder() -> OllamaBuilder {
        OllamaBuilder::default()
    }

    fn endpoint(&self, path: &str) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str(path);
        s
    }

    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        opts: &ChatOptions,
        stream: bool,
    ) -> serde_json::Value {
        let model = opts.model.as_deref().unwrap_or(&self.model);
        let ollama_messages: Vec<serde_json::Value> =
            messages.iter().map(message_to_ollama).collect();
        let mut options = serde_json::Map::new();
        if let Some(t) = opts.temperature {
            options.insert("temperature".into(), serde_json::json!(t));
        }
        if let Some(p) = opts.top_p {
            options.insert("top_p".into(), serde_json::json!(p));
        }
        if let Some(m) = opts.max_tokens {
            options.insert("num_predict".into(), serde_json::json!(m));
        }
        if !opts.stop.is_empty() {
            options.insert("stop".into(), serde_json::json!(opts.stop));
        }
        let mut body = serde_json::json!({
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
        });
        if !options.is_empty() {
            body["options"] = serde_json::Value::Object(options);
        }
        if !tools.is_empty() {
            body["tools"] = tools_to_ollama_format(tools);
        }
        body
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }
    fn provider_type(&self) -> Provider {
        Provider::Ollama
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
            .post(self.endpoint("chat"))
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
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

        let raw: OllamaChatResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "ollama".into(),
            message: format!("response decode: {e}"),
        })?;

        let model_name = raw.model.clone();
        let message = ollama_message_to_cognis(raw.message);
        let usage = if raw.eval_count.is_some() || raw.prompt_eval_count.is_some() {
            Some(Usage {
                prompt_tokens: raw.prompt_eval_count.unwrap_or(0),
                completion_tokens: raw.eval_count.unwrap_or(0),
                total_tokens: raw.prompt_eval_count.unwrap_or(0) + raw.eval_count.unwrap_or(0),
            })
        } else {
            None
        };
        Ok(ChatResponse {
            message,
            usage,
            finish_reason: if raw.done {
                "stop".into()
            } else {
                "length".into()
            },
            model: model_name,
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
            .post(self.endpoint("chat"))
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
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
        // Ollama streams newline-delimited JSON. Each line is a complete
        // OllamaChatResponse-shaped object. We don't accumulate across
        // bytes for simplicity (Ollama typically sends one object per
        // chunk); production code may need a buffered NDJSON parser.
        let chunk_stream = byte_stream.filter_map(|res| async move {
            match res {
                Ok(bytes) => Some(parse_ndjson_chunk(&bytes)),
                Err(e) => Some(Err(CognisError::Network {
                    status_code: None,
                    message: e.to_string(),
                })),
            }
        });

        Ok(RunnableStream::new(chunk_stream))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let start = Instant::now();
        let resp = self.http.get(self.endpoint("tags")).send().await;
        match resp {
            Ok(r) if r.status().is_success() => Ok(HealthStatus::Healthy {
                latency_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(r) => Ok(HealthStatus::Degraded {
                reason: format!("tags endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(HealthStatus::Unhealthy {
                reason: e.to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for `OllamaProvider`.
#[derive(Default)]
pub struct OllamaBuilder {
    base_url: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
}

impl OllamaBuilder {
    /// Override the base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    /// Default model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Construct.
    pub fn build(self) -> Result<OllamaProvider> {
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(std::time::Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(OllamaProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            model: self
                .model
                .unwrap_or_else(|| Provider::Ollama.default_model().to_string()),
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Deserialize, Serialize)]
struct OllamaMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Deserialize, Serialize, Clone)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Deserialize, Serialize, Clone)]
struct OllamaFunctionCall {
    name: String,
    /// Ollama returns args as an OBJECT (not a string like OpenAI).
    /// But scalar values inside may come as strings — we coerce numerically.
    arguments: serde_json::Value,
}

fn message_to_ollama(m: &Message) -> serde_json::Value {
    match m {
        Message::Human(h) => serde_json::json!({"role": "user", "content": h.content}),
        Message::Ai(a) => serde_json::json!({"role": "assistant", "content": a.content}),
        Message::System(s) => serde_json::json!({"role": "system", "content": s.content}),
        Message::Tool(t) => serde_json::json!({"role": "tool", "content": t.content}),
    }
}

fn ollama_message_to_cognis(m: OllamaMessage) -> Message {
    let tool_calls = m
        .tool_calls
        .into_iter()
        .enumerate()
        .map(|(i, tc)| ToolCall {
            id: format!("call_{i}"),
            name: tc.function.name,
            arguments: coerce_numeric_strings(tc.function.arguments),
        })
        .collect();
    Message::Ai(AiMessage {
        content: m.content,
        tool_calls,
        parts: Vec::new(),
    })
}

/// Walk a JSON value; for any string that looks numeric, replace it with a Number.
/// Ollama returns tool args like `{"a": "156", "b": "23"}` even when the schema
/// says `number`. This coercion fixes that at the boundary.
fn coerce_numeric_strings(v: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::String(s) => {
            if let Ok(n) = s.parse::<i64>() {
                Value::Number(n.into())
            } else if let Ok(n) = s.parse::<f64>() {
                serde_json::Number::from_f64(n)
                    .map(Value::Number)
                    .unwrap_or(Value::String(s))
            } else if s == "true" {
                Value::Bool(true)
            } else if s == "false" {
                Value::Bool(false)
            } else {
                Value::String(s)
            }
        }
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, coerce_numeric_strings(v)))
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.into_iter().map(coerce_numeric_strings).collect()),
        other => other,
    }
}

fn tools_to_ollama_format(tools: &[ToolDefinition]) -> serde_json::Value {
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

fn parse_ndjson_chunk(bytes: &[u8]) -> Result<StreamChunk> {
    let s = std::str::from_utf8(bytes).map_err(|e| CognisError::Provider {
        provider: "ollama".into(),
        message: format!("invalid UTF-8: {e}"),
    })?;
    // Take the first non-empty line.
    let line = s.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    if line.is_empty() {
        return Ok(StreamChunk::default());
    }
    let v: OllamaChatResponse = serde_json::from_str(line).map_err(|e| CognisError::Provider {
        provider: "ollama".into(),
        message: format!("stream parse: {e}"),
    })?;
    Ok(StreamChunk {
        content: v.message.content,
        is_delta: !v.done,
        is_done: v.done,
        finish_reason: if v.done { Some("stop".into()) } else { None },
        usage: None,
        tool_calls_delta: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coerce_numeric_strings_recursive() {
        let v = serde_json::json!({
            "a": "156",
            "b": "23.5",
            "nested": {
                "c": "true",
                "d": "hello",
            },
            "arr": ["1", "2", "three"],
        });
        let coerced = coerce_numeric_strings(v);
        assert_eq!(coerced["a"], 156);
        assert_eq!(coerced["b"], 23.5);
        assert_eq!(coerced["nested"]["c"], true);
        assert_eq!(coerced["nested"]["d"], "hello");
        assert_eq!(coerced["arr"][0], 1);
        assert_eq!(coerced["arr"][2], "three");
    }

    #[test]
    fn message_to_ollama_human() {
        let m = Message::human("hi");
        let v = message_to_ollama(&m);
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hi");
    }

    #[test]
    fn ollama_message_to_cognis_assigns_call_ids() {
        let m = OllamaMessage {
            role: "assistant".into(),
            content: "ok".into(),
            tool_calls: vec![OllamaToolCall {
                function: OllamaFunctionCall {
                    name: "search".into(),
                    arguments: serde_json::json!({"q": "rust"}),
                },
            }],
        };
        let cognis = ollama_message_to_cognis(m);
        if let Message::Ai(a) = cognis {
            assert_eq!(a.tool_calls[0].id, "call_0");
            assert_eq!(a.tool_calls[0].name, "search");
        } else {
            panic!("expected Ai");
        }
    }

    #[test]
    fn parse_ndjson_chunk_done() {
        let bytes =
            br#"{"model":"llama3.2","message":{"role":"assistant","content":"hi"},"done":true}
"#;
        let chunk = parse_ndjson_chunk(bytes).unwrap();
        assert_eq!(chunk.content, "hi");
        assert!(chunk.is_done);
    }

    #[test]
    fn builder_with_defaults() {
        let p = OllamaBuilder::default().model("llama3.2").build().unwrap();
        assert_eq!(p.name(), "ollama");
        assert_eq!(p.provider_type(), Provider::Ollama);
    }
}
