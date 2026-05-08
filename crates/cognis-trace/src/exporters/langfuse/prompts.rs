//! Langfuse prompt-management client. Pulls versioned prompts via
//! `GET /api/public/v2/prompts/{name}` and `.../versions/{version}`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::TraceError;
use crate::prompts::{ChatMessageTemplate, Prompt, PromptBody, PromptStore};

use super::client::LangfuseHttp;
use super::config::LangfuseConfig;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WirePrompt {
    Text(WireTextPrompt),
    Chat(WireChatPrompt),
}

#[derive(Debug, Deserialize)]
struct WireTextPrompt {
    name: String,
    version: u32,
    prompt: String,
    #[serde(default)]
    config: serde_json::Value,
    #[serde(default)]
    labels: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WireChatPrompt {
    name: String,
    version: u32,
    prompt: Vec<WireChatMessage>,
    #[serde(default)]
    config: serde_json::Value,
    #[serde(default)]
    labels: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WireChatMessage {
    Message {
        role: String,
        content: String,
        #[allow(dead_code)]
        #[serde(default)]
        r#type: Option<String>,
    },
    Placeholder {
        name: String,
        #[allow(dead_code)]
        #[serde(default)]
        r#type: Option<String>,
    },
}

impl From<WirePrompt> for Prompt {
    fn from(w: WirePrompt) -> Self {
        match w {
            WirePrompt::Text(t) => Prompt {
                name: t.name,
                version: t.version,
                body: PromptBody::Text { prompt: t.prompt },
                config: t.config,
                labels: t.labels,
            },
            WirePrompt::Chat(c) => {
                let messages = c
                    .prompt
                    .into_iter()
                    .map(|m| match m {
                        WireChatMessage::Message { role, content, .. } => {
                            ChatMessageTemplate::Message { role, content }
                        }
                        WireChatMessage::Placeholder { name, .. } => {
                            ChatMessageTemplate::Placeholder { name }
                        }
                    })
                    .collect();
                Prompt {
                    name: c.name,
                    version: c.version,
                    body: PromptBody::Chat { messages },
                    config: c.config,
                    labels: c.labels,
                }
            }
        }
    }
}

#[derive(Debug)]
struct CacheEntry {
    prompt: Prompt,
    inserted: Instant,
}

/// Pulls versioned prompts from Langfuse with a small in-memory cache.
pub struct LangfusePromptClient {
    http: LangfuseHttp,
    cache: Mutex<HashMap<String, CacheEntry>>,
    ttl: Duration,
}

impl LangfusePromptClient {
    /// Construct from config; default cache TTL is 60s.
    pub fn new(cfg: LangfuseConfig) -> Result<Self, TraceError> {
        Ok(Self {
            http: LangfuseHttp::new(&cfg)?,
            cache: Mutex::new(HashMap::new()),
            ttl: Duration::from_secs(60),
        })
    }

    /// Override the cache TTL.
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    fn cached(&self, key: &str) -> Option<Prompt> {
        let map = self.cache.lock().unwrap();
        let e = map.get(key)?;
        if e.inserted.elapsed() < self.ttl {
            Some(e.prompt.clone())
        } else {
            None
        }
    }

    fn store(&self, key: String, prompt: Prompt) {
        let mut map = self.cache.lock().unwrap();
        map.insert(key, CacheEntry { prompt, inserted: Instant::now() });
    }

    async fn fetch(&self, path: &str, cache_key: String) -> Result<Prompt, TraceError> {
        if let Some(p) = self.cached(&cache_key) {
            return Ok(p);
        }
        let resp = self
            .http
            .request(reqwest::Method::GET, path)
            .send()
            .await
            .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(TraceError::BackendStatus {
                backend: "langfuse",
                status,
                body: body.chars().take(512).collect(),
            });
        }
        let wire: WirePrompt = resp
            .json()
            .await
            .map_err(|e| TraceError::Network { backend: "langfuse", source: e })?;
        let p: Prompt = wire.into();
        self.store(cache_key, p.clone());
        Ok(p)
    }
}

#[async_trait]
impl PromptStore for LangfusePromptClient {
    async fn get(&self, name: &str) -> Result<Prompt, TraceError> {
        let path = format!("/api/public/v2/prompts/{}", urlencoding(name));
        self.fetch(&path, format!("name:{name}")).await
    }

    async fn get_version(&self, name: &str, version: u32) -> Result<Prompt, TraceError> {
        let path = format!(
            "/api/public/v2/prompts/{}/versions/{}",
            urlencoding(name),
            version
        );
        self.fetch(&path, format!("name:{name}@v{version}")).await
    }

    async fn get_label(&self, name: &str, label: &str) -> Result<Prompt, TraceError> {
        let path = format!(
            "/api/public/v2/prompts/{}?label={}",
            urlencoding(name),
            urlencoding(label)
        );
        self.fetch(&path, format!("name:{name}#{label}")).await
    }
}

fn urlencoding(s: &str) -> String {
    // Minimal: percent-encode characters that are not URL-safe in path segments.
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        let c = *b;
        let safe = c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.' | b'_' | b'~');
        if safe {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    out
}
