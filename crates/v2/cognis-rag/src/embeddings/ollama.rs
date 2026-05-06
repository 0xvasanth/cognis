//! Ollama embeddings. POST /api/embed with batched input.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use serde::Deserialize;

use cognis2_core::{CognisError, Result};

use super::Embeddings;

const DEFAULT_BASE: &str = "http://localhost:11434/api/";
const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Ollama embeddings client.
#[derive(Debug)]
pub struct OllamaEmbeddings {
    base_url: String,
    model: String,
    http: reqwest::Client,
}

impl OllamaEmbeddings {
    /// New with default base URL + given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self::builder().model(model).build().expect("default OllamaEmbeddings build")
    }

    /// Fluent builder.
    pub fn builder() -> OllamaEmbeddingsBuilder {
        OllamaEmbeddingsBuilder::default()
    }

    fn endpoint(&self) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str("embed");
        s
    }
}

#[async_trait]
impl Embeddings for OllamaEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .http
            .post(self.endpoint())
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

        let raw: OllamaEmbResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "ollama".into(),
            message: format!("response decode: {e}"),
        })?;
        Ok(raw.embeddings)
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Fluent builder for `OllamaEmbeddings`.
#[derive(Default)]
pub struct OllamaEmbeddingsBuilder {
    base_url: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
}

impl OllamaEmbeddingsBuilder {
    /// Override base URL.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Embedding model.
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
    pub fn build(self) -> Result<OllamaEmbeddings> {
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(OllamaEmbeddings {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            model: self.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            http,
        })
    }
}

#[derive(Deserialize)]
struct OllamaEmbResponse {
    embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_with_defaults() {
        let e = OllamaEmbeddings::new("nomic-embed-text");
        assert_eq!(e.model(), "nomic-embed-text");
    }

    #[test]
    fn endpoint_appends_slash() {
        let e = OllamaEmbeddings::builder()
            .base_url("http://localhost:11434/api")
            .model("nomic-embed-text")
            .build()
            .unwrap();
        assert_eq!(e.endpoint(), "http://localhost:11434/api/embed");
    }

    #[test]
    fn empty_input_returns_empty_vec() {
        // Smoke test that empty input doesn't hit the network.
        let e = OllamaEmbeddings::new("test-model");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let v = rt.block_on(e.embed_documents(Vec::new())).unwrap();
        assert!(v.is_empty());
    }
}
