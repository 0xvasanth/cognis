//! OpenAI Embeddings API. POST /v1/embeddings.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis2_core::{CognisError, Result};

use super::Embeddings;

const DEFAULT_BASE: &str = "https://api.openai.com/v1/";
const DEFAULT_MODEL: &str = "text-embedding-3-small";

/// OpenAI Embeddings API client.
#[derive(Debug)]
pub struct OpenAIEmbeddings {
    base_url: String,
    api_key: SecretString,
    model: String,
    dimensions: Option<usize>,
    http: reqwest::Client,
}

impl OpenAIEmbeddings {
    /// New with API key + default model "text-embedding-3-small".
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder()
            .api_key(api_key)
            .build()
            .expect("default OpenAIEmbeddings build")
    }

    /// Fluent builder.
    pub fn builder() -> OpenAIEmbeddingsBuilder {
        OpenAIEmbeddingsBuilder::default()
    }

    fn endpoint(&self) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str("embeddings");
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
        Ok(h)
    }
}

#[async_trait]
impl Embeddings for OpenAIEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });
        if let Some(dim) = self.dimensions {
            // Only newer embedding models accept the `dimensions` parameter
            // (e.g. text-embedding-3-*). Pass it; older models ignore it.
            body["dimensions"] = serde_json::json!(dim);
        }

        let resp = self
            .http
            .post(self.endpoint())
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

        let raw: OpenAIEmbResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "openai".into(),
            message: format!("response decode: {e}"),
        })?;

        // Sort by index defensively (OpenAI returns in order, but spec
        // doesn't guarantee).
        let mut data = raw.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Fluent builder for `OpenAIEmbeddings`.
#[derive(Default)]
pub struct OpenAIEmbeddingsBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    dimensions: Option<usize>,
    timeout_secs: Option<u64>,
}

impl OpenAIEmbeddingsBuilder {
    /// Set the API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Override base URL (for proxies / OpenAI-compatible endpoints).
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Override the model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    /// Request a specific dimensionality (only honored by `text-embedding-3-*`).
    pub fn dimensions(mut self, d: usize) -> Self {
        self.dimensions = Some(d);
        self
    }
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Construct.
    pub fn build(self) -> Result<OpenAIEmbeddings> {
        let api_key = self.api_key.ok_or_else(|| {
            CognisError::Configuration("OpenAIEmbeddings: API key required".into())
        })?;
        let mut http = reqwest::ClientBuilder::new();
        if let Some(t) = self.timeout_secs {
            http = http.timeout(Duration::from_secs(t));
        }
        let http = http
            .build()
            .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?;
        Ok(OpenAIEmbeddings {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into()),
            model: self.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            dimensions: self.dimensions,
            http,
        })
    }
}

#[derive(Deserialize)]
struct OpenAIEmbResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Deserialize, Serialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
    index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_api_key() {
        let err = OpenAIEmbeddingsBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let e = OpenAIEmbeddings::new("sk-test");
        assert_eq!(e.model(), DEFAULT_MODEL);
        assert!(e.dimensions().is_none());
    }

    #[test]
    fn builder_with_dimensions() {
        let e = OpenAIEmbeddings::builder()
            .api_key("sk-test")
            .model("text-embedding-3-large")
            .dimensions(256)
            .build()
            .unwrap();
        assert_eq!(e.model(), "text-embedding-3-large");
        assert_eq!(e.dimensions(), Some(256));
    }

    #[test]
    fn endpoint_appends_slash_when_missing() {
        let e = OpenAIEmbeddings::builder()
            .api_key("sk-test")
            .base_url("https://example.com/v1")
            .build()
            .unwrap();
        assert_eq!(e.endpoint(), "https://example.com/v1/embeddings");
    }
}
