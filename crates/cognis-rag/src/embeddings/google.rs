//! Google Generative Language embeddings.
//!
//! Targets the public REST API at
//! `https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents`.
//! API key is sent as the `key=...` query parameter (Google's convention).
//!
//! Customization:
//! - [`GoogleEmbeddingsBuilder`] — model, base URL, optional task-type
//!   hint, timeout, custom HTTP client.
//! - The `task_type` hint (e.g. `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`,
//!   `SEMANTIC_SIMILARITY`) lets Google's models specialize embeddings
//!   for retrieval vs. generation. Optional; omitted when unset.

#![cfg(feature = "google")]

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use cognis_core::{CognisError, Result};

use super::Embeddings;

const DEFAULT_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MODEL: &str = "text-embedding-004";

/// Google Generative Language embeddings client.
pub struct GoogleEmbeddings {
    base_url: String,
    api_key: SecretString,
    model: String,
    task_type: Option<String>,
    http: reqwest::Client,
}

impl std::fmt::Debug for GoogleEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleEmbeddings")
            .field("model", &self.model)
            .field("task_type", &self.task_type)
            .finish_non_exhaustive()
    }
}

impl GoogleEmbeddings {
    /// New with API key + default model `text-embedding-004`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::builder()
            .api_key(api_key)
            .build()
            .expect("default GoogleEmbeddings build")
    }

    /// Fluent builder.
    pub fn builder() -> GoogleEmbeddingsBuilder {
        GoogleEmbeddingsBuilder::default()
    }

    /// `POST {base}/models/{model}:batchEmbedContents?key={api_key}`.
    fn batch_endpoint(&self) -> String {
        let mut s = self.base_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str("models/");
        s.push_str(&self.model);
        s.push_str(":batchEmbedContents?key=");
        s.push_str(self.api_key.expose_secret());
        s
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(h)
    }

    /// Model id formatted the way Google's API expects in request bodies
    /// (`"models/text-embedding-004"`).
    fn qualified_model(&self) -> String {
        if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        }
    }
}

#[async_trait]
impl Embeddings for GoogleEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Google's batchEmbedContents takes a list of EmbedContentRequests,
        // each with its own `model`, `content`, and optional `taskType`.
        let qualified = self.qualified_model();
        let requests: Vec<EmbedContentRequest> = texts
            .iter()
            .map(|t| EmbedContentRequest {
                model: qualified.clone(),
                content: Content {
                    parts: vec![Part { text: t.clone() }],
                },
                task_type: self.task_type.clone(),
            })
            .collect();

        let body = BatchEmbedRequest { requests };
        let resp = self
            .http
            .post(self.batch_endpoint())
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
        let raw: BatchEmbedResponse = resp.json().await.map_err(|e| CognisError::Provider {
            provider: "google".into(),
            message: format!("response decode: {e}"),
        })?;
        Ok(raw.embeddings.into_iter().map(|e| e.values).collect())
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Fluent builder for [`GoogleEmbeddings`].
#[derive(Default)]
pub struct GoogleEmbeddingsBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    task_type: Option<String>,
    timeout_secs: Option<u64>,
    http: Option<reqwest::Client>,
}

impl GoogleEmbeddingsBuilder {
    /// Set the API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Override base URL.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Override the model (default `text-embedding-004`).
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    /// Set the task-type hint (e.g. `"RETRIEVAL_DOCUMENT"`,
    /// `"RETRIEVAL_QUERY"`, `"SEMANTIC_SIMILARITY"`,
    /// `"CLASSIFICATION"`, `"CLUSTERING"`). Sent as-is.
    pub fn task_type(mut self, t: impl Into<String>) -> Self {
        self.task_type = Some(t.into());
        self
    }
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Override the HTTP client.
    pub fn http_client(mut self, c: reqwest::Client) -> Self {
        self.http = Some(c);
        self
    }
    /// Construct.
    pub fn build(self) -> Result<GoogleEmbeddings> {
        let api_key = self.api_key.ok_or_else(|| {
            CognisError::Configuration("GoogleEmbeddings: API key required".into())
        })?;
        let http = match self.http {
            Some(c) => c,
            None => {
                let mut b = reqwest::ClientBuilder::new();
                if let Some(t) = self.timeout_secs {
                    b = b.timeout(Duration::from_secs(t));
                }
                b.build()
                    .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?
            }
        };
        Ok(GoogleEmbeddings {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            api_key: SecretString::new(api_key.into()),
            model: self.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            task_type: self.task_type,
            http,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BatchEmbedRequest {
    requests: Vec<EmbedContentRequest>,
}

#[derive(Serialize)]
struct EmbedContentRequest {
    model: String,
    content: Content,
    #[serde(rename = "taskType", skip_serializing_if = "Option::is_none")]
    task_type: Option<String>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbeddingValues>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_api_key() {
        let err = GoogleEmbeddingsBuilder::default().build().unwrap_err();
        assert!(format!("{err}").contains("API key"));
    }

    #[test]
    fn builder_with_defaults() {
        let e = GoogleEmbeddings::new("sk-test");
        assert_eq!(e.model(), DEFAULT_MODEL);
    }

    #[test]
    fn task_type_round_trips_through_builder() {
        let e = GoogleEmbeddings::builder()
            .api_key("sk-test")
            .task_type("RETRIEVAL_DOCUMENT")
            .build()
            .unwrap();
        assert_eq!(e.task_type.as_deref(), Some("RETRIEVAL_DOCUMENT"));
    }

    #[test]
    fn endpoint_includes_model_and_key() {
        let e = GoogleEmbeddings::builder()
            .api_key("sk-test")
            .model("text-embedding-004")
            .build()
            .unwrap();
        let url = e.batch_endpoint();
        assert!(url.contains("models/text-embedding-004:batchEmbedContents"));
        assert!(url.contains("key=sk-test"));
    }

    #[test]
    fn qualified_model_prefixes_when_missing() {
        let e = GoogleEmbeddings::new("sk-test");
        assert_eq!(e.qualified_model(), "models/text-embedding-004");
    }

    #[test]
    fn qualified_model_passes_through_when_already_prefixed() {
        let e = GoogleEmbeddings::builder()
            .api_key("sk-test")
            .model("models/embedding-001")
            .build()
            .unwrap();
        assert_eq!(e.qualified_model(), "models/embedding-001");
    }

    #[test]
    fn empty_input_returns_empty_without_http_call() {
        // We can't hit real HTTP in unit tests; this just verifies the
        // empty-vec short-circuit path doesn't panic.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let e = GoogleEmbeddings::new("sk-test");
        let out = rt.block_on(e.embed_documents(Vec::new())).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn batch_request_serializes_with_task_type() {
        let req = BatchEmbedRequest {
            requests: vec![EmbedContentRequest {
                model: "models/text-embedding-004".into(),
                content: Content {
                    parts: vec![Part {
                        text: "hello".into(),
                    }],
                },
                task_type: Some("RETRIEVAL_QUERY".into()),
            }],
        };
        let json = serde_json::to_value(&req).unwrap();
        let r0 = &json["requests"][0];
        assert_eq!(r0["model"], "models/text-embedding-004");
        assert_eq!(r0["content"]["parts"][0]["text"], "hello");
        assert_eq!(r0["taskType"], "RETRIEVAL_QUERY");
    }

    #[test]
    fn batch_request_omits_task_type_when_none() {
        let req = BatchEmbedRequest {
            requests: vec![EmbedContentRequest {
                model: "models/text-embedding-004".into(),
                content: Content {
                    parts: vec![Part {
                        text: "hello".into(),
                    }],
                },
                task_type: None,
            }],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["requests"][0].get("taskType").is_none());
    }
}
