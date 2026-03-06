//! Ollama embedding model implementation.
//!
//! Provides [`OllamaEmbeddings`], an implementation of the [`Embeddings`] trait
//! for the Ollama embeddings API.
//!
//! Ollama runs locally and requires no authentication.

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::{Result, RustChainError};

/// Builder for constructing an [`OllamaEmbeddings`] instance.
#[derive(Debug)]
pub struct OllamaEmbeddingsBuilder {
    model: Option<String>,
    base_url: Option<String>,
}

impl OllamaEmbeddingsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            base_url: None,
        }
    }

    /// Set the model name (default: `"nomic-embed-text"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the base URL (default: `"http://localhost:11434"`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the [`OllamaEmbeddings`] instance.
    pub fn build(self) -> OllamaEmbeddings {
        OllamaEmbeddings {
            model: self.model.unwrap_or_else(|| "nomic-embed-text".into()),
            base_url: self
                .base_url
                .unwrap_or_else(|| "http://localhost:11434".into()),
            client: Client::new(),
        }
    }
}

impl Default for OllamaEmbeddingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Ollama embeddings model.
///
/// Implements the Ollama `/api/embed` endpoint for generating text embeddings
/// using locally-hosted models.
///
/// No authentication is required since Ollama runs locally.
///
/// # Example
///
/// ```no_run
/// use rustchain::embeddings::ollama::OllamaEmbeddings;
///
/// let embeddings = OllamaEmbeddings::builder()
///     .model("nomic-embed-text")
///     .build();
/// ```
#[derive(Debug)]
pub struct OllamaEmbeddings {
    /// The model identifier (e.g. "nomic-embed-text").
    pub model: String,
    /// Base URL for the Ollama server.
    pub base_url: String,
    /// HTTP client.
    client: Client,
}

impl OllamaEmbeddings {
    /// Returns a new builder for `OllamaEmbeddings`.
    pub fn builder() -> OllamaEmbeddingsBuilder {
        OllamaEmbeddingsBuilder::new()
    }

    /// Build the JSON request payload for the Ollama embed API.
    fn build_payload(&self, texts: &[String]) -> Value {
        json!({
            "model": self.model,
            "input": texts,
        })
    }

    /// Call the Ollama embed API and return raw embedding vectors.
    async fn call_api(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);
        let payload = self.build_payload(&texts);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| RustChainError::Other(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(RustChainError::HttpError { status, body });
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| RustChainError::Other(format!("Failed to parse response JSON: {}", e)))?;

        let embeddings_arr = body
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RustChainError::Other("Missing 'embeddings' array in Ollama embed response".into())
            })?;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(embeddings_arr.len());
        for item in embeddings_arr {
            let vec_arr = item.as_array().ok_or_else(|| {
                RustChainError::Other("Expected array for embedding vector".into())
            })?;

            let vec: Vec<f32> = vec_arr
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        RustChainError::Other("Non-numeric value in embedding array".into())
                    })
                })
                .collect::<Result<Vec<f32>>>()?;

            embeddings.push(vec);
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl Embeddings for OllamaEmbeddings {
    /// Embed a list of documents using the Ollama embed API.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.call_api(texts).await
    }

    /// Embed a single query text.
    ///
    /// Delegates to `embed_documents` with a single-element list.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_documents(vec![text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| RustChainError::Other("Empty embedding response for query".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let embeddings = OllamaEmbeddings::builder().build();

        assert_eq!(embeddings.model, "nomic-embed-text");
        assert_eq!(embeddings.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_builder_custom_values() {
        let embeddings = OllamaEmbeddings::builder()
            .model("mxbai-embed-large")
            .base_url("http://remote-host:11434")
            .build();

        assert_eq!(embeddings.model, "mxbai-embed-large");
        assert_eq!(embeddings.base_url, "http://remote-host:11434");
    }

    #[test]
    fn test_build_payload() {
        let embeddings = OllamaEmbeddings::builder().build();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let payload = embeddings.build_payload(&texts);

        assert_eq!(payload["model"], "nomic-embed-text");
        assert_eq!(payload["input"], json!(["hello", "world"]));
    }

    #[tokio::test]
    async fn test_embed_documents_empty() {
        let embeddings = OllamaEmbeddings::builder().build();

        let result = embeddings.embed_documents(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_debug_output() {
        let embeddings = OllamaEmbeddings::builder()
            .model("test-model")
            .base_url("http://test:11434")
            .build();

        let debug_str = format!("{:?}", embeddings);
        assert!(debug_str.contains("OllamaEmbeddings"));
        assert!(debug_str.contains("test-model"));
        assert!(debug_str.contains("http://test:11434"));
    }
}
