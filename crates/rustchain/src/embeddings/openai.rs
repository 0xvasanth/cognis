//! OpenAI embedding model implementation.
//!
//! Provides [`OpenAIEmbeddings`], an implementation of the [`Embeddings`] trait
//! for the OpenAI Embeddings API.

use async_trait::async_trait;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};

use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::{Result, RustChainError};

/// Builder for constructing an [`OpenAIEmbeddings`] instance.
#[derive(Debug)]
pub struct OpenAIEmbeddingsBuilder {
    api_key: Option<SecretString>,
    model: Option<String>,
    dimensions: Option<usize>,
    base_url: Option<String>,
}

impl OpenAIEmbeddingsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            api_key: None,
            model: None,
            dimensions: None,
            base_url: None,
        }
    }

    /// Set the API key. Falls back to `OPENAI_API_KEY` env var.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set the model name (default: `"text-embedding-3-small"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the output dimensions (optional, supported by some models).
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set the base API URL (default: `"https://api.openai.com/v1"`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the [`OpenAIEmbeddings`] instance.
    ///
    /// Returns an error if the API key cannot be resolved from the builder
    /// or environment.
    pub fn build(self) -> Result<OpenAIEmbeddings> {
        let api_key = match self.api_key {
            Some(key) => key,
            None => {
                let key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    RustChainError::Other(
                        "api_key not provided and OPENAI_API_KEY env var not set".into(),
                    )
                })?;
                SecretString::from(key)
            }
        };

        Ok(OpenAIEmbeddings {
            api_key,
            model: self
                .model
                .unwrap_or_else(|| "text-embedding-3-small".into()),
            dimensions: self.dimensions,
            base_url: self
                .base_url
                .unwrap_or_else(|| "https://api.openai.com/v1".into()),
            client: Client::new(),
        })
    }
}

impl Default for OpenAIEmbeddingsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// OpenAI embeddings model.
///
/// Implements the OpenAI Embeddings API for generating text embeddings.
///
/// # Example
///
/// ```no_run
/// use rustchain::embeddings::openai::OpenAIEmbeddings;
///
/// let embeddings = OpenAIEmbeddings::builder()
///     .api_key("sk-...")
///     .model("text-embedding-3-small")
///     .build()
///     .unwrap();
/// ```
pub struct OpenAIEmbeddings {
    /// Secret API key.
    api_key: SecretString,
    /// The model identifier (e.g. "text-embedding-3-small").
    pub model: String,
    /// Optional output dimensions.
    pub dimensions: Option<usize>,
    /// Base URL for the OpenAI API.
    pub base_url: String,
    /// HTTP client.
    client: Client,
}

impl std::fmt::Debug for OpenAIEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbeddings")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAIEmbeddings {
    /// Returns a new builder for `OpenAIEmbeddings`.
    pub fn builder() -> OpenAIEmbeddingsBuilder {
        OpenAIEmbeddingsBuilder::new()
    }

    /// Build the JSON request payload for the OpenAI Embeddings API.
    fn build_payload(&self, texts: &[String]) -> Value {
        let mut payload = json!({
            "model": self.model,
            "input": texts,
        });

        if let Some(dims) = self.dimensions {
            payload["dimensions"] = json!(dims);
        }

        payload
    }

    /// Call the OpenAI Embeddings API and return raw embedding vectors.
    async fn call_api(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url);
        let payload = self.build_payload(&texts);

        let response = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
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

        let data = body.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            RustChainError::Other("Missing 'data' array in OpenAI embeddings response".into())
        })?;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    RustChainError::Other("Missing 'embedding' array in response data item".into())
                })?;

            let vec: Vec<f32> = embedding
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
impl Embeddings for OpenAIEmbeddings {
    /// Embed a list of documents using the OpenAI Embeddings API.
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
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "text-embedding-3-small");
        assert_eq!(embeddings.base_url, "https://api.openai.com/v1");
        assert!(embeddings.dimensions.is_none());
    }

    #[test]
    fn test_builder_custom_values() {
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("test-key")
            .model("text-embedding-3-large")
            .dimensions(256)
            .base_url("https://custom.api.com/v1")
            .build()
            .unwrap();

        assert_eq!(embeddings.model, "text-embedding-3-large");
        assert_eq!(embeddings.base_url, "https://custom.api.com/v1");
        assert_eq!(embeddings.dimensions, Some(256));
    }

    #[test]
    fn test_builder_requires_api_key() {
        // Clear env var to ensure it's not set
        std::env::remove_var("OPENAI_API_KEY");
        let result = OpenAIEmbeddings::builder().build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("OPENAI_API_KEY"));
    }

    #[test]
    fn test_build_payload_without_dimensions() {
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let payload = embeddings.build_payload(&texts);

        assert_eq!(payload["model"], "text-embedding-3-small");
        assert_eq!(payload["input"], json!(["hello", "world"]));
        assert!(payload.get("dimensions").is_none());
    }

    #[test]
    fn test_build_payload_with_dimensions() {
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("test-key")
            .dimensions(512)
            .build()
            .unwrap();

        let texts = vec!["hello".to_string()];
        let payload = embeddings.build_payload(&texts);

        assert_eq!(payload["model"], "text-embedding-3-small");
        assert_eq!(payload["input"], json!(["hello"]));
        assert_eq!(payload["dimensions"], 512);
    }

    #[tokio::test]
    async fn test_embed_documents_empty() {
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("test-key")
            .build()
            .unwrap();

        let result = embeddings.embed_documents(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_debug_does_not_leak_api_key() {
        let embeddings = OpenAIEmbeddings::builder()
            .api_key("super-secret-key")
            .build()
            .unwrap();

        let debug_str = format!("{:?}", embeddings);
        assert!(!debug_str.contains("super-secret-key"));
        assert!(debug_str.contains("OpenAIEmbeddings"));
        assert!(debug_str.contains("text-embedding-3-small"));
    }
}
