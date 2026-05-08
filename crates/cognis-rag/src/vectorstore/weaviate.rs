//! Weaviate vector-store backend (HTTP).
//!
//! Targets the Weaviate REST API (`/v1/objects` + `/v1/graphql`).
//! Documents become objects in a class with properties `text` (string)
//! plus arbitrary user metadata.
//!
//! Customization:
//! - [`WeaviateBuilder`] — base URL, class name, optional API key,
//!   `text_property` override (default `"text"`).

#![cfg(feature = "vectorstore-weaviate")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::embeddings::Embeddings;
use crate::vectorstore::{Filter, SearchResult, VectorStore};

const DEFAULT_BASE: &str = "http://localhost:8080";
const DEFAULT_TEXT_KEY: &str = "text";

/// Weaviate-backed vector store.
pub struct WeaviateProvider {
    base_url: String,
    class: String,
    api_key: Option<SecretString>,
    extra_headers: Vec<(String, String)>,
    text_property: String,
    embeddings: Arc<dyn Embeddings>,
    http: reqwest::Client,
    local_count: std::sync::atomic::AtomicUsize,
}

impl WeaviateProvider {
    /// Fluent builder.
    pub fn builder() -> WeaviateBuilder {
        WeaviateBuilder::default()
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
        if let Some(k) = &self.api_key {
            h.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", k.expose_secret()))
                    .map_err(|e| CognisError::Configuration(format!("invalid api key: {e}")))?,
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
}

#[async_trait]
impl VectorStore for WeaviateProvider {
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        let vectors = self.embeddings.embed_documents(texts.clone()).await?;
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); texts.len()]);
        self.add_vectors(vectors, texts, Some(metadatas)).await
    }

    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        #[derive(Serialize)]
        struct ObjectBody {
            class: String,
            id: String,
            vector: Vec<f32>,
            properties: HashMap<String, serde_json::Value>,
        }
        let mut ids: Vec<String> = Vec::with_capacity(vectors.len());
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); vectors.len()]);
        let url = self.endpoint("v1/objects");
        for ((vec, text), mut props) in vectors.into_iter().zip(texts).zip(metadatas) {
            let id = Uuid::new_v4().to_string();
            props.insert(self.text_property.clone(), serde_json::Value::String(text));
            let body = ObjectBody {
                class: self.class.clone(),
                id: id.clone(),
                vector: vec,
                properties: props,
            };
            let resp = self
                .http
                .post(&url)
                .headers(self.headers()?)
                .json(&body)
                .send()
                .await
                .map_err(|e| CognisError::Internal(format!("weaviate object create: {e}")))?;
            if !resp.status().is_success() {
                let s = resp.status();
                let t = resp.text().await.unwrap_or_default();
                return Err(CognisError::Internal(format!(
                    "weaviate object create: HTTP {s}: {t}"
                )));
            }
            ids.push(id);
        }
        let n = ids.len();
        self.local_count
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        Ok(ids)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let v = self.embeddings.embed_query(query.to_string()).await?;
        self.similarity_search_by_vector(v, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // GraphQL: pull the configured text property + _additional id/
        // distance for each match. Other property fields are returned
        // as-is in the JSON response and surface as metadata.
        let q = format!(
            "{{ Get {{ {class}(nearVector: {{vector: {vec_json}}}, limit: {k}) {{ \
                {text_prop} _additional {{ id distance }} }} }} }}",
            class = self.class,
            text_prop = self.text_property,
            vec_json = serde_json::to_string(&query_vector).unwrap_or("[]".into()),
        );
        #[derive(Serialize)]
        struct Body<'a> {
            query: &'a str,
        }
        let url = self.endpoint("v1/graphql");
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&Body { query: &q })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("weaviate graphql: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "weaviate graphql: HTTP {s}: {t}"
            )));
        }
        let raw: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("weaviate json: {e}")))?;
        Ok(parse_graphql_response(
            &raw,
            &self.class,
            &self.text_property,
        ))
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let count = ids.len();
        for id in &ids {
            let url = self.endpoint(&format!("v1/objects/{}/{id}", self.class));
            let resp = self
                .http
                .delete(&url)
                .headers(self.headers()?)
                .send()
                .await
                .map_err(|e| CognisError::Internal(format!("weaviate delete: {e}")))?;
            if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
                let s = resp.status();
                let t = resp.text().await.unwrap_or_default();
                return Err(CognisError::Internal(format!(
                    "weaviate delete: HTTP {s}: {t}"
                )));
            }
        }
        self.local_count
            .fetch_sub(count, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        // Trait default (post-filter) is fine for the GraphQL path here.
        // For complex filters, users should switch to the trait default.
        let candidates = self.similarity_search(query, k.saturating_mul(4)).await?;
        Ok(candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(k)
            .collect())
    }

    fn len(&self) -> usize {
        self.local_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn parse_graphql_response(
    raw: &serde_json::Value,
    class: &str,
    text_prop: &str,
) -> Vec<SearchResult> {
    let arr = raw
        .get("data")
        .and_then(|v| v.get("Get"))
        .and_then(|v| v.get(class))
        .and_then(|v| v.as_array());
    let arr = match arr {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(arr.len());
    for hit in arr {
        let id = hit
            .get("_additional")
            .and_then(|a| a.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let distance = hit
            .get("_additional")
            .and_then(|a| a.get("distance"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let text = hit
            .get(text_prop)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        // Pull other properties (everything except `_additional` and the text key) as metadata.
        let mut metadata = HashMap::new();
        if let Some(obj) = hit.as_object() {
            for (k, v) in obj {
                if k == "_additional" || k == text_prop {
                    continue;
                }
                metadata.insert(k.clone(), v.clone());
            }
        }
        out.push(SearchResult {
            id,
            text,
            score: 1.0 - distance,
            metadata,
        });
    }
    out
}

/// Fluent builder for [`WeaviateProvider`].
#[derive(Default)]
pub struct WeaviateBuilder {
    base_url: Option<String>,
    class: Option<String>,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    text_property: Option<String>,
    embeddings: Option<Arc<dyn Embeddings>>,
    http: Option<reqwest::Client>,
    timeout_secs: Option<u64>,
}

impl WeaviateBuilder {
    /// Override base URL (default `http://localhost:8080`).
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Set the class name (Weaviate class).
    pub fn class(mut self, c: impl Into<String>) -> Self {
        self.class = Some(c.into());
        self
    }
    /// Set bearer API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
    /// Override the property name for `text` (default `"text"`).
    pub fn text_property(mut self, p: impl Into<String>) -> Self {
        self.text_property = Some(p.into());
        self
    }
    /// Embeddings provider (required).
    pub fn embeddings(mut self, e: Arc<dyn Embeddings>) -> Self {
        self.embeddings = Some(e);
        self
    }
    /// Override HTTP client.
    pub fn http_client(mut self, c: reqwest::Client) -> Self {
        self.http = Some(c);
        self
    }
    /// Timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Build.
    pub fn build(self) -> Result<WeaviateProvider> {
        let embeddings = self.embeddings.ok_or_else(|| {
            CognisError::Configuration("Weaviate: embeddings provider is required".into())
        })?;
        let class = self
            .class
            .ok_or_else(|| CognisError::Configuration("Weaviate: class is required".into()))?;
        let http = match self.http {
            Some(c) => c,
            None => {
                let mut b = reqwest::ClientBuilder::new();
                if let Some(t) = self.timeout_secs {
                    b = b.timeout(std::time::Duration::from_secs(t));
                }
                b.build()
                    .map_err(|e| CognisError::Configuration(format!("HTTP client: {e}")))?
            }
        };
        Ok(WeaviateProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            class,
            api_key: self.api_key.map(|s| SecretString::new(s.into_boxed_str())),
            extra_headers: self.extra_headers,
            text_property: self
                .text_property
                .unwrap_or_else(|| DEFAULT_TEXT_KEY.to_string()),
            embeddings,
            http,
            local_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_graphql_extracts_id_text_and_metadata() {
        let raw = serde_json::json!({
            "data": {
                "Get": {
                    "Articles": [
                        {
                            "text": "doc body",
                            "topic": "rust",
                            "_additional": {
                                "id": "abc-123",
                                "distance": 0.2
                            }
                        }
                    ]
                }
            }
        });
        let out = parse_graphql_response(&raw, "Articles", "text");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "abc-123");
        assert_eq!(out[0].text, "doc body");
        assert!((out[0].score - 0.8).abs() < 1e-5);
        assert_eq!(out[0].metadata.get("topic").unwrap(), "rust");
        assert!(!out[0].metadata.contains_key("text"));
        assert!(!out[0].metadata.contains_key("_additional"));
    }

    #[test]
    fn parse_graphql_handles_missing_class() {
        let raw = serde_json::json!({"data": {"Get": {}}});
        let out = parse_graphql_response(&raw, "Articles", "text");
        assert!(out.is_empty());
    }

    #[test]
    fn builder_validates_required_fields() {
        assert!(WeaviateBuilder::default()
            .class("Articles")
            .build()
            .is_err());
        assert!(WeaviateBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .build()
            .is_err());
    }
}
