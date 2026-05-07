//! Qdrant vector-store backend (HTTP).
//!
//! Targets the Qdrant REST API. Documents are stored as points with a
//! UUID id, the embedding as the vector, and `text + user metadata` as
//! the payload.
//!
//! Customization:
//! - [`QdrantBuilder`] — base URL, collection name, API key, optional
//!   `text_payload_key` (default `"text"`).
//! - Native filter pushdown via the [`Filter`] → Qdrant `filter` JSON
//!   translator. Stores with un-translatable filters fall back to the
//!   trait's post-filter default.

#![cfg(feature = "vectorstore-qdrant")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cognis2_core::{CognisError, Result};

use crate::embeddings::Embeddings;
use crate::vectorstore::{Filter, SearchResult, VectorStore};

const DEFAULT_BASE: &str = "http://localhost:6333";
const DEFAULT_TEXT_KEY: &str = "text";

/// Qdrant-backed vector store.
pub struct QdrantProvider {
    base_url: String,
    collection: String,
    api_key: Option<SecretString>,
    extra_headers: Vec<(String, String)>,
    text_payload_key: String,
    embeddings: Arc<dyn Embeddings>,
    http: reqwest::Client,
    local_count: std::sync::atomic::AtomicUsize,
}

impl QdrantProvider {
    /// Fluent builder.
    pub fn builder() -> QdrantBuilder {
        QdrantBuilder::default()
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
                HeaderName::from_static("api-key"),
                HeaderValue::from_str(k.expose_secret())
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

    async fn upsert_inner(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadatas: Vec<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        #[derive(Serialize)]
        struct Point {
            id: String,
            vector: Vec<f32>,
            payload: HashMap<String, serde_json::Value>,
        }
        #[derive(Serialize)]
        struct Body {
            points: Vec<Point>,
        }
        let points: Vec<Point> = ids
            .iter()
            .zip(vectors)
            .zip(texts)
            .zip(metadatas)
            .map(|(((id, v), text), mut meta)| {
                meta.insert(
                    self.text_payload_key.clone(),
                    serde_json::Value::String(text),
                );
                Point {
                    id: id.clone(),
                    vector: v,
                    payload: meta,
                }
            })
            .collect();
        let url = self.endpoint(&format!("collections/{}/points?wait=true", self.collection));
        let resp = self
            .http
            .put(&url)
            .headers(self.headers()?)
            .json(&Body { points })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("qdrant upsert: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "qdrant upsert: HTTP {s}: {t}"
            )));
        }
        self.local_count
            .fetch_add(ids.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantProvider {
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        let vectors = self.embeddings.embed_documents(texts.clone()).await?;
        let ids: Vec<String> = (0..texts.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); ids.len()]);
        self.upsert_inner(ids.clone(), vectors, texts, metadatas)
            .await?;
        Ok(ids)
    }

    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        let ids: Vec<String> = (0..vectors.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); ids.len()]);
        self.upsert_inner(ids.clone(), vectors, texts, metadatas)
            .await?;
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
        #[derive(Serialize)]
        struct SearchBody {
            vector: Vec<f32>,
            limit: usize,
            with_payload: bool,
        }
        let url = self.endpoint(&format!("collections/{}/points/search", self.collection));
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&SearchBody {
                vector: query_vector,
                limit: k,
                with_payload: true,
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("qdrant search: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "qdrant search: HTTP {s}: {t}"
            )));
        }
        let body: SearchResponse = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("qdrant search json: {e}")))?;
        Ok(parse_search_response(body, &self.text_payload_key))
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        // Translate the Filter into Qdrant's `must`-list of conditions.
        let v = self.embeddings.embed_query(query.to_string()).await?;
        #[derive(Serialize)]
        struct SearchBody<'a> {
            vector: Vec<f32>,
            limit: usize,
            with_payload: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            filter: Option<&'a serde_json::Value>,
        }
        let qf = filter_to_qdrant_json(filter, &self.text_payload_key);
        let url = self.endpoint(&format!("collections/{}/points/search", self.collection));
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&SearchBody {
                vector: v,
                limit: k,
                with_payload: true,
                filter: qf.as_ref(),
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("qdrant search filter: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "qdrant search filter: HTTP {s}: {t}"
            )));
        }
        let body: SearchResponse = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("qdrant search json: {e}")))?;
        Ok(parse_search_response(body, &self.text_payload_key))
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        #[derive(Serialize)]
        struct DeleteBody {
            points: Vec<String>,
        }
        let url = self.endpoint(&format!(
            "collections/{}/points/delete?wait=true",
            self.collection
        ));
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&DeleteBody {
                points: ids.clone(),
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("qdrant delete: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "qdrant delete: HTTP {s}: {t}"
            )));
        }
        self.local_count
            .fetch_sub(ids.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn len(&self) -> usize {
        self.local_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Deserialize)]
struct SearchResponse {
    result: Vec<SearchHit>,
}

#[derive(Deserialize)]
struct SearchHit {
    id: serde_json::Value,
    score: f32,
    #[serde(default)]
    payload: HashMap<String, serde_json::Value>,
}

fn parse_search_response(body: SearchResponse, text_key: &str) -> Vec<SearchResult> {
    body.result
        .into_iter()
        .map(|h| {
            let mut metadata = h.payload;
            let text = metadata
                .remove(text_key)
                .map(|v| match v {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                })
                .unwrap_or_default();
            let id = match h.id {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            SearchResult {
                id,
                text,
                score: h.score,
                metadata,
            }
        })
        .collect()
}

/// Translate a `Filter` into Qdrant's `filter.must` JSON shape. Returns
/// `None` when the filter is empty (so the caller can omit the field).
fn filter_to_qdrant_json(f: &Filter, text_key: &str) -> Option<serde_json::Value> {
    if f.is_empty() {
        return None;
    }
    let mut must: Vec<serde_json::Value> = Vec::new();
    for (k, v) in &f.equals {
        if k == text_key {
            // We never let the user filter on the text payload key —
            // it's reserved by the store.
            continue;
        }
        must.push(serde_json::json!({"key": k, "match": {"value": v}}));
    }
    for (k, allowed) in &f.r#in {
        must.push(serde_json::json!({"key": k, "match": {"any": allowed}}));
    }
    for (k, lo) in &f.gte {
        must.push(serde_json::json!({"key": k, "range": {"gte": lo}}));
    }
    for (k, hi) in &f.lte {
        must.push(serde_json::json!({"key": k, "range": {"lte": hi}}));
    }
    Some(serde_json::json!({"must": must}))
}

/// Fluent builder for [`QdrantProvider`].
#[derive(Default)]
pub struct QdrantBuilder {
    base_url: Option<String>,
    collection: Option<String>,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    text_payload_key: Option<String>,
    embeddings: Option<Arc<dyn Embeddings>>,
    http: Option<reqwest::Client>,
    timeout_secs: Option<u64>,
}

impl QdrantBuilder {
    /// Override base URL.
    pub fn base_url(mut self, u: impl Into<String>) -> Self {
        self.base_url = Some(u.into());
        self
    }
    /// Set collection name.
    pub fn collection(mut self, c: impl Into<String>) -> Self {
        self.collection = Some(c.into());
        self
    }
    /// Set API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
    /// Override the payload key under which `text` is stored
    /// (default `"text"`).
    pub fn text_payload_key(mut self, k: impl Into<String>) -> Self {
        self.text_payload_key = Some(k.into());
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
    /// HTTP timeout in seconds.
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }
    /// Build.
    pub fn build(self) -> Result<QdrantProvider> {
        let embeddings = self.embeddings.ok_or_else(|| {
            CognisError::Configuration("Qdrant: embeddings provider is required".into())
        })?;
        let collection = self.collection.ok_or_else(|| {
            CognisError::Configuration("Qdrant: collection name is required".into())
        })?;
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
        Ok(QdrantProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            collection,
            api_key: self.api_key.map(|s| SecretString::new(s.into_boxed_str())),
            extra_headers: self.extra_headers,
            text_payload_key: self
                .text_payload_key
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
    fn parse_search_response_extracts_payload_text() {
        let body = SearchResponse {
            result: vec![SearchHit {
                id: serde_json::json!("p1"),
                score: 0.95,
                payload: HashMap::from([
                    ("text".to_string(), serde_json::json!("doc body")),
                    ("topic".to_string(), serde_json::json!("rust")),
                ]),
            }],
        };
        let out = parse_search_response(body, "text");
        assert_eq!(out[0].id, "p1");
        assert_eq!(out[0].text, "doc body");
        assert_eq!(out[0].score, 0.95);
        assert_eq!(out[0].metadata.get("topic").unwrap(), "rust");
        assert!(!out[0].metadata.contains_key("text"));
    }

    #[test]
    fn filter_translates_equals_in_gte_lte() {
        let f = Filter::new()
            .equals("topic", "rust")
            .one_of("category", ["a", "b"])
            .gte("score", 0.5)
            .lte("score", 1.0);
        let q = filter_to_qdrant_json(&f, "text").unwrap();
        let must = q["must"].as_array().unwrap();
        assert_eq!(must.len(), 4);
    }

    #[test]
    fn empty_filter_returns_none() {
        assert!(filter_to_qdrant_json(&Filter::new(), "text").is_none());
    }

    #[test]
    fn builder_requires_embeddings_and_collection() {
        assert!(QdrantBuilder::default().collection("c").build().is_err());
        assert!(QdrantBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .build()
            .is_err());
    }

    #[test]
    fn api_key_sets_api_key_header() {
        let p = QdrantBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .collection("c")
            .api_key("sk-q")
            .build()
            .unwrap();
        let h = p.headers().unwrap();
        assert_eq!(h.get("api-key").unwrap(), "sk-q");
    }
}
