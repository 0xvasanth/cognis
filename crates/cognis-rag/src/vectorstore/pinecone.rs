//! Pinecone vector-store backend (HTTP).
//!
//! Targets the serverless Pinecone HTTP API. Each provider instance
//! holds an `index_url` (e.g.
//! `https://my-index-XXX.svc.us-east1-gcp.pinecone.io`), an API key,
//! and an optional namespace.
//!
//! Customization:
//! - [`PineconeBuilder`] — index URL, API key, namespace, custom http
//!   client / timeout.
//! - [`PineconeProvider::with_extra_header`] — pass tenant headers /
//!   custom auth variants without forking.

#![cfg(feature = "vectorstore-pinecone")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::embeddings::Embeddings;
use crate::vectorstore::{Filter, SearchResult, VectorStore};

const DEFAULT_TEXT_KEY: &str = "text";

/// Pinecone-backed vector store.
pub struct PineconeProvider {
    index_url: String,
    api_key: SecretString,
    namespace: Option<String>,
    extra_headers: Vec<(String, String)>,
    text_metadata_key: String,
    embeddings: Arc<dyn Embeddings>,
    http: reqwest::Client,
    local_count: std::sync::atomic::AtomicUsize,
}

impl PineconeProvider {
    /// Fluent builder.
    pub fn builder() -> PineconeBuilder {
        PineconeBuilder::default()
    }

    fn endpoint(&self, path: &str) -> String {
        let mut s = self.index_url.clone();
        if !s.ends_with('/') {
            s.push('/');
        }
        s.push_str(path);
        s
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            HeaderName::from_static("api-key"),
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

    async fn upsert_inner(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadatas: Vec<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        #[derive(Serialize)]
        struct Vector {
            id: String,
            values: Vec<f32>,
            metadata: HashMap<String, serde_json::Value>,
        }
        #[derive(Serialize)]
        struct Body {
            vectors: Vec<Vector>,
            #[serde(skip_serializing_if = "Option::is_none")]
            namespace: Option<String>,
        }
        let vectors: Vec<Vector> = ids
            .iter()
            .zip(vectors)
            .zip(texts)
            .zip(metadatas)
            .map(|(((id, v), text), mut meta)| {
                meta.insert(
                    self.text_metadata_key.clone(),
                    serde_json::Value::String(text),
                );
                Vector {
                    id: id.clone(),
                    values: v,
                    metadata: meta,
                }
            })
            .collect();
        let url = self.endpoint("vectors/upsert");
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&Body {
                vectors,
                namespace: self.namespace.clone(),
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("pinecone upsert: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "pinecone upsert: HTTP {s}: {t}"
            )));
        }
        self.local_count
            .fetch_add(ids.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl VectorStore for PineconeProvider {
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
        struct Body {
            vector: Vec<f32>,
            #[serde(rename = "topK")]
            top_k: usize,
            #[serde(rename = "includeMetadata")]
            include_metadata: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            namespace: Option<String>,
        }
        let url = self.endpoint("query");
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&Body {
                vector: query_vector,
                top_k: k,
                include_metadata: true,
                namespace: self.namespace.clone(),
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("pinecone query: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "pinecone query: HTTP {s}: {t}"
            )));
        }
        let body: QueryResponse = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("pinecone query json: {e}")))?;
        Ok(parse_query_response(body, &self.text_metadata_key))
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        #[derive(Serialize)]
        struct Body {
            ids: Vec<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            namespace: Option<String>,
        }
        let url = self.endpoint("vectors/delete");
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&Body {
                ids: ids.clone(),
                namespace: self.namespace.clone(),
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("pinecone delete: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "pinecone delete: HTTP {s}: {t}"
            )));
        }
        self.local_count
            .fetch_sub(ids.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        // Pinecone supports a metadata filter expression on `query`.
        // For unsupported shapes, fall back to the trait default.
        let vec = self.embeddings.embed_query(query.to_string()).await?;
        let filter_json = filter_to_pinecone_json(filter);
        if filter_json.is_none() {
            return self.similarity_search_by_vector(vec, k).await;
        }
        #[derive(Serialize)]
        struct Body {
            vector: Vec<f32>,
            #[serde(rename = "topK")]
            top_k: usize,
            #[serde(rename = "includeMetadata")]
            include_metadata: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            namespace: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            filter: Option<serde_json::Value>,
        }
        let url = self.endpoint("query");
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&Body {
                vector: vec,
                top_k: k,
                include_metadata: true,
                namespace: self.namespace.clone(),
                filter: filter_json,
            })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("pinecone query (filter): {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "pinecone query: HTTP {s}: {t}"
            )));
        }
        let body: QueryResponse = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("pinecone query json: {e}")))?;
        Ok(parse_query_response(body, &self.text_metadata_key))
    }

    fn len(&self) -> usize {
        self.local_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Deserialize)]
struct QueryResponse {
    matches: Vec<QueryMatch>,
}

#[derive(Deserialize)]
struct QueryMatch {
    id: String,
    score: f32,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

fn parse_query_response(body: QueryResponse, text_key: &str) -> Vec<SearchResult> {
    body.matches
        .into_iter()
        .map(|m| {
            let mut metadata = m.metadata;
            let text = metadata
                .remove(text_key)
                .map(|v| match v {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                })
                .unwrap_or_default();
            SearchResult {
                id: m.id,
                text,
                score: m.score,
                metadata,
            }
        })
        .collect()
}

/// Translate a `Filter` to Pinecone's metadata-filter shape. Returns
/// `None` for empty filters.
fn filter_to_pinecone_json(f: &Filter) -> Option<serde_json::Value> {
    if f.is_empty() {
        return None;
    }
    let mut and: Vec<serde_json::Value> = Vec::new();
    for (k, v) in &f.equals {
        and.push(serde_json::json!({k: {"$eq": v}}));
    }
    for (k, allowed) in &f.r#in {
        and.push(serde_json::json!({k: {"$in": allowed}}));
    }
    for (k, lo) in &f.gte {
        and.push(serde_json::json!({k: {"$gte": lo}}));
    }
    for (k, hi) in &f.lte {
        and.push(serde_json::json!({k: {"$lte": hi}}));
    }
    if and.len() == 1 {
        Some(and.into_iter().next().unwrap())
    } else {
        Some(serde_json::json!({"$and": and}))
    }
}

/// Fluent builder for [`PineconeProvider`].
#[derive(Default)]
pub struct PineconeBuilder {
    index_url: Option<String>,
    api_key: Option<String>,
    namespace: Option<String>,
    extra_headers: Vec<(String, String)>,
    text_metadata_key: Option<String>,
    embeddings: Option<Arc<dyn Embeddings>>,
    http: Option<reqwest::Client>,
    timeout_secs: Option<u64>,
}

impl PineconeBuilder {
    /// Index URL (`https://my-idx-...svc.<region>.pinecone.io`).
    pub fn index_url(mut self, u: impl Into<String>) -> Self {
        self.index_url = Some(u.into());
        self
    }
    /// API key.
    pub fn api_key(mut self, k: impl Into<String>) -> Self {
        self.api_key = Some(k.into());
        self
    }
    /// Optional namespace (multi-tenant indexes).
    pub fn namespace(mut self, n: impl Into<String>) -> Self {
        self.namespace = Some(n.into());
        self
    }
    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }
    /// Override the metadata key used for `text` (default `"text"`).
    pub fn text_metadata_key(mut self, k: impl Into<String>) -> Self {
        self.text_metadata_key = Some(k.into());
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
    pub fn build(self) -> Result<PineconeProvider> {
        let embeddings = self.embeddings.ok_or_else(|| {
            CognisError::Configuration("Pinecone: embeddings provider is required".into())
        })?;
        let index_url = self
            .index_url
            .ok_or_else(|| CognisError::Configuration("Pinecone: index_url is required".into()))?;
        let api_key = self
            .api_key
            .ok_or_else(|| CognisError::Configuration("Pinecone: api_key is required".into()))?;
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
        Ok(PineconeProvider {
            index_url,
            api_key: SecretString::new(api_key.into_boxed_str()),
            namespace: self.namespace,
            extra_headers: self.extra_headers,
            text_metadata_key: self
                .text_metadata_key
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
    fn parse_query_extracts_metadata_text() {
        let body = QueryResponse {
            matches: vec![QueryMatch {
                id: "p1".into(),
                score: 0.7,
                metadata: HashMap::from([
                    ("text".to_string(), serde_json::json!("hello")),
                    ("topic".to_string(), serde_json::json!("rust")),
                ]),
            }],
        };
        let r = parse_query_response(body, "text");
        assert_eq!(r[0].text, "hello");
        assert_eq!(r[0].metadata.get("topic").unwrap(), "rust");
    }

    #[test]
    fn filter_translates_to_and_clauses() {
        let f = Filter::new().equals("topic", "rust").gte("score", 0.5);
        let json = filter_to_pinecone_json(&f).unwrap();
        assert!(json.get("$and").is_some());
    }

    #[test]
    fn single_clause_filter_unwraps() {
        let f = Filter::new().equals("topic", "rust");
        let json = filter_to_pinecone_json(&f).unwrap();
        // No $and wrapper for single clause.
        assert!(json.get("$and").is_none());
        assert_eq!(json["topic"]["$eq"], "rust");
    }

    #[test]
    fn empty_filter_is_none() {
        assert!(filter_to_pinecone_json(&Filter::new()).is_none());
    }

    #[test]
    fn builder_validates_required_fields() {
        assert!(PineconeBuilder::default()
            .index_url("u")
            .api_key("k")
            .build()
            .is_err()); // missing embeddings
        assert!(PineconeBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .api_key("k")
            .build()
            .is_err()); // missing url
    }
}
