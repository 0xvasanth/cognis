//! Chroma vector-store backend (HTTP).
//!
//! Targets the Chroma OSS HTTP API (`/api/v1`). The store holds a
//! collection name + `Arc<dyn Embeddings>` and translates VectorStore
//! trait calls into the corresponding Chroma REST calls.
//!
//! Customization:
//! - [`ChromaBuilder`] — base URL, collection name, optional auth
//!   token, custom `reqwest::Client`.
//! - [`ChromaProvider::with_extra_header`] — add tenant headers / auth
//!   variants without forking.

#![cfg(feature = "vectorstore-chroma")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::embeddings::Embeddings;
use crate::vectorstore::{Filter, SearchResult, VectorStore};

const DEFAULT_BASE: &str = "http://localhost:8000/api/v1";

/// Chroma-backed vector store.
pub struct ChromaProvider {
    base_url: String,
    collection: String,
    auth_token: Option<SecretString>,
    extra_headers: Vec<(String, String)>,
    embeddings: Arc<dyn Embeddings>,
    http: reqwest::Client,
    /// Cached collection-id; populated lazily.
    collection_id: tokio::sync::OnceCell<String>,
    /// Local document count — Chroma returns this on add but having
    /// our own counter avoids an extra round-trip for `len()`.
    local_count: std::sync::atomic::AtomicUsize,
}

impl ChromaProvider {
    /// Fluent builder.
    pub fn builder() -> ChromaBuilder {
        ChromaBuilder::default()
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
        if let Some(t) = &self.auth_token {
            h.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", t.expose_secret()))
                    .map_err(|e| CognisError::Configuration(format!("invalid auth token: {e}")))?,
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

    /// Resolve the collection id (creating it if needed).
    async fn ensure_collection(&self) -> Result<String> {
        if let Some(id) = self.collection_id.get() {
            return Ok(id.clone());
        }
        // Look up by name; on 404 create.
        #[derive(Deserialize)]
        struct CollectionView {
            id: String,
        }
        let url = self.endpoint(&format!("collections/{}", self.collection));
        let resp = self
            .http
            .get(&url)
            .headers(self.headers()?)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("chroma get collection: {e}")))?;
        let id = if resp.status().is_success() {
            resp.json::<CollectionView>()
                .await
                .map_err(|e| CognisError::Serialization(format!("chroma collection json: {e}")))?
                .id
        } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
            #[derive(Serialize)]
            struct CreateBody<'a> {
                name: &'a str,
                #[serde(skip_serializing_if = "Option::is_none")]
                metadata: Option<HashMap<String, serde_json::Value>>,
            }
            let create_url = self.endpoint("collections");
            let body = CreateBody {
                name: &self.collection,
                metadata: None,
            };
            let create_resp = self
                .http
                .post(&create_url)
                .headers(self.headers()?)
                .json(&body)
                .send()
                .await
                .map_err(|e| CognisError::Internal(format!("chroma create collection: {e}")))?;
            if !create_resp.status().is_success() {
                let s = create_resp.status();
                let t = create_resp.text().await.unwrap_or_default();
                return Err(CognisError::Internal(format!(
                    "chroma create collection: HTTP {s}: {t}"
                )));
            }
            create_resp
                .json::<CollectionView>()
                .await
                .map_err(|e| CognisError::Serialization(format!("chroma create json: {e}")))?
                .id
        } else {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "chroma get collection: HTTP {s}: {t}"
            )));
        };
        let _ = self.collection_id.set(id.clone());
        Ok(id)
    }

    async fn add_inner(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadatas: Vec<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        let collection_id = self.ensure_collection().await?;
        #[derive(Serialize)]
        struct AddBody {
            ids: Vec<String>,
            embeddings: Vec<Vec<f32>>,
            documents: Vec<String>,
            metadatas: Vec<HashMap<String, serde_json::Value>>,
        }
        let url = self.endpoint(&format!("collections/{collection_id}/add"));
        let body = AddBody {
            ids: ids.clone(),
            embeddings: vectors,
            documents: texts,
            metadatas,
        };
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("chroma add: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!("chroma add: HTTP {s}: {t}")));
        }
        self.local_count
            .fetch_add(ids.len(), std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl VectorStore for ChromaProvider {
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        let vectors = self.embeddings.embed_documents(texts.clone()).await?;
        let mut ids: Vec<String> = (0..texts.len())
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let metadatas = metadata.unwrap_or_else(|| vec![HashMap::new(); ids.len()]);
        let texts_clone = texts.clone();
        self.add_inner(ids.clone(), vectors, texts_clone, metadatas)
            .await?;
        ids.shrink_to_fit();
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
        self.add_inner(ids.clone(), vectors, texts, metadatas)
            .await?;
        Ok(ids)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let vec = self.embeddings.embed_query(query.to_string()).await?;
        self.similarity_search_by_vector(vec, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let collection_id = self.ensure_collection().await?;
        #[derive(Serialize)]
        struct QueryBody {
            query_embeddings: Vec<Vec<f32>>,
            n_results: usize,
        }
        let url = self.endpoint(&format!("collections/{collection_id}/query"));
        let body = QueryBody {
            query_embeddings: vec![query_vector],
            n_results: k,
        };
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("chroma query: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "chroma query: HTTP {s}: {t}"
            )));
        }
        let q: QueryResponseLike = resp
            .json()
            .await
            .map_err(|e| CognisError::Serialization(format!("chroma query json: {e}")))?;
        Ok(parse_query_response(q))
    }

    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        // Chroma supports `where` filters natively, but the trait's
        // default impl (post-filter) is correct. Override only when we
        // can map the Filter cleanly to Chroma's syntax.
        let vec = self.embeddings.embed_query(query.to_string()).await?;
        let candidates = self
            .similarity_search_by_vector(vec, k.saturating_mul(4))
            .await?;
        Ok(candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(k)
            .collect())
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let collection_id = self.ensure_collection().await?;
        #[derive(Serialize)]
        struct DeleteBody {
            ids: Vec<String>,
        }
        let url = self.endpoint(&format!("collections/{collection_id}/delete"));
        let resp = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&DeleteBody { ids: ids.clone() })
            .send()
            .await
            .map_err(|e| CognisError::Internal(format!("chroma delete: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(CognisError::Internal(format!(
                "chroma delete: HTTP {s}: {t}"
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

fn parse_query_response(q: QueryResponseLike) -> Vec<SearchResult> {
    let row_ids = q.ids.into_iter().next().unwrap_or_default();
    let row_docs = q.documents.into_iter().next().unwrap_or_default();
    let row_metas = q.metadatas.into_iter().next().unwrap_or_default();
    let row_dists = q.distances.into_iter().next().unwrap_or_default();
    row_ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| SearchResult {
            id,
            text: row_docs.get(i).cloned().unwrap_or_default(),
            score: row_dists
                .get(i)
                .copied()
                .map(|d| 1.0 - d) // chroma returns distance; convert to similarity
                .unwrap_or(0.0),
            metadata: row_metas.get(i).cloned().flatten().unwrap_or_default(),
        })
        .collect()
}

#[derive(Deserialize)]
struct QueryResponseLike {
    ids: Vec<Vec<String>>,
    documents: Vec<Vec<String>>,
    #[serde(default)]
    metadatas: Vec<Vec<Option<HashMap<String, serde_json::Value>>>>,
    #[serde(default)]
    distances: Vec<Vec<f32>>,
}

/// Fluent builder for [`ChromaProvider`].
#[derive(Default)]
pub struct ChromaBuilder {
    base_url: Option<String>,
    collection: Option<String>,
    auth_token: Option<String>,
    extra_headers: Vec<(String, String)>,
    embeddings: Option<Arc<dyn Embeddings>>,
    http: Option<reqwest::Client>,
    timeout_secs: Option<u64>,
}

impl ChromaBuilder {
    /// Override the base URL (default `http://localhost:8000/api/v1`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set the collection name.
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.collection = Some(name.into());
        self
    }

    /// Set bearer auth token.
    pub fn auth_token(mut self, t: impl Into<String>) -> Self {
        self.auth_token = Some(t.into());
        self
    }

    /// Add an extra header.
    pub fn extra_header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.extra_headers.push((k.into(), v.into()));
        self
    }

    /// Embeddings provider (required).
    pub fn embeddings(mut self, e: Arc<dyn Embeddings>) -> Self {
        self.embeddings = Some(e);
        self
    }

    /// Override the HTTP client (e.g. for a custom connector / proxy).
    pub fn http_client(mut self, c: reqwest::Client) -> Self {
        self.http = Some(c);
        self
    }

    /// HTTP timeout in seconds (used only when no custom http client is supplied).
    pub fn timeout_secs(mut self, s: u64) -> Self {
        self.timeout_secs = Some(s);
        self
    }

    /// Build.
    pub fn build(self) -> Result<ChromaProvider> {
        let embeddings = self.embeddings.ok_or_else(|| {
            CognisError::Configuration("Chroma: embeddings provider is required".into())
        })?;
        let collection = self.collection.ok_or_else(|| {
            CognisError::Configuration("Chroma: collection name is required".into())
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
        Ok(ChromaProvider {
            base_url: self.base_url.unwrap_or_else(|| DEFAULT_BASE.to_string()),
            collection,
            auth_token: self
                .auth_token
                .map(|s| SecretString::new(s.into_boxed_str())),
            extra_headers: self.extra_headers,
            embeddings,
            http,
            collection_id: tokio::sync::OnceCell::new(),
            local_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_response_extracts_first_row() {
        let q = QueryResponseLike {
            ids: vec![vec!["a".into(), "b".into()]],
            documents: vec![vec!["doc-a".into(), "doc-b".into()]],
            metadatas: vec![vec![
                Some(HashMap::from([("k".into(), serde_json::json!(1))])),
                None,
            ]],
            distances: vec![vec![0.1, 0.4]],
        };
        let out = parse_query_response(q);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "a");
        assert!((out[0].score - 0.9).abs() < 1e-5);
        assert_eq!(out[1].metadata.len(), 0);
    }

    #[test]
    fn builder_requires_embeddings_and_collection() {
        let res = ChromaBuilder::default().collection("c").build();
        assert!(res.is_err());
        let res = ChromaBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .build();
        assert!(res.is_err());
    }

    #[test]
    fn builder_succeeds_with_required_fields() {
        let p = ChromaBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .collection("c")
            .build()
            .unwrap();
        assert_eq!(p.collection, "c");
        assert_eq!(p.base_url, DEFAULT_BASE);
    }

    #[test]
    fn extra_header_round_trips() {
        let p = ChromaBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .collection("c")
            .extra_header("X-Tenant", "t1")
            .build()
            .unwrap();
        let h = p.headers().unwrap();
        assert_eq!(h.get("x-tenant").unwrap(), "t1");
    }

    #[test]
    fn auth_token_sets_authorization_header() {
        let p = ChromaBuilder::default()
            .embeddings(Arc::new(crate::embeddings::FakeEmbeddings::new(4)))
            .collection("c")
            .auth_token("sk-test")
            .build()
            .unwrap();
        let h = p.headers().unwrap();
        assert_eq!(h.get(AUTHORIZATION).unwrap(), "Bearer sk-test");
    }
}
