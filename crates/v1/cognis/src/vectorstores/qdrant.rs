//! Qdrant vector store implementation.
//!
//! Provides a `QdrantVectorStore` that implements the `VectorStore` trait,
//! backed by a pluggable `QdrantClient` trait for abstracting HTTP/gRPC calls.
//! Includes a `MockQdrantClient` for testing without a running Qdrant instance.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;
use cognis_core::vectorstores::base::VectorStore;

// ---------------------------------------------------------------------------
// Distance metric
// ---------------------------------------------------------------------------

/// Distance metric used for vector similarity comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    /// Cosine similarity (normalized dot product).
    #[default]
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot product (inner product).
    DotProduct,
}

// ---------------------------------------------------------------------------
// QdrantConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Qdrant instance.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Base URL of the Qdrant REST API.
    pub url: String,
    /// Name of the collection to operate on.
    pub collection_name: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Optional gRPC port (default Qdrant gRPC port is 6334).
    pub grpc_port: Option<u16>,
    /// Whether to prefer gRPC over REST.
    pub prefer_grpc: bool,
    /// Distance metric for the collection.
    pub distance: DistanceMetric,
}

impl QdrantConfig {
    /// Create a new config with the given collection name and sensible defaults.
    pub fn new(collection_name: impl Into<String>) -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            collection_name: collection_name.into(),
            api_key: None,
            grpc_port: None,
            prefer_grpc: false,
            distance: DistanceMetric::default(),
        }
    }

    /// Set the base URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the gRPC port.
    pub fn with_grpc_port(mut self, port: u16) -> Self {
        self.grpc_port = Some(port);
        self
    }

    /// Prefer gRPC transport.
    pub fn with_prefer_grpc(mut self, prefer: bool) -> Self {
        self.prefer_grpc = prefer;
        self
    }

    /// Set the distance metric.
    pub fn with_distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }
}

// ---------------------------------------------------------------------------
// QdrantPoint
// ---------------------------------------------------------------------------

/// A point stored in a Qdrant collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantPoint {
    /// Unique identifier for the point.
    pub id: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Arbitrary JSON payload (metadata).
    pub payload: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// QdrantFilter
// ---------------------------------------------------------------------------

/// A single filter condition on a payload field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantFieldCondition {
    /// The payload field name.
    pub key: String,
    /// The value to match against.
    pub value: Value,
}

/// Filter for constraining search results by metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QdrantFilter {
    /// All conditions must match.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must: Vec<QdrantFieldCondition>,
    /// At least one condition must match.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub should: Vec<QdrantFieldCondition>,
    /// No condition must match.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must_not: Vec<QdrantFieldCondition>,
}

impl QdrantFilter {
    /// Create an empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a must-match condition.
    pub fn must(mut self, key: impl Into<String>, value: Value) -> Self {
        self.must.push(QdrantFieldCondition {
            key: key.into(),
            value,
        });
        self
    }

    /// Add a should-match condition.
    pub fn should(mut self, key: impl Into<String>, value: Value) -> Self {
        self.should.push(QdrantFieldCondition {
            key: key.into(),
            value,
        });
        self
    }

    /// Add a must-not-match condition.
    pub fn must_not(mut self, key: impl Into<String>, value: Value) -> Self {
        self.must_not.push(QdrantFieldCondition {
            key: key.into(),
            value,
        });
        self
    }

    /// Returns true if this filter has no conditions.
    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Collection configuration
// ---------------------------------------------------------------------------

/// Configuration for creating a Qdrant collection.
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    /// Dimensionality of the vectors.
    pub vector_size: usize,
    /// Distance metric.
    pub distance: DistanceMetric,
}

// ---------------------------------------------------------------------------
// QdrantClient trait
// ---------------------------------------------------------------------------

/// Abstraction over Qdrant API calls, enabling mocking for tests.
#[async_trait]
pub trait QdrantClient: Send + Sync {
    /// Insert or update points in a collection.
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()>;

    /// Search for the nearest vectors, optionally filtered.
    async fn search_points(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<&QdrantFilter>,
    ) -> Result<Vec<(QdrantPoint, f32)>>;

    /// Delete points by their IDs.
    async fn delete_points(&self, collection: &str, ids: &[String]) -> Result<bool>;

    /// Retrieve points by their IDs.
    async fn get_points(&self, collection: &str, ids: &[String]) -> Result<Vec<QdrantPoint>>;

    /// Create a new collection with the given configuration.
    async fn create_collection(&self, collection: &str, config: CollectionConfig) -> Result<()>;
}

// ---------------------------------------------------------------------------
// MockQdrantClient
// ---------------------------------------------------------------------------

/// In-memory mock implementation of `QdrantClient` for testing.
pub struct MockQdrantClient {
    /// Points stored per collection.
    collections: RwLock<HashMap<String, Vec<QdrantPoint>>>,
    /// Distance metric per collection.
    distances: RwLock<HashMap<String, DistanceMetric>>,
}

impl MockQdrantClient {
    /// Create a new empty mock client.
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            distances: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MockQdrantClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute similarity score between two vectors using the given metric.
fn compute_score(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a * norm_b)
            }
        }
        DistanceMetric::Euclidean => {
            let dist: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            // Convert distance to similarity: higher is more similar.
            1.0 / (1.0 + dist)
        }
        DistanceMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
    }
}

/// Check whether a point's payload matches a filter.
fn matches_filter(payload: &HashMap<String, Value>, filter: &QdrantFilter) -> bool {
    // Must: all conditions must match.
    let must_ok = filter.must.iter().all(|cond| {
        payload
            .get(&cond.key)
            .map(|v| v == &cond.value)
            .unwrap_or(false)
    });

    // Should: at least one condition must match (if any are specified).
    let should_ok = filter.should.is_empty()
        || filter.should.iter().any(|cond| {
            payload
                .get(&cond.key)
                .map(|v| v == &cond.value)
                .unwrap_or(false)
        });

    // Must not: no condition may match.
    let must_not_ok = filter.must_not.iter().all(|cond| {
        payload
            .get(&cond.key)
            .map(|v| v != &cond.value)
            .unwrap_or(true)
    });

    must_ok && should_ok && must_not_ok
}

#[async_trait]
impl QdrantClient for MockQdrantClient {
    async fn upsert_points(&self, collection: &str, points: Vec<QdrantPoint>) -> Result<()> {
        let mut collections = self.collections.write().await;
        let coll = collections
            .entry(collection.to_string())
            .or_insert_with(Vec::new);

        for point in points {
            // Remove existing point with the same ID, then insert.
            coll.retain(|p| p.id != point.id);
            coll.push(point);
        }
        Ok(())
    }

    async fn search_points(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<&QdrantFilter>,
    ) -> Result<Vec<(QdrantPoint, f32)>> {
        let collections = self.collections.read().await;
        let distances = self.distances.read().await;
        let metric = distances
            .get(collection)
            .copied()
            .unwrap_or(DistanceMetric::Cosine);

        let Some(coll) = collections.get(collection) else {
            return Ok(vec![]);
        };

        let mut scored: Vec<(QdrantPoint, f32)> = coll
            .iter()
            .filter(|p| {
                filter
                    .map(|f| matches_filter(&p.payload, f))
                    .unwrap_or(true)
            })
            .map(|p| {
                let score = compute_score(vector, &p.vector, metric);
                (p.clone(), score)
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    async fn delete_points(&self, collection: &str, ids: &[String]) -> Result<bool> {
        let mut collections = self.collections.write().await;
        let Some(coll) = collections.get_mut(collection) else {
            return Ok(false);
        };
        let before = coll.len();
        coll.retain(|p| !ids.contains(&p.id));
        Ok(coll.len() < before)
    }

    async fn get_points(&self, collection: &str, ids: &[String]) -> Result<Vec<QdrantPoint>> {
        let collections = self.collections.read().await;
        let Some(coll) = collections.get(collection) else {
            return Ok(vec![]);
        };
        Ok(coll
            .iter()
            .filter(|p| ids.contains(&p.id))
            .cloned()
            .collect())
    }

    async fn create_collection(&self, collection: &str, config: CollectionConfig) -> Result<()> {
        let mut collections = self.collections.write().await;
        collections
            .entry(collection.to_string())
            .or_insert_with(Vec::new);

        let mut distances = self.distances.write().await;
        distances.insert(collection.to_string(), config.distance);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QdrantVectorStore
// ---------------------------------------------------------------------------

/// Vector store backed by Qdrant (or a mock thereof).
///
/// Wraps a `QdrantClient` implementation and an `Embeddings` model to provide
/// the full `VectorStore` trait interface.
pub struct QdrantVectorStore {
    client: Arc<dyn QdrantClient>,
    embeddings: Arc<dyn Embeddings>,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    /// Create a new Qdrant vector store.
    pub fn new(
        client: Arc<dyn QdrantClient>,
        embeddings: Arc<dyn Embeddings>,
        config: QdrantConfig,
    ) -> Self {
        Self {
            client,
            embeddings,
            config,
        }
    }

    /// Create a Qdrant vector store pre-populated with documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        client: Arc<dyn QdrantClient>,
        embeddings: Arc<dyn Embeddings>,
        config: QdrantConfig,
    ) -> Result<Self> {
        let store = Self::new(client, embeddings, config);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Ensure the collection exists, creating it if necessary.
    pub async fn ensure_collection(&self, vector_size: usize) -> Result<()> {
        self.client
            .create_collection(
                &self.config.collection_name,
                CollectionConfig {
                    vector_size,
                    distance: self.config.distance,
                },
            )
            .await
    }

    /// Perform a similarity search with an optional metadata filter, returning scores.
    pub async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: Option<&QdrantFilter>,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let results = self
            .client
            .search_points(&self.config.collection_name, &query_embedding, k, filter)
            .await?;

        Ok(results
            .into_iter()
            .map(|(point, score)| {
                let content = point
                    .payload
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = point.payload.clone();
                metadata.remove("page_content");

                let doc = Document::new(content)
                    .with_id(point.id)
                    .with_metadata(metadata);

                (doc, score)
            })
            .collect())
    }

    /// Return a reference to the underlying config.
    pub fn config(&self) -> &QdrantConfig {
        &self.config
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;

        let mut points = Vec::with_capacity(texts.len());
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let mut payload: HashMap<String, Value> = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            // Store page_content in the payload so we can reconstruct documents.
            payload.insert("page_content".to_string(), Value::String(text.clone()));

            points.push(QdrantPoint {
                id: id.clone(),
                vector: embeddings_vec[i].clone(),
                payload,
            });

            result_ids.push(id);
        }

        self.client
            .upsert_points(&self.config.collection_name, points)
            .await?;

        Ok(result_ids)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
        ids: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        let id_refs: Option<Vec<String>> = ids.or_else(|| {
            let doc_ids: Vec<String> = documents.iter().filter_map(|d| d.id.clone()).collect();
            if doc_ids.len() == documents.len() {
                Some(doc_ids)
            } else {
                None
            }
        });
        let id_slice_ref: Option<&[String]> = id_refs.as_deref();
        self.add_texts(&texts, Some(&metadatas), id_slice_ref).await
    }

    async fn delete(&self, ids: Option<&[String]>) -> Result<bool> {
        let Some(ids) = ids else {
            return Ok(false);
        };
        self.client
            .delete_points(&self.config.collection_name, ids)
            .await
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let points = self
            .client
            .get_points(&self.config.collection_name, ids)
            .await?;

        Ok(points
            .into_iter()
            .map(|point| {
                let content = point
                    .payload
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = point.payload.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(point.id)
                    .with_metadata(metadata)
            })
            .collect())
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let results = self.similarity_search_with_score(query, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }

    async fn similarity_search_with_score(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        self.similarity_search_with_filter(query, k, None).await
    }

    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        let results = self
            .client
            .search_points(&self.config.collection_name, embedding, k, None)
            .await?;

        Ok(results
            .into_iter()
            .map(|(point, _)| {
                let content = point
                    .payload
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = point.payload.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(point.id)
                    .with_metadata(metadata)
            })
            .collect())
    }

    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let results = self
            .client
            .search_points(
                &self.config.collection_name,
                &query_embedding,
                fetch_k,
                None,
            )
            .await?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        let candidate_embeddings: Vec<Vec<f64>> = results
            .iter()
            .map(|(p, _)| p.vector.iter().map(|&v| v as f64).collect())
            .collect();
        let query_emb_f64: Vec<f64> = query_embedding.iter().map(|&v| v as f64).collect();

        let mmr_indices = cognis_core::vectorstores::utils::maximal_marginal_relevance(
            &query_emb_f64,
            &candidate_embeddings,
            lambda_mult as f64,
            k,
        );

        let docs = mmr_indices
            .into_iter()
            .filter_map(|idx| results.get(idx))
            .map(|(point, _)| {
                let content = point
                    .payload
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = point.payload.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(point.id.clone())
                    .with_metadata(metadata)
            })
            .collect();

        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    fn make_store() -> QdrantVectorStore {
        let client = Arc::new(MockQdrantClient::new());
        let embeddings = make_embeddings();
        let config = QdrantConfig::new("test_collection");
        QdrantVectorStore::new(client, embeddings, config)
    }

    fn make_store_with_metric(
        metric: DistanceMetric,
    ) -> (QdrantVectorStore, Arc<MockQdrantClient>) {
        let client = Arc::new(MockQdrantClient::new());
        let embeddings = make_embeddings();
        let config = QdrantConfig::new("test_collection").with_distance(metric);
        let store = QdrantVectorStore::new(client.clone(), embeddings, config);
        (store, client)
    }

    // ---- Test 1: Add and search documents ----
    #[tokio::test]
    async fn test_add_and_search_documents() {
        let store = make_store();
        let docs = vec![
            Document::new("Rust is fast").with_id("d1"),
            Document::new("Python is dynamic").with_id("d2"),
            Document::new("Rust has zero-cost abstractions").with_id("d3"),
        ];
        let ids = store.add_documents(docs, None).await.unwrap();
        assert_eq!(ids.len(), 3);

        let results = store.similarity_search("Rust", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    // ---- Test 2: Similarity search with scores ----
    #[tokio::test]
    async fn test_similarity_search_with_scores() {
        let store = make_store();
        let texts = vec!["cat".into(), "dog".into(), "fish".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search_with_score("cat", 3).await.unwrap();
        assert_eq!(results.len(), 3);
        // First result should be "cat" with highest score.
        assert_eq!(results[0].0.page_content, "cat");
        // Scores in descending order.
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    // ---- Test 3: Metadata filtering with must ----
    #[tokio::test]
    async fn test_metadata_filter_must() {
        let store = make_store();
        let texts = vec!["apple".into(), "banana".into(), "cherry".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("color".into(), Value::String("red".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("color".into(), Value::String("yellow".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("color".into(), Value::String("red".into()));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = QdrantFilter::new().must("color", Value::String("red".into()));
        let results = store
            .similarity_search_with_filter("fruit", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            assert_eq!(
                doc.metadata.get("color").unwrap(),
                &Value::String("red".into())
            );
        }
    }

    // ---- Test 4: Metadata filtering with should ----
    #[tokio::test]
    async fn test_metadata_filter_should() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("type".into(), Value::String("x".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("type".into(), Value::String("y".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("type".into(), Value::String("z".into()));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = QdrantFilter::new()
            .should("type", Value::String("x".into()))
            .should("type", Value::String("z".into()));
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let t = doc.metadata.get("type").unwrap().as_str().unwrap();
            assert!(t == "x" || t == "z");
        }
    }

    // ---- Test 5: Metadata filtering with must_not ----
    #[tokio::test]
    async fn test_metadata_filter_must_not() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("draft".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("published".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("draft".into()));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = QdrantFilter::new().must_not("status", Value::String("draft".into()));
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0.metadata.get("status").unwrap(),
            &Value::String("published".into())
        );
    }

    // ---- Test 6: Delete documents ----
    #[tokio::test]
    async fn test_delete_documents() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&[ids[1].clone()])).await.unwrap();
        assert!(deleted);

        let remaining = store.similarity_search("a", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|d| d.page_content != "b"));
    }

    // ---- Test 7: Config defaults ----
    #[tokio::test]
    async fn test_config_defaults() {
        let config = QdrantConfig::new("my_collection");
        assert_eq!(config.url, "http://localhost:6333");
        assert_eq!(config.collection_name, "my_collection");
        assert!(config.api_key.is_none());
        assert!(config.grpc_port.is_none());
        assert!(!config.prefer_grpc);
        assert_eq!(config.distance, DistanceMetric::Cosine);
    }

    // ---- Test 8: Distance metric - Euclidean ----
    #[tokio::test]
    async fn test_euclidean_distance_metric() {
        let (store, client) = make_store_with_metric(DistanceMetric::Euclidean);
        client
            .create_collection(
                "test_collection",
                CollectionConfig {
                    vector_size: 16,
                    distance: DistanceMetric::Euclidean,
                },
            )
            .await
            .unwrap();

        let texts = vec!["near".into(), "far".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search_with_score("near", 2).await.unwrap();
        assert_eq!(results.len(), 2);
        // First result should be the most similar.
        assert_eq!(results[0].0.page_content, "near");
        assert!(results[0].1 >= results[1].1);
    }

    // ---- Test 9: Distance metric - DotProduct ----
    #[tokio::test]
    async fn test_dot_product_distance_metric() {
        let (store, client) = make_store_with_metric(DistanceMetric::DotProduct);
        client
            .create_collection(
                "test_collection",
                CollectionConfig {
                    vector_size: 16,
                    distance: DistanceMetric::DotProduct,
                },
            )
            .await
            .unwrap();

        let texts = vec!["hello".into(), "world".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store
            .similarity_search_with_score("hello", 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.page_content, "hello");
    }

    // ---- Test 10: Empty collection search ----
    #[tokio::test]
    async fn test_empty_collection_search() {
        let store = make_store();
        let results = store.similarity_search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Test 11: Large batch upsert ----
    #[tokio::test]
    async fn test_large_batch_upsert() {
        let store = make_store();
        let texts: Vec<String> = (0..100).map(|i| format!("document_{}", i)).collect();
        let ids = store.add_texts(&texts, None, None).await.unwrap();
        assert_eq!(ids.len(), 100);

        let results = store.similarity_search("document_50", 5).await.unwrap();
        assert_eq!(results.len(), 5);
    }

    // ---- Test 12: Mock client upsert and get ----
    #[tokio::test]
    async fn test_mock_client_upsert_and_get() {
        let client = MockQdrantClient::new();
        let points = vec![
            QdrantPoint {
                id: "p1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: HashMap::new(),
            },
            QdrantPoint {
                id: "p2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                payload: HashMap::new(),
            },
        ];

        client.upsert_points("coll", points).await.unwrap();

        let retrieved = client
            .get_points("coll", &["p1".into(), "p2".into()])
            .await
            .unwrap();
        assert_eq!(retrieved.len(), 2);

        // Upsert with same ID replaces.
        let updated = vec![QdrantPoint {
            id: "p1".to_string(),
            vector: vec![0.5, 0.5, 0.0],
            payload: HashMap::new(),
        }];
        client.upsert_points("coll", updated).await.unwrap();

        let after = client.get_points("coll", &["p1".into()]).await.unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].vector, vec![0.5, 0.5, 0.0]);
    }

    // ---- Test 13: Filter combinations (must + must_not) ----
    #[tokio::test]
    async fn test_filter_combinations() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("category".into(), Value::String("food".into()));
                m.insert("organic".into(), Value::Bool(true));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("category".into(), Value::String("food".into()));
                m.insert("organic".into(), Value::Bool(false));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("category".into(), Value::String("drink".into()));
                m.insert("organic".into(), Value::Bool(true));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("category".into(), Value::String("drink".into()));
                m.insert("organic".into(), Value::Bool(false));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        // Must be food AND must not be organic
        let filter = QdrantFilter::new()
            .must("category", Value::String("food".into()))
            .must_not("organic", Value::Bool(true));
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.page_content, "b");
    }

    // ---- Test 14: get_by_ids returns correct documents ----
    #[tokio::test]
    async fn test_get_by_ids() {
        let store = make_store();
        let texts = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let custom_ids = vec!["id-a".to_string(), "id-b".to_string(), "id-c".to_string()];
        store
            .add_texts(&texts, None, Some(&custom_ids))
            .await
            .unwrap();

        let docs = store
            .get_by_ids(&["id-a".into(), "id-c".into()])
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);

        let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
        assert!(contents.contains(&"alpha"));
        assert!(contents.contains(&"gamma"));
    }

    // ---- Test 15: Delete with None returns false ----
    #[tokio::test]
    async fn test_delete_none_returns_false() {
        let store = make_store();
        let result = store.delete(None).await.unwrap();
        assert!(!result);
    }

    // ---- Test 16: from_documents constructor ----
    #[tokio::test]
    async fn test_from_documents_constructor() {
        let client = Arc::new(MockQdrantClient::new());
        let embeddings = make_embeddings();
        let config = QdrantConfig::new("test_collection");

        let docs = vec![
            Document::new("hello world").with_id("h1"),
            Document::new("goodbye world").with_id("g1"),
        ];

        let store = QdrantVectorStore::from_documents(docs, client, embeddings, config)
            .await
            .unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }
}
