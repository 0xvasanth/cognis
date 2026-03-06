//! Pinecone vector store implementation.
//!
//! Provides a `PineconeVectorStore` that implements the `VectorStore` trait,
//! backed by a pluggable `PineconeClient` trait for abstracting API calls.
//! Includes a `MockPineconeClient` for testing without a running Pinecone instance.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::Result;
use rustchain_core::vectorstores::base::VectorStore;

// ---------------------------------------------------------------------------
// Distance metric
// ---------------------------------------------------------------------------

/// Distance metric used by a Pinecone index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PineconeMetric {
    /// Cosine similarity.
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Dot product (inner product).
    DotProduct,
}

impl Default for PineconeMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

// ---------------------------------------------------------------------------
// PineconeConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Pinecone index.
#[derive(Debug, Clone)]
pub struct PineconeConfig {
    /// Pinecone API key.
    pub api_key: String,
    /// Pinecone environment (e.g. "us-east-1-aws").
    pub environment: String,
    /// Name of the Pinecone index.
    pub index_name: String,
    /// Optional namespace for vector isolation.
    pub namespace: Option<String>,
    /// Distance metric for the index.
    pub metric: PineconeMetric,
    /// Dimensionality of vectors in the index.
    pub dimension: usize,
}

impl PineconeConfig {
    /// Create a new config with the given index name and required fields.
    pub fn new(
        api_key: impl Into<String>,
        environment: impl Into<String>,
        index_name: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            environment: environment.into(),
            index_name: index_name.into(),
            namespace: None,
            metric: PineconeMetric::default(),
            dimension,
        }
    }

    /// Set the namespace.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: PineconeMetric) -> Self {
        self.metric = metric;
        self
    }
}

// ---------------------------------------------------------------------------
// PineconeVector
// ---------------------------------------------------------------------------

/// A vector stored in a Pinecone index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeVector {
    /// Unique identifier for the vector.
    pub id: String,
    /// The embedding values.
    pub values: Vec<f32>,
    /// Arbitrary metadata associated with the vector.
    pub metadata: HashMap<String, Value>,
    /// Optional sparse vector values (for hybrid search).
    pub sparse_values: Option<PineconeSparseValues>,
}

/// Sparse vector representation for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeSparseValues {
    /// Indices of non-zero dimensions.
    pub indices: Vec<u32>,
    /// Values at those indices.
    pub values: Vec<f32>,
}

// ---------------------------------------------------------------------------
// PineconeQueryResult
// ---------------------------------------------------------------------------

/// A single result from a Pinecone query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeQueryResult {
    /// The vector ID.
    pub id: String,
    /// Similarity score.
    pub score: f32,
    /// Metadata associated with the vector.
    pub metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// PineconeFilter
// ---------------------------------------------------------------------------

/// Metadata filter for Pinecone queries.
///
/// Supports Pinecone's metadata filtering operators:
/// `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeFilter {
    /// Filter conditions: field name -> operator -> value.
    pub conditions: Vec<PineconeFilterCondition>,
}

/// A single filter condition on a metadata field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeFilterCondition {
    /// The metadata field name.
    pub field: String,
    /// The filter operator.
    pub operator: PineconeFilterOperator,
    /// The value to compare against.
    pub value: Value,
}

/// Supported Pinecone metadata filter operators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PineconeFilterOperator {
    /// Equal to.
    #[serde(rename = "$eq")]
    Eq,
    /// Not equal to.
    #[serde(rename = "$ne")]
    Ne,
    /// Greater than.
    #[serde(rename = "$gt")]
    Gt,
    /// Greater than or equal to.
    #[serde(rename = "$gte")]
    Gte,
    /// Less than.
    #[serde(rename = "$lt")]
    Lt,
    /// Less than or equal to.
    #[serde(rename = "$lte")]
    Lte,
    /// In set.
    #[serde(rename = "$in")]
    In,
    /// Not in set.
    #[serde(rename = "$nin")]
    Nin,
}

impl PineconeFilter {
    /// Create an empty filter.
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a filter condition.
    pub fn condition(
        mut self,
        field: impl Into<String>,
        operator: PineconeFilterOperator,
        value: Value,
    ) -> Self {
        self.conditions.push(PineconeFilterCondition {
            field: field.into(),
            operator,
            value,
        });
        self
    }

    /// Shorthand for `$eq`.
    pub fn eq(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Eq, value)
    }

    /// Shorthand for `$ne`.
    pub fn ne(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Ne, value)
    }

    /// Shorthand for `$gt`.
    pub fn gt(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Gt, value)
    }

    /// Shorthand for `$gte`.
    pub fn gte(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Gte, value)
    }

    /// Shorthand for `$lt`.
    pub fn lt(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Lt, value)
    }

    /// Shorthand for `$lte`.
    pub fn lte(self, field: impl Into<String>, value: Value) -> Self {
        self.condition(field, PineconeFilterOperator::Lte, value)
    }

    /// Shorthand for `$in`.
    pub fn r#in(self, field: impl Into<String>, values: Vec<Value>) -> Self {
        self.condition(field, PineconeFilterOperator::In, Value::Array(values))
    }

    /// Shorthand for `$nin`.
    pub fn nin(self, field: impl Into<String>, values: Vec<Value>) -> Self {
        self.condition(field, PineconeFilterOperator::Nin, Value::Array(values))
    }

    /// Returns true if this filter has no conditions.
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }
}

impl Default for PineconeFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// IndexStats
// ---------------------------------------------------------------------------

/// Statistics about a Pinecone index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PineconeIndexStats {
    /// Total number of vectors across all namespaces.
    pub total_vector_count: usize,
    /// Vector count per namespace.
    pub namespaces: HashMap<String, usize>,
    /// Dimensionality of vectors in the index.
    pub dimension: usize,
}

// ---------------------------------------------------------------------------
// PineconeClient trait
// ---------------------------------------------------------------------------

/// Abstraction over Pinecone API calls, enabling mocking for tests.
#[async_trait]
pub trait PineconeClient: Send + Sync {
    /// Upsert vectors into a namespace.
    async fn upsert(&self, namespace: &str, vectors: Vec<PineconeVector>) -> Result<()>;

    /// Query for similar vectors.
    async fn query(
        &self,
        namespace: &str,
        vector: &[f32],
        top_k: usize,
        filter: Option<&PineconeFilter>,
        include_metadata: bool,
    ) -> Result<Vec<PineconeQueryResult>>;

    /// Delete vectors by their IDs.
    async fn delete(&self, namespace: &str, ids: &[String]) -> Result<()>;

    /// Fetch vectors by their IDs.
    async fn fetch(&self, namespace: &str, ids: &[String]) -> Result<Vec<PineconeVector>>;

    /// Describe index statistics.
    async fn describe_index_stats(&self) -> Result<PineconeIndexStats>;
}

// ---------------------------------------------------------------------------
// MockPineconeClient
// ---------------------------------------------------------------------------

/// In-memory mock implementation of `PineconeClient` for testing.
///
/// Vectors are stored per namespace, providing full namespace isolation.
pub struct MockPineconeClient {
    /// Vectors stored per namespace.
    namespaces: RwLock<HashMap<String, Vec<PineconeVector>>>,
    /// Distance metric to use for scoring.
    metric: PineconeMetric,
    /// Dimensionality of vectors.
    dimension: usize,
}

impl MockPineconeClient {
    /// Create a new empty mock client.
    pub fn new(metric: PineconeMetric, dimension: usize) -> Self {
        Self {
            namespaces: RwLock::new(HashMap::new()),
            metric,
            dimension,
        }
    }
}

/// Compute similarity score between two vectors using the given metric.
fn compute_score(a: &[f32], b: &[f32], metric: PineconeMetric) -> f32 {
    match metric {
        PineconeMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a * norm_b)
            }
        }
        PineconeMetric::Euclidean => {
            let dist: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            1.0 / (1.0 + dist)
        }
        PineconeMetric::DotProduct => a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
    }
}

/// Check whether a vector's metadata matches a Pinecone filter.
fn matches_filter(metadata: &HashMap<String, Value>, filter: &PineconeFilter) -> bool {
    filter.conditions.iter().all(|cond| {
        let Some(field_val) = metadata.get(&cond.field) else {
            return false;
        };

        match &cond.operator {
            PineconeFilterOperator::Eq => field_val == &cond.value,
            PineconeFilterOperator::Ne => field_val != &cond.value,
            PineconeFilterOperator::Gt => compare_values(field_val, &cond.value) == Some(std::cmp::Ordering::Greater),
            PineconeFilterOperator::Gte => {
                matches!(
                    compare_values(field_val, &cond.value),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
            }
            PineconeFilterOperator::Lt => compare_values(field_val, &cond.value) == Some(std::cmp::Ordering::Less),
            PineconeFilterOperator::Lte => {
                matches!(
                    compare_values(field_val, &cond.value),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }
            PineconeFilterOperator::In => {
                if let Value::Array(arr) = &cond.value {
                    arr.contains(field_val)
                } else {
                    false
                }
            }
            PineconeFilterOperator::Nin => {
                if let Value::Array(arr) = &cond.value {
                    !arr.contains(field_val)
                } else {
                    true
                }
            }
        }
    })
}

/// Compare two JSON values numerically or lexicographically.
fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64()?;
            let fb = nb.as_f64()?;
            fa.partial_cmp(&fb)
        }
        (Value::String(sa), Value::String(sb)) => Some(sa.cmp(sb)),
        _ => None,
    }
}

#[async_trait]
impl PineconeClient for MockPineconeClient {
    async fn upsert(&self, namespace: &str, vectors: Vec<PineconeVector>) -> Result<()> {
        let mut namespaces = self.namespaces.write().await;
        let ns = namespaces
            .entry(namespace.to_string())
            .or_insert_with(Vec::new);

        for vector in vectors {
            ns.retain(|v| v.id != vector.id);
            ns.push(vector);
        }
        Ok(())
    }

    async fn query(
        &self,
        namespace: &str,
        vector: &[f32],
        top_k: usize,
        filter: Option<&PineconeFilter>,
        _include_metadata: bool,
    ) -> Result<Vec<PineconeQueryResult>> {
        let namespaces = self.namespaces.read().await;
        let Some(ns) = namespaces.get(namespace) else {
            return Ok(vec![]);
        };

        let mut scored: Vec<PineconeQueryResult> = ns
            .iter()
            .filter(|v| {
                filter
                    .map(|f| matches_filter(&v.metadata, f))
                    .unwrap_or(true)
            })
            .map(|v| PineconeQueryResult {
                id: v.id.clone(),
                score: compute_score(vector, &v.values, self.metric),
                metadata: v.metadata.clone(),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn delete(&self, namespace: &str, ids: &[String]) -> Result<()> {
        let mut namespaces = self.namespaces.write().await;
        if let Some(ns) = namespaces.get_mut(namespace) {
            ns.retain(|v| !ids.contains(&v.id));
        }
        Ok(())
    }

    async fn fetch(&self, namespace: &str, ids: &[String]) -> Result<Vec<PineconeVector>> {
        let namespaces = self.namespaces.read().await;
        let Some(ns) = namespaces.get(namespace) else {
            return Ok(vec![]);
        };
        Ok(ns.iter().filter(|v| ids.contains(&v.id)).cloned().collect())
    }

    async fn describe_index_stats(&self) -> Result<PineconeIndexStats> {
        let namespaces = self.namespaces.read().await;
        let mut ns_counts = HashMap::new();
        let mut total = 0;
        for (name, vectors) in namespaces.iter() {
            ns_counts.insert(name.clone(), vectors.len());
            total += vectors.len();
        }
        Ok(PineconeIndexStats {
            total_vector_count: total,
            namespaces: ns_counts,
            dimension: self.dimension,
        })
    }
}

// ---------------------------------------------------------------------------
// PineconeVectorStore
// ---------------------------------------------------------------------------

/// Vector store backed by Pinecone (or a mock thereof).
///
/// Wraps a `PineconeClient` implementation and an `Embeddings` model to provide
/// the full `VectorStore` trait interface.
pub struct PineconeVectorStore {
    client: Arc<dyn PineconeClient>,
    embeddings: Arc<dyn Embeddings>,
    config: PineconeConfig,
}

impl PineconeVectorStore {
    /// Create a new Pinecone vector store.
    pub fn new(
        client: Arc<dyn PineconeClient>,
        embeddings: Arc<dyn Embeddings>,
        config: PineconeConfig,
    ) -> Self {
        Self {
            client,
            embeddings,
            config,
        }
    }

    /// Create a Pinecone vector store pre-populated with documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        client: Arc<dyn PineconeClient>,
        embeddings: Arc<dyn Embeddings>,
        config: PineconeConfig,
    ) -> Result<Self> {
        let store = Self::new(client, embeddings, config);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Return the effective namespace (defaults to empty string).
    fn namespace(&self) -> &str {
        self.config
            .namespace
            .as_deref()
            .unwrap_or("")
    }

    /// Perform a similarity search with an optional metadata filter, returning scores.
    pub async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: Option<&PineconeFilter>,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let results = self
            .client
            .query(self.namespace(), &query_embedding, k, filter, true)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| {
                let content = r
                    .metadata
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = r.metadata.clone();
                metadata.remove("page_content");

                let doc = Document::new(content)
                    .with_id(r.id)
                    .with_metadata(metadata);

                (doc, r.score)
            })
            .collect())
    }

    /// Return a reference to the underlying config.
    pub fn config(&self) -> &PineconeConfig {
        &self.config
    }

    /// Get index statistics.
    pub async fn describe_index_stats(&self) -> Result<PineconeIndexStats> {
        self.client.describe_index_stats().await
    }
}

#[async_trait]
impl VectorStore for PineconeVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;

        let mut vectors = Vec::with_capacity(texts.len());
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let mut metadata: HashMap<String, Value> = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            metadata.insert(
                "page_content".to_string(),
                Value::String(text.clone()),
            );

            vectors.push(PineconeVector {
                id: id.clone(),
                values: embeddings_vec[i].clone(),
                metadata,
                sparse_values: None,
            });

            result_ids.push(id);
        }

        self.client
            .upsert(self.namespace(), vectors)
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
        self.client.delete(self.namespace(), ids).await?;
        Ok(true)
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let vectors = self.client.fetch(self.namespace(), ids).await?;

        Ok(vectors
            .into_iter()
            .map(|v| {
                let content = v
                    .metadata
                    .get("page_content")
                    .and_then(|val| val.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = v.metadata.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(v.id)
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
            .query(self.namespace(), embedding, k, None, true)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| {
                let content = r
                    .metadata
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = r.metadata.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(r.id)
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
            .query(self.namespace(), &query_embedding, fetch_k, None, true)
            .await?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        // Fetch full vectors to get embedding values for MMR computation.
        let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        let full_vectors = self.client.fetch(self.namespace(), &result_ids).await?;

        let candidate_embeddings: Vec<Vec<f64>> = full_vectors
            .iter()
            .map(|v| v.values.iter().map(|&val| val as f64).collect())
            .collect();
        let query_emb_f64: Vec<f64> = query_embedding.iter().map(|&v| v as f64).collect();

        let mmr_indices = rustchain_core::vectorstores::utils::maximal_marginal_relevance(
            &query_emb_f64,
            &candidate_embeddings,
            lambda_mult as f64,
            k,
        );

        let docs = mmr_indices
            .into_iter()
            .filter_map(|idx| results.get(idx))
            .map(|r| {
                let content = r
                    .metadata
                    .get("page_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = r.metadata.clone();
                metadata.remove("page_content");

                Document::new(content)
                    .with_id(r.id.clone())
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
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    fn make_client() -> Arc<MockPineconeClient> {
        Arc::new(MockPineconeClient::new(PineconeMetric::Cosine, 16))
    }

    fn make_config() -> PineconeConfig {
        PineconeConfig::new("test-key", "us-east-1-aws", "test-index", 16)
    }

    fn make_store() -> PineconeVectorStore {
        PineconeVectorStore::new(make_client(), make_embeddings(), make_config())
    }

    fn make_store_with_namespace(ns: &str) -> (PineconeVectorStore, Arc<MockPineconeClient>) {
        let client = make_client();
        let config = make_config().with_namespace(ns);
        let store = PineconeVectorStore::new(client.clone(), make_embeddings(), config);
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

        let results = store
            .similarity_search_with_score("cat", 3)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        // First result should be "cat" with highest score.
        assert_eq!(results[0].0.page_content, "cat");
        // Scores in descending order.
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    // ---- Test 3: Namespace isolation ----
    #[tokio::test]
    async fn test_namespace_isolation() {
        let client = make_client();
        let embeddings = make_embeddings();

        let config_a = make_config().with_namespace("ns-a");
        let store_a = PineconeVectorStore::new(client.clone(), embeddings.clone(), config_a);

        let config_b = make_config().with_namespace("ns-b");
        let store_b = PineconeVectorStore::new(client.clone(), embeddings.clone(), config_b);

        store_a
            .add_texts(&["hello from A".into()], None, None)
            .await
            .unwrap();
        store_b
            .add_texts(&["hello from B".into()], None, None)
            .await
            .unwrap();

        let results_a = store_a.similarity_search("hello", 10).await.unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0].page_content, "hello from A");

        let results_b = store_b.similarity_search("hello", 10).await.unwrap();
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].page_content, "hello from B");
    }

    // ---- Test 4: Metadata filtering with $eq ----
    #[tokio::test]
    async fn test_metadata_filter_eq() {
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

        let filter = PineconeFilter::new().eq("color", Value::String("red".into()));
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

    // ---- Test 5: Metadata filtering with $in ----
    #[tokio::test]
    async fn test_metadata_filter_in() {
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

        let filter = PineconeFilter::new().r#in(
            "type",
            vec![Value::String("x".into()), Value::String("z".into())],
        );
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

    // ---- Test 6: Metadata filtering with $gt ----
    #[tokio::test]
    async fn test_metadata_filter_gt() {
        let store = make_store();
        let texts = vec!["low".into(), "mid".into(), "high".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::Number(serde_json::Number::from(10)));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::Number(serde_json::Number::from(50)));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::Number(serde_json::Number::from(90)));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = PineconeFilter::new().gt("score", Value::Number(serde_json::Number::from(40)));
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let s = doc.metadata.get("score").unwrap().as_i64().unwrap();
            assert!(s > 40);
        }
    }

    // ---- Test 7: Delete documents ----
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

    // ---- Test 8: Config construction ----
    #[tokio::test]
    async fn test_config_construction() {
        let config = PineconeConfig::new("my-key", "us-west-2-aws", "my-index", 128);
        assert_eq!(config.api_key, "my-key");
        assert_eq!(config.environment, "us-west-2-aws");
        assert_eq!(config.index_name, "my-index");
        assert_eq!(config.dimension, 128);
        assert!(config.namespace.is_none());
        assert_eq!(config.metric, PineconeMetric::Cosine);

        let config = config
            .with_namespace("my-ns")
            .with_metric(PineconeMetric::DotProduct);
        assert_eq!(config.namespace.as_deref(), Some("my-ns"));
        assert_eq!(config.metric, PineconeMetric::DotProduct);
    }

    // ---- Test 9: Fetch by IDs ----
    #[tokio::test]
    async fn test_fetch_by_ids() {
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

    // ---- Test 10: Empty index search ----
    #[tokio::test]
    async fn test_empty_index_search() {
        let store = make_store();
        let results = store.similarity_search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Test 11: Default namespace ----
    #[tokio::test]
    async fn test_default_namespace() {
        let store = make_store();
        assert_eq!(store.namespace(), "");

        let texts = vec!["hello".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let stats = store.describe_index_stats().await.unwrap();
        assert!(stats.namespaces.contains_key(""));
        assert_eq!(stats.total_vector_count, 1);
    }

    // ---- Test 12: Multiple namespaces via describe_index_stats ----
    #[tokio::test]
    async fn test_multiple_namespaces_stats() {
        let client = make_client();
        let embeddings = make_embeddings();

        let config_a = make_config().with_namespace("ns-alpha");
        let store_a = PineconeVectorStore::new(client.clone(), embeddings.clone(), config_a);

        let config_b = make_config().with_namespace("ns-beta");
        let store_b = PineconeVectorStore::new(client.clone(), embeddings.clone(), config_b);

        store_a
            .add_texts(&["doc1".into(), "doc2".into()], None, None)
            .await
            .unwrap();
        store_b
            .add_texts(&["doc3".into()], None, None)
            .await
            .unwrap();

        let stats = store_a.describe_index_stats().await.unwrap();
        assert_eq!(stats.total_vector_count, 3);
        assert_eq!(*stats.namespaces.get("ns-alpha").unwrap(), 2);
        assert_eq!(*stats.namespaces.get("ns-beta").unwrap(), 1);
    }

    // ---- Test 13: from_documents constructor ----
    #[tokio::test]
    async fn test_from_documents_constructor() {
        let client = make_client();
        let embeddings = make_embeddings();
        let config = make_config();

        let docs = vec![
            Document::new("hello world").with_id("h1"),
            Document::new("goodbye world").with_id("g1"),
        ];

        let store = PineconeVectorStore::from_documents(docs, client, embeddings, config)
            .await
            .unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }

    // ---- Test 14: Batch upsert ----
    #[tokio::test]
    async fn test_batch_upsert() {
        let store = make_store();
        let texts: Vec<String> = (0..100).map(|i| format!("document_{}", i)).collect();
        let ids = store.add_texts(&texts, None, None).await.unwrap();
        assert_eq!(ids.len(), 100);

        let results = store.similarity_search("document_50", 5).await.unwrap();
        assert_eq!(results.len(), 5);

        let stats = store.describe_index_stats().await.unwrap();
        assert_eq!(stats.total_vector_count, 100);
    }

    // ---- Test 15: Delete with None returns false ----
    #[tokio::test]
    async fn test_delete_none_returns_false() {
        let store = make_store();
        let result = store.delete(None).await.unwrap();
        assert!(!result);
    }

    // ---- Test 16: Metadata filtering with $ne ----
    #[tokio::test]
    async fn test_metadata_filter_ne() {
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

        let filter = PineconeFilter::new().ne("status", Value::String("draft".into()));
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
}
