//! Weaviate vector store implementation.
//!
//! Provides a `WeaviateVectorStore` that implements the `VectorStore` trait,
//! backed by a pluggable `WeaviateClient` trait for abstracting HTTP calls.
//! Includes a `MockWeaviateClient` for testing without a running Weaviate instance.

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
// WeaviateConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Weaviate instance.
#[derive(Debug, Clone)]
pub struct WeaviateConfig {
    /// Base URL of the Weaviate REST API (default: "http://localhost:8080").
    pub url: String,
    /// The Weaviate class (collection) name to operate on.
    pub class_name: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// The property name used to store document text content (default: "text").
    pub text_key: String,
    /// Additional HTTP headers to include in requests.
    pub additional_headers: HashMap<String, String>,
}

impl WeaviateConfig {
    /// Create a new config with the given class name and sensible defaults.
    pub fn new(class_name: impl Into<String>) -> Self {
        Self {
            url: "http://localhost:8080".to_string(),
            class_name: class_name.into(),
            api_key: None,
            text_key: "text".to_string(),
            additional_headers: HashMap::new(),
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

    /// Set the text key used for storing document content.
    pub fn with_text_key(mut self, text_key: impl Into<String>) -> Self {
        self.text_key = text_key.into();
        self
    }

    /// Add an additional HTTP header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_headers.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// WeaviateObject
// ---------------------------------------------------------------------------

/// An object stored in a Weaviate class/collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviateObject {
    /// Unique identifier for the object.
    pub id: String,
    /// The Weaviate class this object belongs to.
    pub class: String,
    /// Arbitrary JSON properties (metadata + content).
    pub properties: HashMap<String, Value>,
    /// The embedding vector.
    pub vector: Vec<f32>,
}

// ---------------------------------------------------------------------------
// WeaviateWhereFilter
// ---------------------------------------------------------------------------

/// Filter operators supported by Weaviate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeaviateOperator {
    /// Exact equality.
    Equal,
    /// Not equal.
    NotEqual,
    /// Greater than (numeric).
    GreaterThan,
    /// Less than (numeric).
    LessThan,
    /// Wildcard/prefix matching (string).
    Like,
    /// Array contains any of the given values.
    ContainsAny,
    /// Array contains all of the given values.
    ContainsAll,
    /// Logical AND — all operands must match.
    And,
    /// Logical OR — at least one operand must match.
    Or,
}

/// Weaviate-style where filter for constraining search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviateWhereFilter {
    /// The filter operator.
    pub operator: WeaviateOperator,
    /// Property path for leaf conditions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<String>,
    /// The value to compare against (for leaf conditions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// Child filters for compound operators (And/Or).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub operands: Vec<WeaviateWhereFilter>,
}

impl WeaviateWhereFilter {
    /// Create a leaf filter condition.
    pub fn new(operator: WeaviateOperator, path: Vec<String>, value: Value) -> Self {
        Self {
            operator,
            path,
            value: Some(value),
            operands: Vec::new(),
        }
    }

    /// Create a compound (And/Or) filter from operands.
    pub fn compound(operator: WeaviateOperator, operands: Vec<WeaviateWhereFilter>) -> Self {
        Self {
            operator,
            path: Vec::new(),
            value: None,
            operands,
        }
    }
}

// ---------------------------------------------------------------------------
// WeaviateSearchResult
// ---------------------------------------------------------------------------

/// A single result returned from a Weaviate vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviateSearchResult {
    /// Object identifier.
    pub id: String,
    /// Object properties.
    pub properties: HashMap<String, Value>,
    /// The embedding vector (if requested).
    pub vector: Option<Vec<f32>>,
    /// Distance metric value (lower = closer).
    pub distance: Option<f32>,
    /// Certainty score (higher = more certain, 0.0-1.0).
    pub certainty: Option<f32>,
}

// ---------------------------------------------------------------------------
// WeaviateClassConfig
// ---------------------------------------------------------------------------

/// Configuration for creating a Weaviate class/schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviateClassConfig {
    /// Class name.
    pub class: String,
    /// Property definitions.
    pub properties: Vec<WeaviatePropertyConfig>,
}

/// Configuration for a single property in a Weaviate class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaviatePropertyConfig {
    /// Property name.
    pub name: String,
    /// Data type (e.g., "text", "int", "number", "boolean").
    pub data_type: Vec<String>,
}

// ---------------------------------------------------------------------------
// WeaviateClient trait
// ---------------------------------------------------------------------------

/// Abstraction over Weaviate API calls, enabling mocking for tests.
#[async_trait]
pub trait WeaviateClient: Send + Sync {
    /// Batch insert objects into a Weaviate class.
    async fn batch_create_objects(&self, class: &str, objects: Vec<WeaviateObject>) -> Result<()>;

    /// Perform a vector similarity search.
    async fn search(
        &self,
        class: &str,
        vector: &[f32],
        limit: usize,
        where_filter: Option<&WeaviateWhereFilter>,
        properties: Option<&[String]>,
    ) -> Result<Vec<WeaviateSearchResult>>;

    /// Delete objects by their IDs.
    async fn delete_objects(&self, class: &str, ids: &[String]) -> Result<bool>;

    /// Retrieve objects by their IDs.
    async fn get_objects(&self, class: &str, ids: &[String]) -> Result<Vec<WeaviateObject>>;

    /// Create a class/schema in Weaviate.
    async fn create_class(&self, class_config: WeaviateClassConfig) -> Result<()>;
}

// ---------------------------------------------------------------------------
// MockWeaviateClient
// ---------------------------------------------------------------------------

/// In-memory mock implementation of `WeaviateClient` for testing.
pub struct MockWeaviateClient {
    /// Objects stored per class.
    classes: RwLock<HashMap<String, Vec<WeaviateObject>>>,
}

impl MockWeaviateClient {
    /// Create a new empty mock client.
    pub fn new() -> Self {
        Self {
            classes: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MockWeaviateClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Check whether an object's properties match a Weaviate where filter.
fn matches_where_filter(props: &HashMap<String, Value>, filter: &WeaviateWhereFilter) -> bool {
    match filter.operator {
        WeaviateOperator::And => filter
            .operands
            .iter()
            .all(|op| matches_where_filter(props, op)),
        WeaviateOperator::Or => filter
            .operands
            .iter()
            .any(|op| matches_where_filter(props, op)),
        _ => {
            // Leaf condition — need path and value.
            let prop_name = match filter.path.first() {
                Some(p) => p,
                None => return false,
            };
            let filter_val = match &filter.value {
                Some(v) => v,
                None => return false,
            };
            let prop_val = match props.get(prop_name) {
                Some(v) => v,
                None => return false,
            };

            match filter.operator {
                WeaviateOperator::Equal => prop_val == filter_val,
                WeaviateOperator::NotEqual => prop_val != filter_val,
                WeaviateOperator::GreaterThan => match (prop_val.as_f64(), filter_val.as_f64()) {
                    (Some(a), Some(b)) => a > b,
                    _ => false,
                },
                WeaviateOperator::LessThan => match (prop_val.as_f64(), filter_val.as_f64()) {
                    (Some(a), Some(b)) => a < b,
                    _ => false,
                },
                WeaviateOperator::Like => {
                    // Simple prefix/wildcard matching: "foo*" matches "foobar".
                    match (prop_val.as_str(), filter_val.as_str()) {
                        (Some(pv), Some(fv)) => {
                            if let Some(prefix) = fv.strip_suffix('*') {
                                pv.starts_with(prefix)
                            } else if let Some(suffix) = fv.strip_prefix('*') {
                                pv.ends_with(suffix)
                            } else {
                                pv == fv
                            }
                        }
                        _ => false,
                    }
                }
                WeaviateOperator::ContainsAny => {
                    match (prop_val.as_array(), filter_val.as_array()) {
                        (Some(arr), Some(targets)) => targets.iter().any(|t| arr.contains(t)),
                        _ => false,
                    }
                }
                WeaviateOperator::ContainsAll => {
                    match (prop_val.as_array(), filter_val.as_array()) {
                        (Some(arr), Some(targets)) => targets.iter().all(|t| arr.contains(t)),
                        _ => false,
                    }
                }
                // And/Or handled above.
                WeaviateOperator::And | WeaviateOperator::Or => unreachable!(),
            }
        }
    }
}

#[async_trait]
impl WeaviateClient for MockWeaviateClient {
    async fn batch_create_objects(&self, class: &str, objects: Vec<WeaviateObject>) -> Result<()> {
        let mut classes = self.classes.write().await;
        let coll = classes.entry(class.to_string()).or_default();

        for obj in objects {
            // Replace existing object with the same ID.
            coll.retain(|o| o.id != obj.id);
            coll.push(obj);
        }
        Ok(())
    }

    async fn search(
        &self,
        class: &str,
        vector: &[f32],
        limit: usize,
        where_filter: Option<&WeaviateWhereFilter>,
        _properties: Option<&[String]>,
    ) -> Result<Vec<WeaviateSearchResult>> {
        let classes = self.classes.read().await;
        let Some(coll) = classes.get(class) else {
            return Ok(vec![]);
        };

        let mut scored: Vec<(WeaviateSearchResult, f32)> = coll
            .iter()
            .filter(|obj| {
                where_filter
                    .map(|f| matches_where_filter(&obj.properties, f))
                    .unwrap_or(true)
            })
            .map(|obj| {
                let sim = cosine_similarity(vector, &obj.vector);
                // Weaviate distance = 1 - cosine_similarity for cosine metric.
                let distance = 1.0 - sim;
                // Certainty maps to (1 + cosine) / 2 in Weaviate.
                let certainty = (1.0 + sim) / 2.0;
                let result = WeaviateSearchResult {
                    id: obj.id.clone(),
                    properties: obj.properties.clone(),
                    vector: Some(obj.vector.clone()),
                    distance: Some(distance),
                    certainty: Some(certainty),
                };
                (result, sim)
            })
            .collect();

        // Sort descending by similarity (ascending distance).
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored.into_iter().map(|(r, _)| r).collect())
    }

    async fn delete_objects(&self, class: &str, ids: &[String]) -> Result<bool> {
        let mut classes = self.classes.write().await;
        let Some(coll) = classes.get_mut(class) else {
            return Ok(false);
        };
        let before = coll.len();
        coll.retain(|o| !ids.contains(&o.id));
        Ok(coll.len() < before)
    }

    async fn get_objects(&self, class: &str, ids: &[String]) -> Result<Vec<WeaviateObject>> {
        let classes = self.classes.read().await;
        let Some(coll) = classes.get(class) else {
            return Ok(vec![]);
        };
        Ok(coll
            .iter()
            .filter(|o| ids.contains(&o.id))
            .cloned()
            .collect())
    }

    async fn create_class(&self, class_config: WeaviateClassConfig) -> Result<()> {
        let mut classes = self.classes.write().await;
        classes.entry(class_config.class).or_default();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WeaviateVectorStore
// ---------------------------------------------------------------------------

/// Vector store backed by Weaviate (or a mock thereof).
///
/// Wraps a `WeaviateClient` implementation and an `Embeddings` model to provide
/// the full `VectorStore` trait interface.
pub struct WeaviateVectorStore {
    client: Arc<dyn WeaviateClient>,
    embeddings: Arc<dyn Embeddings>,
    config: WeaviateConfig,
}

impl WeaviateVectorStore {
    /// Create a new Weaviate vector store.
    pub fn new(
        client: Arc<dyn WeaviateClient>,
        embeddings: Arc<dyn Embeddings>,
        config: WeaviateConfig,
    ) -> Self {
        Self {
            client,
            embeddings,
            config,
        }
    }

    /// Create a Weaviate vector store pre-populated with documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        client: Arc<dyn WeaviateClient>,
        embeddings: Arc<dyn Embeddings>,
        config: WeaviateConfig,
    ) -> Result<Self> {
        let store = Self::new(client, embeddings, config);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Perform a similarity search with an optional Weaviate where filter, returning scores.
    pub async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: Option<&WeaviateWhereFilter>,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let results = self
            .client
            .search(&self.config.class_name, &query_embedding, k, filter, None)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| {
                let content = r
                    .properties
                    .get(&self.config.text_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = r.properties.clone();
                metadata.remove(&self.config.text_key);

                // Use certainty as the score (higher = better).
                let score = r.certainty.unwrap_or(0.0);

                let doc = Document::new(content).with_id(r.id).with_metadata(metadata);

                (doc, score)
            })
            .collect())
    }

    /// Return a reference to the underlying config.
    pub fn config(&self) -> &WeaviateConfig {
        &self.config
    }
}

/// Convert a `WeaviateSearchResult` to a `Document` using the given text key.
fn result_to_document(result: WeaviateSearchResult, text_key: &str) -> Document {
    let content = result
        .properties
        .get(text_key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut metadata = result.properties;
    metadata.remove(text_key);

    Document::new(content)
        .with_id(result.id)
        .with_metadata(metadata)
}

#[async_trait]
impl VectorStore for WeaviateVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;

        let mut objects = Vec::with_capacity(texts.len());
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let mut properties: HashMap<String, Value> = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            // Store text content under the configured text key.
            properties.insert(self.config.text_key.clone(), Value::String(text.clone()));

            objects.push(WeaviateObject {
                id: id.clone(),
                class: self.config.class_name.clone(),
                properties,
                vector: embeddings_vec[i].clone(),
            });

            result_ids.push(id);
        }

        self.client
            .batch_create_objects(&self.config.class_name, objects)
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
            .delete_objects(&self.config.class_name, ids)
            .await
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let objects = self
            .client
            .get_objects(&self.config.class_name, ids)
            .await?;

        Ok(objects
            .into_iter()
            .map(|obj| {
                let content = obj
                    .properties
                    .get(&self.config.text_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = obj.properties.clone();
                metadata.remove(&self.config.text_key);

                Document::new(content)
                    .with_id(obj.id)
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
            .search(&self.config.class_name, embedding, k, None, None)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| result_to_document(r, &self.config.text_key))
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
            .search(
                &self.config.class_name,
                &query_embedding,
                fetch_k,
                None,
                None,
            )
            .await?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        let candidate_embeddings: Vec<Vec<f64>> = results
            .iter()
            .filter_map(|r| {
                r.vector
                    .as_ref()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
            })
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
            .map(|r| result_to_document(r.clone(), &self.config.text_key))
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

    fn make_store() -> WeaviateVectorStore {
        let client = Arc::new(MockWeaviateClient::new());
        let embeddings = make_embeddings();
        let config = WeaviateConfig::new("TestClass");
        WeaviateVectorStore::new(client, embeddings, config)
    }

    fn make_store_with_text_key(text_key: &str) -> WeaviateVectorStore {
        let client = Arc::new(MockWeaviateClient::new());
        let embeddings = make_embeddings();
        let config = WeaviateConfig::new("TestClass").with_text_key(text_key);
        WeaviateVectorStore::new(client, embeddings, config)
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

    // ---- Test 3: Where filter - Equal ----
    #[tokio::test]
    async fn test_where_filter_equal() {
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

        let filter = WeaviateWhereFilter::new(
            WeaviateOperator::Equal,
            vec!["color".into()],
            Value::String("red".into()),
        );
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

    // ---- Test 4: Where filter - GreaterThan ----
    #[tokio::test]
    async fn test_where_filter_greater_than() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
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

        let filter = WeaviateWhereFilter::new(
            WeaviateOperator::GreaterThan,
            vec!["score".into()],
            Value::Number(serde_json::Number::from(40)),
        );
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let score = doc.metadata.get("score").unwrap().as_f64().unwrap();
            assert!(score > 40.0);
        }
    }

    // ---- Test 5: And compound filter ----
    #[tokio::test]
    async fn test_and_compound_filter() {
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

        // Must be food AND organic.
        let filter = WeaviateWhereFilter::compound(
            WeaviateOperator::And,
            vec![
                WeaviateWhereFilter::new(
                    WeaviateOperator::Equal,
                    vec!["category".into()],
                    Value::String("food".into()),
                ),
                WeaviateWhereFilter::new(
                    WeaviateOperator::Equal,
                    vec!["organic".into()],
                    Value::Bool(true),
                ),
            ],
        );
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.page_content, "a");
    }

    // ---- Test 6: Or compound filter ----
    #[tokio::test]
    async fn test_or_compound_filter() {
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

        let filter = WeaviateWhereFilter::compound(
            WeaviateOperator::Or,
            vec![
                WeaviateWhereFilter::new(
                    WeaviateOperator::Equal,
                    vec!["type".into()],
                    Value::String("x".into()),
                ),
                WeaviateWhereFilter::new(
                    WeaviateOperator::Equal,
                    vec!["type".into()],
                    Value::String("z".into()),
                ),
            ],
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

    // ---- Test 7: Delete objects ----
    #[tokio::test]
    async fn test_delete_objects() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&[ids[1].clone()])).await.unwrap();
        assert!(deleted);

        let remaining = store.similarity_search("a", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|d| d.page_content != "b"));
    }

    // ---- Test 8: Config defaults ----
    #[tokio::test]
    async fn test_config_defaults() {
        let config = WeaviateConfig::new("MyClass");
        assert_eq!(config.url, "http://localhost:8080");
        assert_eq!(config.class_name, "MyClass");
        assert!(config.api_key.is_none());
        assert_eq!(config.text_key, "text");
        assert!(config.additional_headers.is_empty());
    }

    // ---- Test 9: Get by IDs ----
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

    // ---- Test 10: Empty collection ----
    #[tokio::test]
    async fn test_empty_collection_search() {
        let store = make_store();
        let results = store.similarity_search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Test 11: Text key customization ----
    #[tokio::test]
    async fn test_text_key_customization() {
        let store = make_store_with_text_key("content");
        let texts = vec!["hello world".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
        // The "content" key should not appear in metadata.
        assert!(!results[0].metadata.contains_key("content"));
    }

    // ---- Test 12: from_documents constructor ----
    #[tokio::test]
    async fn test_from_documents_constructor() {
        let client = Arc::new(MockWeaviateClient::new());
        let embeddings = make_embeddings();
        let config = WeaviateConfig::new("TestClass");

        let docs = vec![
            Document::new("hello world").with_id("h1"),
            Document::new("goodbye world").with_id("g1"),
        ];

        let store = WeaviateVectorStore::from_documents(docs, client, embeddings, config)
            .await
            .unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }

    // ---- Test 13: Batch create via mock client ----
    #[tokio::test]
    async fn test_batch_create_mock_client() {
        let client = MockWeaviateClient::new();
        let objects = vec![
            WeaviateObject {
                id: "o1".to_string(),
                class: "Test".to_string(),
                properties: {
                    let mut m = HashMap::new();
                    m.insert("text".into(), Value::String("first".into()));
                    m
                },
                vector: vec![1.0, 0.0, 0.0],
            },
            WeaviateObject {
                id: "o2".to_string(),
                class: "Test".to_string(),
                properties: {
                    let mut m = HashMap::new();
                    m.insert("text".into(), Value::String("second".into()));
                    m
                },
                vector: vec![0.0, 1.0, 0.0],
            },
        ];

        client.batch_create_objects("Test", objects).await.unwrap();

        let retrieved = client
            .get_objects("Test", &["o1".into(), "o2".into()])
            .await
            .unwrap();
        assert_eq!(retrieved.len(), 2);

        // Upsert with same ID replaces.
        let updated = vec![WeaviateObject {
            id: "o1".to_string(),
            class: "Test".to_string(),
            properties: {
                let mut m = HashMap::new();
                m.insert("text".into(), Value::String("updated".into()));
                m
            },
            vector: vec![0.5, 0.5, 0.0],
        }];
        client.batch_create_objects("Test", updated).await.unwrap();

        let after = client.get_objects("Test", &["o1".into()]).await.unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0].properties.get("text").unwrap(),
            &Value::String("updated".into())
        );
    }

    // ---- Test 14: Like filter (prefix matching) ----
    #[tokio::test]
    async fn test_like_filter_prefix() {
        let store = make_store();
        let texts = vec![
            "apple pie".into(),
            "apple sauce".into(),
            "banana split".into(),
        ];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("name".into(), Value::String("apple pie".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("name".into(), Value::String("apple sauce".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("name".into(), Value::String("banana split".into()));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = WeaviateWhereFilter::new(
            WeaviateOperator::Like,
            vec!["name".into()],
            Value::String("apple*".into()),
        );
        let results = store
            .similarity_search_with_filter("fruit", 10, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let name = doc.metadata.get("name").unwrap().as_str().unwrap();
            assert!(name.starts_with("apple"));
        }
    }

    // ---- Test 15: Mock client search with property selection ----
    #[tokio::test]
    async fn test_search_with_property_selection() {
        let client = MockWeaviateClient::new();
        let objects = vec![WeaviateObject {
            id: "p1".to_string(),
            class: "Article".to_string(),
            properties: {
                let mut m = HashMap::new();
                m.insert("title".into(), Value::String("Hello".into()));
                m.insert("body".into(), Value::String("World".into()));
                m
            },
            vector: vec![1.0, 0.0, 0.0],
        }];

        client
            .batch_create_objects("Article", objects)
            .await
            .unwrap();

        // Search with specific property selection (mock returns all, but API accepts it).
        let results = client
            .search(
                "Article",
                &[1.0, 0.0, 0.0],
                10,
                None,
                Some(&["title".to_string()]),
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "p1");
        assert!(results[0].certainty.is_some());
        assert!(results[0].distance.is_some());
    }

    // ---- Test 16: Delete with None returns false ----
    #[tokio::test]
    async fn test_delete_none_returns_false() {
        let store = make_store();
        let result = store.delete(None).await.unwrap();
        assert!(!result);
    }
}
