//! ChromaDB vector store implementation.
//!
//! Provides a `ChromaVectorStore` that implements the `VectorStore` trait,
//! backed by a pluggable `ChromaClient` trait for abstracting HTTP calls.
//! Includes a `MockChromaClient` for testing without a running Chroma instance.

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
// ChromaConfig
// ---------------------------------------------------------------------------

/// Configuration for connecting to a ChromaDB instance.
#[derive(Clone)]
pub struct ChromaConfig {
    /// Base URL of the Chroma REST API.
    pub url: String,
    /// Name of the collection to operate on.
    pub collection_name: String,
    /// Optional tenant name for multi-tenancy.
    pub tenant: Option<String>,
    /// Optional database name.
    pub database: Option<String>,
    /// Optional embedding function for automatic embedding generation.
    pub embedding_function: Option<Arc<dyn Embeddings>>,
}

impl std::fmt::Debug for ChromaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChromaConfig")
            .field("url", &self.url)
            .field("collection_name", &self.collection_name)
            .field("tenant", &self.tenant)
            .field("database", &self.database)
            .field("embedding_function", &self.embedding_function.as_ref().map(|_| "..."))
            .finish()
    }
}

impl ChromaConfig {
    /// Create a new config with the given collection name and sensible defaults.
    pub fn new(collection_name: impl Into<String>) -> Self {
        Self {
            url: "http://localhost:8000".to_string(),
            collection_name: collection_name.into(),
            tenant: None,
            database: None,
            embedding_function: None,
        }
    }

    /// Set the base URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set the tenant name.
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Set the database name.
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    /// Set the embedding function.
    pub fn with_embedding_function(mut self, embedding_function: Arc<dyn Embeddings>) -> Self {
        self.embedding_function = Some(embedding_function);
        self
    }
}

// ---------------------------------------------------------------------------
// ChromaWhereFilter — Chroma-style metadata filtering
// ---------------------------------------------------------------------------

/// A single comparison operator for metadata filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChromaWhereOperator {
    /// Equal to.
    #[serde(rename = "$eq")]
    Eq(Value),
    /// Not equal to.
    #[serde(rename = "$ne")]
    Ne(Value),
    /// Greater than.
    #[serde(rename = "$gt")]
    Gt(Value),
    /// Greater than or equal to.
    #[serde(rename = "$gte")]
    Gte(Value),
    /// Less than.
    #[serde(rename = "$lt")]
    Lt(Value),
    /// Less than or equal to.
    #[serde(rename = "$lte")]
    Lte(Value),
    /// In a list of values.
    #[serde(rename = "$in")]
    In(Vec<Value>),
    /// Not in a list of values.
    #[serde(rename = "$nin")]
    Nin(Vec<Value>),
}

/// A Chroma-style metadata filter expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChromaWhereFilter {
    /// A single field condition: `{ field: { $op: value } }`.
    Condition {
        /// The metadata field name.
        field: String,
        /// The comparison operator.
        operator: ChromaWhereOperator,
    },
    /// All sub-filters must match.
    #[serde(rename = "$and")]
    And(Vec<ChromaWhereFilter>),
    /// At least one sub-filter must match.
    #[serde(rename = "$or")]
    Or(Vec<ChromaWhereFilter>),
}

impl ChromaWhereFilter {
    /// Create an equality condition: `{ field: { $eq: value } }`.
    pub fn eq(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Eq(value),
        }
    }

    /// Create a not-equal condition: `{ field: { $ne: value } }`.
    pub fn ne(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Ne(value),
        }
    }

    /// Create a greater-than condition: `{ field: { $gt: value } }`.
    pub fn gt(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Gt(value),
        }
    }

    /// Create a greater-than-or-equal condition: `{ field: { $gte: value } }`.
    pub fn gte(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Gte(value),
        }
    }

    /// Create a less-than condition: `{ field: { $lt: value } }`.
    pub fn lt(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Lt(value),
        }
    }

    /// Create a less-than-or-equal condition: `{ field: { $lte: value } }`.
    pub fn lte(field: impl Into<String>, value: Value) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Lte(value),
        }
    }

    /// Create an in-list condition: `{ field: { $in: [values] } }`.
    pub fn r#in(field: impl Into<String>, values: Vec<Value>) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::In(values),
        }
    }

    /// Create a not-in-list condition: `{ field: { $nin: [values] } }`.
    pub fn nin(field: impl Into<String>, values: Vec<Value>) -> Self {
        Self::Condition {
            field: field.into(),
            operator: ChromaWhereOperator::Nin(values),
        }
    }

    /// Combine filters with `$and`.
    pub fn and(filters: Vec<ChromaWhereFilter>) -> Self {
        Self::And(filters)
    }

    /// Combine filters with `$or`.
    pub fn or(filters: Vec<ChromaWhereFilter>) -> Self {
        Self::Or(filters)
    }
}

// ---------------------------------------------------------------------------
// ChromaWhereDocumentFilter — document content filtering
// ---------------------------------------------------------------------------

/// A Chroma-style document content filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChromaWhereDocumentFilter {
    /// Document must contain the given string.
    #[serde(rename = "$contains")]
    Contains(String),
    /// Document must not contain the given string.
    #[serde(rename = "$not_contains")]
    NotContains(String),
}

// ---------------------------------------------------------------------------
// ChromaQueryResult
// ---------------------------------------------------------------------------

/// Result returned from a Chroma query operation.
#[derive(Debug, Clone, Default)]
pub struct ChromaQueryResult {
    /// Lists of IDs, one per query embedding.
    pub ids: Vec<Vec<String>>,
    /// Optional embeddings for each result.
    pub embeddings: Option<Vec<Vec<Vec<f32>>>>,
    /// Optional document contents for each result.
    pub documents: Option<Vec<Vec<String>>>,
    /// Optional metadata for each result.
    pub metadatas: Option<Vec<Vec<HashMap<String, Value>>>>,
    /// Optional distances/scores for each result.
    pub distances: Option<Vec<Vec<f32>>>,
}

// ---------------------------------------------------------------------------
// ChromaClient trait
// ---------------------------------------------------------------------------

/// Abstraction over Chroma API calls, enabling mocking for tests.
#[async_trait]
pub trait ChromaClient: Send + Sync {
    /// Create a collection with optional metadata.
    async fn create_collection(
        &self,
        name: &str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Result<()>;

    /// Add documents to a collection.
    async fn add(
        &self,
        collection: &str,
        ids: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<HashMap<String, Value>>>,
    ) -> Result<()>;

    /// Query a collection by embeddings.
    async fn query(
        &self,
        collection: &str,
        query_embeddings: Vec<Vec<f32>>,
        n_results: usize,
        where_filter: Option<&ChromaWhereFilter>,
        where_document: Option<&ChromaWhereDocumentFilter>,
    ) -> Result<ChromaQueryResult>;

    /// Delete documents from a collection.
    async fn delete(
        &self,
        collection: &str,
        ids: Option<Vec<String>>,
        where_filter: Option<&ChromaWhereFilter>,
    ) -> Result<()>;

    /// Get documents from a collection by IDs or filter.
    async fn get(
        &self,
        collection: &str,
        ids: Option<Vec<String>>,
        where_filter: Option<&ChromaWhereFilter>,
    ) -> Result<ChromaQueryResult>;

    /// Update documents in a collection.
    async fn update(
        &self,
        collection: &str,
        ids: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<HashMap<String, Value>>>,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Internal record type for MockChromaClient
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ChromaRecord {
    id: String,
    embedding: Vec<f32>,
    document: String,
    metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// MockChromaClient
// ---------------------------------------------------------------------------

/// In-memory mock implementation of `ChromaClient` for testing.
pub struct MockChromaClient {
    /// Records stored per collection.
    collections: RwLock<HashMap<String, Vec<ChromaRecord>>>,
    /// Metadata per collection.
    collection_metadata: RwLock<HashMap<String, HashMap<String, Value>>>,
}

impl MockChromaClient {
    /// Create a new empty mock client.
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            collection_metadata: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for MockChromaClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare two JSON values numerically, returning an ordering if possible.
fn compare_numeric(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    let a_f = a.as_f64().or_else(|| a.as_i64().map(|i| i as f64))?;
    let b_f = b.as_f64().or_else(|| b.as_i64().map(|i| i as f64))?;
    a_f.partial_cmp(&b_f)
}

/// Evaluate whether a metadata value satisfies a where operator.
fn evaluate_operator(field_value: Option<&Value>, operator: &ChromaWhereOperator) -> bool {
    match operator {
        ChromaWhereOperator::Eq(v) => field_value.map(|fv| fv == v).unwrap_or(false),
        ChromaWhereOperator::Ne(v) => field_value.map(|fv| fv != v).unwrap_or(true),
        ChromaWhereOperator::Gt(v) => field_value
            .and_then(|fv| compare_numeric(fv, v))
            .map(|ord| ord == std::cmp::Ordering::Greater)
            .unwrap_or(false),
        ChromaWhereOperator::Gte(v) => field_value
            .and_then(|fv| compare_numeric(fv, v))
            .map(|ord| ord != std::cmp::Ordering::Less)
            .unwrap_or(false),
        ChromaWhereOperator::Lt(v) => field_value
            .and_then(|fv| compare_numeric(fv, v))
            .map(|ord| ord == std::cmp::Ordering::Less)
            .unwrap_or(false),
        ChromaWhereOperator::Lte(v) => field_value
            .and_then(|fv| compare_numeric(fv, v))
            .map(|ord| ord != std::cmp::Ordering::Greater)
            .unwrap_or(false),
        ChromaWhereOperator::In(values) => {
            field_value.map(|fv| values.contains(fv)).unwrap_or(false)
        }
        ChromaWhereOperator::Nin(values) => {
            field_value.map(|fv| !values.contains(fv)).unwrap_or(true)
        }
    }
}

/// Evaluate whether a record's metadata matches a where filter.
fn matches_where_filter(metadata: &HashMap<String, Value>, filter: &ChromaWhereFilter) -> bool {
    match filter {
        ChromaWhereFilter::Condition { field, operator } => {
            evaluate_operator(metadata.get(field), operator)
        }
        ChromaWhereFilter::And(filters) => filters.iter().all(|f| matches_where_filter(metadata, f)),
        ChromaWhereFilter::Or(filters) => filters.iter().any(|f| matches_where_filter(metadata, f)),
    }
}

/// Evaluate whether a document matches a document content filter.
fn matches_document_filter(document: &str, filter: &ChromaWhereDocumentFilter) -> bool {
    match filter {
        ChromaWhereDocumentFilter::Contains(s) => document.contains(s.as_str()),
        ChromaWhereDocumentFilter::NotContains(s) => !document.contains(s.as_str()),
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

#[async_trait]
impl ChromaClient for MockChromaClient {
    async fn create_collection(
        &self,
        name: &str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        collections
            .entry(name.to_string())
            .or_insert_with(Vec::new);

        if let Some(meta) = metadata {
            let mut col_meta = self.collection_metadata.write().await;
            col_meta.insert(name.to_string(), meta);
        }
        Ok(())
    }

    async fn add(
        &self,
        collection: &str,
        ids: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<HashMap<String, Value>>>,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        let coll = collections
            .entry(collection.to_string())
            .or_insert_with(Vec::new);

        for (i, id) in ids.iter().enumerate() {
            let embedding = embeddings
                .as_ref()
                .and_then(|e| e.get(i))
                .cloned()
                .unwrap_or_default();
            let document = documents
                .as_ref()
                .and_then(|d| d.get(i))
                .cloned()
                .unwrap_or_default();
            let metadata = metadatas
                .as_ref()
                .and_then(|m| m.get(i))
                .cloned()
                .unwrap_or_default();

            // Remove existing record with the same ID, then insert.
            coll.retain(|r| r.id != *id);
            coll.push(ChromaRecord {
                id: id.clone(),
                embedding,
                document,
                metadata,
            });
        }
        Ok(())
    }

    async fn query(
        &self,
        collection: &str,
        query_embeddings: Vec<Vec<f32>>,
        n_results: usize,
        where_filter: Option<&ChromaWhereFilter>,
        where_document: Option<&ChromaWhereDocumentFilter>,
    ) -> Result<ChromaQueryResult> {
        let collections = self.collections.read().await;
        let Some(coll) = collections.get(collection) else {
            return Ok(ChromaQueryResult::default());
        };

        let mut all_ids = Vec::new();
        let mut all_documents = Vec::new();
        let mut all_metadatas = Vec::new();
        let mut all_distances = Vec::new();
        let mut all_embeddings = Vec::new();

        for query_embedding in &query_embeddings {
            let mut scored: Vec<(&ChromaRecord, f32)> = coll
                .iter()
                .filter(|r| {
                    if let Some(wf) = where_filter {
                        if !matches_where_filter(&r.metadata, wf) {
                            return false;
                        }
                    }
                    if let Some(df) = where_document {
                        if !matches_document_filter(&r.document, df) {
                            return false;
                        }
                    }
                    true
                })
                .map(|r| {
                    let score = cosine_similarity(query_embedding, &r.embedding);
                    (r, score)
                })
                .collect();

            // Sort descending by score (higher similarity first).
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(n_results);

            // Chroma returns distances (lower = more similar), so convert from similarity.
            let ids: Vec<String> = scored.iter().map(|(r, _)| r.id.clone()).collect();
            let docs: Vec<String> = scored.iter().map(|(r, _)| r.document.clone()).collect();
            let metas: Vec<HashMap<String, Value>> =
                scored.iter().map(|(r, _)| r.metadata.clone()).collect();
            let dists: Vec<f32> = scored.iter().map(|(_, score)| 1.0 - score).collect();
            let embs: Vec<Vec<f32>> = scored.iter().map(|(r, _)| r.embedding.clone()).collect();

            all_ids.push(ids);
            all_documents.push(docs);
            all_metadatas.push(metas);
            all_distances.push(dists);
            all_embeddings.push(embs);
        }

        Ok(ChromaQueryResult {
            ids: all_ids,
            embeddings: Some(all_embeddings),
            documents: Some(all_documents),
            metadatas: Some(all_metadatas),
            distances: Some(all_distances),
        })
    }

    async fn delete(
        &self,
        collection: &str,
        ids: Option<Vec<String>>,
        where_filter: Option<&ChromaWhereFilter>,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        let Some(coll) = collections.get_mut(collection) else {
            return Ok(());
        };

        coll.retain(|r| {
            // If IDs are specified, check membership.
            if let Some(ref id_list) = ids {
                if id_list.contains(&r.id) {
                    return false;
                }
            }
            // If a where filter is specified, check match.
            if let Some(wf) = where_filter {
                if matches_where_filter(&r.metadata, wf) {
                    return false;
                }
            }
            // Keep the record if neither condition matched.
            // If both ids and where_filter are None, keep everything.
            ids.is_some() || where_filter.is_some()
        });
        Ok(())
    }

    async fn get(
        &self,
        collection: &str,
        ids: Option<Vec<String>>,
        where_filter: Option<&ChromaWhereFilter>,
    ) -> Result<ChromaQueryResult> {
        let collections = self.collections.read().await;
        let Some(coll) = collections.get(collection) else {
            return Ok(ChromaQueryResult::default());
        };

        let filtered: Vec<&ChromaRecord> = coll
            .iter()
            .filter(|r| {
                if let Some(ref id_list) = ids {
                    if !id_list.contains(&r.id) {
                        return false;
                    }
                }
                if let Some(wf) = where_filter {
                    if !matches_where_filter(&r.metadata, wf) {
                        return false;
                    }
                }
                true
            })
            .collect();

        let result_ids: Vec<String> = filtered.iter().map(|r| r.id.clone()).collect();
        let result_docs: Vec<String> = filtered.iter().map(|r| r.document.clone()).collect();
        let result_metas: Vec<HashMap<String, Value>> =
            filtered.iter().map(|r| r.metadata.clone()).collect();
        let result_embs: Vec<Vec<f32>> = filtered.iter().map(|r| r.embedding.clone()).collect();

        Ok(ChromaQueryResult {
            ids: vec![result_ids],
            embeddings: Some(vec![result_embs]),
            documents: Some(vec![result_docs]),
            metadatas: Some(vec![result_metas]),
            distances: None,
        })
    }

    async fn update(
        &self,
        collection: &str,
        ids: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<HashMap<String, Value>>>,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        let Some(coll) = collections.get_mut(collection) else {
            return Ok(());
        };

        for (i, id) in ids.iter().enumerate() {
            if let Some(record) = coll.iter_mut().find(|r| r.id == *id) {
                if let Some(ref embs) = embeddings {
                    if let Some(emb) = embs.get(i) {
                        record.embedding = emb.clone();
                    }
                }
                if let Some(ref docs) = documents {
                    if let Some(doc) = docs.get(i) {
                        record.document = doc.clone();
                    }
                }
                if let Some(ref metas) = metadatas {
                    if let Some(meta) = metas.get(i) {
                        record.metadata = meta.clone();
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChromaVectorStore
// ---------------------------------------------------------------------------

/// Vector store backed by ChromaDB (or a mock thereof).
///
/// Wraps a `ChromaClient` implementation and an `Embeddings` model to provide
/// the full `VectorStore` trait interface.
pub struct ChromaVectorStore {
    client: Arc<dyn ChromaClient>,
    embeddings: Arc<dyn Embeddings>,
    config: ChromaConfig,
}

impl ChromaVectorStore {
    /// Create a new Chroma vector store.
    pub fn new(
        client: Arc<dyn ChromaClient>,
        embeddings: Arc<dyn Embeddings>,
        config: ChromaConfig,
    ) -> Self {
        Self {
            client,
            embeddings,
            config,
        }
    }

    /// Create a Chroma vector store pre-populated with documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        client: Arc<dyn ChromaClient>,
        embeddings: Arc<dyn Embeddings>,
        config: ChromaConfig,
    ) -> Result<Self> {
        let store = Self::new(client, embeddings, config);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Perform a similarity search with optional metadata and document filters, returning scores.
    pub async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        where_filter: Option<&ChromaWhereFilter>,
        where_document: Option<&ChromaWhereDocumentFilter>,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let result = self
            .client
            .query(
                &self.config.collection_name,
                vec![query_embedding],
                k,
                where_filter,
                where_document,
            )
            .await?;

        let ids = result.ids.first().cloned().unwrap_or_default();
        let documents = result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .cloned()
            .unwrap_or_default();
        let metadatas = result
            .metadatas
            .as_ref()
            .and_then(|m| m.first())
            .cloned()
            .unwrap_or_default();
        let distances = result
            .distances
            .as_ref()
            .and_then(|d| d.first())
            .cloned()
            .unwrap_or_default();

        let mut docs = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let content = documents.get(i).cloned().unwrap_or_default();
            let metadata = metadatas.get(i).cloned().unwrap_or_default();
            // Convert distance to similarity score (higher is better).
            let score = 1.0 - distances.get(i).copied().unwrap_or(1.0);

            let doc = Document::new(content)
                .with_id(id.clone())
                .with_metadata(metadata);
            docs.push((doc, score));
        }

        Ok(docs)
    }

    /// Delete documents by metadata filter.
    pub async fn delete_by_filter(&self, where_filter: &ChromaWhereFilter) -> Result<()> {
        self.client
            .delete(&self.config.collection_name, None, Some(where_filter))
            .await
    }

    /// Return a reference to the underlying config.
    pub fn config(&self) -> &ChromaConfig {
        &self.config
    }
}

#[async_trait]
impl VectorStore for ChromaVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;

        let mut result_ids = Vec::with_capacity(texts.len());
        let mut doc_texts = Vec::with_capacity(texts.len());
        let mut doc_metadatas = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let metadata: HashMap<String, Value> = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            result_ids.push(id);
            doc_texts.push(text.clone());
            doc_metadatas.push(metadata);
        }

        self.client
            .add(
                &self.config.collection_name,
                result_ids.clone(),
                Some(embeddings_vec),
                Some(doc_texts),
                Some(doc_metadatas),
            )
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
            .delete(
                &self.config.collection_name,
                Some(ids.to_vec()),
                None,
            )
            .await?;
        Ok(true)
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let result = self
            .client
            .get(&self.config.collection_name, Some(ids.to_vec()), None)
            .await?;

        let result_ids = result.ids.first().cloned().unwrap_or_default();
        let documents = result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .cloned()
            .unwrap_or_default();
        let metadatas = result
            .metadatas
            .as_ref()
            .and_then(|m| m.first())
            .cloned()
            .unwrap_or_default();

        let mut docs = Vec::new();
        for (i, id) in result_ids.iter().enumerate() {
            let content = documents.get(i).cloned().unwrap_or_default();
            let metadata = metadatas.get(i).cloned().unwrap_or_default();
            let doc = Document::new(content)
                .with_id(id.clone())
                .with_metadata(metadata);
            docs.push(doc);
        }

        Ok(docs)
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
        self.similarity_search_with_filter(query, k, None, None).await
    }

    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        let result = self
            .client
            .query(
                &self.config.collection_name,
                vec![embedding.to_vec()],
                k,
                None,
                None,
            )
            .await?;

        let ids = result.ids.first().cloned().unwrap_or_default();
        let documents = result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .cloned()
            .unwrap_or_default();
        let metadatas = result
            .metadatas
            .as_ref()
            .and_then(|m| m.first())
            .cloned()
            .unwrap_or_default();

        let mut docs = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let content = documents.get(i).cloned().unwrap_or_default();
            let metadata = metadatas.get(i).cloned().unwrap_or_default();
            let doc = Document::new(content)
                .with_id(id.clone())
                .with_metadata(metadata);
            docs.push(doc);
        }

        Ok(docs)
    }

    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let result = self
            .client
            .query(
                &self.config.collection_name,
                vec![query_embedding.clone()],
                fetch_k,
                None,
                None,
            )
            .await?;

        let ids = result.ids.first().cloned().unwrap_or_default();
        let documents = result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .cloned()
            .unwrap_or_default();
        let metadatas = result
            .metadatas
            .as_ref()
            .and_then(|m| m.first())
            .cloned()
            .unwrap_or_default();
        let embeddings = result
            .embeddings
            .as_ref()
            .and_then(|e| e.first())
            .cloned()
            .unwrap_or_default();

        if ids.is_empty() {
            return Ok(vec![]);
        }

        let candidate_embeddings: Vec<Vec<f64>> = embeddings
            .iter()
            .map(|e| e.iter().map(|&v| v as f64).collect())
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
            .filter_map(|idx| {
                let id = ids.get(idx)?;
                let content = documents.get(idx).cloned().unwrap_or_default();
                let metadata = metadatas.get(idx).cloned().unwrap_or_default();
                Some(
                    Document::new(content)
                        .with_id(id.clone())
                        .with_metadata(metadata),
                )
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

    fn make_store() -> ChromaVectorStore {
        let client = Arc::new(MockChromaClient::new());
        let embeddings = make_embeddings();
        let config = ChromaConfig::new("test_collection");
        ChromaVectorStore::new(client, embeddings, config)
    }

    fn make_store_with_client() -> (ChromaVectorStore, Arc<MockChromaClient>) {
        let client = Arc::new(MockChromaClient::new());
        let embeddings = make_embeddings();
        let config = ChromaConfig::new("test_collection");
        let store = ChromaVectorStore::new(client.clone(), embeddings, config);
        (store, client)
    }

    // ---- Test 1: Add and query documents ----
    #[tokio::test]
    async fn test_add_and_query_documents() {
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

    // ---- Test 3: Metadata filtering with $eq ----
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

        let filter = ChromaWhereFilter::eq("color", Value::String("red".into()));
        let results = store
            .similarity_search_with_filter("fruit", 10, Some(&filter), None)
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

    // ---- Test 4: Metadata filtering with $gt ----
    #[tokio::test]
    async fn test_metadata_filter_gt() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::from(10));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::from(20));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("score".into(), Value::from(30));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = ChromaWhereFilter::gt("score", Value::from(15));
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let score = doc.metadata.get("score").unwrap().as_i64().unwrap();
            assert!(score > 15);
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

        let filter = ChromaWhereFilter::r#in(
            "type",
            vec![Value::String("x".into()), Value::String("z".into())],
        );
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let t = doc.metadata.get("type").unwrap().as_str().unwrap();
            assert!(t == "x" || t == "z");
        }
    }

    // ---- Test 6: Document content filtering with $contains ----
    #[tokio::test]
    async fn test_document_filter_contains() {
        let store = make_store();
        let texts = vec![
            "Rust programming language".into(),
            "Python scripting".into(),
            "Rust systems programming".into(),
        ];
        store.add_texts(&texts, None, None).await.unwrap();

        let filter = ChromaWhereDocumentFilter::Contains("Rust".into());
        let results = store
            .similarity_search_with_filter("programming", 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            assert!(doc.page_content.contains("Rust"));
        }
    }

    // ---- Test 7: $and compound filter ----
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

        // Must be food AND organic
        let filter = ChromaWhereFilter::and(vec![
            ChromaWhereFilter::eq("category", Value::String("food".into())),
            ChromaWhereFilter::eq("organic", Value::Bool(true)),
        ]);
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.page_content, "a");
    }

    // ---- Test 8: $or compound filter ----
    #[tokio::test]
    async fn test_or_compound_filter() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("active".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("inactive".into()));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("status".into(), Value::String("pending".into()));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = ChromaWhereFilter::or(vec![
            ChromaWhereFilter::eq("status", Value::String("active".into())),
            ChromaWhereFilter::eq("status", Value::String("pending".into())),
        ]);
        let results = store
            .similarity_search_with_filter("query", 10, Some(&filter), None)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        for (doc, _) in &results {
            let status = doc.metadata.get("status").unwrap().as_str().unwrap();
            assert!(status == "active" || status == "pending");
        }
    }

    // ---- Test 9: Delete by IDs ----
    #[tokio::test]
    async fn test_delete_by_ids() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&[ids[1].clone()])).await.unwrap();
        assert!(deleted);

        let remaining = store.similarity_search("a", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|d| d.page_content != "b"));
    }

    // ---- Test 10: Delete by filter ----
    #[tokio::test]
    async fn test_delete_by_filter() {
        let store = make_store();
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let metadatas = vec![
            {
                let mut m = HashMap::new();
                m.insert("keep".into(), Value::Bool(true));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("keep".into(), Value::Bool(false));
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("keep".into(), Value::Bool(true));
                m
            },
        ];
        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();

        let filter = ChromaWhereFilter::eq("keep", Value::Bool(false));
        store.delete_by_filter(&filter).await.unwrap();

        let remaining = store.similarity_search("query", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        for doc in &remaining {
            assert_eq!(doc.metadata.get("keep").unwrap(), &Value::Bool(true));
        }
    }

    // ---- Test 11: Update documents ----
    #[tokio::test]
    async fn test_update_documents() {
        let (store, client) = make_store_with_client();
        let texts = vec!["original".into()];
        let ids = store
            .add_texts(&texts, None, Some(&["doc1".to_string()]))
            .await
            .unwrap();
        assert_eq!(ids, vec!["doc1"]);

        // Update the document content.
        client
            .update(
                "test_collection",
                vec!["doc1".to_string()],
                None,
                Some(vec!["updated content".to_string()]),
                None,
            )
            .await
            .unwrap();

        let docs = store.get_by_ids(&["doc1".to_string()]).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "updated content");
    }

    // ---- Test 12: Config defaults ----
    #[tokio::test]
    async fn test_config_defaults() {
        let config = ChromaConfig::new("my_collection");
        assert_eq!(config.url, "http://localhost:8000");
        assert_eq!(config.collection_name, "my_collection");
        assert!(config.tenant.is_none());
        assert!(config.database.is_none());
        assert!(config.embedding_function.is_none());
    }

    // ---- Test 13: Get by IDs ----
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

    // ---- Test 14: Empty collection query ----
    #[tokio::test]
    async fn test_empty_collection_query() {
        let store = make_store();
        let results = store.similarity_search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ---- Test 15: from_documents constructor ----
    #[tokio::test]
    async fn test_from_documents_constructor() {
        let client = Arc::new(MockChromaClient::new());
        let embeddings = make_embeddings();
        let config = ChromaConfig::new("test_collection");

        let docs = vec![
            Document::new("hello world").with_id("h1"),
            Document::new("goodbye world").with_id("g1"),
        ];

        let store = ChromaVectorStore::from_documents(docs, client, embeddings, config)
            .await
            .unwrap();

        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }

    // ---- Test 16: Collection metadata ----
    #[tokio::test]
    async fn test_collection_metadata() {
        let client = Arc::new(MockChromaClient::new());

        let mut metadata = HashMap::new();
        metadata.insert(
            "description".into(),
            Value::String("test collection".into()),
        );
        metadata.insert("version".into(), Value::from(1));

        client
            .create_collection("my_collection", Some(metadata.clone()))
            .await
            .unwrap();

        // Verify collection was created by adding and retrieving a document.
        client
            .add(
                "my_collection",
                vec!["doc1".to_string()],
                Some(vec![vec![1.0, 0.0]]),
                Some(vec!["test doc".to_string()]),
                None,
            )
            .await
            .unwrap();

        let result = client
            .get("my_collection", Some(vec!["doc1".to_string()]), None)
            .await
            .unwrap();

        assert_eq!(result.ids.first().unwrap().len(), 1);
        assert_eq!(
            result
                .documents
                .as_ref()
                .unwrap()
                .first()
                .unwrap()
                .first()
                .unwrap(),
            "test doc"
        );

        // Verify collection metadata is stored.
        let col_meta = client.collection_metadata.read().await;
        assert_eq!(col_meta.get("my_collection").unwrap(), &metadata);
    }

    // ---- Test 17: Batch operations ----
    #[tokio::test]
    async fn test_batch_operations() {
        let store = make_store();
        let texts: Vec<String> = (0..50).map(|i| format!("document_{}", i)).collect();
        let ids = store.add_texts(&texts, None, None).await.unwrap();
        assert_eq!(ids.len(), 50);

        // Query to verify all documents are stored.
        let results = store
            .similarity_search("document_25", 5)
            .await
            .unwrap();
        assert_eq!(results.len(), 5);

        // Delete a batch of documents.
        let to_delete: Vec<String> = ids[0..10].to_vec();
        let deleted = store.delete(Some(&to_delete)).await.unwrap();
        assert!(deleted);

        // Verify deletion.
        let remaining = store.similarity_search("document", 50).await.unwrap();
        assert_eq!(remaining.len(), 40);
    }
}
