//! Time-weighted retriever that scores documents by combining similarity with
//! recency via exponential decay.
//!
//! Documents that were added more recently receive a higher time bonus. The
//! combined score is `similarity + (1.0 - decay_rate).powf(hours_since_access)`.

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;
use cognis_core::retrievers::BaseRetriever;

/// A document with associated temporal metadata.
#[derive(Debug, Clone)]
pub struct TimedDocument {
    /// The underlying document.
    pub document: Document,
    /// When this document was added to the store.
    pub created_at: SystemTime,
    /// When this document was last accessed (returned as a result).
    pub last_accessed: SystemTime,
    /// Number of times this document has been accessed.
    pub access_count: usize,
}

impl TimedDocument {
    /// Create a new timed document with current timestamps.
    pub fn new(document: Document) -> Self {
        let now = SystemTime::now();
        Self {
            document,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Create a timed document with specific timestamps (useful for testing).
    pub fn with_timestamps(
        document: Document,
        created_at: SystemTime,
        last_accessed: SystemTime,
    ) -> Self {
        Self {
            document,
            created_at,
            last_accessed,
            access_count: 0,
        }
    }
}

/// A retriever that combines vector similarity with time-based decay to favor
/// more recently accessed documents.
///
/// The combined score for a document is:
///   `similarity + (1.0 - decay_rate).powf(hours_since_last_access)`
///
/// # Example
///
/// ```rust,ignore
/// use cognis::retrievers::time_weighted::TimeWeightedRetriever;
/// use std::sync::Arc;
///
/// let retriever = TimeWeightedRetriever::new(embeddings)
///     .with_decay_rate(0.01)
///     .with_k(4);
///
/// retriever.add_documents(vec![doc1, doc2]).await;
/// let results = retriever.get_relevant_documents("query").await?;
/// ```
pub struct TimeWeightedRetriever {
    /// Embeddings model for computing similarity.
    embeddings: Arc<dyn Embeddings>,
    /// The document store with temporal metadata.
    documents: Arc<RwLock<Vec<TimedDocument>>>,
    /// Exponential decay rate (default: 0.01).
    decay_rate: f64,
    /// Number of documents to return (default: 4).
    k: usize,
}

impl TimeWeightedRetriever {
    /// Create a new time-weighted retriever.
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            documents: Arc::new(RwLock::new(Vec::new())),
            decay_rate: 0.01,
            k: 4,
        }
    }

    /// Set the decay rate (higher values = faster decay of older documents).
    pub fn with_decay_rate(mut self, decay_rate: f64) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    /// Set the number of results to return.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Add documents to the store with the current timestamp.
    pub async fn add_documents(&self, docs: Vec<Document>) {
        let mut store = self.documents.write().await;
        for doc in docs {
            store.push(TimedDocument::new(doc));
        }
    }

    /// Add a single timed document directly (useful for testing or restoring state).
    pub async fn add_timed_document(&self, timed_doc: TimedDocument) {
        let mut store = self.documents.write().await;
        store.push(timed_doc);
    }

    /// Get the number of documents in the store.
    pub async fn len(&self) -> usize {
        self.documents.read().await.len()
    }

    /// Check if the document store is empty.
    pub async fn is_empty(&self) -> bool {
        self.documents.read().await.is_empty()
    }

    /// Compute the combined score from similarity and time decay.
    ///
    /// `combined_score = similarity + (1.0 - decay_rate).powf(hours_since)`
    pub fn combined_score(&self, similarity: f64, hours_since: f64) -> f64 {
        let time_score = (1.0 - self.decay_rate).powf(hours_since);
        similarity + time_score
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum();
        let norm_a: f64 = a
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = b
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute hours elapsed since the given time.
    fn hours_since(time: SystemTime) -> f64 {
        SystemTime::now()
            .duration_since(time)
            .unwrap_or_default()
            .as_secs_f64()
            / 3600.0
    }
}

#[async_trait]
impl BaseRetriever for TimeWeightedRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        let query_embedding = self.embeddings.embed_query(query).await?;

        let store = self.documents.read().await;
        if store.is_empty() {
            return Ok(vec![]);
        }

        // Embed all document contents.
        let doc_texts: Vec<String> = store
            .iter()
            .map(|td| td.document.page_content.clone())
            .collect();
        let doc_embeddings = self.embeddings.embed_documents(doc_texts).await?;

        // Score each document.
        let mut scored: Vec<(usize, f64)> = store
            .iter()
            .enumerate()
            .zip(doc_embeddings.iter())
            .map(|((idx, timed_doc), embedding)| {
                let similarity = Self::cosine_similarity(&query_embedding, embedding);
                let hours = Self::hours_since(timed_doc.last_accessed);
                let score = self.combined_score(similarity, hours);
                (idx, score)
            })
            .collect();

        // Sort by score descending.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k.
        let top_indices: Vec<usize> = scored.iter().take(self.k).map(|(idx, _)| *idx).collect();

        drop(store);

        // Update access metadata.
        let mut store = self.documents.write().await;
        let now = SystemTime::now();
        let mut results = Vec::with_capacity(top_indices.len());
        for idx in &top_indices {
            if let Some(timed_doc) = store.get_mut(*idx) {
                timed_doc.last_accessed = now;
                timed_doc.access_count += 1;
                results.push(timed_doc.document.clone());
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A mock embeddings model that returns simple embeddings based on
    /// document content for testing purposes.
    struct MockEmbeddings;

    #[async_trait]
    impl Embeddings for MockEmbeddings {
        async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| text_to_embedding(t)).collect())
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(text_to_embedding(text))
        }
    }

    /// Convert text to a simple deterministic embedding for testing.
    /// Uses the sum of byte values in different positions to create a 3D vector.
    fn text_to_embedding(text: &str) -> Vec<f32> {
        let bytes = text.as_bytes();
        let dim0: f32 = bytes.iter().map(|b| *b as f32).sum::<f32>() / 1000.0;
        let dim1: f32 = bytes
            .iter()
            .enumerate()
            .map(|(i, b)| (i as f32 + 1.0) * (*b as f32))
            .sum::<f32>()
            / 10000.0;
        let dim2: f32 = bytes.len() as f32 / 100.0;
        vec![dim0, dim1, dim2]
    }

    #[tokio::test]
    async fn test_empty_store_returns_empty() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings);

        let docs = retriever.get_relevant_documents("query").await.unwrap();
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn test_add_documents() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings);

        retriever
            .add_documents(vec![Document::new("doc1"), Document::new("doc2")])
            .await;

        assert_eq!(retriever.len().await, 2);
        assert!(!retriever.is_empty().await);
    }

    #[tokio::test]
    async fn test_retrieve_returns_up_to_k() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_k(2);

        retriever
            .add_documents(vec![
                Document::new("alpha"),
                Document::new("beta"),
                Document::new("gamma"),
                Document::new("delta"),
            ])
            .await;

        let docs = retriever.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_retrieve_returns_all_when_fewer_than_k() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_k(10);

        retriever
            .add_documents(vec![Document::new("only_one")])
            .await;

        let docs = retriever.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "only_one");
    }

    #[tokio::test]
    async fn test_combined_score_calculation() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_decay_rate(0.01);

        // At 0 hours, time component is 1.0
        let score = retriever.combined_score(0.5, 0.0);
        assert!((score - 1.5).abs() < 1e-10);

        // At some hours, time component decays
        let score_later = retriever.combined_score(0.5, 24.0);
        assert!(score_later < score);
        assert!(score_later > 0.5); // Still positive
    }

    #[tokio::test]
    async fn test_high_decay_rate_penalizes_old_documents() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_decay_rate(0.5);

        // With 50% decay rate, after 1 hour the time score is 0.5
        let score = retriever.combined_score(0.0, 1.0);
        assert!((score - 0.5).abs() < 1e-10);

        // After 2 hours: (0.5)^2 = 0.25
        let score2 = retriever.combined_score(0.0, 2.0);
        assert!((score2 - 0.25).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_zero_decay_rate_no_time_penalty() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_decay_rate(0.0);

        // With 0 decay, time component is always 1.0
        let score = retriever.combined_score(0.5, 1000.0);
        assert!((score - 1.5).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let similarity = TimeWeightedRetriever::cosine_similarity(&a, &a);
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = TimeWeightedRetriever::cosine_similarity(&a, &b);
        assert!(similarity.abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_cosine_similarity_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let similarity = TimeWeightedRetriever::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[tokio::test]
    async fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let similarity = TimeWeightedRetriever::cosine_similarity(&a, &b);
        assert_eq!(similarity, 0.0);
    }

    #[tokio::test]
    async fn test_access_count_increments() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_k(10);

        retriever.add_documents(vec![Document::new("doc")]).await;

        retriever.get_relevant_documents("q1").await.unwrap();
        retriever.get_relevant_documents("q2").await.unwrap();

        let store = retriever.documents.read().await;
        assert_eq!(store[0].access_count, 2);
    }

    #[tokio::test]
    async fn test_older_documents_scored_lower() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings)
            .with_k(2)
            .with_decay_rate(0.99);

        // Add a document with an old timestamp.
        let old_doc = TimedDocument::with_timestamps(
            Document::new("old_doc"),
            SystemTime::now() - Duration::from_secs(3600 * 24), // 24 hours ago
            SystemTime::now() - Duration::from_secs(3600 * 24),
        );
        retriever.add_timed_document(old_doc).await;

        // Add a recent document with similar content.
        retriever
            .add_documents(vec![Document::new("new_doc")])
            .await;

        let docs = retriever.get_relevant_documents("doc").await.unwrap();
        // The new doc should be ranked first due to less time decay.
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "new_doc");
    }

    #[tokio::test]
    async fn test_with_k_builder() {
        let embeddings = Arc::new(MockEmbeddings);
        let retriever = TimeWeightedRetriever::new(embeddings).with_k(7);

        retriever
            .add_documents((0..10).map(|i| Document::new(format!("doc{i}"))).collect())
            .await;

        let docs = retriever.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs.len(), 7);
    }

    #[tokio::test]
    async fn test_timed_document_creation() {
        let doc = Document::new("test content");
        let timed = TimedDocument::new(doc.clone());

        assert_eq!(timed.document.page_content, "test content");
        assert_eq!(timed.access_count, 0);

        // created_at and last_accessed should be very close to now.
        let elapsed = timed.created_at.elapsed().unwrap();
        assert!(elapsed < Duration::from_secs(1));
    }
}
