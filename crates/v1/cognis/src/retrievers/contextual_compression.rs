//! Contextual compression retriever that filters and compresses documents
//! based on query relevance.
//!
//! Provides two compression strategies:
//! - [`LLMCompressor`] — Uses a chat model to extract relevant parts from documents.
//! - [`EmbeddingsFilter`] — Uses embeddings similarity to filter irrelevant documents.
//!
//! The [`ContextualCompressionRetriever`] wraps a base retriever and applies a
//! [`DocumentCompressor`] to post-process retrieved documents.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::retrievers::BaseRetriever;

/// Trait for compressing or filtering documents based on a query.
#[async_trait]
pub trait DocumentCompressor: Send + Sync {
    /// Compress or filter a list of documents given a query.
    ///
    /// Implementations may shorten document content, remove irrelevant documents,
    /// or both.
    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>>;

    /// The name of this compressor.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// LLMCompressor
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT_TEMPLATE: &str = "Given the following question and document, extract only the parts of the document that are relevant to answering the question. If the document is not relevant at all, respond with exactly \"NO_RELEVANT_CONTENT\".\n\nQuestion: {query}\n\nDocument:\n{document}\n\nRelevant parts:";

/// Uses a [`BaseChatModel`] to extract query-relevant passages from each document.
///
/// Documents whose content is deemed entirely irrelevant are dropped.
///
/// # Example
///
/// ```rust,ignore
/// let compressor = LLMCompressor::new(model)
///     .with_prompt_template("Custom prompt: {query}\n{document}");
/// let compressed = compressor.compress_documents(&docs, "query").await?;
/// ```
pub struct LLMCompressor {
    /// The chat model used for compression.
    model: Arc<dyn BaseChatModel>,
    /// Prompt template with `{query}` and `{document}` placeholders.
    prompt_template: String,
}

impl LLMCompressor {
    /// Create a new `LLMCompressor` with the given chat model.
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        Self {
            model,
            prompt_template: DEFAULT_PROMPT_TEMPLATE.to_string(),
        }
    }

    /// Set a custom prompt template.
    ///
    /// The template must contain `{query}` and `{document}` placeholders.
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }

    /// Format the prompt for a given query and document content.
    fn format_prompt(&self, query: &str, document: &str) -> String {
        self.prompt_template
            .replace("{query}", query)
            .replace("{document}", document)
    }
}

#[async_trait]
impl DocumentCompressor for LLMCompressor {
    fn name(&self) -> &str {
        "LLMCompressor"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>> {
        let mut compressed = Vec::new();
        for doc in documents {
            let prompt = self.format_prompt(query, &doc.page_content);
            let message = Message::Human(HumanMessage::new(prompt));
            let ai_msg = self.model.invoke_messages(&[message], None).await?;
            let text = ai_msg.base.content.text();
            let trimmed = text.trim();
            if !trimmed.is_empty() && trimmed != "NO_RELEVANT_CONTENT" {
                let mut compressed_doc = doc.clone();
                compressed_doc.page_content = trimmed.to_string();
                compressed.push(compressed_doc);
            }
        }
        Ok(compressed)
    }
}

// ---------------------------------------------------------------------------
// EmbeddingsFilter
// ---------------------------------------------------------------------------

/// Filters documents by computing cosine similarity between the query embedding
/// and each document embedding, keeping only those above a threshold or the
/// top-k most similar.
///
/// # Example
///
/// ```rust,ignore
/// let filter = EmbeddingsFilter::new(embeddings)
///     .with_similarity_threshold(0.75)
///     .with_top_k(5);
/// let filtered = filter.compress_documents(&docs, "query").await?;
/// ```
pub struct EmbeddingsFilter {
    /// The embeddings model.
    embeddings: Arc<dyn Embeddings>,
    /// Minimum cosine similarity to keep a document. Default: 0.0 (keep all).
    similarity_threshold: f32,
    /// If set, keep at most this many documents (by descending similarity).
    top_k: Option<usize>,
}

impl EmbeddingsFilter {
    /// Create a new `EmbeddingsFilter` with the given embeddings model.
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            similarity_threshold: 0.0,
            top_k: None,
        }
    }

    /// Set the minimum cosine similarity threshold.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set the maximum number of documents to keep.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[async_trait]
impl DocumentCompressor for EmbeddingsFilter {
    fn name(&self) -> &str {
        "EmbeddingsFilter"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // Embed query and documents.
        let query_embedding = self.embeddings.embed_query(query).await?;
        let doc_texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let doc_embeddings = self.embeddings.embed_documents(doc_texts).await?;

        // Score each document.
        let mut scored: Vec<(usize, f32)> = doc_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
            .filter(|(_, sim)| *sim >= self.similarity_threshold)
            .collect();

        // Sort by descending similarity.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top_k limit.
        if let Some(k) = self.top_k {
            scored.truncate(k);
        }

        Ok(scored
            .into_iter()
            .map(|(i, _)| documents[i].clone())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// ContextualCompressionRetriever
// ---------------------------------------------------------------------------

/// A retriever that retrieves documents from a base retriever and then
/// compresses/filters them using a [`DocumentCompressor`].
///
/// # Example
///
/// ```rust,ignore
/// let retriever = ContextualCompressionRetriever::new(base_retriever, compressor);
/// let docs = retriever.get_relevant_documents("query").await?;
/// ```
pub struct ContextualCompressionRetriever {
    /// The base retriever to fetch initial documents.
    base_retriever: Box<dyn BaseRetriever>,
    /// The compressor/filter to apply to retrieved documents.
    compressor: Box<dyn DocumentCompressor>,
}

impl ContextualCompressionRetriever {
    /// Create a new `ContextualCompressionRetriever`.
    pub fn new(
        base_retriever: Box<dyn BaseRetriever>,
        compressor: Box<dyn DocumentCompressor>,
    ) -> Self {
        Self {
            base_retriever,
            compressor,
        }
    }

    /// Builder: set the base retriever.
    pub fn with_base_retriever(mut self, retriever: Box<dyn BaseRetriever>) -> Self {
        self.base_retriever = retriever;
        self
    }

    /// Builder: set the document compressor.
    pub fn with_compressor(mut self, compressor: Box<dyn DocumentCompressor>) -> Self {
        self.compressor = compressor;
        self
    }
}

#[async_trait]
impl BaseRetriever for ContextualCompressionRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        let docs = self.base_retriever.get_relevant_documents(query).await?;
        self.compressor.compress_documents(&docs, query).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::AIMessage;
    use cognis_core::outputs::{ChatGeneration, ChatResult};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // -- Mock Retriever --

    struct MockRetriever {
        docs: Vec<Document>,
    }

    impl MockRetriever {
        fn new(contents: &[&str]) -> Self {
            Self {
                docs: contents.iter().map(|c| Document::new(*c)).collect(),
            }
        }
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    // -- Mock Chat Model --

    /// A fake chat model that returns a predetermined response for each invocation.
    struct FakeChatModel {
        /// Responses to return in order. If exhausted, returns "NO_RELEVANT_CONTENT".
        responses: Mutex<Vec<String>>,
    }

    impl FakeChatModel {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for FakeChatModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let response = {
                let mut resps = self.responses.lock().unwrap();
                if resps.is_empty() {
                    "NO_RELEVANT_CONTENT".to_string()
                } else {
                    resps.remove(0)
                }
            };
            let ai_msg = AIMessage::new(&response);
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai_msg)],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "fake"
        }
    }

    // -- Mock Embeddings --

    /// A fake embeddings model that returns predetermined embeddings.
    struct FakeEmbeddings {
        /// Query embedding to return.
        query_embedding: Vec<f32>,
        /// Document embeddings to return (one per document, in order).
        doc_embeddings: Vec<Vec<f32>>,
    }

    #[async_trait]
    impl Embeddings for FakeEmbeddings {
        async fn embed_documents(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            Ok(self.doc_embeddings.clone())
        }

        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.query_embedding.clone())
        }
    }

    // -- Tests --

    #[tokio::test]
    async fn test_llm_compressor_extracts_relevant_text() {
        let model = Arc::new(FakeChatModel::new(vec![
            "relevant part A".into(),
            "relevant part B".into(),
        ]));
        let compressor = LLMCompressor::new(model);
        let docs = vec![Document::new("full doc A"), Document::new("full doc B")];
        let result = compressor.compress_documents(&docs, "query").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "relevant part A");
        assert_eq!(result[1].page_content, "relevant part B");
    }

    #[tokio::test]
    async fn test_llm_compressor_filters_irrelevant_documents() {
        let model = Arc::new(FakeChatModel::new(vec![
            "relevant content".into(),
            "NO_RELEVANT_CONTENT".into(),
            "also relevant".into(),
        ]));
        let compressor = LLMCompressor::new(model);
        let docs = vec![
            Document::new("doc 1"),
            Document::new("doc 2"),
            Document::new("doc 3"),
        ];
        let result = compressor.compress_documents(&docs, "query").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "relevant content");
        assert_eq!(result[1].page_content, "also relevant");
    }

    #[tokio::test]
    async fn test_llm_compressor_preserves_metadata() {
        let model = Arc::new(FakeChatModel::new(vec!["compressed".into()]));
        let compressor = LLMCompressor::new(model);
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), serde_json::json!("test.pdf"));
        let doc = Document::new("original content").with_metadata(metadata.clone());
        let result = compressor
            .compress_documents(&[doc], "query")
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "compressed");
        assert_eq!(result[0].metadata, metadata);
    }

    #[tokio::test]
    async fn test_embeddings_filter_with_threshold() {
        // Query embedding: [1, 0, 0]
        // Doc embeddings: [1,0,0] (sim=1.0), [0,1,0] (sim=0.0), [0.7,0.7,0] (sim~0.707)
        let embeddings = Arc::new(FakeEmbeddings {
            query_embedding: vec![1.0, 0.0, 0.0],
            doc_embeddings: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.707, 0.707, 0.0],
            ],
        });
        let filter = EmbeddingsFilter::new(embeddings).with_similarity_threshold(0.5);
        let docs = vec![
            Document::new("identical"),
            Document::new("orthogonal"),
            Document::new("partial"),
        ];
        let result = filter.compress_documents(&docs, "query").await.unwrap();
        // Only "identical" (1.0) and "partial" (~0.707) should pass threshold of 0.5.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "identical");
        assert_eq!(result[1].page_content, "partial");
    }

    #[tokio::test]
    async fn test_embeddings_filter_with_top_k() {
        let embeddings = Arc::new(FakeEmbeddings {
            query_embedding: vec![1.0, 0.0],
            doc_embeddings: vec![
                vec![1.0, 0.0], // sim=1.0
                vec![0.5, 0.5], // sim~0.707
                vec![0.9, 0.1], // sim~0.994
                vec![0.0, 1.0], // sim=0.0
            ],
        });
        let filter = EmbeddingsFilter::new(embeddings).with_top_k(2);
        let docs = vec![
            Document::new("doc_a"),
            Document::new("doc_b"),
            Document::new("doc_c"),
            Document::new("doc_d"),
        ];
        let result = filter.compress_documents(&docs, "query").await.unwrap();
        assert_eq!(result.len(), 2);
        // Top 2 by similarity: doc_a (1.0) and doc_c (~0.994)
        assert_eq!(result[0].page_content, "doc_a");
        assert_eq!(result[1].page_content, "doc_c");
    }

    #[tokio::test]
    async fn test_contextual_compression_retriever_with_llm() {
        let base = Box::new(MockRetriever::new(&["full document 1", "full document 2"]));
        let model = Arc::new(FakeChatModel::new(vec![
            "compressed 1".into(),
            "compressed 2".into(),
        ]));
        let compressor = Box::new(LLMCompressor::new(model));
        let retriever = ContextualCompressionRetriever::new(base, compressor);
        let docs = retriever.get_relevant_documents("query").await.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "compressed 1");
        assert_eq!(docs[1].page_content, "compressed 2");
    }

    #[tokio::test]
    async fn test_contextual_compression_retriever_with_embeddings_filter() {
        let base = Box::new(MockRetriever::new(&["relevant", "irrelevant"]));
        let embeddings = Arc::new(FakeEmbeddings {
            query_embedding: vec![1.0, 0.0],
            doc_embeddings: vec![
                vec![0.9, 0.1], // high similarity
                vec![0.0, 1.0], // low similarity
            ],
        });
        let filter = Box::new(EmbeddingsFilter::new(embeddings).with_similarity_threshold(0.5));
        let retriever = ContextualCompressionRetriever::new(base, filter);
        let docs = retriever.get_relevant_documents("query").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "relevant");
    }

    #[tokio::test]
    async fn test_empty_document_list() {
        let model = Arc::new(FakeChatModel::new(vec![]));
        let compressor = LLMCompressor::new(model);
        let result = compressor.compress_documents(&[], "query").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_all_documents_filtered_out() {
        let model = Arc::new(FakeChatModel::new(vec![
            "NO_RELEVANT_CONTENT".into(),
            "NO_RELEVANT_CONTENT".into(),
        ]));
        let compressor = LLMCompressor::new(model);
        let docs = vec![Document::new("doc 1"), Document::new("doc 2")];
        let result = compressor.compress_documents(&docs, "query").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_custom_prompt_template() {
        let model = Arc::new(FakeChatModel::new(vec!["answer".into()]));
        let custom_template = "Q: {query}\nDoc: {document}\nExtract:";
        let compressor = LLMCompressor::new(model).with_prompt_template(custom_template);
        // Verify the prompt is correctly formatted.
        let formatted = compressor.format_prompt("test query", "test doc");
        assert_eq!(formatted, "Q: test query\nDoc: test doc\nExtract:");
        // Verify it still works end-to-end.
        let docs = vec![Document::new("test doc")];
        let result = compressor
            .compress_documents(&docs, "test query")
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "answer");
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let base = Box::new(MockRetriever::new(&["doc"]));
        let model = Arc::new(FakeChatModel::new(vec!["compressed".into()]));
        let compressor = Box::new(LLMCompressor::new(model));

        // Build using chained builder methods.
        let new_base = Box::new(MockRetriever::new(&["new doc"]));
        let new_model = Arc::new(FakeChatModel::new(vec!["new compressed".into()]));
        let new_compressor = Box::new(LLMCompressor::new(new_model));

        let retriever = ContextualCompressionRetriever::new(base, compressor)
            .with_base_retriever(new_base)
            .with_compressor(new_compressor);

        let docs = retriever.get_relevant_documents("query").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "new compressed");
    }

    #[tokio::test]
    async fn test_embeddings_filter_empty_documents() {
        let embeddings = Arc::new(FakeEmbeddings {
            query_embedding: vec![1.0, 0.0],
            doc_embeddings: vec![],
        });
        let filter = EmbeddingsFilter::new(embeddings);
        let result = filter.compress_documents(&[], "query").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_cosine_similarity_function() {
        // Identical vectors -> 1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        // Orthogonal vectors -> 0.0
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        // Opposite vectors -> -1.0
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // Empty vectors -> 0.0
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
}
