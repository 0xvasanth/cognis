//! Document transformer pipeline for processing, filtering, and enriching documents.
//!
//! This module provides:
//! - [`DocumentTransformer`] trait for transforming document collections
//! - [`EmbeddingsRedundantFilter`] to remove near-duplicate documents using embeddings
//! - [`LLMDocumentTransformer`] to transform documents using a chat model
//! - [`DocumentTransformerPipeline`] to chain multiple transformers in sequence
//! - [`MetadataEnricher`] to add computed metadata fields to documents

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::Message;

/// Trait for transforming a collection of documents.
#[async_trait]
pub trait DocumentTransformer: Send + Sync {
    /// Transform a slice of documents into a new collection.
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>>;

    /// The name of this transformer.
    fn name(&self) -> &str;
}

// ─── EmbeddingsRedundantFilter ───

/// Removes near-duplicate documents by comparing embedding similarity.
///
/// Documents whose cosine similarity exceeds the configured threshold are
/// considered redundant. The first occurrence is always kept.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::EmbeddingsRedundantFilter;
///
/// let filter = EmbeddingsRedundantFilter::new(embeddings)
///     .with_similarity_threshold(0.90);
/// let unique = filter.transform_documents(&docs).await?;
/// ```
pub struct EmbeddingsRedundantFilter {
    embeddings: Arc<dyn Embeddings>,
    similarity_threshold: f32,
}

impl EmbeddingsRedundantFilter {
    /// Create a new filter with the given embeddings provider and default
    /// similarity threshold of 0.95.
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            similarity_threshold: 0.95,
        }
    }

    /// Set the cosine similarity threshold above which documents are considered
    /// duplicates. Must be in `[0.0, 1.0]`.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[async_trait]
impl DocumentTransformer for EmbeddingsRedundantFilter {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let embeddings = self.embeddings.embed_documents(texts).await?;

        let mut keep_indices: Vec<usize> = Vec::new();

        for (i, emb_i) in embeddings.iter().enumerate() {
            let is_duplicate = keep_indices
                .iter()
                .any(|&j| cosine_similarity(emb_i, &embeddings[j]) >= self.similarity_threshold);
            if !is_duplicate {
                keep_indices.push(i);
            }
        }

        Ok(keep_indices
            .into_iter()
            .map(|i| documents[i].clone())
            .collect())
    }

    fn name(&self) -> &str {
        "EmbeddingsRedundantFilter"
    }
}

// ─── LLMDocumentTransformer ───

/// Transforms each document using a chat model (LLM).
///
/// The prompt template must contain `{document}` which will be replaced with
/// each document's page content before sending to the model.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::LLMDocumentTransformer;
///
/// let transformer = LLMDocumentTransformer::new(model)
///     .with_prompt("Summarize the following text:\n\n{document}");
/// let summaries = transformer.transform_documents(&docs).await?;
/// ```
pub struct LLMDocumentTransformer {
    model: Arc<dyn BaseChatModel>,
    prompt_template: String,
}

impl LLMDocumentTransformer {
    /// Create a new transformer with the given chat model and a default prompt
    /// that extracts key information.
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        Self {
            model,
            prompt_template:
                "Extract the key information from the following document:\n\n{document}".to_string(),
        }
    }

    /// Set a custom prompt template. Must contain `{document}` as a placeholder.
    pub fn with_prompt(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = template.into();
        self
    }
}

#[async_trait]
impl DocumentTransformer for LLMDocumentTransformer {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());

        for doc in documents {
            let prompt = self
                .prompt_template
                .replace("{document}", &doc.page_content);
            let messages = vec![Message::human(prompt)];
            let ai_msg = self.model.invoke_messages(&messages, None).await?;
            let content = ai_msg.base.content.text();

            let mut new_doc = Document::new(content);
            new_doc.metadata = doc.metadata.clone();
            new_doc.id = doc.id.clone();
            new_doc.doc_type = doc.doc_type.clone();
            results.push(new_doc);
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "LLMDocumentTransformer"
    }
}

// ─── DocumentTransformerPipeline ───

/// Chains multiple [`DocumentTransformer`]s in sequence.
///
/// The output of each transformer is fed as input to the next one.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::DocumentTransformerPipeline;
///
/// let pipeline = DocumentTransformerPipeline::new(vec![
///     Box::new(enricher),
///     Box::new(filter),
/// ]);
/// let result = pipeline.transform_documents(&docs).await?;
/// ```
pub struct DocumentTransformerPipeline {
    transformers: Vec<Box<dyn DocumentTransformer>>,
}

impl DocumentTransformerPipeline {
    /// Create a pipeline from an ordered list of transformers.
    pub fn new(transformers: Vec<Box<dyn DocumentTransformer>>) -> Self {
        Self { transformers }
    }
}

#[async_trait]
impl DocumentTransformer for DocumentTransformerPipeline {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut docs = documents.to_vec();
        for transformer in &self.transformers {
            docs = transformer.transform_documents(&docs).await?;
        }
        Ok(docs)
    }

    fn name(&self) -> &str {
        "DocumentTransformerPipeline"
    }
}

// ─── MetadataEnricher ───

/// Adds computed metadata fields to each document.
///
/// Supported fields (all optional, enabled via builder):
/// - `word_count` — number of whitespace-delimited words
/// - `char_count` — number of characters
/// - `language` — simple language detection based on character frequency
/// - `hash` — deterministic hash of the page content
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::MetadataEnricher;
///
/// let enricher = MetadataEnricher::new()
///     .with_word_count()
///     .with_char_count()
///     .with_hash();
/// let enriched = enricher.transform_documents(&docs).await?;
/// ```
pub struct MetadataEnricher {
    word_count: bool,
    char_count: bool,
    language: bool,
    hash: bool,
}

impl MetadataEnricher {
    /// Create a new enricher with all fields disabled.
    pub fn new() -> Self {
        Self {
            word_count: false,
            char_count: false,
            language: false,
            hash: false,
        }
    }

    /// Enable word count metadata.
    pub fn with_word_count(mut self) -> Self {
        self.word_count = true;
        self
    }

    /// Enable character count metadata.
    pub fn with_char_count(mut self) -> Self {
        self.char_count = true;
        self
    }

    /// Enable simple language detection metadata.
    pub fn with_language(mut self) -> Self {
        self.language = true;
        self
    }

    /// Enable content hash metadata.
    pub fn with_hash(mut self) -> Self {
        self.hash = true;
        self
    }
}

impl Default for MetadataEnricher {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple language detection based on Unicode script analysis.
///
/// This is a very rough heuristic: it counts Latin, CJK, Cyrillic, and Arabic
/// characters, then picks the most likely language family.
fn detect_language(text: &str) -> &'static str {
    let mut latin = 0u32;
    let mut cjk = 0u32;
    let mut cyrillic = 0u32;
    let mut arabic = 0u32;

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || matches!(ch, '\u{00C0}'..='\u{024F}') {
            latin += 1;
        } else if matches!(ch, '\u{4E00}'..='\u{9FFF}' | '\u{3040}'..='\u{30FF}') {
            cjk += 1;
        } else if matches!(ch, '\u{0400}'..='\u{04FF}') {
            cyrillic += 1;
        } else if matches!(ch, '\u{0600}'..='\u{06FF}') {
            arabic += 1;
        }
    }

    let max = latin.max(cjk).max(cyrillic).max(arabic);
    if max == 0 {
        return "unknown";
    }
    if max == cjk {
        "cjk"
    } else if max == cyrillic {
        "cyrillic"
    } else if max == arabic {
        "arabic"
    } else {
        "latin"
    }
}

/// Compute a deterministic hash of a string, returned as a hex string.
fn content_hash(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[async_trait]
impl DocumentTransformer for MetadataEnricher {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let mut results = Vec::with_capacity(documents.len());

        for doc in documents {
            let mut new_doc = doc.clone();

            if self.word_count {
                let count = doc.page_content.split_whitespace().count();
                new_doc
                    .metadata
                    .insert("word_count".to_string(), Value::from(count as u64));
            }

            if self.char_count {
                let count = doc.page_content.chars().count();
                new_doc
                    .metadata
                    .insert("char_count".to_string(), Value::from(count as u64));
            }

            if self.language {
                let lang = detect_language(&doc.page_content);
                new_doc
                    .metadata
                    .insert("language".to_string(), Value::from(lang));
            }

            if self.hash {
                let h = content_hash(&doc.page_content);
                new_doc.metadata.insert("hash".to_string(), Value::from(h));
            }

            results.push(new_doc);
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "MetadataEnricher"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::embeddings_fake::{DeterministicFakeEmbedding, FakeConstantEmbedding};
    use rustchain_core::language_models::fake::FakeListChatModel;
    use std::collections::HashMap;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_metadata(content: &str, key: &str, value: &str) -> Document {
        let mut metadata = HashMap::new();
        metadata.insert(key.to_string(), Value::from(value));
        Document::new(content).with_metadata(metadata)
    }

    // ─── EmbeddingsRedundantFilter tests ───

    #[tokio::test]
    async fn test_redundant_filter_removes_duplicates() {
        // DeterministicFakeEmbedding returns the same vector for identical texts,
        // so duplicate documents should be removed, keeping only the first.
        let embeddings = Arc::new(DeterministicFakeEmbedding::new(64));
        let filter = EmbeddingsRedundantFilter::new(embeddings);

        let docs = vec![
            make_doc("hello world"),
            make_doc("hello world"),
            make_doc("hello world"),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "hello world");
    }

    #[tokio::test]
    async fn test_redundant_filter_keeps_unique_docs() {
        // DeterministicFakeEmbedding returns different vectors for different texts.
        let embeddings = Arc::new(DeterministicFakeEmbedding::new(64));
        let filter = EmbeddingsRedundantFilter::new(embeddings);

        let docs = vec![
            make_doc("The quick brown fox jumps over the lazy dog"),
            make_doc("Machine learning is a subset of artificial intelligence"),
            make_doc("Rust is a systems programming language"),
        ];
        let result = filter.transform_documents(&docs).await.unwrap();
        // All documents should be kept since they have different content
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_redundant_filter_configurable_threshold() {
        // With threshold 1.0, only exact vector duplicates are removed.
        let embeddings = Arc::new(FakeConstantEmbedding::new(8));
        let filter = EmbeddingsRedundantFilter::new(embeddings).with_similarity_threshold(1.0);

        let docs = vec![make_doc("hello"), make_doc("world")];
        // Constant embeddings yield cosine similarity of exactly 1.0 (all zeros
        // produces 0.0 via our guard, so only truly identical non-zero vectors
        // hit 1.0). With FakeConstantEmbedding (all zeros), cosine_similarity
        // returns 0.0, which is below 1.0, so both are kept.
        let result = filter.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_redundant_filter_empty_documents() {
        let embeddings = Arc::new(DeterministicFakeEmbedding::new(8));
        let filter = EmbeddingsRedundantFilter::new(embeddings);
        let result = filter.transform_documents(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    // ─── LLMDocumentTransformer tests ───

    #[tokio::test]
    async fn test_llm_transformer_summarizes_docs() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "Summary of doc 1".into(),
            "Summary of doc 2".into(),
        ]));
        let transformer = LLMDocumentTransformer::new(model).with_prompt("Summarize: {document}");

        let docs = vec![
            make_doc("Long document content about various topics..."),
            make_doc("Another document with different content..."),
        ];
        let result = transformer.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "Summary of doc 1");
        assert_eq!(result[1].page_content, "Summary of doc 2");
    }

    #[tokio::test]
    async fn test_llm_transformer_preserves_metadata() {
        let model = Arc::new(FakeListChatModel::new(vec!["transformed".into()]));
        let transformer = LLMDocumentTransformer::new(model);

        let docs = vec![make_doc_with_metadata("original", "source", "test.txt")];
        let result = transformer.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "transformed");
        assert_eq!(
            result[0].metadata.get("source").and_then(|v| v.as_str()),
            Some("test.txt")
        );
    }

    // ─── DocumentTransformerPipeline tests ───

    #[tokio::test]
    async fn test_pipeline_chains_transformers() {
        let enricher = MetadataEnricher::new().with_word_count().with_char_count();
        // Use a second enricher that adds a hash
        let hasher = MetadataEnricher::new().with_hash();

        let pipeline = DocumentTransformerPipeline::new(vec![Box::new(enricher), Box::new(hasher)]);

        let docs = vec![make_doc("hello world")];
        let result = pipeline.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        // Should have word_count, char_count from first, and hash from second
        assert!(result[0].metadata.contains_key("word_count"));
        assert!(result[0].metadata.contains_key("char_count"));
        assert!(result[0].metadata.contains_key("hash"));
    }

    #[tokio::test]
    async fn test_pipeline_with_single_transformer() {
        let enricher = MetadataEnricher::new().with_word_count();
        let pipeline = DocumentTransformerPipeline::new(vec![Box::new(enricher)]);

        let docs = vec![make_doc("one two three")];
        let result = pipeline.transform_documents(&docs).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0]
                .metadata
                .get("word_count")
                .and_then(|v| v.as_u64()),
            Some(3)
        );
    }

    #[tokio::test]
    async fn test_pipeline_empty_documents() {
        let enricher = MetadataEnricher::new().with_word_count();
        let pipeline = DocumentTransformerPipeline::new(vec![Box::new(enricher)]);

        let result = pipeline.transform_documents(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    // ─── MetadataEnricher tests ───

    #[tokio::test]
    async fn test_enricher_adds_word_count() {
        let enricher = MetadataEnricher::new().with_word_count();
        let docs = vec![make_doc("the quick brown fox")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("word_count")
                .and_then(|v| v.as_u64()),
            Some(4)
        );
    }

    #[tokio::test]
    async fn test_enricher_adds_char_count_and_hash() {
        let enricher = MetadataEnricher::new().with_char_count().with_hash();
        let docs = vec![make_doc("hello")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("char_count")
                .and_then(|v| v.as_u64()),
            Some(5)
        );
        assert!(result[0].metadata.contains_key("hash"));
        let hash_val = result[0]
            .metadata
            .get("hash")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(hash_val.len(), 16); // 16 hex chars from u64
    }

    #[tokio::test]
    async fn test_enricher_preserves_existing_metadata() {
        let enricher = MetadataEnricher::new().with_word_count();
        let docs = vec![make_doc_with_metadata("some text", "source", "file.txt")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("source").and_then(|v| v.as_str()),
            Some("file.txt")
        );
        assert_eq!(
            result[0]
                .metadata
                .get("word_count")
                .and_then(|v| v.as_u64()),
            Some(2)
        );
    }

    #[tokio::test]
    async fn test_enricher_language_detection() {
        let enricher = MetadataEnricher::new().with_language();

        let latin_docs = vec![make_doc("Hello world, this is English text")];
        let result = enricher.transform_documents(&latin_docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("language").and_then(|v| v.as_str()),
            Some("latin")
        );
    }

    // ─── Helper function tests ───

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_detect_language_latin() {
        assert_eq!(detect_language("Hello world"), "latin");
    }

    #[test]
    fn test_detect_language_cjk() {
        assert_eq!(detect_language("\u{4F60}\u{597D}\u{4E16}\u{754C}"), "cjk");
    }

    #[test]
    fn test_detect_language_unknown() {
        assert_eq!(detect_language("123 456"), "unknown");
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = content_hash("hello");
        let h2 = content_hash("hello");
        assert_eq!(h1, h2);

        let h3 = content_hash("world");
        assert_ne!(h1, h3);
    }
}
