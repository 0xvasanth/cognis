//! QA (Question Answering) chain with retrieval integration and citation support.
//!
//! Provides configurable QA chains that can use Stuff, MapReduce, or Refine strategies
//! to answer questions from a set of documents. Includes a `RetrievalQAChain` that
//! combines a retriever with the QA chain for end-to-end RAG workflows.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

use super::documents::DocumentFormatter;

// ---------------------------------------------------------------------------
// QAChainType
// ---------------------------------------------------------------------------

/// The strategy used by a QA chain to combine documents and produce an answer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QAChainType {
    /// Concatenate all documents into a single context block and answer in one pass.
    #[default]
    Stuff,
    /// Map each document independently, then reduce the intermediate answers.
    MapReduce,
    /// Iteratively refine the answer by processing documents one at a time.
    Refine,
}

impl std::fmt::Display for QAChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stuff => write!(f, "stuff"),
            Self::MapReduce => write!(f, "map_reduce"),
            Self::Refine => write!(f, "refine"),
        }
    }
}

// ---------------------------------------------------------------------------
// QAConfig
// ---------------------------------------------------------------------------

/// Configuration for a QA chain, built using the builder pattern.
#[derive(Debug, Clone)]
pub struct QAConfig {
    /// The chain strategy to use.
    pub chain_type: QAChainType,
    /// Whether to include source documents in the result.
    pub return_source_documents: bool,
    /// Maximum number of source documents to use.
    pub max_source_docs: usize,
    /// Whether to emit verbose logging.
    pub verbose: bool,
}

impl Default for QAConfig {
    fn default() -> Self {
        Self {
            chain_type: QAChainType::Stuff,
            return_source_documents: true,
            max_source_docs: 4,
            verbose: false,
        }
    }
}

impl QAConfig {
    /// Create a new builder for `QAConfig`.
    pub fn builder() -> QAConfigBuilder {
        QAConfigBuilder::default()
    }
}

/// Builder for [`QAConfig`].
#[derive(Debug, Clone)]
pub struct QAConfigBuilder {
    chain_type: QAChainType,
    return_source_documents: bool,
    max_source_docs: usize,
    verbose: bool,
}

impl Default for QAConfigBuilder {
    fn default() -> Self {
        let config = QAConfig::default();
        Self {
            chain_type: config.chain_type,
            return_source_documents: config.return_source_documents,
            max_source_docs: config.max_source_docs,
            verbose: config.verbose,
        }
    }
}

impl QAConfigBuilder {
    /// Set the chain type strategy.
    pub fn chain_type(mut self, chain_type: QAChainType) -> Self {
        self.chain_type = chain_type;
        self
    }

    /// Set whether to return source documents.
    pub fn return_source_documents(mut self, return_source_documents: bool) -> Self {
        self.return_source_documents = return_source_documents;
        self
    }

    /// Set the maximum number of source documents.
    pub fn max_source_docs(mut self, max_source_docs: usize) -> Self {
        self.max_source_docs = max_source_docs;
        self
    }

    /// Set verbose mode.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the `QAConfig`.
    pub fn build(self) -> QAConfig {
        QAConfig {
            chain_type: self.chain_type,
            return_source_documents: self.return_source_documents,
            max_source_docs: self.max_source_docs,
            verbose: self.verbose,
        }
    }
}

// ---------------------------------------------------------------------------
// QAResult
// ---------------------------------------------------------------------------

/// The result of a QA chain invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAResult {
    /// The generated answer text.
    pub answer: String,
    /// The source documents used to generate the answer.
    pub source_documents: Vec<Document>,
    /// Optional confidence score in the range [0.0, 1.0].
    pub confidence: Option<f64>,
    /// The chain type used to produce this result.
    pub chain_type: QAChainType,
}

// ---------------------------------------------------------------------------
// Citation / CitedAnswer
// ---------------------------------------------------------------------------

/// A single inline citation referencing a source document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    /// The source identifier (e.g., file name, URL).
    pub source: String,
    /// A snippet from the document's content that supports the citation.
    pub page_content_snippet: String,
    /// The index of the cited document in the source documents list.
    pub doc_index: usize,
}

/// An answer with inline citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitedAnswer {
    /// The answer text (may contain inline citation markers like [1], [2]).
    pub answer: String,
    /// The extracted citations.
    pub citations: Vec<Citation>,
}

impl CitedAnswer {
    /// Create a `CitedAnswer` by extracting citations from an answer that uses
    /// `[N]` markers referencing the source documents list.
    ///
    /// For each `[N]` marker found in `answer`, a citation is generated from
    /// the corresponding document (1-indexed).
    pub fn from_answer_and_docs(answer: &str, documents: &[Document]) -> Self {
        let mut citations = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Find all [N] patterns in the answer
        let mut remaining = answer;
        while let Some(start) = remaining.find('[') {
            let after = &remaining[start + 1..];
            if let Some(end) = after.find(']') {
                let inside = &after[..end];
                if let Ok(n) = inside.parse::<usize>() {
                    if n >= 1 && n <= documents.len() && seen.insert(n) {
                        let doc = &documents[n - 1];
                        let source = doc
                            .metadata
                            .get("source")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let snippet_len = doc.page_content.len().min(100);
                        let page_content_snippet = doc.page_content[..snippet_len].to_string();
                        citations.push(Citation {
                            source,
                            page_content_snippet,
                            doc_index: n - 1,
                        });
                    }
                }
                remaining = &after[end + 1..];
            } else {
                break;
            }
        }

        Self {
            answer: answer.to_string(),
            citations,
        }
    }
}

// ---------------------------------------------------------------------------
// QAChain
// ---------------------------------------------------------------------------

/// Default template for formatting each document in the context.
const DEFAULT_DOCUMENT_PROMPT: &str = "Document {doc_index}:\n{page_content}";

/// Default template for the QA question prompt.
const DEFAULT_QA_PROMPT: &str = "Use the following documents to answer the question. \
If you cannot find the answer in the documents, say so.\n\n\
{context}\n\n\
Question: {question}\n\n\
Answer:";

/// A QA chain that formats documents and a question into an answer.
///
/// This chain does not call an LLM — it formats the prompt that would be sent
/// to one, making it composable and testable. For end-to-end retrieval + LLM
/// answering, see [`RetrievalQAChain`].
pub struct QAChain {
    /// Configuration for the chain.
    pub config: QAConfig,
    /// Template for formatting each individual document.
    /// Supports `{page_content}`, `{metadata.KEY}`, and `{doc_index}` placeholders.
    pub document_prompt: String,
    /// Template for the final QA prompt.
    /// Supports `{context}` and `{question}` placeholders.
    pub qa_prompt: String,
}

impl QAChain {
    /// Create a new `QAChain` with the given config and default prompt templates.
    pub fn new(config: QAConfig) -> Self {
        Self {
            config,
            document_prompt: DEFAULT_DOCUMENT_PROMPT.to_string(),
            qa_prompt: DEFAULT_QA_PROMPT.to_string(),
        }
    }

    /// Set a custom document prompt template.
    pub fn with_document_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.document_prompt = prompt.into();
        self
    }

    /// Set a custom QA prompt template.
    pub fn with_qa_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.qa_prompt = prompt.into();
        self
    }

    /// Format a single document using the document prompt template.
    fn format_document(&self, doc: &Document, index: usize) -> String {
        let formatted = DocumentFormatter::format_document(doc, &self.document_prompt);
        formatted.replace("{doc_index}", &(index + 1).to_string())
    }

    /// Format multiple documents into a single context string.
    fn format_documents(&self, documents: &[Document]) -> String {
        let max_docs = self.config.max_source_docs.min(documents.len());
        let docs = &documents[..max_docs];
        docs.iter()
            .enumerate()
            .map(|(i, doc)| self.format_document(doc, i))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Format the full QA prompt with context and question.
    fn format_qa_prompt(&self, context: &str, question: &str) -> String {
        self.qa_prompt
            .replace("{context}", context)
            .replace("{question}", question)
    }

    /// Format documents and a question into a QA result.
    ///
    /// The `answer` field contains the formatted prompt (suitable for sending to an LLM).
    /// Source documents are included based on the config.
    pub fn answer(&self, question: &str, documents: &[Document]) -> Result<QAResult> {
        if question.is_empty() {
            return Err(RustChainError::Other(
                "Question must not be empty".to_string(),
            ));
        }

        let max_docs = self.config.max_source_docs.min(documents.len());
        let used_docs: Vec<Document> = documents[..max_docs].to_vec();

        let context = self.format_documents(&used_docs);
        let answer = self.format_qa_prompt(&context, question);

        let source_documents = if self.config.return_source_documents {
            used_docs
        } else {
            Vec::new()
        };

        Ok(QAResult {
            answer,
            source_documents,
            confidence: None,
            chain_type: self.config.chain_type,
        })
    }

    /// Answer a question given pre-formatted context text.
    ///
    /// Returns the formatted prompt string.
    pub fn answer_with_context(&self, question: &str, context: &str) -> Result<String> {
        if question.is_empty() {
            return Err(RustChainError::Other(
                "Question must not be empty".to_string(),
            ));
        }
        Ok(self.format_qa_prompt(context, question))
    }

    /// Extract citations from a generated answer that uses `[N]` markers.
    pub fn extract_citations(&self, answer: &str, documents: &[Document]) -> CitedAnswer {
        CitedAnswer::from_answer_and_docs(answer, documents)
    }
}

#[async_trait]
impl Runnable for QAChain {
    fn name(&self) -> &str {
        "QAChain"
    }

    /// Invoke the QA chain.
    ///
    /// Expected input: `{"question": "...", "documents": [...]}`
    /// Output: serialized `QAResult`
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let question = input
            .get("question")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "object with 'question' string field".into(),
                got: format!("{}", input),
            })?;

        let documents: Vec<Document> = if let Some(docs_val) = input.get("documents") {
            serde_json::from_value(docs_val.clone())?
        } else {
            Vec::new()
        };

        let result = self.answer(question, &documents)?;
        serde_json::to_value(&result).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// RetrievalQAChain (combines retriever + QAChain)
// ---------------------------------------------------------------------------

/// A chain that retrieves relevant documents and then applies a QA chain to answer
/// the question. This is the main entry point for retrieval-augmented QA.
pub struct RetrievalQAChain {
    /// The retriever used to fetch relevant documents.
    pub retriever: Arc<dyn BaseRetriever>,
    /// The QA chain used to format and answer.
    pub qa_chain: QAChain,
}

impl RetrievalQAChain {
    /// Create a new `RetrievalQAChain`.
    pub fn new(retriever: Arc<dyn BaseRetriever>, qa_chain: QAChain) -> Self {
        Self {
            retriever,
            qa_chain,
        }
    }

    /// Retrieve documents for the question and produce a QA result.
    pub async fn run(&self, question: &str) -> Result<QAResult> {
        let docs = self.retriever.get_relevant_documents(question).await?;
        self.qa_chain.answer(question, &docs)
    }
}

#[async_trait]
impl Runnable for RetrievalQAChain {
    fn name(&self) -> &str {
        "RetrievalQAChain"
    }

    /// Invoke the retrieval QA chain.
    ///
    /// Expected input: `Value::String(question)` or `{"question": "..."}`
    /// Output: serialized `QAResult`
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let question = if let Some(s) = input.as_str() {
            s.to_string()
        } else if let Some(s) = input.get("question").and_then(|v| v.as_str()) {
            s.to_string()
        } else {
            return Err(RustChainError::TypeMismatch {
                expected: "String or object with 'question' field".into(),
                got: format!("{}", input),
            });
        };

        let result = self.run(&question).await?;
        serde_json::to_value(&result).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Create a QA chain with the given configuration and default prompt templates.
pub fn create_qa_chain(config: QAConfig) -> QAChain {
    QAChain::new(config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    // -- Helpers --

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_source(content: &str, source: &str) -> Document {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(source));
        Document::new(content).with_metadata(metadata)
    }

    fn default_chain() -> QAChain {
        create_qa_chain(QAConfig::default())
    }

    struct MockRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    fn mock_retriever(docs: Vec<Document>) -> Arc<dyn BaseRetriever> {
        Arc::new(MockRetriever { docs })
    }

    // -- QAChainType tests --

    #[test]
    fn test_chain_type_default_is_stuff() {
        assert_eq!(QAChainType::default(), QAChainType::Stuff);
    }

    #[test]
    fn test_chain_type_display() {
        assert_eq!(QAChainType::Stuff.to_string(), "stuff");
        assert_eq!(QAChainType::MapReduce.to_string(), "map_reduce");
        assert_eq!(QAChainType::Refine.to_string(), "refine");
    }

    #[test]
    fn test_chain_type_serialize() {
        let json = serde_json::to_string(&QAChainType::Stuff).unwrap();
        assert_eq!(json, "\"Stuff\"");
        let deserialized: QAChainType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, QAChainType::Stuff);
    }

    // -- QAConfig tests --

    #[test]
    fn test_config_defaults() {
        let config = QAConfig::default();
        assert_eq!(config.chain_type, QAChainType::Stuff);
        assert!(config.return_source_documents);
        assert_eq!(config.max_source_docs, 4);
        assert!(!config.verbose);
    }

    #[test]
    fn test_config_builder() {
        let config = QAConfig::builder()
            .chain_type(QAChainType::Refine)
            .return_source_documents(false)
            .max_source_docs(10)
            .verbose(true)
            .build();

        assert_eq!(config.chain_type, QAChainType::Refine);
        assert!(!config.return_source_documents);
        assert_eq!(config.max_source_docs, 10);
        assert!(config.verbose);
    }

    #[test]
    fn test_config_builder_defaults() {
        let config = QAConfig::builder().build();
        assert_eq!(config.chain_type, QAChainType::Stuff);
        assert!(config.return_source_documents);
        assert_eq!(config.max_source_docs, 4);
    }

    // -- QAChain tests --

    #[test]
    fn test_answer_basic() {
        let chain = default_chain();
        let docs = vec![make_doc("Rust is a systems language.")];
        let result = chain.answer("What is Rust?", &docs).unwrap();

        assert!(result.answer.contains("What is Rust?"));
        assert!(result.answer.contains("Rust is a systems language."));
        assert_eq!(result.chain_type, QAChainType::Stuff);
        assert_eq!(result.source_documents.len(), 1);
    }

    #[test]
    fn test_answer_multiple_docs() {
        let chain = default_chain();
        let docs = vec![
            make_doc("Document one content."),
            make_doc("Document two content."),
            make_doc("Document three content."),
        ];
        let result = chain.answer("Tell me about docs", &docs).unwrap();

        assert!(result.answer.contains("Document one content."));
        assert!(result.answer.contains("Document two content."));
        assert!(result.answer.contains("Document three content."));
        assert_eq!(result.source_documents.len(), 3);
    }

    #[test]
    fn test_answer_max_source_docs_limit() {
        let config = QAConfig::builder().max_source_docs(2).build();
        let chain = create_qa_chain(config);
        let docs = vec![
            make_doc("Doc A"),
            make_doc("Doc B"),
            make_doc("Doc C"),
            make_doc("Doc D"),
        ];
        let result = chain.answer("question?", &docs).unwrap();

        assert_eq!(result.source_documents.len(), 2);
        assert!(result.answer.contains("Doc A"));
        assert!(result.answer.contains("Doc B"));
        assert!(!result.answer.contains("Doc C"));
    }

    #[test]
    fn test_answer_no_source_documents() {
        let config = QAConfig::builder().return_source_documents(false).build();
        let chain = create_qa_chain(config);
        let docs = vec![make_doc("Some content")];
        let result = chain.answer("question?", &docs).unwrap();

        assert!(result.source_documents.is_empty());
        assert!(result.answer.contains("Some content"));
    }

    #[test]
    fn test_answer_empty_docs() {
        let chain = default_chain();
        let result = chain.answer("question?", &[]).unwrap();

        assert!(result.source_documents.is_empty());
        assert!(result.answer.contains("question?"));
    }

    #[test]
    fn test_answer_empty_question_error() {
        let chain = default_chain();
        let result = chain.answer("", &[make_doc("content")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_answer_with_context() {
        let chain = default_chain();
        let result = chain
            .answer_with_context("What is Rust?", "Rust is a programming language.")
            .unwrap();

        assert!(result.contains("What is Rust?"));
        assert!(result.contains("Rust is a programming language."));
    }

    #[test]
    fn test_answer_with_context_empty_question_error() {
        let chain = default_chain();
        let result = chain.answer_with_context("", "some context");
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_document_prompt() {
        let chain = default_chain().with_document_prompt("Source [{doc_index}]: {page_content}");
        let docs = vec![make_doc("Hello world")];
        let result = chain.answer("test?", &docs).unwrap();

        assert!(result.answer.contains("Source [1]: Hello world"));
    }

    #[test]
    fn test_custom_qa_prompt() {
        let chain = default_chain().with_qa_prompt("Context: {context}\nQ: {question}\nA:");
        let docs = vec![make_doc("content here")];
        let result = chain.answer("what?", &docs).unwrap();

        assert!(result.answer.starts_with("Context:"));
        assert!(result.answer.contains("Q: what?"));
        assert!(result.answer.ends_with("A:"));
    }

    #[test]
    fn test_document_prompt_with_metadata() {
        let chain = default_chain().with_document_prompt("{page_content} (from {metadata.source})");
        let docs = vec![make_doc_with_source("content", "wiki.txt")];
        let result = chain.answer("q?", &docs).unwrap();

        assert!(result.answer.contains("content (from wiki.txt)"));
    }

    #[test]
    fn test_chain_type_in_result() {
        let config = QAConfig::builder()
            .chain_type(QAChainType::MapReduce)
            .build();
        let chain = create_qa_chain(config);
        let result = chain.answer("q?", &[make_doc("d")]).unwrap();
        assert_eq!(result.chain_type, QAChainType::MapReduce);
    }

    // -- Citation tests --

    #[test]
    fn test_citation_from_answer() {
        let docs = vec![
            make_doc_with_source("First doc content here", "source1.txt"),
            make_doc_with_source("Second doc content here", "source2.txt"),
        ];
        let cited = CitedAnswer::from_answer_and_docs(
            "Based on [1] and also [2], the answer is yes.",
            &docs,
        );

        assert_eq!(cited.citations.len(), 2);
        assert_eq!(cited.citations[0].source, "source1.txt");
        assert_eq!(cited.citations[0].doc_index, 0);
        assert_eq!(cited.citations[1].source, "source2.txt");
        assert_eq!(cited.citations[1].doc_index, 1);
    }

    #[test]
    fn test_citation_no_markers() {
        let docs = vec![make_doc("content")];
        let cited = CitedAnswer::from_answer_and_docs("No citations here.", &docs);
        assert!(cited.citations.is_empty());
    }

    #[test]
    fn test_citation_out_of_range() {
        let docs = vec![make_doc("content")];
        let cited = CitedAnswer::from_answer_and_docs("Reference [5] is invalid.", &docs);
        assert!(cited.citations.is_empty());
    }

    #[test]
    fn test_citation_deduplication() {
        let docs = vec![make_doc_with_source("content", "src.txt")];
        let cited = CitedAnswer::from_answer_and_docs("See [1] and again [1].", &docs);
        assert_eq!(cited.citations.len(), 1);
    }

    #[test]
    fn test_citation_snippet_truncation() {
        let long_content = "x".repeat(200);
        let docs = vec![make_doc(&long_content)];
        let cited = CitedAnswer::from_answer_and_docs("See [1].", &docs);
        assert_eq!(cited.citations[0].page_content_snippet.len(), 100);
    }

    #[test]
    fn test_extract_citations_via_chain() {
        let chain = default_chain();
        let docs = vec![make_doc_with_source("content", "file.txt")];
        let cited = chain.extract_citations("Answer [1] here.", &docs);
        assert_eq!(cited.citations.len(), 1);
        assert_eq!(cited.citations[0].source, "file.txt");
    }

    // -- Runnable (QAChain) tests --

    #[tokio::test]
    async fn test_qa_chain_runnable_invoke() {
        let chain = default_chain();
        let input = json!({
            "question": "What is Rust?",
            "documents": [{"page_content": "Rust is a language."}]
        });
        let result = chain.invoke(input, None).await.unwrap();
        let qa_result: QAResult = serde_json::from_value(result).unwrap();
        assert!(qa_result.answer.contains("What is Rust?"));
        assert_eq!(qa_result.source_documents.len(), 1);
    }

    #[tokio::test]
    async fn test_qa_chain_runnable_no_documents() {
        let chain = default_chain();
        let input = json!({ "question": "What is Rust?" });
        let result = chain.invoke(input, None).await.unwrap();
        let qa_result: QAResult = serde_json::from_value(result).unwrap();
        assert!(qa_result.source_documents.is_empty());
    }

    #[tokio::test]
    async fn test_qa_chain_runnable_missing_question() {
        let chain = default_chain();
        let input = json!({ "documents": [] });
        let result = chain.invoke(input, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_qa_chain_runnable_name() {
        let chain = default_chain();
        assert_eq!(chain.name(), "QAChain");
    }

    // -- RetrievalQAChain tests --

    #[tokio::test]
    async fn test_retrieval_qa_run() {
        let docs = vec![make_doc("Rust is safe."), make_doc("Rust is fast.")];
        let retriever = mock_retriever(docs);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let result = chain.run("What is Rust?").await.unwrap();
        assert!(result.answer.contains("Rust is safe."));
        assert!(result.answer.contains("Rust is fast."));
        assert_eq!(result.source_documents.len(), 2);
    }

    #[tokio::test]
    async fn test_retrieval_qa_run_with_max_docs() {
        let docs = vec![
            make_doc("A"),
            make_doc("B"),
            make_doc("C"),
            make_doc("D"),
            make_doc("E"),
        ];
        let retriever = mock_retriever(docs);
        let config = QAConfig::builder().max_source_docs(2).build();
        let qa_chain = create_qa_chain(config);
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let result = chain.run("q?").await.unwrap();
        assert_eq!(result.source_documents.len(), 2);
    }

    #[tokio::test]
    async fn test_retrieval_qa_runnable_string_input() {
        let docs = vec![make_doc("content")];
        let retriever = mock_retriever(docs);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let result = chain
            .invoke(Value::String("What is Rust?".into()), None)
            .await
            .unwrap();
        let qa_result: QAResult = serde_json::from_value(result).unwrap();
        assert!(qa_result.answer.contains("What is Rust?"));
    }

    #[tokio::test]
    async fn test_retrieval_qa_runnable_object_input() {
        let docs = vec![make_doc("content")];
        let retriever = mock_retriever(docs);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let input = json!({ "question": "What is Rust?" });
        let result = chain.invoke(input, None).await.unwrap();
        let qa_result: QAResult = serde_json::from_value(result).unwrap();
        assert!(qa_result.answer.contains("What is Rust?"));
    }

    #[tokio::test]
    async fn test_retrieval_qa_runnable_invalid_input() {
        let retriever = mock_retriever(vec![]);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let result = chain.invoke(json!(42), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retrieval_qa_runnable_name() {
        let retriever = mock_retriever(vec![]);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);
        assert_eq!(chain.name(), "RetrievalQAChain");
    }

    #[tokio::test]
    async fn test_retrieval_qa_empty_results() {
        let retriever = mock_retriever(vec![]);
        let qa_chain = default_chain();
        let chain = RetrievalQAChain::new(retriever, qa_chain);

        let result = chain.run("unknown?").await.unwrap();
        assert!(result.source_documents.is_empty());
    }

    // -- QAResult serialization --

    #[test]
    fn test_qa_result_serialization() {
        let result = QAResult {
            answer: "The answer.".to_string(),
            source_documents: vec![make_doc("doc content")],
            confidence: Some(0.95),
            chain_type: QAChainType::Stuff,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["answer"], "The answer.");
        assert_eq!(json["confidence"], 0.95);
        assert_eq!(json["chain_type"], "Stuff");
    }

    #[test]
    fn test_qa_result_deserialization() {
        let json = json!({
            "answer": "yes",
            "source_documents": [],
            "confidence": null,
            "chain_type": "Refine"
        });
        let result: QAResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.answer, "yes");
        assert!(result.confidence.is_none());
        assert_eq!(result.chain_type, QAChainType::Refine);
    }

    // -- Factory function --

    #[test]
    fn test_create_qa_chain_factory() {
        let config = QAConfig::builder()
            .chain_type(QAChainType::MapReduce)
            .max_source_docs(8)
            .build();
        let chain = create_qa_chain(config);
        assert_eq!(chain.config.chain_type, QAChainType::MapReduce);
        assert_eq!(chain.config.max_source_docs, 8);
    }
}
