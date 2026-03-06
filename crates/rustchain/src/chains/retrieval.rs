use std::sync::Arc;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::retrievers::BaseRetriever;

/// Default prompt template for the RetrievalQA chain.
const DEFAULT_PROMPT_TEMPLATE: &str =
    "Use the following context to answer the question.\n\n\
     Context:\n{context}\n\n\
     Question: {query}\n\n\
     Answer:";

/// The result of a retrieval QA call that includes source documents.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The generated answer text.
    pub answer: String,
    /// The source documents that were retrieved and used as context.
    pub source_documents: Vec<Document>,
}

/// A Retrieval-Augmented Generation (RAG) chain that retrieves relevant documents
/// and uses them as context for answering questions via an LLM.
///
/// The chain performs two steps:
/// 1. Retrieve relevant documents for the query using the configured retriever.
/// 2. Format a prompt with the retrieved context and query, then call the LLM.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use rustchain::chains::retrieval::RetrievalQAChain;
///
/// # async fn example(
/// #     retriever: Arc<dyn rustchain_core::retrievers::BaseRetriever>,
/// #     llm: Arc<dyn rustchain_core::language_models::chat_model::BaseChatModel>,
/// # ) {
/// let chain = RetrievalQAChain::new(retriever, llm)
///     .with_k(3)
///     .with_prompt_template("Context: {context}\nQ: {query}\nA:");
///
/// let answer = chain.call("What is Rust?").await.unwrap();
/// # }
/// ```
pub struct RetrievalQAChain {
    /// The retriever used to fetch relevant documents.
    retriever: Arc<dyn BaseRetriever>,
    /// The chat model used to generate answers.
    llm: Arc<dyn BaseChatModel>,
    /// Optional custom prompt template. Uses `{context}` and `{query}` placeholders.
    prompt_template: Option<String>,
    /// Number of documents to retrieve. Defaults to 4.
    k: usize,
}

impl RetrievalQAChain {
    /// Create a new `RetrievalQAChain` with the given retriever and LLM.
    ///
    /// Uses the default prompt template and retrieves 4 documents.
    pub fn new(retriever: Arc<dyn BaseRetriever>, llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            retriever,
            llm,
            prompt_template: None,
            k: 4,
        }
    }

    /// Set a custom prompt template.
    ///
    /// The template must contain `{context}` and `{query}` placeholders.
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }

    /// Set the number of documents to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Concatenate document contents into a single string with separators.
    ///
    /// Each document's `page_content` is joined with double newlines.
    pub fn stuff_documents(docs: &[Document]) -> String {
        docs.iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get the effective prompt template.
    fn effective_template(&self) -> &str {
        self.prompt_template
            .as_deref()
            .unwrap_or(DEFAULT_PROMPT_TEMPLATE)
    }

    /// Format the prompt by replacing `{context}` and `{query}` placeholders.
    fn format_prompt(&self, context: &str, query: &str) -> String {
        self.effective_template()
            .replace("{context}", context)
            .replace("{query}", query)
    }

    /// Retrieve documents, format a prompt with context, call the LLM, and return the answer.
    pub async fn call(&self, query: &str) -> Result<String> {
        let result = self.call_with_sources(query).await?;
        Ok(result.answer)
    }

    /// Retrieve documents, format a prompt with context, call the LLM, and return
    /// both the answer and the source documents used.
    pub async fn call_with_sources(&self, query: &str) -> Result<RetrievalResult> {
        // Step 1: Retrieve relevant documents
        let docs = self.retriever.get_relevant_documents(query).await?;
        // Limit to k documents
        let docs: Vec<Document> = docs.into_iter().take(self.k).collect();

        // Step 2: Stuff documents into context
        let context = Self::stuff_documents(&docs);

        // Step 3: Format prompt and call LLM
        let prompt = self.format_prompt(&context, query);
        let messages = vec![Message::Human(HumanMessage::new(&prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let answer = ai_msg.base.content.text();

        Ok(RetrievalResult {
            answer,
            source_documents: docs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use async_trait::async_trait;
    use rustchain_core::language_models::fake::FakeListChatModel;
    use serde_json::json;

    /// A mock retriever that returns a fixed set of documents.
    struct MockRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_metadata(content: &str, source: &str) -> Document {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(source));
        Document::new(content).with_metadata(metadata)
    }

    fn fake_llm(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn mock_retriever(docs: Vec<Document>) -> Arc<dyn BaseRetriever> {
        Arc::new(MockRetriever { docs })
    }

    #[test]
    fn test_stuff_documents_formatting() {
        let docs = vec![
            make_doc("First document content."),
            make_doc("Second document content."),
            make_doc("Third document content."),
        ];
        let result = RetrievalQAChain::stuff_documents(&docs);
        assert_eq!(
            result,
            "First document content.\n\nSecond document content.\n\nThird document content."
        );
    }

    #[test]
    fn test_stuff_documents_empty() {
        let docs: Vec<Document> = vec![];
        let result = RetrievalQAChain::stuff_documents(&docs);
        assert_eq!(result, "");
    }

    #[test]
    fn test_stuff_documents_single() {
        let docs = vec![make_doc("Only one document.")];
        let result = RetrievalQAChain::stuff_documents(&docs);
        assert_eq!(result, "Only one document.");
    }

    #[tokio::test]
    async fn test_basic_call_with_mock() {
        let docs = vec![
            make_doc("Rust is a systems programming language."),
            make_doc("Rust focuses on safety and performance."),
        ];
        let retriever = mock_retriever(docs);
        let llm = fake_llm(vec!["Rust is a systems programming language focused on safety."]);

        let chain = RetrievalQAChain::new(retriever, llm);
        let answer = chain.call("What is Rust?").await.unwrap();
        assert_eq!(answer, "Rust is a systems programming language focused on safety.");
    }

    #[tokio::test]
    async fn test_call_with_sources_returns_documents() {
        let docs = vec![
            make_doc_with_metadata("Document A content.", "source_a.txt"),
            make_doc_with_metadata("Document B content.", "source_b.txt"),
        ];
        let retriever = mock_retriever(docs.clone());
        let llm = fake_llm(vec!["Answer based on docs."]);

        let chain = RetrievalQAChain::new(retriever, llm);
        let result = chain.call_with_sources("test query").await.unwrap();

        assert_eq!(result.answer, "Answer based on docs.");
        assert_eq!(result.source_documents.len(), 2);
        assert_eq!(result.source_documents[0].page_content, "Document A content.");
        assert_eq!(result.source_documents[1].page_content, "Document B content.");
        assert_eq!(
            result.source_documents[0].metadata.get("source").unwrap(),
            &json!("source_a.txt")
        );
    }

    #[tokio::test]
    async fn test_custom_prompt_template() {
        let docs = vec![make_doc("Some context here.")];
        let retriever = mock_retriever(docs);
        // Use ParrotFakeChatModel to verify the prompt content
        let llm: Arc<dyn BaseChatModel> =
            Arc::new(rustchain_core::language_models::fake::ParrotFakeChatModel::new());

        let chain = RetrievalQAChain::new(retriever, llm)
            .with_prompt_template("CONTEXT: {context} | QUERY: {query}");

        let answer = chain.call("my question").await.unwrap();
        assert!(answer.contains("CONTEXT: Some context here."));
        assert!(answer.contains("QUERY: my question"));
    }

    #[tokio::test]
    async fn test_empty_retrieval_results() {
        let retriever = mock_retriever(vec![]);
        let llm: Arc<dyn BaseChatModel> =
            Arc::new(rustchain_core::language_models::fake::ParrotFakeChatModel::new());

        let chain = RetrievalQAChain::new(retriever, llm);
        let result = chain.call_with_sources("unknown topic").await.unwrap();

        assert_eq!(result.source_documents.len(), 0);
        // The prompt should still be sent with empty context
        assert!(result.answer.contains("Context:\n\n"));
        assert!(result.answer.contains("Question: unknown topic"));
    }

    #[tokio::test]
    async fn test_with_k_limits_documents() {
        let docs = vec![
            make_doc("Doc 1"),
            make_doc("Doc 2"),
            make_doc("Doc 3"),
            make_doc("Doc 4"),
            make_doc("Doc 5"),
        ];
        let retriever = mock_retriever(docs);
        let llm = fake_llm(vec!["limited answer"]);

        let chain = RetrievalQAChain::new(retriever, llm).with_k(2);
        let result = chain.call_with_sources("query").await.unwrap();

        assert_eq!(result.source_documents.len(), 2);
        assert_eq!(result.source_documents[0].page_content, "Doc 1");
        assert_eq!(result.source_documents[1].page_content, "Doc 2");
    }

    #[test]
    fn test_default_prompt_template_has_placeholders() {
        let chain = RetrievalQAChain::new(
            mock_retriever(vec![]),
            fake_llm(vec!["x"]),
        );
        let template = chain.effective_template();
        assert!(template.contains("{context}"));
        assert!(template.contains("{query}"));
    }
}
