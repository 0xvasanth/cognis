use std::sync::Arc;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};

/// Default map prompt template applied to each document.
const DEFAULT_MAP_PROMPT: &str = "Summarize the following text:\n\n{text}\n\nSummary:";

/// Default reduce prompt template applied to the combined summaries.
const DEFAULT_REDUCE_PROMPT: &str =
    "Combine the following summaries into a coherent final answer:\n\n{summaries}\n\nFinal answer:";

/// A map-reduce chain for processing multiple documents.
///
/// The chain operates in two phases:
/// 1. **Map phase**: Each document is individually processed using the `map_prompt`.
/// 2. **Reduce phase**: All map results are combined and processed using the `reduce_prompt`
///    to produce a single final answer.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use rustchain::chains::map_reduce::MapReduceChain;
/// use rustchain_core::documents::Document;
///
/// # async fn example(llm: Arc<dyn rustchain_core::language_models::chat_model::BaseChatModel>) {
/// let chain = MapReduceChain::new(llm)
///     .with_map_prompt("Extract key points from:\n\n{text}\n\nKey points:")
///     .with_reduce_prompt("Merge these key points:\n\n{summaries}\n\nMerged:");
///
/// let docs = vec![
///     Document::new("First document content."),
///     Document::new("Second document content."),
/// ];
/// let result = chain.call(&docs).await.unwrap();
/// # }
/// ```
pub struct MapReduceChain {
    /// The chat model used for both map and reduce phases.
    llm: Arc<dyn BaseChatModel>,
    /// Prompt template for the map phase. Must contain `{text}`.
    map_prompt: String,
    /// Prompt template for the reduce phase. Must contain `{summaries}`.
    reduce_prompt: String,
    /// Maximum number of concurrent map operations (currently reserved for future use).
    max_concurrency: usize,
}

impl MapReduceChain {
    /// Create a new `MapReduceChain` with default prompts.
    pub fn new(llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            llm,
            map_prompt: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt: DEFAULT_REDUCE_PROMPT.to_string(),
            max_concurrency: 5,
        }
    }

    /// Set a custom map prompt template. Must contain `{text}`.
    pub fn with_map_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.map_prompt = prompt.into();
        self
    }

    /// Set a custom reduce prompt template. Must contain `{summaries}`.
    pub fn with_reduce_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.reduce_prompt = prompt.into();
        self
    }

    /// Set the maximum concurrency for the map phase.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Format the map prompt by replacing `{text}` with the document content.
    fn format_map_prompt(&self, text: &str) -> String {
        self.map_prompt.replace("{text}", text)
    }

    /// Format the reduce prompt by replacing `{summaries}` with the combined summaries.
    fn format_reduce_prompt(&self, summaries: &str) -> String {
        self.reduce_prompt.replace("{summaries}", summaries)
    }

    /// Run the map-reduce chain over the given documents.
    ///
    /// 1. **Map**: For each document, formats the map prompt with the document content
    ///    and calls the LLM to produce a per-document result.
    /// 2. **Reduce**: Combines all map results into a single string, formats the reduce
    ///    prompt, and calls the LLM for the final answer.
    pub async fn call(&self, documents: &[Document]) -> Result<String> {
        // Map phase: process each document individually
        let mut map_results = Vec::with_capacity(documents.len());
        for doc in documents {
            let prompt = self.format_map_prompt(&doc.page_content);
            let messages = vec![Message::Human(HumanMessage::new(&prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            map_results.push(ai_msg.base.content.text());
        }

        // Reduce phase: combine all map results and produce final answer
        let combined = map_results.join("\n\n");
        let reduce_prompt = self.format_reduce_prompt(&combined);
        let messages = vec![Message::Human(HumanMessage::new(&reduce_prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;

        Ok(ai_msg.base.content.text())
    }

    /// Run only the map phase and return per-document results.
    pub async fn map(&self, documents: &[Document]) -> Result<Vec<String>> {
        let mut results = Vec::with_capacity(documents.len());
        for doc in documents {
            let prompt = self.format_map_prompt(&doc.page_content);
            let messages = vec![Message::Human(HumanMessage::new(&prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            results.push(ai_msg.base.content.text());
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::{FakeListChatModel, ParrotFakeChatModel};

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    #[tokio::test]
    async fn test_map_phase_produces_per_doc_summaries() {
        // 2 docs -> 2 map calls, each returns a different response
        let llm = fake_model(vec!["Summary of doc 1", "Summary of doc 2", "Final"]);
        let chain = MapReduceChain::new(llm);

        let docs = vec![
            make_doc("Document one content"),
            make_doc("Document two content"),
        ];
        let map_results = chain.map(&docs).await.unwrap();

        assert_eq!(map_results.len(), 2);
        assert_eq!(map_results[0], "Summary of doc 1");
        assert_eq!(map_results[1], "Summary of doc 2");
    }

    #[tokio::test]
    async fn test_reduce_phase_combines_summaries() {
        // Use ParrotFakeChatModel to verify the reduce prompt includes combined summaries
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = MapReduceChain::new(llm);

        let reduce_prompt = chain.format_reduce_prompt("Summary A\n\nSummary B");
        assert!(reduce_prompt.contains("Summary A"));
        assert!(reduce_prompt.contains("Summary B"));
        assert!(reduce_prompt.contains("Combine the following summaries"));
    }

    #[tokio::test]
    async fn test_end_to_end_call() {
        // 3 responses: 2 for map phase, 1 for reduce phase
        let llm = fake_model(vec!["mapped-1", "mapped-2", "final-answer"]);
        let chain = MapReduceChain::new(llm);

        let docs = vec![make_doc("First doc"), make_doc("Second doc")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "final-answer");
    }

    #[tokio::test]
    async fn test_custom_prompts() {
        // Use ParrotFakeChatModel so we can verify the prompts are formatted correctly
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = MapReduceChain::new(llm)
            .with_map_prompt("EXTRACT: {text} END")
            .with_reduce_prompt("MERGE: {summaries} DONE");

        let docs = vec![make_doc("hello world")];
        // Map phase: parrot echoes "EXTRACT: hello world END"
        // Reduce phase: parrot echoes "MERGE: EXTRACT: hello world END DONE"
        let result = chain.call(&docs).await.unwrap();

        assert!(result.contains("MERGE:"));
        assert!(result.contains("DONE"));
    }

    #[tokio::test]
    async fn test_single_document() {
        let llm = fake_model(vec!["single-map", "single-reduce"]);
        let chain = MapReduceChain::new(llm);

        let docs = vec![make_doc("Only document")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "single-reduce");
    }

    #[tokio::test]
    async fn test_empty_documents() {
        let llm = fake_model(vec!["reduce-of-nothing"]);
        let chain = MapReduceChain::new(llm);

        let docs: Vec<Document> = vec![];
        let result = chain.call(&docs).await.unwrap();

        // With no docs, the map phase produces nothing, reduce is called with empty summaries
        assert_eq!(result, "reduce-of-nothing");
    }

    #[tokio::test]
    async fn test_max_concurrency_builder() {
        let llm = fake_model(vec!["x"]);
        let chain = MapReduceChain::new(llm).with_max_concurrency(10);
        assert_eq!(chain.max_concurrency, 10);
    }
}
