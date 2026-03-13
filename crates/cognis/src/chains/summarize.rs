use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis_core::documents::Document;
use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// Default prompt templates
// ---------------------------------------------------------------------------

/// Default prompt for the stuff summarization chain.
const DEFAULT_STUFF_PROMPT: &str =
    "Write a concise summary of the following:\n\n\"{text}\"\n\nCONCISE SUMMARY:";

/// Default map prompt for the map-reduce summarization chain.
const DEFAULT_MAP_SUMMARIZE_PROMPT: &str =
    "Write a concise summary of the following:\n\n\"{text}\"\n\nCONCISE SUMMARY:";

/// Default reduce prompt for the map-reduce summarization chain.
const DEFAULT_REDUCE_SUMMARIZE_PROMPT: &str =
    "The following is a set of summaries:\n\n{summaries}\n\n\
     Take these and distill them into a final, consolidated summary of the main themes.\n\n\
     FINAL SUMMARY:";

/// Default initial prompt for the refine summarization chain.
const DEFAULT_INITIAL_SUMMARIZE_PROMPT: &str =
    "Write a concise summary of the following:\n\n\"{text}\"\n\nCONCISE SUMMARY:";

/// Default refine prompt for the refine summarization chain.
const DEFAULT_REFINE_SUMMARIZE_PROMPT: &str = "Your job is to produce a final summary.\n\
     We have provided an existing summary up to a certain point:\n\n\
     {existing_summary}\n\n\
     We have the opportunity to refine the existing summary (only if needed) \
     with some more context below.\n\n\
     \"{text}\"\n\n\
     Given the new context, refine the original summary. \
     If the context is not useful, return the original summary.\n\n\
     REFINED SUMMARY:";

// ===========================================================================
// StuffSummarizationChain
// ===========================================================================

/// Summarization chain that "stuffs" all documents into a single prompt.
///
/// This is the simplest summarization strategy: concatenate all document contents
/// and send them to the LLM in a single call. Best for small inputs that fit
/// within the model's context window.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use cognis::chains::summarize::StuffSummarizationChain;
/// use cognis_core::documents::Document;
///
/// # async fn example(llm: Arc<dyn cognis_core::language_models::chat_model::BaseChatModel>) {
/// let chain = StuffSummarizationChain::new(llm);
/// let docs = vec![
///     Document::new("First document."),
///     Document::new("Second document."),
/// ];
/// let summary = chain.call(&docs).await.unwrap();
/// # }
/// ```
pub struct StuffSummarizationChain {
    /// The chat model used for summarization.
    llm: Arc<dyn BaseChatModel>,
    /// Prompt template. Must contain `{text}`.
    prompt: String,
    /// Separator used to join document contents.
    document_separator: String,
}

impl StuffSummarizationChain {
    /// Create a new `StuffSummarizationChain` with the default prompt.
    pub fn new(llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            llm,
            prompt: DEFAULT_STUFF_PROMPT.to_string(),
            document_separator: "\n\n".to_string(),
        }
    }

    /// Set a custom prompt template. Must contain `{text}`.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Set a custom document separator. Default: `"\n\n"`.
    pub fn with_document_separator(mut self, sep: impl Into<String>) -> Self {
        self.document_separator = sep.into();
        self
    }

    /// Format the prompt by replacing `{text}` with the combined document content.
    fn format_prompt(&self, text: &str) -> String {
        self.prompt.replace("{text}", text)
    }

    /// Run the stuff summarization chain over the given documents.
    ///
    /// All document contents are joined with the document separator and sent
    /// to the LLM in a single call.
    pub async fn call(&self, documents: &[Document]) -> Result<String> {
        let combined: String = documents
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>()
            .join(&self.document_separator);

        let prompt = self.format_prompt(&combined);
        let messages = vec![Message::Human(HumanMessage::new(&prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;

        Ok(ai_msg.base.content.text())
    }
}

#[async_trait]
impl Runnable for StuffSummarizationChain {
    fn name(&self) -> &str {
        "StuffSummarizationChain"
    }

    /// Invoke with JSON input `{ "documents": [...] }`.
    ///
    /// Each element in `documents` can be either a string or a Document object
    /// with a `page_content` field. Returns `{ "summary": "..." }`.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let docs = parse_documents_from_input(&input)?;
        let summary = self.call(&docs).await?;
        Ok(json!({ "summary": summary }))
    }
}

// ===========================================================================
// MapReduceSummarizationChain
// ===========================================================================

/// Summarization chain using the map-reduce strategy.
///
/// 1. **Map phase**: Each document is independently summarized.
/// 2. **Reduce phase**: All individual summaries are combined and distilled
///    into a final summary.
///
/// Optionally supports recursive reduction: if the combined summaries still
/// exceed a configurable token estimate, they are split and reduced again.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use cognis::chains::summarize::MapReduceSummarizationChain;
/// use cognis_core::documents::Document;
///
/// # async fn example(llm: Arc<dyn cognis_core::language_models::chat_model::BaseChatModel>) {
/// let chain = MapReduceSummarizationChain::new(llm);
/// let docs = vec![
///     Document::new("Document one content."),
///     Document::new("Document two content."),
/// ];
/// let summary = chain.call(&docs).await.unwrap();
/// # }
/// ```
pub struct MapReduceSummarizationChain {
    /// The chat model used for both map and reduce phases.
    llm: Arc<dyn BaseChatModel>,
    /// Prompt template for the map phase. Must contain `{text}`.
    map_prompt: String,
    /// Prompt template for the reduce phase. Must contain `{summaries}`.
    reduce_prompt: String,
    /// Maximum character length for the combined summaries before triggering
    /// recursive reduction. `0` means no recursive reduction.
    max_reduce_length: usize,
}

impl MapReduceSummarizationChain {
    /// Create a new `MapReduceSummarizationChain` with default prompts.
    pub fn new(llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            llm,
            map_prompt: DEFAULT_MAP_SUMMARIZE_PROMPT.to_string(),
            reduce_prompt: DEFAULT_REDUCE_SUMMARIZE_PROMPT.to_string(),
            max_reduce_length: 0,
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

    /// Set the maximum character length for combined summaries before recursive
    /// reduction is triggered. Set to `0` to disable recursive reduction (default).
    pub fn with_max_reduce_length(mut self, max_len: usize) -> Self {
        self.max_reduce_length = max_len;
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

    /// Run the map-reduce summarization chain over the given documents.
    pub async fn call(&self, documents: &[Document]) -> Result<String> {
        // Map phase: summarize each document independently
        let mut map_results = Vec::with_capacity(documents.len());
        for doc in documents {
            let prompt = self.format_map_prompt(&doc.page_content);
            let messages = vec![Message::Human(HumanMessage::new(&prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            map_results.push(ai_msg.base.content.text());
        }

        // Reduce phase: combine summaries, optionally recursive
        self.reduce(&map_results).await
    }

    /// Run only the map phase and return per-document summaries.
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

    /// Reduce summaries into a final summary, optionally recursively.
    fn reduce<'a>(
        &'a self,
        summaries: &'a [String],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let combined = summaries.join("\n\n");

            // If recursive reduction is enabled and the combined text is too long,
            // split into chunks and reduce in rounds.
            if self.max_reduce_length > 0 && combined.len() > self.max_reduce_length {
                let chunks = split_into_chunks(summaries, self.max_reduce_length);
                let mut intermediate = Vec::with_capacity(chunks.len());
                for chunk in &chunks {
                    let joined = chunk.join("\n\n");
                    let prompt = self.format_reduce_prompt(&joined);
                    let messages = vec![Message::Human(HumanMessage::new(&prompt))];
                    let ai_msg = self.llm.invoke_messages(&messages, None).await?;
                    intermediate.push(ai_msg.base.content.text());
                }
                // Recursively reduce the intermediate results
                return self.reduce(&intermediate).await;
            }

            let prompt = self.format_reduce_prompt(&combined);
            let messages = vec![Message::Human(HumanMessage::new(&prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            Ok(ai_msg.base.content.text())
        })
    }
}

#[async_trait]
impl Runnable for MapReduceSummarizationChain {
    fn name(&self) -> &str {
        "MapReduceSummarizationChain"
    }

    /// Invoke with JSON input `{ "documents": [...] }`.
    /// Returns `{ "summary": "..." }`.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let docs = parse_documents_from_input(&input)?;
        let summary = self.call(&docs).await?;
        Ok(json!({ "summary": summary }))
    }
}

// ===========================================================================
// RefineSummarizationChain
// ===========================================================================

/// Summarization chain using the refine strategy.
///
/// Documents are processed sequentially:
/// 1. The first document is summarized using the `initial_prompt`.
/// 2. Each subsequent document refines the running summary using the `refine_prompt`.
///
/// This approach preserves more detail than map-reduce because each step has
/// access to the evolving summary, but it cannot be parallelised.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use cognis::chains::summarize::RefineSummarizationChain;
/// use cognis_core::documents::Document;
///
/// # async fn example(llm: Arc<dyn cognis_core::language_models::chat_model::BaseChatModel>) {
/// let chain = RefineSummarizationChain::new(llm);
/// let docs = vec![
///     Document::new("First part of the text."),
///     Document::new("Second part adds more detail."),
/// ];
/// let summary = chain.call(&docs).await.unwrap();
/// # }
/// ```
pub struct RefineSummarizationChain {
    /// The chat model used for all steps.
    llm: Arc<dyn BaseChatModel>,
    /// Prompt template for the first document. Must contain `{text}`.
    initial_prompt: String,
    /// Prompt template for subsequent documents. Must contain `{text}` and `{existing_summary}`.
    refine_prompt: String,
}

impl RefineSummarizationChain {
    /// Create a new `RefineSummarizationChain` with default prompts.
    pub fn new(llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            llm,
            initial_prompt: DEFAULT_INITIAL_SUMMARIZE_PROMPT.to_string(),
            refine_prompt: DEFAULT_REFINE_SUMMARIZE_PROMPT.to_string(),
        }
    }

    /// Set a custom initial prompt template. Must contain `{text}`.
    pub fn with_initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = prompt.into();
        self
    }

    /// Set a custom refine prompt template. Must contain `{text}` and `{existing_summary}`.
    pub fn with_refine_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.refine_prompt = prompt.into();
        self
    }

    /// Format the initial prompt by replacing `{text}`.
    fn format_initial_prompt(&self, text: &str) -> String {
        self.initial_prompt.replace("{text}", text)
    }

    /// Format the refine prompt by replacing `{text}` and `{existing_summary}`.
    fn format_refine_prompt(&self, text: &str, existing_summary: &str) -> String {
        self.refine_prompt
            .replace("{text}", text)
            .replace("{existing_summary}", existing_summary)
    }

    /// Run the refine summarization chain over the given documents.
    ///
    /// Returns an error if no documents are provided.
    pub async fn call(&self, documents: &[Document]) -> Result<String> {
        if documents.is_empty() {
            return Err(CognisError::Other(
                "RefineSummarizationChain requires at least one document".into(),
            ));
        }

        // Process first document with initial prompt
        let first_prompt = self.format_initial_prompt(&documents[0].page_content);
        let messages = vec![Message::Human(HumanMessage::new(&first_prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let mut current_summary = ai_msg.base.content.text();

        // Refine with subsequent documents
        for doc in &documents[1..] {
            let refine_prompt = self.format_refine_prompt(&doc.page_content, &current_summary);
            let messages = vec![Message::Human(HumanMessage::new(&refine_prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            current_summary = ai_msg.base.content.text();
        }

        Ok(current_summary)
    }
}

#[async_trait]
impl Runnable for RefineSummarizationChain {
    fn name(&self) -> &str {
        "RefineSummarizationChain"
    }

    /// Invoke with JSON input `{ "documents": [...] }`.
    /// Returns `{ "summary": "..." }`.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let docs = parse_documents_from_input(&input)?;
        let summary = self.call(&docs).await?;
        Ok(json!({ "summary": summary }))
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse a `Vec<Document>` from a JSON input with a `"documents"` key.
///
/// Each element can be either:
/// - A plain string (interpreted as `page_content`)
/// - An object with at least a `"page_content"` field
fn parse_documents_from_input(input: &Value) -> Result<Vec<Document>> {
    let arr = input
        .get("documents")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CognisError::TypeMismatch {
            expected: "JSON object with 'documents' array".into(),
            got: format!("{}", input),
        })?;

    let mut docs = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            Value::String(s) => {
                docs.push(Document::new(s.as_str()));
            }
            Value::Object(_) => {
                let doc: Document = serde_json::from_value(item.clone())?;
                docs.push(doc);
            }
            _ => {
                return Err(CognisError::TypeMismatch {
                    expected: "string or Document object".into(),
                    got: format!("{}", item),
                });
            }
        }
    }

    Ok(docs)
}

/// Split summaries into chunks where each chunk's combined length does not
/// exceed `max_length`.
fn split_into_chunks(summaries: &[String], max_length: usize) -> Vec<Vec<String>> {
    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut current_chunk: Vec<String> = Vec::new();
    let mut current_length: usize = 0;

    for summary in summaries {
        let sep_len = if current_chunk.is_empty() { 0 } else { 2 }; // "\n\n"
        if current_length + sep_len + summary.len() > max_length && !current_chunk.is_empty() {
            chunks.push(std::mem::take(&mut current_chunk));
            current_length = 0;
        }
        current_length += if current_chunk.is_empty() { 0 } else { 2 } + summary.len();
        current_chunk.push(summary.clone());
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::{FakeListChatModel, ParrotFakeChatModel};

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    // -----------------------------------------------------------------------
    // StuffSummarizationChain tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_stuff_basic_summarization() {
        let llm = fake_model(vec!["This is a summary."]);
        let chain = StuffSummarizationChain::new(llm);

        let docs = vec![make_doc("Some content to summarize.")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "This is a summary.");
    }

    #[tokio::test]
    async fn test_stuff_multiple_documents_joined() {
        // Use ParrotFakeChatModel to verify documents are joined in the prompt
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = StuffSummarizationChain::new(llm);

        let docs = vec![make_doc("Doc A"), make_doc("Doc B"), make_doc("Doc C")];
        let result = chain.call(&docs).await.unwrap();

        // Parrot echoes the prompt, so it should contain all three doc contents
        assert!(result.contains("Doc A"));
        assert!(result.contains("Doc B"));
        assert!(result.contains("Doc C"));
    }

    #[tokio::test]
    async fn test_stuff_custom_prompt() {
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = StuffSummarizationChain::new(llm).with_prompt("CUSTOM SUMMARIZE: {text} END");

        let docs = vec![make_doc("hello world")];
        let result = chain.call(&docs).await.unwrap();

        assert!(result.contains("CUSTOM SUMMARIZE:"));
        assert!(result.contains("hello world"));
        assert!(result.contains("END"));
    }

    #[tokio::test]
    async fn test_stuff_empty_documents() {
        // Empty doc list should still work -- sends empty text
        let llm = fake_model(vec!["empty summary"]);
        let chain = StuffSummarizationChain::new(llm);

        let docs: Vec<Document> = vec![];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "empty summary");
    }

    #[tokio::test]
    async fn test_stuff_implements_runnable() {
        let llm = fake_model(vec!["runnable summary"]);
        let chain = StuffSummarizationChain::new(llm);

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "StuffSummarizationChain");

        let input = json!({ "documents": ["doc content"] });
        let result = runnable.invoke(input, None).await.unwrap();

        assert_eq!(result["summary"], "runnable summary");
    }

    // -----------------------------------------------------------------------
    // MapReduceSummarizationChain tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_map_reduce_with_multiple_documents() {
        // 3 docs -> 3 map calls + 1 reduce call = 4 LLM calls
        let llm = fake_model(vec![
            "summary of doc 1",
            "summary of doc 2",
            "summary of doc 3",
            "final combined summary",
        ]);
        let chain = MapReduceSummarizationChain::new(llm);

        let docs = vec![
            make_doc("First document content"),
            make_doc("Second document content"),
            make_doc("Third document content"),
        ];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "final combined summary");
    }

    #[tokio::test]
    async fn test_map_reduce_produces_map_then_reduce_outputs() {
        // 2 docs -> 2 map calls + 1 reduce call
        let llm = fake_model(vec!["mapped-A", "mapped-B", "reduced-final"]);
        let chain = MapReduceSummarizationChain::new(llm.clone());

        let docs = vec![make_doc("Doc A"), make_doc("Doc B")];

        // Verify map phase independently
        let map_results = chain.map(&docs).await.unwrap();
        assert_eq!(map_results.len(), 2);
        assert_eq!(map_results[0], "mapped-A");
        assert_eq!(map_results[1], "mapped-B");

        // Need a fresh model for the full call since FakeListChatModel consumed responses
        let llm2 = fake_model(vec!["mapped-X", "mapped-Y", "final-reduced"]);
        let chain2 = MapReduceSummarizationChain::new(llm2);
        let result = chain2.call(&docs).await.unwrap();
        assert_eq!(result, "final-reduced");
    }

    #[tokio::test]
    async fn test_map_reduce_single_document() {
        // 1 doc -> 1 map call + 1 reduce call
        let llm = fake_model(vec!["single-map", "single-reduce"]);
        let chain = MapReduceSummarizationChain::new(llm);

        let docs = vec![make_doc("Only document")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "single-reduce");
    }

    #[tokio::test]
    async fn test_map_reduce_empty_documents() {
        // 0 docs -> 0 map calls + 1 reduce call with empty summaries
        let llm = fake_model(vec!["reduce-of-nothing"]);
        let chain = MapReduceSummarizationChain::new(llm);

        let docs: Vec<Document> = vec![];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "reduce-of-nothing");
    }

    #[tokio::test]
    async fn test_map_reduce_implements_runnable() {
        let llm = fake_model(vec!["m1", "m2", "reduced"]);
        let chain = MapReduceSummarizationChain::new(llm);

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "MapReduceSummarizationChain");

        let input = json!({ "documents": ["doc1", "doc2"] });
        let result = runnable.invoke(input, None).await.unwrap();

        assert_eq!(result["summary"], "reduced");
    }

    #[tokio::test]
    async fn test_map_reduce_large_document_set() {
        // 10 documents -> 10 map calls + 1 reduce call
        let mut responses: Vec<&str> = Vec::new();
        let map_responses: Vec<String> = (0..10).map(|i| format!("summary-{}", i)).collect();
        for r in &map_responses {
            responses.push(r.as_str());
        }
        // We need to collect into owned strings for the fake model
        let mut all_responses: Vec<String> = map_responses;
        all_responses.push("grand-summary".to_string());

        let llm = Arc::new(FakeListChatModel::new(all_responses));
        let chain = MapReduceSummarizationChain::new(llm);

        let docs: Vec<Document> = (0..10)
            .map(|i| make_doc(&format!("Document number {} with some content.", i)))
            .collect();

        let result = chain.call(&docs).await.unwrap();
        assert_eq!(result, "grand-summary");
    }

    #[tokio::test]
    async fn test_map_reduce_custom_prompts() {
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = MapReduceSummarizationChain::new(llm)
            .with_map_prompt("MAP: {text} ENDMAP")
            .with_reduce_prompt("REDUCE: {summaries} ENDREDUCE");

        let docs = vec![make_doc("test content")];
        let result = chain.call(&docs).await.unwrap();

        // The reduce prompt will contain the parrot-echoed map prompt
        assert!(result.contains("REDUCE:"));
        assert!(result.contains("ENDREDUCE"));
    }

    // -----------------------------------------------------------------------
    // RefineSummarizationChain tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_refine_iterates_through_documents() {
        // 3 docs: 1 initial + 2 refine = 3 LLM calls
        let llm = fake_model(vec!["initial-summary", "refined-once", "refined-twice"]);
        let chain = RefineSummarizationChain::new(llm);

        let docs = vec![make_doc("Doc 1"), make_doc("Doc 2"), make_doc("Doc 3")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "refined-twice");
    }

    #[tokio::test]
    async fn test_refine_single_document_no_refinement() {
        // Single doc uses only the initial prompt
        let llm = fake_model(vec!["only-summary"]);
        let chain = RefineSummarizationChain::new(llm);

        let docs = vec![make_doc("Single document")];
        let result = chain.call(&docs).await.unwrap();

        assert_eq!(result, "only-summary");
    }

    #[tokio::test]
    async fn test_refine_empty_documents_returns_error() {
        let llm = fake_model(vec!["unused"]);
        let chain = RefineSummarizationChain::new(llm);

        let result = chain.call(&[]).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one document"));
    }

    #[tokio::test]
    async fn test_refine_implements_runnable() {
        let llm = fake_model(vec!["summary-result"]);
        let chain = RefineSummarizationChain::new(llm);

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "RefineSummarizationChain");

        let input = json!({ "documents": ["content"] });
        let result = runnable.invoke(input, None).await.unwrap();

        assert_eq!(result["summary"], "summary-result");
    }

    #[tokio::test]
    async fn test_refine_custom_prompts() {
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = RefineSummarizationChain::new(llm)
            .with_initial_prompt("INIT: {text}")
            .with_refine_prompt("REFINE: {existing_summary} + {text}");

        let docs = vec![make_doc("alpha"), make_doc("beta")];
        let result = chain.call(&docs).await.unwrap();

        assert!(result.contains("REFINE:"));
        assert!(result.contains("beta"));
    }

    // -----------------------------------------------------------------------
    // Runnable input parsing tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_runnable_with_document_objects_input() {
        let llm = fake_model(vec!["parsed-summary"]);
        let chain = StuffSummarizationChain::new(llm);

        let input = json!({
            "documents": [
                { "page_content": "Document from JSON object" }
            ]
        });
        let result = chain.invoke(input, None).await.unwrap();
        assert_eq!(result["summary"], "parsed-summary");
    }

    #[tokio::test]
    async fn test_runnable_invalid_input() {
        let llm = fake_model(vec!["unused"]);
        let chain = StuffSummarizationChain::new(llm);

        // Missing "documents" key
        let result = chain.invoke(json!({"text": "hello"}), None).await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_into_chunks() {
        let summaries: Vec<String> = vec![
            "short".to_string(),
            "also short".to_string(),
            "another one".to_string(),
        ];
        // max_length = 20 -> first two fit ("short\n\nalso short" = 18), third is separate
        let chunks = split_into_chunks(&summaries, 20);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 1);
    }
}
