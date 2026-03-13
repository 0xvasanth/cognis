use std::sync::Arc;

use cognis_core::documents::Document;
use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};

/// Default prompt template for the first document.
const DEFAULT_INITIAL_PROMPT: &str = "Answer the question based on the text.\n\n\
     Text: {text}\n\
     Question: {question}\n\
     Answer:";

/// Default prompt template for refining the answer with subsequent documents.
const DEFAULT_REFINE_PROMPT: &str = "Refine the answer using additional context.\n\n\
     Existing answer: {existing_answer}\n\
     New context: {text}\n\
     Question: {question}\n\
     Refined answer:";

/// A refine chain that iteratively improves an answer by processing documents sequentially.
///
/// Unlike map-reduce which processes documents independently, the refine chain processes
/// documents one at a time, passing the running answer forward so each step can refine it
/// with new information.
///
/// The chain operates in two modes:
/// 1. **Initial**: The first document is processed with the `initial_prompt`.
/// 2. **Refine**: Each subsequent document is processed with the `refine_prompt`, which
///    includes the existing answer from the previous step.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use cognis::chains::refine::RefineChain;
/// use cognis_core::documents::Document;
///
/// # async fn example(llm: Arc<dyn cognis_core::language_models::chat_model::BaseChatModel>) {
/// let chain = RefineChain::new(llm);
///
/// let docs = vec![
///     Document::new("Rust was released in 2015."),
///     Document::new("Rust is known for memory safety."),
/// ];
/// let answer = chain.call("What is Rust?", &docs).await.unwrap();
/// # }
/// ```
pub struct RefineChain {
    /// The chat model used for all refinement steps.
    llm: Arc<dyn BaseChatModel>,
    /// Prompt template for the first document. Must contain `{text}` and `{question}`.
    initial_prompt: String,
    /// Prompt template for subsequent documents. Must contain `{text}`, `{question}`,
    /// and `{existing_answer}`.
    refine_prompt: String,
}

impl RefineChain {
    /// Create a new `RefineChain` with default prompts.
    pub fn new(llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            llm,
            initial_prompt: DEFAULT_INITIAL_PROMPT.to_string(),
            refine_prompt: DEFAULT_REFINE_PROMPT.to_string(),
        }
    }

    /// Set a custom initial prompt template. Must contain `{text}` and `{question}`.
    pub fn with_initial_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.initial_prompt = prompt.into();
        self
    }

    /// Set a custom refine prompt template. Must contain `{text}`, `{question}`,
    /// and `{existing_answer}`.
    pub fn with_refine_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.refine_prompt = prompt.into();
        self
    }

    /// Format the initial prompt by replacing `{text}` and `{question}`.
    fn format_initial_prompt(&self, text: &str, question: &str) -> String {
        self.initial_prompt
            .replace("{text}", text)
            .replace("{question}", question)
    }

    /// Format the refine prompt by replacing `{text}`, `{question}`, and `{existing_answer}`.
    fn format_refine_prompt(&self, text: &str, question: &str, existing_answer: &str) -> String {
        self.refine_prompt
            .replace("{text}", text)
            .replace("{question}", question)
            .replace("{existing_answer}", existing_answer)
    }

    /// Run the refine chain over the given documents.
    ///
    /// 1. Process the first document with the initial prompt.
    /// 2. For each subsequent document, refine the answer using the refine prompt.
    /// 3. Return the final refined answer.
    ///
    /// Returns an error if no documents are provided.
    pub async fn call(&self, question: &str, documents: &[Document]) -> Result<String> {
        if documents.is_empty() {
            return Err(CognisError::Other(
                "RefineChain requires at least one document".into(),
            ));
        }

        // Process first document with initial prompt
        let first_prompt = self.format_initial_prompt(&documents[0].page_content, question);
        let messages = vec![Message::Human(HumanMessage::new(&first_prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let mut current_answer = ai_msg.base.content.text();

        // Refine with subsequent documents
        for doc in &documents[1..] {
            let refine_prompt =
                self.format_refine_prompt(&doc.page_content, question, &current_answer);
            let messages = vec![Message::Human(HumanMessage::new(&refine_prompt))];
            let ai_msg = self.llm.invoke_messages(&messages, None).await?;
            current_answer = ai_msg.base.content.text();
        }

        Ok(current_answer)
    }
}

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

    #[tokio::test]
    async fn test_single_doc_uses_initial_prompt_only() {
        // With a single document, only the initial prompt is used (1 LLM call)
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = RefineChain::new(llm);

        let docs = vec![make_doc("Rust is a language")];
        let result = chain.call("What is Rust?", &docs).await.unwrap();

        // The parrot model echoes the prompt, so the result should contain the initial template
        assert!(result.contains("Answer the question based on the text"));
        assert!(result.contains("Rust is a language"));
        assert!(result.contains("What is Rust?"));
    }

    #[tokio::test]
    async fn test_multiple_docs_refine_iteratively() {
        // 3 docs: 1 initial call + 2 refine calls = 3 LLM calls
        let llm = fake_model(vec!["initial-answer", "refined-once", "refined-twice"]);
        let chain = RefineChain::new(llm);

        let docs = vec![make_doc("Doc 1"), make_doc("Doc 2"), make_doc("Doc 3")];
        let result = chain.call("question?", &docs).await.unwrap();

        // The final answer should be the last refinement
        assert_eq!(result, "refined-twice");
    }

    #[tokio::test]
    async fn test_end_to_end_call() {
        let llm = fake_model(vec!["answer-v1", "answer-v2"]);
        let chain = RefineChain::new(llm);

        let docs = vec![make_doc("Context A"), make_doc("Context B")];
        let result = chain.call("What is the answer?", &docs).await.unwrap();

        assert_eq!(result, "answer-v2");
    }

    #[tokio::test]
    async fn test_custom_prompts() {
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = RefineChain::new(llm)
            .with_initial_prompt("INIT: {text} Q: {question}")
            .with_refine_prompt("REFINE: {existing_answer} + {text} Q: {question}");

        let docs = vec![make_doc("alpha"), make_doc("beta")];
        let result = chain.call("test?", &docs).await.unwrap();

        // Second call uses refine prompt with the initial answer echoed back
        assert!(result.contains("REFINE:"));
        assert!(result.contains("beta"));
        assert!(result.contains("test?"));
    }

    #[tokio::test]
    async fn test_empty_documents_returns_error() {
        let llm = fake_model(vec!["unused"]);
        let chain = RefineChain::new(llm);

        let result = chain.call("question?", &[]).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least one document"));
    }

    #[tokio::test]
    async fn test_refine_prompt_includes_existing_answer() {
        // Use ParrotFakeChatModel to verify the refine prompt format
        let llm: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());
        let chain = RefineChain::new(llm);

        let refine_formatted = chain.format_refine_prompt("new text", "my question", "old answer");
        assert!(refine_formatted.contains("Existing answer: old answer"));
        assert!(refine_formatted.contains("New context: new text"));
        assert!(refine_formatted.contains("Question: my question"));
    }
}
