use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

const DEFAULT_CONDENSE_PROMPT: &str = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:";

const DEFAULT_QA_PROMPT: &str = "Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:";

/// Result returned by [`ConversationalRetrievalChain::call_with_sources`].
#[derive(Debug, Clone)]
pub struct ConversationalRetrievalResult {
    /// The generated answer.
    pub answer: String,
    /// The documents retrieved for the condensed question.
    pub source_documents: Vec<Document>,
    /// The standalone question produced by the condensation step.
    pub condensed_question: String,
}

/// A chain that combines conversational history condensation with document
/// retrieval and question answering.
///
/// On each call the chain:
/// 1. Condenses the follow-up question into a standalone question using the
///    chat history.
/// 2. Retrieves relevant documents for the standalone question.
/// 3. Formats a QA prompt with the retrieved context and the question.
/// 4. Calls the LLM to produce an answer.
/// 5. Updates the internal chat history.
pub struct ConversationalRetrievalChain {
    retriever: Arc<dyn BaseRetriever>,
    llm: Arc<dyn BaseChatModel>,
    condense_llm: Option<Arc<dyn BaseChatModel>>,
    chat_history: Arc<RwLock<Vec<Message>>>,
    k: usize,
    condense_prompt: String,
    qa_prompt: String,
}

impl ConversationalRetrievalChain {
    /// Create a new chain with the given retriever and LLM.
    pub fn new(retriever: Arc<dyn BaseRetriever>, llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            retriever,
            llm,
            condense_llm: None,
            chat_history: Arc::new(RwLock::new(Vec::new())),
            k: 4,
            condense_prompt: DEFAULT_CONDENSE_PROMPT.to_string(),
            qa_prompt: DEFAULT_QA_PROMPT.to_string(),
        }
    }

    /// Set a separate LLM for the question condensation step.
    pub fn with_condense_llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.condense_llm = Some(llm);
        self
    }

    /// Set the number of documents to retrieve (default: 4).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a custom condense prompt template.
    ///
    /// The template must contain `{chat_history}` and `{question}` placeholders.
    pub fn with_condense_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.condense_prompt = prompt.into();
        self
    }

    /// Set a custom QA prompt template.
    ///
    /// The template must contain `{context}` and `{question}` placeholders.
    pub fn with_qa_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.qa_prompt = prompt.into();
        self
    }

    /// Condense a follow-up question into a standalone question using the chat
    /// history.
    ///
    /// If the chat history is empty the question is returned as-is.
    pub async fn condense_question(&self, question: &str) -> Result<String> {
        let history = self.chat_history.read().await;
        if history.is_empty() {
            return Ok(question.to_string());
        }

        // Format the chat history as text
        let history_text: String = history
            .iter()
            .map(|m| {
                let role = match m.message_type() {
                    rustchain_core::messages::MessageType::Human => "Human",
                    rustchain_core::messages::MessageType::Ai => "AI",
                    _ => "Other",
                };
                format!("{}: {}", role, m.content().text())
            })
            .collect::<Vec<_>>()
            .join("\n");
        drop(history);

        let prompt = self
            .condense_prompt
            .replace("{chat_history}", &history_text)
            .replace("{question}", question);

        let condense_model = self.condense_llm.as_ref().unwrap_or(&self.llm);
        let messages = vec![Message::Human(HumanMessage::new(&prompt))];
        let ai_msg = condense_model.invoke_messages(&messages, None).await?;
        Ok(ai_msg.base.content.text())
    }

    /// Run the full retrieval-augmented conversation chain.
    ///
    /// Returns the answer as a plain string and updates the internal chat
    /// history.
    pub async fn call(&self, question: &str) -> Result<String> {
        let result = self.call_with_sources(question).await?;
        Ok(result.answer)
    }

    /// Run the chain and return the answer along with source documents and the
    /// condensed question.
    pub async fn call_with_sources(
        &self,
        question: &str,
    ) -> Result<ConversationalRetrievalResult> {
        // Step 1: Condense the question
        let condensed = self.condense_question(question).await?;

        // Step 2: Retrieve documents
        let mut docs = self.retriever.get_relevant_documents(&condensed).await?;
        docs.truncate(self.k);

        // Step 3: Format QA prompt
        let context = docs
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let qa_input = self
            .qa_prompt
            .replace("{context}", &context)
            .replace("{question}", &condensed);

        // Step 4: Call LLM
        let messages = vec![Message::Human(HumanMessage::new(&qa_input))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let answer = ai_msg.base.content.text();

        // Step 5: Update chat history
        {
            let mut history = self.chat_history.write().await;
            history.push(Message::human(question));
            history.push(Message::ai(&answer));
        }

        Ok(ConversationalRetrievalResult {
            answer,
            source_documents: docs,
            condensed_question: condensed,
        })
    }

    /// Clear the conversation history.
    pub async fn clear_history(&self) {
        let mut history = self.chat_history.write().await;
        history.clear();
    }
}

#[async_trait]
impl Runnable for ConversationalRetrievalChain {
    fn name(&self) -> &str {
        "ConversationalRetrievalChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let question = input
            .get("input")
            .and_then(|v| v.as_str())
            .or_else(|| input.as_str())
            .ok_or_else(|| rustchain_core::error::RustChainError::TypeMismatch {
                expected: "Object with 'input' string key or a plain string".into(),
                got: format!("{}", input),
            })?
            .to_string();

        let result = self.call_with_sources(&question).await?;
        let source_docs: Vec<Value> = result
            .source_documents
            .iter()
            .map(|d| serde_json::to_value(d).unwrap_or(Value::Null))
            .collect();

        Ok(json!({
            "answer": result.answer,
            "source_documents": source_docs,
            "condensed_question": result.condensed_question,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::documents::Document;
    use rustchain_core::error::Result;
    use rustchain_core::language_models::fake::FakeListChatModel;
    use rustchain_core::retrievers::BaseRetriever;

    struct MockRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    fn make_retriever(docs: Vec<Document>) -> Arc<dyn BaseRetriever> {
        Arc::new(MockRetriever { docs })
    }

    fn make_docs() -> Vec<Document> {
        vec![
            Document::new("Rust is a systems programming language."),
            Document::new("Rust was first released in 2015."),
        ]
    }

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    #[tokio::test]
    async fn test_first_question_skips_condensation() {
        // With no history the condensed question should be the original question.
        let chain = ConversationalRetrievalChain::new(
            make_retriever(make_docs()),
            fake_model(vec!["answer about Rust"]),
        );

        let result = chain.call_with_sources("What is Rust?").await.unwrap();
        // condensed_question should be the original since history was empty
        assert_eq!(result.condensed_question, "What is Rust?");
        assert_eq!(result.answer, "answer about Rust");
    }

    #[tokio::test]
    async fn test_followup_condenses_with_history() {
        // The condense LLM produces a standalone question from history.
        // First call uses response index 0, condensation uses index 1, QA uses index 2.
        let llm = fake_model(vec![
            "Rust is great",           // first call QA answer
            "What year was Rust released?", // condensation output
            "Rust was released in 2015",    // second call QA answer
        ]);

        let chain = ConversationalRetrievalChain::new(
            make_retriever(make_docs()),
            llm,
        );

        // First call -- no condensation
        let r1 = chain.call_with_sources("Tell me about Rust").await.unwrap();
        assert_eq!(r1.condensed_question, "Tell me about Rust");
        assert_eq!(r1.answer, "Rust is great");

        // Second call -- should condense
        let r2 = chain.call_with_sources("When was it released?").await.unwrap();
        // The condensed question comes from the LLM
        assert_eq!(r2.condensed_question, "What year was Rust released?");
        assert_eq!(r2.answer, "Rust was released in 2015");
    }

    #[tokio::test]
    async fn test_call_returns_answer_and_updates_history() {
        let chain = ConversationalRetrievalChain::new(
            make_retriever(make_docs()),
            fake_model(vec!["the answer"]),
        );

        let answer = chain.call("my question").await.unwrap();
        assert_eq!(answer, "the answer");

        // History should contain 2 messages: human + ai
        let history = chain.chat_history.read().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content().text(), "my question");
        assert_eq!(history[1].content().text(), "the answer");
    }

    #[tokio::test]
    async fn test_call_with_sources_returns_docs_and_condensed() {
        let docs = vec![
            Document::new("doc1 content").with_id("d1"),
            Document::new("doc2 content").with_id("d2"),
        ];
        let chain = ConversationalRetrievalChain::new(
            make_retriever(docs.clone()),
            fake_model(vec!["sourced answer"]),
        );

        let result = chain.call_with_sources("question?").await.unwrap();
        assert_eq!(result.answer, "sourced answer");
        assert_eq!(result.source_documents.len(), 2);
        assert_eq!(result.source_documents[0].page_content, "doc1 content");
        assert_eq!(result.source_documents[1].page_content, "doc2 content");
        assert_eq!(result.condensed_question, "question?");
    }

    #[tokio::test]
    async fn test_clear_history_resets() {
        let chain = ConversationalRetrievalChain::new(
            make_retriever(make_docs()),
            fake_model(vec!["a1", "condensed q", "a2"]),
        );

        // First call populates history
        chain.call("first question").await.unwrap();
        {
            let history = chain.chat_history.read().await;
            assert_eq!(history.len(), 2);
        }

        // Clear
        chain.clear_history().await;
        {
            let history = chain.chat_history.read().await;
            assert!(history.is_empty());
        }

        // Next call should skip condensation again since history is empty
        let r = chain.call_with_sources("new question").await.unwrap();
        assert_eq!(r.condensed_question, "new question");
    }

    #[tokio::test]
    async fn test_with_condense_llm() {
        let main_llm = fake_model(vec!["main answer 1", "main answer 2"]);
        let condense_llm = fake_model(vec!["standalone question from condense llm"]);

        let chain = ConversationalRetrievalChain::new(
            make_retriever(make_docs()),
            main_llm,
        )
        .with_condense_llm(condense_llm);

        // First call -- no condensation needed
        chain.call("first q").await.unwrap();

        // Second call -- uses the condense LLM
        let r = chain.call_with_sources("follow up").await.unwrap();
        assert_eq!(r.condensed_question, "standalone question from condense llm");
        assert_eq!(r.answer, "main answer 2");
    }

    #[tokio::test]
    async fn test_k_limits_documents() {
        let docs = vec![
            Document::new("doc1"),
            Document::new("doc2"),
            Document::new("doc3"),
            Document::new("doc4"),
            Document::new("doc5"),
        ];
        let chain = ConversationalRetrievalChain::new(
            make_retriever(docs),
            fake_model(vec!["answer"]),
        )
        .with_k(2);

        let result = chain.call_with_sources("q").await.unwrap();
        assert_eq!(result.source_documents.len(), 2);
    }
}
