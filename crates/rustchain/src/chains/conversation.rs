//! Conversation chain that manages chat history and uses configurable prompt templates.
//!
//! The [`ConversationChain`] wraps a chat model with a memory backend so that
//! multi-turn conversations are handled transparently. Each call to [`predict`]
//! loads the conversation history from memory, formats a prompt, sends it to the
//! model, saves the exchange, and returns the response.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::{BaseChatModel, ChatStream};
use rustchain_core::messages::{HumanMessage, Message, SystemMessage};
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

use crate::memory::BaseMemory;

/// Default prompt template used when none is provided.
const DEFAULT_PROMPT_TEMPLATE: &str = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:";

/// A conversation chain that manages chat history via a pluggable memory backend
/// and formats prompts with configurable templates.
///
/// # Example
///
/// ```rust,no_run
/// use rustchain::chains::ConversationChain;
/// use rustchain::memory::ConversationBufferMemory;
/// # use std::sync::Arc;
///
/// # async fn example(model: Arc<dyn rustchain_core::language_models::chat_model::BaseChatModel>) {
/// let chain = ConversationChain::builder()
///     .llm(model)
///     .memory(Box::new(ConversationBufferMemory::new().with_return_messages(false)))
///     .system_prompt("You are a helpful assistant.")
///     .build();
///
/// let response = chain.predict("Hello!").await.unwrap();
/// # }
/// ```
pub struct ConversationChain {
    /// The chat model used for generation.
    llm: Arc<dyn BaseChatModel>,
    /// Conversation memory backend.
    memory: Box<dyn BaseMemory>,
    /// Optional system message prepended to every request.
    system_prompt: Option<String>,
    /// Prompt template with `{history}` and `{input}` placeholders.
    prompt_template: String,
    /// Key used for output in the result JSON. Default: `"response"`.
    output_key: String,
    /// Key used for the human input. Default: `"input"`.
    input_key: String,
    /// Whether to log verbose information.
    verbose: bool,
}

/// Builder for [`ConversationChain`].
pub struct ConversationChainBuilder {
    llm: Option<Arc<dyn BaseChatModel>>,
    memory: Option<Box<dyn BaseMemory>>,
    system_prompt: Option<String>,
    prompt_template: Option<String>,
    output_key: String,
    input_key: String,
    verbose: bool,
}

impl ConversationChainBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            llm: None,
            memory: None,
            system_prompt: None,
            prompt_template: None,
            output_key: "response".to_string(),
            input_key: "input".to_string(),
            verbose: false,
        }
    }

    /// Set the chat model (required).
    pub fn llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the chat model (required). Alias for [`llm`] for backwards compatibility.
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(model);
        self
    }

    /// Set the memory backend.
    pub fn memory(mut self, memory: Box<dyn BaseMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set an optional system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set a custom prompt template with `{history}` and `{input}` placeholders.
    pub fn prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }

    /// Set the output key for the result map. Default: `"response"`.
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set the input key. Default: `"input"`.
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Enable verbose logging.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the [`ConversationChain`].
    ///
    /// # Panics
    ///
    /// Panics if `llm` is not set.
    pub fn build(self) -> ConversationChain {
        use crate::memory::ConversationBufferMemory;

        let memory = self.memory.unwrap_or_else(|| {
            Box::new(ConversationBufferMemory::new().with_return_messages(false))
        });

        ConversationChain {
            llm: self.llm.expect("llm is required for ConversationChain"),
            memory,
            system_prompt: self.system_prompt,
            prompt_template: self
                .prompt_template
                .unwrap_or_else(|| DEFAULT_PROMPT_TEMPLATE.to_string()),
            output_key: self.output_key,
            input_key: self.input_key,
            verbose: self.verbose,
        }
    }
}

impl Default for ConversationChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationChain {
    /// Create a new builder.
    pub fn builder() -> ConversationChainBuilder {
        ConversationChainBuilder::new()
    }

    /// Build the messages to send to the model from the prompt template, memory,
    /// and optional extra context variables.
    async fn build_messages(
        &self,
        input: &str,
        extra_context: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Message>> {
        // Load history from memory
        let mem_vars = self.memory.load_memory_variables().await?;
        let history = mem_vars
            .get(self.memory.memory_key())
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Build template variables
        let mut vars: HashMap<String, String> = HashMap::new();
        vars.insert("history".to_string(), history);
        vars.insert("input".to_string(), input.to_string());

        if let Some(ctx) = extra_context {
            for (k, v) in ctx {
                vars.insert(k.clone(), v.clone());
            }
        }

        // Format the prompt template
        let formatted = self.format_template(&self.prompt_template, &vars)?;

        if self.verbose {
            eprintln!("[ConversationChain] Formatted prompt:\n{}", formatted);
        }

        // Build message list
        let mut messages = Vec::new();
        if let Some(ref sys) = self.system_prompt {
            messages.push(Message::System(SystemMessage::new(sys)));
        }
        messages.push(Message::Human(HumanMessage::new(&formatted)));

        Ok(messages)
    }

    /// Format a template string by replacing `{variable}` placeholders.
    fn format_template(&self, template: &str, vars: &HashMap<String, String>) -> Result<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let mut missing: Vec<String> = Vec::new();

        let result = re.replace_all(template, |caps: &regex::Captures| {
            let key = &caps[1];
            match vars.get(key) {
                Some(val) => val.clone(),
                None => {
                    missing.push(key.to_string());
                    String::new()
                }
            }
        });

        if !missing.is_empty() {
            return Err(RustChainError::InvalidKey(format!(
                "Missing template variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(result.into_owned())
    }

    /// Run the conversation chain with the given input and return the AI response.
    pub async fn predict(&self, input: &str) -> Result<String> {
        let messages = self.build_messages(input, None).await?;
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let response = ai_msg.base.content.text();

        // Save to memory
        let input_msg = Message::human(input);
        let output_msg = Message::ai(&response);
        self.memory.save_context(&input_msg, &output_msg).await?;

        Ok(response)
    }

    /// Run the conversation chain with extra context variables substituted into the prompt.
    pub async fn predict_with_context(
        &self,
        input: &str,
        context: HashMap<String, String>,
    ) -> Result<String> {
        let messages = self.build_messages(input, Some(&context)).await?;
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let response = ai_msg.base.content.text();

        // Save to memory
        let input_msg = Message::human(input);
        let output_msg = Message::ai(&response);
        self.memory.save_context(&input_msg, &output_msg).await?;

        Ok(response)
    }

    /// Stream the conversation response.
    pub async fn stream(&self, input: &str) -> Result<ChatStream> {
        let messages = self.build_messages(input, None).await?;
        self.llm._stream(&messages, None).await
    }

    /// Clear the conversation history.
    pub async fn clear_history(&self) -> Result<()> {
        self.memory.clear().await
    }
}

#[async_trait]
impl Runnable for ConversationChain {
    fn name(&self) -> &str {
        "ConversationChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let input_str = match &input {
            Value::String(s) => s.clone(),
            Value::Object(map) => map
                .get(&self.input_key)
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    RustChainError::InvalidKey(format!(
                        "Input object missing '{}' key",
                        self.input_key
                    ))
                })?
                .to_string(),
            _ => {
                return Err(RustChainError::TypeMismatch {
                    expected: "String or Object with input key".into(),
                    got: format!("{}", input),
                });
            }
        };

        let response = self.predict(&input_str).await?;
        Ok(json!({ &self.output_key: response }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{ConversationBufferMemory, ConversationWindowMemory};
    use rustchain_core::language_models::fake::FakeListChatModel;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    // 1. Basic conversation with buffer memory
    #[tokio::test]
    async fn test_basic_conversation_buffer_memory() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Hello! How can I help you?"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let response = chain.predict("Hi there!").await.unwrap();
        assert_eq!(response, "Hello! How can I help you?");
    }

    // 2. Conversation with window memory
    #[tokio::test]
    async fn test_conversation_window_memory() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Response 1", "Response 2", "Response 3"]))
            .memory(Box::new(
                ConversationWindowMemory::new(1).with_return_messages(false),
            ))
            .build();

        let r1 = chain.predict("Turn 1").await.unwrap();
        assert_eq!(r1, "Response 1");

        let r2 = chain.predict("Turn 2").await.unwrap();
        assert_eq!(r2, "Response 2");

        // Window memory with k=1 only keeps last turn
        let mem_vars = chain.memory.load_memory_variables().await.unwrap();
        let history = mem_vars
            .get(chain.memory.memory_key())
            .unwrap()
            .as_str()
            .unwrap();
        // Should contain Turn 2 and Response 2 but NOT Turn 1
        assert!(history.contains("Turn 2"));
        assert!(history.contains("Response 2"));
        assert!(!history.contains("Turn 1"));
    }

    // 3. Custom system prompt
    #[tokio::test]
    async fn test_custom_system_prompt() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["I am a pirate assistant!"]))
            .system_prompt("You are a pirate. Always respond in pirate speak.")
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        assert!(chain.system_prompt.is_some());
        assert_eq!(
            chain.system_prompt.as_ref().unwrap(),
            "You are a pirate. Always respond in pirate speak."
        );

        let response = chain.predict("Hello").await.unwrap();
        assert_eq!(response, "I am a pirate assistant!");
    }

    // 4. Custom prompt template
    #[tokio::test]
    async fn test_custom_prompt_template() {
        let custom_template = "{history}\nUser: {input}\nBot:";
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Custom template works!"]))
            .prompt_template(custom_template)
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        assert_eq!(chain.prompt_template, custom_template);

        let response = chain.predict("Test input").await.unwrap();
        assert_eq!(response, "Custom template works!");
    }

    // 5. Builder pattern
    #[tokio::test]
    async fn test_builder_pattern() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["ok"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .system_prompt("System")
            .prompt_template("{history}\n{input}")
            .output_key("answer")
            .input_key("question")
            .verbose(true)
            .build();

        assert_eq!(chain.output_key, "answer");
        assert_eq!(chain.input_key, "question");
        assert!(chain.verbose);
        assert_eq!(chain.system_prompt.as_deref(), Some("System"));
        assert_eq!(chain.prompt_template, "{history}\n{input}");
    }

    // 6. Multi-turn conversation (history accumulation)
    #[tokio::test]
    async fn test_multi_turn_conversation() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec![
                "I'm doing well!",
                "The weather is sunny.",
                "Goodbye!",
            ]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let r1 = chain.predict("How are you?").await.unwrap();
        assert_eq!(r1, "I'm doing well!");

        let r2 = chain.predict("What's the weather?").await.unwrap();
        assert_eq!(r2, "The weather is sunny.");

        // Verify history accumulated
        let mem_vars = chain.memory.load_memory_variables().await.unwrap();
        let history = mem_vars
            .get(chain.memory.memory_key())
            .unwrap()
            .as_str()
            .unwrap();
        assert!(history.contains("How are you?"));
        assert!(history.contains("I'm doing well!"));
        assert!(history.contains("What's the weather?"));
        assert!(history.contains("The weather is sunny."));
    }

    // 7. Clear memory
    #[tokio::test]
    async fn test_clear_memory() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Hello!", "Hi again!"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        chain.predict("Hi").await.unwrap();

        // Verify memory has content
        let vars = chain.memory.load_memory_variables().await.unwrap();
        let history = vars
            .get(chain.memory.memory_key())
            .unwrap()
            .as_str()
            .unwrap();
        assert!(!history.is_empty());

        // Clear memory
        chain.clear_history().await.unwrap();

        // Verify memory is empty
        let vars = chain.memory.load_memory_variables().await.unwrap();
        let history = vars
            .get(chain.memory.memory_key())
            .unwrap()
            .as_str()
            .unwrap();
        assert!(history.is_empty());
    }

    // 8. Input/output key customization
    #[tokio::test]
    async fn test_custom_input_output_keys() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Custom key response"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .input_key("question")
            .output_key("answer")
            .build();

        // Use Runnable interface with custom input key
        let result = chain
            .invoke(json!({"question": "What is 2+2?"}), None)
            .await
            .unwrap();
        assert_eq!(result["answer"], "Custom key response");
    }

    // 9. Context variables
    #[tokio::test]
    async fn test_context_variables() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Context used!"]))
            .prompt_template("{history}\nContext: {context}\nHuman: {input}\nAI:")
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let mut context = HashMap::new();
        context.insert("context".to_string(), "The user likes Rust".to_string());

        let response = chain
            .predict_with_context("Tell me something", context)
            .await
            .unwrap();
        assert_eq!(response, "Context used!");
    }

    // 10. Default configuration
    #[tokio::test]
    async fn test_default_configuration() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Default response"]))
            .build();

        assert_eq!(chain.output_key, "response");
        assert_eq!(chain.input_key, "input");
        assert!(!chain.verbose);
        assert!(chain.system_prompt.is_none());
        assert_eq!(chain.prompt_template, DEFAULT_PROMPT_TEMPLATE);

        let response = chain.predict("Hello").await.unwrap();
        assert_eq!(response, "Default response");
    }

    // 11. Runnable trait implementation
    #[tokio::test]
    async fn test_runnable_trait() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Runnable response"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "ConversationChain");

        // Invoke with a string
        let result = runnable
            .invoke(Value::String("Hello".into()), None)
            .await
            .unwrap();
        assert_eq!(result["response"], "Runnable response");
    }

    // 12. Runnable trait with object input
    #[tokio::test]
    async fn test_runnable_with_object_input() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["Object input response"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let result = chain
            .invoke(json!({"input": "Hello from object"}), None)
            .await
            .unwrap();
        assert_eq!(result["response"], "Object input response");
    }

    // 13. Empty history handling
    #[tokio::test]
    async fn test_empty_history_handling() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["First message response"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        // On the first call, history should be empty but the chain should still work
        let mem_vars = chain.memory.load_memory_variables().await.unwrap();
        let history = mem_vars
            .get(chain.memory.memory_key())
            .unwrap()
            .as_str()
            .unwrap();
        assert!(history.is_empty());

        let response = chain.predict("First message").await.unwrap();
        assert_eq!(response, "First message response");
    }

    // 14. Missing input key in object returns error
    #[tokio::test]
    async fn test_missing_input_key_error() {
        let chain = ConversationChain::builder()
            .llm(fake_model(vec!["response"]))
            .memory(Box::new(
                ConversationBufferMemory::new().with_return_messages(false),
            ))
            .build();

        let result = chain.invoke(json!({"wrong_key": "value"}), None).await;
        assert!(result.is_err());
    }
}
