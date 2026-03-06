//! Output fixing parser that uses an LLM to correct malformed output.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message, SystemMessage};
use rustchain_core::output_parsers::OutputParser;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

/// An output parser that wraps an inner parser and uses a chat model to fix
/// malformed output when parsing fails.
///
/// On parse:
/// 1. Try the inner parser first.
/// 2. If it fails, send the malformed output, the error message, and format
///    instructions to the LLM, asking it to fix the output.
/// 3. Parse the LLM's corrected response with the inner parser.
/// 4. If that also fails, return the error.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::output_parsers::OutputFixingParser;
///
/// let parser = OutputFixingParser::builder()
///     .parser(json_parser)
///     .llm(model)
///     .build();
/// let result = parser.parse(r#"{"name": "Alice",}"#); // trailing comma
/// ```
pub struct OutputFixingParser {
    /// The inner parser to attempt first.
    parser: Box<dyn OutputParser>,
    /// The chat model used to fix malformed output.
    llm: Arc<dyn BaseChatModel>,
}

/// Builder for [`OutputFixingParser`].
pub struct OutputFixingParserBuilder {
    parser: Option<Box<dyn OutputParser>>,
    llm: Option<Arc<dyn BaseChatModel>>,
}

impl OutputFixingParserBuilder {
    /// Set the inner output parser.
    pub fn parser(mut self, parser: impl OutputParser + 'static) -> Self {
        self.parser = Some(Box::new(parser));
        self
    }

    /// Set the chat model for fixing output.
    pub fn llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Build the [`OutputFixingParser`].
    ///
    /// # Panics
    ///
    /// Panics if `parser` or `llm` has not been set.
    pub fn build(self) -> OutputFixingParser {
        OutputFixingParser {
            parser: self.parser.expect("parser is required"),
            llm: self.llm.expect("llm is required"),
        }
    }
}

impl OutputFixingParser {
    /// Create a new builder.
    pub fn builder() -> OutputFixingParserBuilder {
        OutputFixingParserBuilder {
            parser: None,
            llm: None,
        }
    }

    /// Create directly from an inner parser and LLM.
    pub fn new(parser: impl OutputParser + 'static, llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            parser: Box::new(parser),
            llm,
        }
    }

    /// Attempt to fix malformed output by sending it to the LLM.
    async fn fix_output(&self, malformed: &str, error: &RustChainError) -> Result<Value> {
        let format_instructions = self.parser.get_format_instructions().unwrap_or_default();

        let system_msg = Message::System(SystemMessage::new(
            "You are a helpful assistant that fixes malformed output. \
             Return ONLY the corrected output with no additional text.",
        ));

        let user_content = format!(
            "The following output failed to parse:\n\n```\n{}\n```\n\n\
             Error: {}\n\n\
             Please fix the output so it conforms to the expected format.\n\n\
             {}",
            malformed, error, format_instructions
        );
        let user_msg = Message::Human(HumanMessage::new(&user_content));

        let ai_msg = self
            .llm
            .invoke_messages(&[system_msg, user_msg], None)
            .await?;
        let fixed_text = ai_msg.base.content.text();
        self.parser.parse(&fixed_text)
    }
}

impl OutputParser for OutputFixingParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // Synchronous parse -- try inner parser only.
        // LLM-based fixing requires async; callers should use the Runnable interface.
        self.parser.parse(text)
    }

    fn get_format_instructions(&self) -> Option<String> {
        self.parser.get_format_instructions()
    }

    fn parser_type(&self) -> &str {
        "output_fixing_parser"
    }
}

#[async_trait]
impl Runnable for OutputFixingParser {
    fn name(&self) -> &str {
        "OutputFixingParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        match self.parser.parse(&text) {
            Ok(v) => Ok(v),
            Err(e) => self.fix_output(&text, &e).await,
        }
    }
}
