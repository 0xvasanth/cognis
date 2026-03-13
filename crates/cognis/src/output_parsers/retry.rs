//! Retry output parser that retries parsing with LLM feedback.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use cognis_core::output_parsers::OutputParser;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// An output parser that retries parsing up to N times, sending the error
/// feedback to an LLM for correction on each retry.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::output_parsers::RetryOutputParser;
///
/// let parser = RetryOutputParser::builder()
///     .parser(json_parser)
///     .llm(model)
///     .max_retries(3)
///     .include_original_output(true)
///     .build();
/// ```
pub struct RetryOutputParser {
    /// The inner parser to attempt.
    parser: Box<dyn OutputParser>,
    /// The chat model used to correct output on retry.
    llm: Arc<dyn BaseChatModel>,
    /// Maximum number of retry attempts (default: 3).
    max_retries: usize,
    /// Whether to include the original malformed output in the retry prompt.
    include_original_output: bool,
}

/// Builder for [`RetryOutputParser`].
pub struct RetryOutputParserBuilder {
    parser: Option<Box<dyn OutputParser>>,
    llm: Option<Arc<dyn BaseChatModel>>,
    max_retries: usize,
    include_original_output: bool,
}

impl RetryOutputParserBuilder {
    /// Set the inner output parser.
    pub fn parser(mut self, parser: impl OutputParser + 'static) -> Self {
        self.parser = Some(Box::new(parser));
        self
    }

    /// Set the chat model for retry correction.
    pub fn llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the maximum number of retries (default: 3).
    pub fn max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }

    /// Whether to include the original malformed output in retry prompts (default: true).
    pub fn include_original_output(mut self, include: bool) -> Self {
        self.include_original_output = include;
        self
    }

    /// Build the [`RetryOutputParser`].
    ///
    /// # Panics
    ///
    /// Panics if `parser` or `llm` has not been set.
    pub fn build(self) -> RetryOutputParser {
        RetryOutputParser {
            parser: self.parser.expect("parser is required"),
            llm: self.llm.expect("llm is required"),
            max_retries: self.max_retries,
            include_original_output: self.include_original_output,
        }
    }
}

impl RetryOutputParser {
    /// Create a new builder.
    pub fn builder() -> RetryOutputParserBuilder {
        RetryOutputParserBuilder {
            parser: None,
            llm: None,
            max_retries: 3,
            include_original_output: true,
        }
    }

    /// Create directly with all parameters.
    pub fn new(
        parser: impl OutputParser + 'static,
        llm: Arc<dyn BaseChatModel>,
        max_retries: usize,
        include_original_output: bool,
    ) -> Self {
        Self {
            parser: Box::new(parser),
            llm,
            max_retries,
            include_original_output,
        }
    }

    /// Ask the LLM to fix the output, incorporating error feedback.
    async fn retry_with_llm(
        &self,
        original_output: &str,
        last_error: &CognisError,
    ) -> Result<String> {
        let format_instructions = self.parser.get_format_instructions().unwrap_or_default();

        let system_msg = Message::System(SystemMessage::new(
            "You are a helpful assistant that corrects malformed output. \
             Return ONLY the corrected output with no additional text or explanation.",
        ));

        let mut user_content = String::new();

        if self.include_original_output {
            user_content.push_str(&format!(
                "The following output was produced but failed to parse:\n\n```\n{}\n```\n\n",
                original_output
            ));
        }

        user_content.push_str(&format!(
            "Parse error: {}\n\n\
             Please produce a corrected version that matches the expected format.\n\n\
             {}",
            last_error, format_instructions
        ));

        let user_msg = Message::Human(HumanMessage::new(&user_content));
        let ai_msg = self
            .llm
            .invoke_messages(&[system_msg, user_msg], None)
            .await?;
        Ok(ai_msg.base.content.text())
    }
}

impl OutputParser for RetryOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // Synchronous parse -- try inner parser only.
        // Retries require async; callers should use the Runnable interface.
        self.parser.parse(text)
    }

    fn get_format_instructions(&self) -> Option<String> {
        self.parser.get_format_instructions()
    }

    fn parser_type(&self) -> &str {
        "retry_output_parser"
    }
}

#[async_trait]
impl Runnable for RetryOutputParser {
    fn name(&self) -> &str {
        "RetryOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // Initial attempt
        match self.parser.parse(&text) {
            Ok(v) => return Ok(v),
            Err(e) if self.max_retries == 0 => return Err(e),
            Err(mut last_error) => {
                let mut current_text = text.clone();

                for _attempt in 0..self.max_retries {
                    // Ask LLM to fix
                    let fixed_text = self.retry_with_llm(&current_text, &last_error).await?;

                    match self.parser.parse(&fixed_text) {
                        Ok(v) => return Ok(v),
                        Err(e) => {
                            current_text = fixed_text;
                            last_error = e;
                        }
                    }
                }

                Err(CognisError::OutputParserError {
                    message: format!(
                        "Failed to parse after {} retries: {}",
                        self.max_retries, last_error
                    ),
                    observation: Some(current_text),
                    llm_output: None,
                })
            }
        }
    }
}
