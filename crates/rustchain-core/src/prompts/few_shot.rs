use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};
use crate::messages::Message;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::PromptTemplate;
use super::example_selector::BaseExampleSelector;
use super::message::MessagePromptTemplate;
use super::string_formatter::TemplateFormat;

/// A few-shot prompt template that formats examples between a prefix and suffix.
///
/// Supports both static examples and dynamic example selection.
pub struct FewShotPromptTemplate {
    pub examples: Option<Vec<HashMap<String, Value>>>,
    pub example_selector: Option<Arc<dyn BaseExampleSelector>>,
    pub example_prompt: PromptTemplate,
    pub prefix: String,
    pub suffix: String,
    pub example_separator: String,
    pub input_variables: Vec<String>,
    pub template_format: TemplateFormat,
}

impl FewShotPromptTemplate {
    pub fn new(
        examples: Vec<HashMap<String, Value>>,
        example_prompt: PromptTemplate,
        suffix: impl Into<String>,
    ) -> Self {
        let suffix = suffix.into();
        let suffix_vars =
            super::string_formatter::get_template_variables(&suffix, TemplateFormat::FString);

        Self {
            examples: Some(examples),
            example_selector: None,
            example_prompt,
            prefix: String::new(),
            suffix,
            example_separator: "\n\n".into(),
            input_variables: suffix_vars,
            template_format: TemplateFormat::FString,
        }
    }

    /// Create a few-shot template with a dynamic example selector.
    pub fn with_example_selector(
        example_selector: Arc<dyn BaseExampleSelector>,
        example_prompt: PromptTemplate,
        suffix: impl Into<String>,
    ) -> Self {
        let suffix = suffix.into();
        let suffix_vars =
            super::string_formatter::get_template_variables(&suffix, TemplateFormat::FString);

        Self {
            examples: None,
            example_selector: Some(example_selector),
            example_prompt,
            prefix: String::new(),
            suffix,
            example_separator: "\n\n".into(),
            input_variables: suffix_vars,
            template_format: TemplateFormat::FString,
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.example_separator = sep.into();
        self
    }

    /// Get examples — either static or from the selector.
    async fn get_examples(
        &self,
        kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        if let Some(examples) = &self.examples {
            Ok(examples.clone())
        } else if let Some(selector) = &self.example_selector {
            selector.select_examples(kwargs).await
        } else {
            Err(RustChainError::Other(
                "FewShotPromptTemplate has neither examples nor example_selector".into(),
            ))
        }
    }

    /// Format the few-shot prompt into a string.
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<String> {
        let examples = self
            .examples
            .as_ref()
            .ok_or_else(|| {
                RustChainError::Other(
                    "Use format_async for FewShotPromptTemplate with example_selector".into(),
                )
            })?;
        self.format_with_examples(examples, kwargs)
    }

    /// Format with async example selection support.
    pub async fn format_async(&self, kwargs: &HashMap<String, Value>) -> Result<String> {
        let examples = self.get_examples(kwargs).await?;
        self.format_with_examples(&examples, kwargs)
    }

    fn format_with_examples(
        &self,
        examples: &[HashMap<String, Value>],
        kwargs: &HashMap<String, Value>,
    ) -> Result<String> {
        let example_strings: Vec<String> = examples
            .iter()
            .map(|ex| self.example_prompt.format(ex))
            .collect::<Result<Vec<_>>>()?;

        let mut pieces: Vec<&str> = Vec::new();
        if !self.prefix.is_empty() {
            pieces.push(&self.prefix);
        }
        for s in &example_strings {
            pieces.push(s);
        }
        pieces.push(&self.suffix);

        let template = pieces.join(&self.example_separator);
        super::string_formatter::format_template(&template, self.template_format, kwargs)
    }
}

#[async_trait]
impl Runnable for FewShotPromptTemplate {
    fn name(&self) -> &str {
        "FewShotPromptTemplate"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(RustChainError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let text = self.format_async(&kwargs).await?;
        Ok(Value::String(text))
    }
}

/// A few-shot template for chat messages, formatting examples as message sequences.
pub struct FewShotChatMessagePromptTemplate {
    pub examples: Option<Vec<HashMap<String, Value>>>,
    pub example_selector: Option<Arc<dyn BaseExampleSelector>>,
    pub example_prompt: MessagePromptTemplate,
    pub input_variables: Vec<String>,
}

impl FewShotChatMessagePromptTemplate {
    pub fn new(
        examples: Vec<HashMap<String, Value>>,
        example_prompt: MessagePromptTemplate,
    ) -> Self {
        let input_variables = example_prompt.input_variables().to_vec();
        Self {
            examples: Some(examples),
            example_selector: None,
            example_prompt,
            input_variables,
        }
    }

    /// Create with a dynamic example selector.
    pub fn with_example_selector(
        example_selector: Arc<dyn BaseExampleSelector>,
        example_prompt: MessagePromptTemplate,
    ) -> Self {
        let input_variables = example_prompt.input_variables().to_vec();
        Self {
            examples: None,
            example_selector: Some(example_selector),
            example_prompt,
            input_variables,
        }
    }

    /// Get examples — either static or from the selector.
    async fn get_examples(
        &self,
        kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        if let Some(examples) = &self.examples {
            Ok(examples.clone())
        } else if let Some(selector) = &self.example_selector {
            selector.select_examples(kwargs).await
        } else {
            Err(RustChainError::Other(
                "FewShotChatMessagePromptTemplate has neither examples nor example_selector"
                    .into(),
            ))
        }
    }

    /// Format examples as a flat list of messages.
    pub fn format_messages(
        &self,
        kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<Message>> {
        let examples = self.examples.as_ref().ok_or_else(|| {
            RustChainError::Other(
                "Use format_messages_async for dynamic example selection".into(),
            )
        })?;
        self.format_messages_with_examples(examples, kwargs)
    }

    /// Format with async example selection support.
    pub async fn format_messages_async(
        &self,
        kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<Message>> {
        let examples = self.get_examples(kwargs).await?;
        self.format_messages_with_examples(&examples, kwargs)
    }

    fn format_messages_with_examples(
        &self,
        examples: &[HashMap<String, Value>],
        _kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<Message>> {
        let mut messages = Vec::new();
        for example in examples {
            messages.extend(self.example_prompt.format_messages(example)?);
        }
        Ok(messages)
    }
}

#[async_trait]
impl Runnable for FewShotChatMessagePromptTemplate {
    fn name(&self) -> &str {
        "FewShotChatMessagePromptTemplate"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(RustChainError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let messages = self.format_messages_async(&kwargs).await?;
        serde_json::to_value(&messages)
            .map_err(|e| RustChainError::Other(format!("Failed to serialize messages: {}", e)))
    }
}
