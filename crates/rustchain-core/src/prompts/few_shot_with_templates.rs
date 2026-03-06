//! Few-shot prompt with template-based prefix and suffix.
//!
//! Mirrors Python `langchain_core.prompts.few_shot_with_templates`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::PromptTemplate;
use super::example_selector::BaseExampleSelector;

/// A few-shot prompt where the prefix and suffix are full `PromptTemplate`s.
///
/// This allows more flexibility than `FewShotPromptTemplate` where
/// prefix/suffix are plain strings. Both the prefix and suffix can contain
/// their own template variables and partial variables.
pub struct FewShotPromptWithTemplates {
    /// Static examples (mutually exclusive with `example_selector`).
    pub examples: Option<Vec<HashMap<String, Value>>>,
    /// Dynamic example selector (mutually exclusive with `examples`).
    pub example_selector: Option<Arc<dyn BaseExampleSelector>>,
    /// Template for formatting each example.
    pub example_prompt: PromptTemplate,
    /// Optional prefix template rendered before examples.
    pub prefix: Option<PromptTemplate>,
    /// Suffix template rendered after examples.
    pub suffix: PromptTemplate,
    /// Separator between examples.
    pub example_separator: String,
    /// Input variables that the prompt expects.
    pub input_variables: Vec<String>,
}

impl FewShotPromptWithTemplates {
    /// Create a new few-shot prompt with templates.
    ///
    /// The suffix is required; prefix is optional.
    pub fn new(example_prompt: PromptTemplate, suffix: PromptTemplate) -> Self {
        let input_variables = suffix.input_variables.clone();
        Self {
            examples: None,
            example_selector: None,
            example_prompt,
            prefix: None,
            suffix,
            example_separator: "\n\n".to_string(),
            input_variables,
        }
    }

    /// Set static examples.
    pub fn with_examples(mut self, examples: Vec<HashMap<String, Value>>) -> Self {
        self.examples = Some(examples);
        self
    }

    /// Set a dynamic example selector.
    pub fn with_example_selector(mut self, selector: Arc<dyn BaseExampleSelector>) -> Self {
        self.example_selector = Some(selector);
        self
    }

    /// Set a prefix template.
    pub fn with_prefix(mut self, prefix: PromptTemplate) -> Self {
        // Merge prefix input variables
        for var in &prefix.input_variables {
            if !self.input_variables.contains(var) {
                self.input_variables.push(var.clone());
            }
        }
        self.prefix = Some(prefix);
        self
    }

    /// Set the separator between examples.
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.example_separator = sep.into();
        self
    }

    /// Get examples, either from static list or from selector.
    async fn get_examples(
        &self,
        kwargs: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>> {
        if let Some(ref examples) = self.examples {
            Ok(examples.clone())
        } else if let Some(ref selector) = self.example_selector {
            selector.select_examples(kwargs).await
        } else {
            Err(RustChainError::Other(
                "No examples or example_selector provided".into(),
            ))
        }
    }

    /// Format the prompt synchronously (requires static examples).
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<String> {
        let examples = self.examples.as_ref().ok_or_else(|| {
            RustChainError::Other(
                "Use format_async for FewShotPromptWithTemplates with example_selector".into(),
            )
        })?;
        self.format_with_examples(examples, kwargs)
    }

    /// Format the prompt with async example selection support.
    pub async fn format_async(&self, kwargs: &HashMap<String, Value>) -> Result<String> {
        let examples = self.get_examples(kwargs).await?;
        self.format_with_examples(&examples, kwargs)
    }

    fn format_with_examples(
        &self,
        examples: &[HashMap<String, Value>],
        kwargs: &HashMap<String, Value>,
    ) -> Result<String> {
        let mut pieces = Vec::new();

        // Format prefix template
        if let Some(ref prefix) = self.prefix {
            pieces.push(prefix.format(kwargs)?);
        }

        // Format each example
        for example in examples {
            pieces.push(self.example_prompt.format(example)?);
        }

        // Format suffix template
        pieces.push(self.suffix.format(kwargs)?);

        Ok(pieces.join(&self.example_separator))
    }
}

#[async_trait]
impl Runnable for FewShotPromptWithTemplates {
    fn name(&self) -> &str {
        "FewShotPromptWithTemplates"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example_prompt() -> PromptTemplate {
        PromptTemplate::from_template("Q: {question}\nA: {answer}")
    }

    fn make_examples() -> Vec<HashMap<String, Value>> {
        vec![
            HashMap::from([
                ("question".into(), Value::String("What is 2+2?".into())),
                ("answer".into(), Value::String("4".into())),
            ]),
            HashMap::from([
                ("question".into(), Value::String("What is 3+3?".into())),
                ("answer".into(), Value::String("6".into())),
            ]),
        ]
    }

    #[test]
    fn test_format_with_prefix_and_suffix_templates() {
        let prefix = PromptTemplate::from_template("Answer math questions as a {role}.");
        let suffix = PromptTemplate::from_template("Q: {input}\nA:");
        let example_prompt = make_example_prompt();

        let prompt = FewShotPromptWithTemplates::new(example_prompt, suffix)
            .with_prefix(prefix)
            .with_examples(make_examples());

        let kwargs = HashMap::from([
            ("role".into(), Value::String("teacher".into())),
            ("input".into(), Value::String("What is 4+4?".into())),
        ]);

        let result = prompt.format(&kwargs).unwrap();
        assert!(result.starts_with("Answer math questions as a teacher."));
        assert!(result.contains("Q: What is 2+2?\nA: 4"));
        assert!(result.contains("Q: What is 3+3?\nA: 6"));
        assert!(result.ends_with("Q: What is 4+4?\nA:"));
    }

    #[test]
    fn test_format_without_prefix() {
        let suffix = PromptTemplate::from_template("Q: {input}\nA:");
        let example_prompt = make_example_prompt();

        let prompt =
            FewShotPromptWithTemplates::new(example_prompt, suffix).with_examples(make_examples());

        let kwargs = HashMap::from([("input".into(), Value::String("What is 5+5?".into()))]);

        let result = prompt.format(&kwargs).unwrap();
        assert!(result.starts_with("Q: What is 2+2?"));
        assert!(result.ends_with("Q: What is 5+5?\nA:"));
    }

    #[test]
    fn test_custom_separator() {
        let suffix = PromptTemplate::from_template("Q: {input}\nA:");
        let example_prompt = make_example_prompt();

        let prompt = FewShotPromptWithTemplates::new(example_prompt, suffix)
            .with_examples(make_examples())
            .with_separator("\n---\n");

        let kwargs = HashMap::from([("input".into(), Value::String("test".into()))]);

        let result = prompt.format(&kwargs).unwrap();
        assert!(result.contains("\n---\n"));
    }

    #[test]
    fn test_input_variables_include_prefix_vars() {
        let prefix = PromptTemplate::from_template("Context: {context}");
        let suffix = PromptTemplate::from_template("{input}");
        let example_prompt = make_example_prompt();

        let prompt = FewShotPromptWithTemplates::new(example_prompt, suffix).with_prefix(prefix);

        assert!(prompt.input_variables.contains(&"input".to_string()));
        assert!(prompt.input_variables.contains(&"context".to_string()));
    }

    #[test]
    fn test_format_no_examples_or_selector_errors() {
        let suffix = PromptTemplate::from_template("{input}");
        let example_prompt = make_example_prompt();

        let prompt = FewShotPromptWithTemplates::new(example_prompt, suffix);
        let kwargs = HashMap::from([("input".into(), Value::String("test".into()))]);
        assert!(prompt.format(&kwargs).is_err());
    }

    #[tokio::test]
    async fn test_invoke() {
        let suffix = PromptTemplate::from_template("Q: {input}\nA:");
        let example_prompt = make_example_prompt();

        let prompt =
            FewShotPromptWithTemplates::new(example_prompt, suffix).with_examples(make_examples());

        let input = serde_json::json!({"input": "What is 7+7?"});
        let result = prompt.invoke(input, None).await.unwrap();

        let text = result.as_str().unwrap();
        assert!(text.contains("Q: What is 2+2?"));
        assert!(text.contains("Q: What is 7+7?"));
    }

    #[tokio::test]
    async fn test_invoke_rejects_non_object() {
        let suffix = PromptTemplate::from_template("{input}");
        let example_prompt = make_example_prompt();

        let prompt = FewShotPromptWithTemplates::new(example_prompt, suffix).with_examples(vec![]);

        let result = prompt.invoke(Value::String("bad".into()), None).await;
        assert!(result.is_err());
    }
}
