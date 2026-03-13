//! Structured prompt template for language models.
//!
//! Mirrors Python `langchain_core.prompts.structured`.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::string_formatter::{format_template, get_template_variables, TemplateFormat};

/// A prompt template that includes a schema for structured output.
///
/// When used with a language model that supports structured output,
/// the schema is automatically applied to constrain the model's response.
pub struct StructuredPrompt {
    /// The underlying chat prompt messages as template strings.
    pub messages: Vec<(String, String)>, // (role, template)
    /// The schema for structured output (JSON Schema or type descriptor).
    pub schema: Value,
    /// Additional kwargs for structured output configuration.
    pub structured_output_kwargs: HashMap<String, Value>,
    /// Input variables extracted from templates.
    pub input_variables: Vec<String>,
}

impl StructuredPrompt {
    /// Create a new structured prompt from message tuples and a schema.
    ///
    /// Input variables are automatically extracted from the message templates.
    pub fn new(messages: Vec<(String, String)>, schema: Value) -> Self {
        let mut seen = std::collections::HashSet::new();
        let input_variables: Vec<String> = messages
            .iter()
            .flat_map(|(_, tmpl)| get_template_variables(tmpl, TemplateFormat::FString))
            .filter(|v| seen.insert(v.clone()))
            .collect();

        Self {
            messages,
            schema,
            structured_output_kwargs: HashMap::new(),
            input_variables,
        }
    }

    /// Add additional keyword arguments for structured output configuration.
    pub fn with_kwargs(mut self, kwargs: HashMap<String, Value>) -> Self {
        self.structured_output_kwargs = kwargs;
        self
    }

    /// Format the messages with the given variables.
    pub fn format(&self, kwargs: &HashMap<String, Value>) -> Result<Vec<(String, String)>> {
        self.messages
            .iter()
            .map(|(role, template)| {
                let formatted = format_template(template, TemplateFormat::FString, kwargs)?;
                Ok((role.clone(), formatted))
            })
            .collect()
    }

    /// Get the schema for structured output.
    pub fn get_schema(&self) -> &Value {
        &self.schema
    }
}

#[async_trait]
impl Runnable for StructuredPrompt {
    fn name(&self) -> &str {
        "StructuredPrompt"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let kwargs: HashMap<String, Value> = match input {
            Value::Object(map) => map.into_iter().collect(),
            _ => {
                return Err(CognisError::TypeMismatch {
                    expected: "Object".into(),
                    got: "non-Object".into(),
                });
            }
        };
        let formatted = self.format(&kwargs)?;
        Ok(serde_json::json!({
            "messages": formatted.iter().map(|(role, content)| {
                serde_json::json!({"role": role, "content": content})
            }).collect::<Vec<_>>(),
            "schema": self.schema,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_extracts_variables() {
        let prompt = StructuredPrompt::new(
            vec![
                ("system".into(), "You are a {role}.".into()),
                ("human".into(), "Extract info from: {text}".into()),
            ],
            serde_json::json!({"type": "object"}),
        );
        assert!(prompt.input_variables.contains(&"role".to_string()));
        assert!(prompt.input_variables.contains(&"text".to_string()));
        assert_eq!(prompt.input_variables.len(), 2);
    }

    #[test]
    fn test_new_deduplicates_variables() {
        let prompt = StructuredPrompt::new(
            vec![
                ("system".into(), "Hello {name}".into()),
                ("human".into(), "Goodbye {name}".into()),
            ],
            serde_json::json!({}),
        );
        assert_eq!(prompt.input_variables, vec!["name".to_string()]);
    }

    #[test]
    fn test_format_messages() {
        let prompt = StructuredPrompt::new(
            vec![
                ("system".into(), "You are a {role}.".into()),
                ("human".into(), "Parse: {text}".into()),
            ],
            serde_json::json!({"type": "object"}),
        );
        let mut kwargs = HashMap::new();
        kwargs.insert("role".into(), Value::String("parser".into()));
        kwargs.insert("text".into(), Value::String("hello world".into()));

        let result = prompt.format(&kwargs).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("system".into(), "You are a parser.".into()));
        assert_eq!(result[1], ("human".into(), "Parse: hello world".into()));
    }

    #[test]
    fn test_format_missing_variable() {
        let prompt = StructuredPrompt::new(
            vec![("human".into(), "Hello {name}".into())],
            serde_json::json!({}),
        );
        let kwargs = HashMap::new();
        assert!(prompt.format(&kwargs).is_err());
    }

    #[test]
    fn test_get_schema() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let prompt = StructuredPrompt::new(vec![], schema.clone());
        assert_eq!(prompt.get_schema(), &schema);
    }

    #[test]
    fn test_with_kwargs() {
        let prompt = StructuredPrompt::new(vec![], serde_json::json!({})).with_kwargs(
            HashMap::from([("method".into(), Value::String("json_mode".into()))]),
        );
        assert_eq!(
            prompt.structured_output_kwargs.get("method"),
            Some(&Value::String("json_mode".into()))
        );
    }

    #[tokio::test]
    async fn test_invoke_returns_messages_and_schema() {
        let schema = serde_json::json!({"type": "object"});
        let prompt = StructuredPrompt::new(
            vec![("human".into(), "Extract from: {text}".into())],
            schema.clone(),
        );
        let input = serde_json::json!({"text": "some data"});
        let result = prompt.invoke(input, None).await.unwrap();

        assert_eq!(result["schema"], schema);
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "human");
        assert_eq!(messages[0]["content"], "Extract from: some data");
    }

    #[tokio::test]
    async fn test_invoke_rejects_non_object() {
        let prompt = StructuredPrompt::new(vec![], serde_json::json!({}));
        let result = prompt.invoke(Value::String("bad".into()), None).await;
        assert!(result.is_err());
    }
}
