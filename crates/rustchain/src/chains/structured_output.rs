use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::output_parsers::base::OutputParser;
use rustchain_core::output_parsers::json::JsonOutputParser;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

/// A chain that wraps a chat model and a JSON output parser to produce
/// structured (typed) output conforming to a given JSON schema.
///
/// The chain:
/// 1. Formats the user's prompt template with input variables
/// 2. Appends format instructions derived from the JSON schema
/// 3. Sends the combined prompt to the chat model
/// 4. Parses the model's text response through [`JsonOutputParser`]
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use serde_json::json;
/// use rustchain::chains::structured_output::StructuredOutputChain;
/// use rustchain_core::language_models::fake::FakeListChatModel;
/// use rustchain_core::runnables::base::Runnable;
///
/// # async fn example() {
/// let model = Arc::new(FakeListChatModel::new(vec![
///     r#"{"name": "Alice", "age": 30}"#.to_string(),
/// ]));
///
/// let schema = json!({
///     "type": "object",
///     "properties": {
///         "name": { "type": "string" },
///         "age": { "type": "integer" }
///     },
///     "required": ["name", "age"]
/// });
///
/// let chain = StructuredOutputChain::builder()
///     .model(model)
///     .schema(schema)
///     .prompt("Extract person info from: {text}")
///     .build();
///
/// let result = chain
///     .invoke(json!({"text": "Alice is 30 years old"}), None)
///     .await
///     .unwrap();
///
/// assert_eq!(result["name"], "Alice");
/// assert_eq!(result["age"], 30);
/// # }
/// ```
pub struct StructuredOutputChain {
    model: Arc<dyn BaseChatModel>,
    parser: JsonOutputParser,
    prompt_template: String,
    output_key: Option<String>,
}

/// Builder for [`StructuredOutputChain`].
pub struct StructuredOutputChainBuilder {
    model: Option<Arc<dyn BaseChatModel>>,
    schema: Option<Value>,
    prompt_template: Option<String>,
    output_key: Option<String>,
}

impl StructuredOutputChainBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            model: None,
            schema: None,
            prompt_template: None,
            output_key: None,
        }
    }

    /// Set the chat model (required).
    pub fn model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the JSON schema describing the expected output shape (required).
    ///
    /// The schema is used both to generate format instructions appended to
    /// the prompt and to validate the model's response structure.
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the prompt template with `{variable}` placeholders (required).
    ///
    /// Format instructions will be automatically appended after the template
    /// text when invoking the chain.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt_template = Some(prompt.into());
        self
    }

    /// Set an optional output key. When set, the parsed JSON is wrapped in
    /// `{ output_key: <parsed> }`. When `None` (the default), the parsed
    /// JSON value is returned directly.
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = Some(key.into());
        self
    }

    /// Build the [`StructuredOutputChain`].
    ///
    /// # Panics
    ///
    /// Panics if `model`, `schema`, or `prompt` have not been set.
    pub fn build(self) -> StructuredOutputChain {
        let schema = self
            .schema
            .expect("schema is required for StructuredOutputChain");
        StructuredOutputChain {
            model: self
                .model
                .expect("model is required for StructuredOutputChain"),
            parser: JsonOutputParser::with_schema(schema),
            prompt_template: self
                .prompt_template
                .expect("prompt is required for StructuredOutputChain"),
            output_key: self.output_key,
        }
    }
}

impl Default for StructuredOutputChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuredOutputChain {
    /// Create a new builder.
    pub fn builder() -> StructuredOutputChainBuilder {
        StructuredOutputChainBuilder::new()
    }

    /// Return the JSON schema used for this chain.
    pub fn schema(&self) -> Option<&Value> {
        self.parser.schema.as_ref()
    }

    /// Return the format instructions that will be appended to the prompt.
    pub fn format_instructions(&self) -> Option<String> {
        self.parser.get_format_instructions()
    }

    /// Format the prompt template by replacing `{variable}` placeholders with
    /// values from the input JSON object, then append format instructions.
    fn format_prompt(&self, input: &Value) -> Result<String> {
        let re = Regex::new(r"\{(\w+)\}").unwrap();
        let obj = input
            .as_object()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "JSON object".into(),
                got: format!("{}", input),
            })?;

        let mut missing: Vec<String> = Vec::new();
        let result = re.replace_all(&self.prompt_template, |caps: &regex::Captures| {
            let key = &caps[1];
            match obj.get(key) {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => {
                    missing.push(key.to_string());
                    String::new()
                }
            }
        });

        if !missing.is_empty() {
            return Err(RustChainError::InvalidKey(format!(
                "Missing input variable(s): {}",
                missing.join(", ")
            )));
        }

        let mut prompt = result.into_owned();

        // Append format instructions from the parser
        if let Some(instructions) = self.parser.get_format_instructions() {
            prompt.push_str("\n\n");
            prompt.push_str(&instructions);
        }

        Ok(prompt)
    }
}

#[async_trait]
impl Runnable for StructuredOutputChain {
    fn name(&self) -> &str {
        "StructuredOutputChain"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        // 1. Format prompt with variables and schema instructions
        let formatted = self.format_prompt(&input)?;

        // 2. Send to the chat model
        let messages = vec![Message::Human(HumanMessage::new(&formatted))];
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        let text = ai_msg.base.content.text();

        // 3. Parse through JSON output parser
        let parsed = self.parser.parse(&text)?;

        // 4. Wrap in output key if specified
        match &self.output_key {
            Some(key) => Ok(json!({ key: parsed })),
            None => Ok(parsed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::{FakeListChatModel, GenericFakeChatModel};
    use rustchain_core::messages::AIMessage;

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn person_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        })
    }

    #[tokio::test]
    async fn test_basic_structured_output() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "Alice", "age": 30}"#]))
            .schema(person_schema())
            .prompt("Extract person info from: {text}")
            .build();

        let result = chain
            .invoke(json!({"text": "Alice is 30 years old"}), None)
            .await
            .unwrap();

        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[tokio::test]
    async fn test_structured_output_with_output_key() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "Bob", "age": 25}"#]))
            .schema(person_schema())
            .prompt("Extract: {text}")
            .output_key("person")
            .build();

        let result = chain
            .invoke(json!({"text": "Bob is 25"}), None)
            .await
            .unwrap();

        assert_eq!(result["person"]["name"], "Bob");
        assert_eq!(result["person"]["age"], 25);
    }

    #[tokio::test]
    async fn test_structured_output_with_markdown_fences() {
        let response = "```json\n{\"name\": \"Carol\", \"age\": 40}\n```";
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![response]))
            .schema(person_schema())
            .prompt("Extract: {text}")
            .build();

        let result = chain
            .invoke(json!({"text": "Carol is 40"}), None)
            .await
            .unwrap();

        assert_eq!(result["name"], "Carol");
        assert_eq!(result["age"], 40);
    }

    #[tokio::test]
    async fn test_structured_output_invalid_json_error() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec!["this is not json"]))
            .schema(person_schema())
            .prompt("Extract: {text}")
            .build();

        let result = chain.invoke(json!({"text": "something"}), None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Failed to parse JSON"),
            "Expected JSON parse error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_structured_output_missing_variable() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "X", "age": 1}"#]))
            .schema(person_schema())
            .prompt("Extract from {text} in {language}")
            .build();

        let result = chain.invoke(json!({"text": "something"}), None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("language"),
            "Error should mention missing variable: {err}"
        );
    }

    #[tokio::test]
    async fn test_structured_output_complex_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string" },
                        "confidence": { "type": "number" }
                    }
                }
            },
            "required": ["title", "tags"]
        });

        let response = r#"{"title": "Rust Guide", "tags": ["rust", "programming"], "metadata": {"source": "web", "confidence": 0.95}}"#;
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![response]))
            .schema(schema)
            .prompt("Analyze: {input}")
            .build();

        let result = chain
            .invoke(json!({"input": "A guide about Rust programming"}), None)
            .await
            .unwrap();

        assert_eq!(result["title"], "Rust Guide");
        assert_eq!(result["tags"][0], "rust");
        assert_eq!(result["tags"][1], "programming");
        assert_eq!(result["metadata"]["confidence"], 0.95);
    }

    #[tokio::test]
    async fn test_structured_output_with_generic_fake_model() {
        let model = Arc::new(GenericFakeChatModel::from_messages(vec![AIMessage::new(
            r#"{"name": "Dave", "age": 35}"#,
        )]));

        let chain = StructuredOutputChain::builder()
            .model(model)
            .schema(person_schema())
            .prompt("Extract: {text}")
            .build();

        let result = chain
            .invoke(json!({"text": "Dave is 35"}), None)
            .await
            .unwrap();

        assert_eq!(result["name"], "Dave");
        assert_eq!(result["age"], 35);
    }

    #[tokio::test]
    async fn test_structured_output_as_runnable() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "Eve", "age": 28}"#]))
            .schema(person_schema())
            .prompt("Extract: {text}")
            .build();

        let runnable: &dyn Runnable = &chain;
        assert_eq!(runnable.name(), "StructuredOutputChain");

        let result = runnable
            .invoke(json!({"text": "Eve is 28"}), None)
            .await
            .unwrap();

        assert_eq!(result["name"], "Eve");
        assert_eq!(result["age"], 28);
    }

    #[tokio::test]
    async fn test_format_instructions_included() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "X", "age": 1}"#]))
            .schema(person_schema())
            .prompt("Extract: {text}")
            .build();

        let instructions = chain.format_instructions();
        assert!(instructions.is_some());
        let instructions = instructions.unwrap();
        assert!(instructions.contains("JSON"));
        assert!(instructions.contains("schema"));
    }

    #[tokio::test]
    async fn test_schema_accessor() {
        let schema = person_schema();
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{}"#]))
            .schema(schema.clone())
            .prompt("test {x}")
            .build();

        assert_eq!(chain.schema(), Some(&schema));
    }

    #[tokio::test]
    async fn test_structured_output_multiple_variables() {
        let chain = StructuredOutputChain::builder()
            .model(fake_model(vec![r#"{"name": "Frank", "age": 50}"#]))
            .schema(person_schema())
            .prompt("Extract {entity_type} from: {text}")
            .build();

        let result = chain
            .invoke(
                json!({"entity_type": "person", "text": "Frank is 50 years old"}),
                None,
            )
            .await
            .unwrap();

        assert_eq!(result["name"], "Frank");
        assert_eq!(result["age"], 50);
    }
}
