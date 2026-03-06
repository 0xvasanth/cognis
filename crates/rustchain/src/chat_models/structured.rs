//! Structured output chat model wrapper.
//!
//! Provides [`StructuredOutputChatModel`], a wrapper around any [`BaseChatModel`]
//! that extracts structured JSON output from the model's response, either via
//! tool calling or JSON mode.

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::{AIMessage, Message};
use rustchain_core::outputs::{ChatGeneration, ChatResult};
use rustchain_core::tools::ToolSchema;

/// The method used to enforce structured output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredOutputMethod {
    /// Use tool calling: bind a tool whose schema matches the output, then
    /// extract the tool call arguments as the structured result.
    ToolCalling,
    /// Use JSON mode (provider-native). Currently supported by OpenAI.
    JsonMode,
}

impl StructuredOutputMethod {
    /// Parse from a string, defaulting to `ToolCalling`.
    pub fn from_str_or_default(s: Option<&str>) -> Self {
        match s {
            Some("json_mode") => Self::JsonMode,
            _ => Self::ToolCalling,
        }
    }
}

/// A chat model wrapper that enforces structured JSON output.
///
/// For `ToolCalling` method:
///   - Converts the schema into a [`ToolSchema`], binds it on the inner model
///     using `bind_tools`, and forces the model to call that tool.
///   - On `_generate`, invokes the bound model and extracts the tool call
///     arguments as the structured output, placing the JSON string in the
///     AIMessage content.
///
/// For `JsonMode` method:
///   - Delegates to the inner model with a system instruction to respond in
///     JSON matching the provided schema. Provider-specific `response_format`
///     support can be added as needed.
pub struct StructuredOutputChatModel {
    /// The inner model, optionally with tools already bound.
    inner: Box<dyn BaseChatModel>,
    /// The JSON schema describing the expected output structure.
    schema: Value,
    /// The method used to extract structured output.
    method: StructuredOutputMethod,
    /// The tool name used when method is ToolCalling.
    tool_name: String,
    /// Whether to include the raw AI message alongside the parsed output.
    include_raw: bool,
}

impl StructuredOutputChatModel {
    /// Extract the structured JSON from a `ChatResult` produced by a
    /// tool-calling invocation.
    fn extract_tool_call_output(&self, result: &ChatResult) -> Result<Value> {
        let gen = result
            .generations
            .first()
            .ok_or_else(|| RustChainError::Other("No generations returned".into()))?;

        let ai_msg = match &gen.message {
            Message::Ai(ai) => ai,
            _ => {
                return Err(RustChainError::Other(
                    "Expected AIMessage in generation".into(),
                ))
            }
        };

        // Find the tool call matching our schema tool name
        for tc in &ai_msg.tool_calls {
            if tc.name == self.tool_name {
                return serde_json::to_value(&tc.args).map_err(|e| {
                    RustChainError::Other(format!("Failed to serialize tool call args: {}", e))
                });
            }
        }

        Err(RustChainError::Other(format!(
            "No tool call found with name '{}'. Tool calls present: {:?}",
            self.tool_name,
            ai_msg
                .tool_calls
                .iter()
                .map(|tc| &tc.name)
                .collect::<Vec<_>>()
        )))
    }
}

#[async_trait]
impl BaseChatModel for StructuredOutputChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        match self.method {
            StructuredOutputMethod::ToolCalling => {
                let result = self.inner._generate(messages, stop).await?;
                let structured_output = self.extract_tool_call_output(&result)?;

                // Build a new AIMessage with the structured JSON as content
                let json_string = serde_json::to_string(&structured_output).map_err(|e| {
                    RustChainError::Other(format!("JSON serialization error: {}", e))
                })?;

                let mut ai_message = AIMessage::new(&json_string);

                // Preserve usage metadata and id from the original message
                if let Some(gen) = result.generations.first() {
                    if let Message::Ai(ref original) = gen.message {
                        ai_message.usage_metadata = original.usage_metadata.clone();
                        ai_message.base.id = original.base.id.clone();
                        if self.include_raw {
                            ai_message.tool_calls = original.tool_calls.clone();
                        }
                    }
                }

                let generation = ChatGeneration::new(ai_message);
                Ok(ChatResult {
                    generations: vec![generation],
                    llm_output: result.llm_output,
                })
            }
            StructuredOutputMethod::JsonMode => {
                // For JSON mode, prepend a system message instructing JSON output
                let schema_str = serde_json::to_string_pretty(&self.schema)
                    .unwrap_or_else(|_| self.schema.to_string());
                let system_instruction = format!(
                    "Respond with valid JSON matching this schema:\n{}",
                    schema_str
                );

                let mut augmented_messages = vec![Message::System(
                    rustchain_core::messages::SystemMessage::new(&system_instruction),
                )];
                augmented_messages.extend_from_slice(messages);

                let result = self.inner._generate(&augmented_messages, stop).await?;

                // Validate that the response content is valid JSON
                if let Some(gen) = result.generations.first() {
                    if let Message::Ai(ref ai) = gen.message {
                        let content = ai.base.content.text();
                        if !content.is_empty() {
                            // Try to parse to validate it's JSON
                            let _: Value = serde_json::from_str(&content).map_err(|e| {
                                RustChainError::Other(format!(
                                    "Model response is not valid JSON: {}. Content: {}",
                                    e, content
                                ))
                            })?;
                        }
                    }
                }

                Ok(result)
            }
        }
    }

    fn llm_type(&self) -> &str {
        "structured_output"
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        // Streaming with structured output is complex; delegate to inner model
        // for now. Full structured streaming would require accumulating chunks.
        self.inner._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        self.inner.bind_tools(tools, tool_choice)
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

/// Create a [`StructuredOutputChatModel`] wrapping the given model.
///
/// This is the primary entry point for structured output. It takes a model,
/// binds a tool with the given schema (for `tool_calling` method), and returns
/// a new model that extracts structured JSON from the response.
///
/// # Arguments
/// * `model` - The inner chat model to wrap.
/// * `schema` - JSON Schema describing the expected output structure.
///   Must contain a `"title"` field (used as the tool name) or a default
///   name `"structured_output"` will be used.
/// * `method` - `"tool_calling"` (default) or `"json_mode"`.
/// * `include_raw` - Whether to preserve original tool calls on the output message.
///
/// # Example
/// ```ignore
/// let structured = with_structured_output(
///     model,
///     json!({
///         "title": "Person",
///         "type": "object",
///         "properties": {
///             "name": {"type": "string"},
///             "age": {"type": "integer"}
///         },
///         "required": ["name", "age"]
///     }),
///     Some("tool_calling"),
///     false,
/// )?;
/// ```
pub fn with_structured_output(
    model: Box<dyn BaseChatModel>,
    schema: Value,
    method: Option<&str>,
    include_raw: bool,
) -> Result<Box<dyn BaseChatModel>> {
    let method = StructuredOutputMethod::from_str_or_default(method);
    let tool_name = schema
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("structured_output")
        .to_string();

    let description = schema
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("Structured output tool")
        .to_string();

    let inner = match method {
        StructuredOutputMethod::ToolCalling => {
            let tool_schema = ToolSchema {
                name: tool_name.clone(),
                description,
                parameters: Some(schema.clone()),
                extras: None,
            };
            // Bind the tool and force the model to use it
            model.bind_tools(&[tool_schema], Some(ToolChoice::Tool(tool_name.clone())))?
        }
        StructuredOutputMethod::JsonMode => {
            // For JSON mode, don't bind tools — just pass through
            model
        }
    };

    Ok(Box::new(StructuredOutputChatModel {
        inner,
        schema,
        method,
        tool_name,
        include_raw,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{HumanMessage, ToolCall};
    use serde_json::json;
    use std::collections::HashMap;

    /// A mock chat model that returns a predefined tool call response.
    struct MockToolCallModel {
        tool_calls: Vec<ToolCall>,
        bound_tools: Vec<ToolSchema>,
        tool_choice: Option<ToolChoice>,
    }

    impl MockToolCallModel {
        fn new(tool_calls: Vec<ToolCall>) -> Self {
            Self {
                tool_calls,
                bound_tools: Vec::new(),
                tool_choice: None,
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for MockToolCallModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let mut ai_message = AIMessage::new("");
            ai_message.tool_calls = self.tool_calls.clone();
            let generation = ChatGeneration::new(ai_message);
            Ok(ChatResult {
                generations: vec![generation],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "mock"
        }

        fn bind_tools(
            &self,
            tools: &[ToolSchema],
            tool_choice: Option<ToolChoice>,
        ) -> Result<Box<dyn BaseChatModel>> {
            Ok(Box::new(MockToolCallModel {
                tool_calls: self.tool_calls.clone(),
                bound_tools: tools.to_vec(),
                tool_choice,
            }))
        }
    }

    /// A mock chat model that returns plain text content (for json_mode tests).
    struct MockTextModel {
        content: String,
    }

    impl MockTextModel {
        fn new(content: &str) -> Self {
            Self {
                content: content.to_string(),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for MockTextModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let ai_message = AIMessage::new(&self.content);
            let generation = ChatGeneration::new(ai_message);
            Ok(ChatResult {
                generations: vec![generation],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "mock_text"
        }
    }

    #[tokio::test]
    async fn test_structured_output_tool_calling_extracts_args() {
        let mut args = HashMap::new();
        args.insert("name".to_string(), json!("Alice"));
        args.insert("age".to_string(), json!(30));

        let mock = MockToolCallModel::new(vec![ToolCall {
            name: "Person".to_string(),
            args,
            id: Some("call_1".to_string()),
        }]);

        let schema = json!({
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let structured =
            with_structured_output(Box::new(mock), schema, Some("tool_calling"), false).unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Who is Alice?"))];
        let result = structured._generate(&messages, None).await.unwrap();

        assert_eq!(result.generations.len(), 1);

        if let Message::Ai(ref ai) = result.generations[0].message {
            let parsed: Value = serde_json::from_str(&ai.base.content.text()).unwrap();
            assert_eq!(parsed["name"], "Alice");
            assert_eq!(parsed["age"], 30);
            // include_raw is false, so tool_calls should be empty
            assert!(ai.tool_calls.is_empty());
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[tokio::test]
    async fn test_structured_output_tool_calling_with_include_raw() {
        let mut args = HashMap::new();
        args.insert("city".to_string(), json!("Paris"));

        let mock = MockToolCallModel::new(vec![ToolCall {
            name: "Location".to_string(),
            args,
            id: Some("call_2".to_string()),
        }]);

        let schema = json!({
            "title": "Location",
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        });

        let structured = with_structured_output(
            Box::new(mock),
            schema,
            Some("tool_calling"),
            true, // include_raw
        )
        .unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Where?"))];
        let result = structured._generate(&messages, None).await.unwrap();

        if let Message::Ai(ref ai) = result.generations[0].message {
            let parsed: Value = serde_json::from_str(&ai.base.content.text()).unwrap();
            assert_eq!(parsed["city"], "Paris");
            // include_raw is true, so original tool_calls should be preserved
            assert_eq!(ai.tool_calls.len(), 1);
            assert_eq!(ai.tool_calls[0].name, "Location");
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[tokio::test]
    async fn test_structured_output_json_mode() {
        let mock = MockTextModel::new(r#"{"name": "Bob", "age": 25}"#);

        let schema = json!({
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });

        let structured =
            with_structured_output(Box::new(mock), schema, Some("json_mode"), false).unwrap();

        let messages = vec![Message::Human(HumanMessage::new("Tell me about Bob"))];
        let result = structured._generate(&messages, None).await.unwrap();

        if let Message::Ai(ref ai) = result.generations[0].message {
            let parsed: Value = serde_json::from_str(&ai.base.content.text()).unwrap();
            assert_eq!(parsed["name"], "Bob");
            assert_eq!(parsed["age"], 25);
        } else {
            panic!("Expected AIMessage");
        }
    }

    #[tokio::test]
    async fn test_structured_output_no_matching_tool_call_errors() {
        let mut args = HashMap::new();
        args.insert("x".to_string(), json!(1));

        let mock = MockToolCallModel::new(vec![ToolCall {
            name: "WrongTool".to_string(),
            args,
            id: None,
        }]);

        let schema = json!({
            "title": "ExpectedTool",
            "type": "object",
            "properties": {"x": {"type": "integer"}}
        });

        let structured =
            with_structured_output(Box::new(mock), schema, Some("tool_calling"), false).unwrap();

        let messages = vec![Message::Human(HumanMessage::new("test"))];
        let result = structured._generate(&messages, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No tool call found with name 'ExpectedTool'"));
    }

    #[tokio::test]
    async fn test_structured_output_json_mode_invalid_json_errors() {
        let mock = MockTextModel::new("this is not json");

        let schema = json!({"title": "Test", "type": "object"});

        let structured =
            with_structured_output(Box::new(mock), schema, Some("json_mode"), false).unwrap();

        let messages = vec![Message::Human(HumanMessage::new("test"))];
        let result = structured._generate(&messages, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn test_structured_output_method_parsing() {
        assert_eq!(
            StructuredOutputMethod::from_str_or_default(None),
            StructuredOutputMethod::ToolCalling
        );
        assert_eq!(
            StructuredOutputMethod::from_str_or_default(Some("tool_calling")),
            StructuredOutputMethod::ToolCalling
        );
        assert_eq!(
            StructuredOutputMethod::from_str_or_default(Some("json_mode")),
            StructuredOutputMethod::JsonMode
        );
    }

    #[test]
    fn test_with_structured_output_default_tool_name() {
        let mock = MockToolCallModel::new(vec![]);
        let schema = json!({"type": "object"}); // no "title" field

        let result = with_structured_output(Box::new(mock), schema, Some("tool_calling"), false);

        // Should succeed, using default name "structured_output"
        assert!(result.is_ok());
    }

    #[test]
    fn test_llm_type_returns_structured_output() {
        let mock = MockTextModel::new("");
        let schema = json!({"title": "Test", "type": "object"});

        let structured =
            with_structured_output(Box::new(mock), schema, Some("json_mode"), false).unwrap();

        assert_eq!(structured.llm_type(), "structured_output");
    }
}
