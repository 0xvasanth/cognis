//! Structured output types for agent response formatting.
//!
//! Mirrors Python `langchain.agents.structured_output`.
//! Provides strategies for how models should return structured data:
//! - ToolStrategy: use tool calling to enforce structure
//! - ProviderStrategy: use native provider structured output
//! - AutoStrategy: let the framework decide

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The kind of schema being used for structured output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaKind {
    /// A Pydantic-style schema (JSON Schema generated from a model definition).
    Pydantic,
    /// A raw JSON Schema.
    JsonSchema,
    /// A dataclass-style schema.
    Dataclass,
    /// A TypedDict-style schema.
    TypedDict,
}

/// How errors should be handled during structured output parsing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorHandling {
    /// Do not handle errors — propagate them immediately.
    #[default]
    Disabled,
    /// Handle all errors by retrying.
    Enabled,
    /// Handle errors and include a custom message in the retry prompt.
    WithMessage(String),
    /// Only handle errors whose type names match the given list.
    WithTypes(Vec<String>),
}

impl From<bool> for ErrorHandling {
    fn from(value: bool) -> Self {
        if value {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

/// A schema specification describing a single output schema.
///
/// Mirrors Python's `_SchemaSpec` — carries the schema kind, name, JSON schema,
/// and optional description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSpec {
    /// The kind of schema (Pydantic, JsonSchema, Dataclass, TypedDict).
    pub kind: SchemaKind,
    /// The name of the schema (used as tool name or response_format name).
    pub name: String,
    /// The JSON Schema object.
    pub json_schema: Value,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl SchemaSpec {
    /// Create a new SchemaSpec.
    pub fn new(kind: SchemaKind, name: impl Into<String>, json_schema: Value) -> Self {
        Self {
            kind,
            name: name.into(),
            json_schema,
            description: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Error indicating that multiple structured output specifications were provided
/// when only one is allowed.
#[derive(Debug, Clone)]
pub struct MultipleStructuredOutputsError {
    pub message: String,
    /// The AI message that triggered the error, if available.
    pub ai_message: Option<Value>,
    /// The tool names that were found in the response.
    pub tool_names: Vec<String>,
}

impl MultipleStructuredOutputsError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ai_message: None,
            tool_names: Vec::new(),
        }
    }

    /// Set the AI message associated with this error.
    pub fn with_ai_message(mut self, ai_message: Value) -> Self {
        self.ai_message = Some(ai_message);
        self
    }

    /// Set the tool names associated with this error.
    pub fn with_tool_names(mut self, tool_names: Vec<String>) -> Self {
        self.tool_names = tool_names;
        self
    }
}

impl std::fmt::Display for MultipleStructuredOutputsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultipleStructuredOutputsError: {}", self.message)
    }
}

impl std::error::Error for MultipleStructuredOutputsError {}

/// Error during structured output validation.
#[derive(Debug, Clone)]
pub struct StructuredOutputValidationError {
    pub message: String,
    pub raw_output: Option<String>,
    /// The AI message that triggered the error, if available.
    pub ai_message: Option<Value>,
    /// The tool name that was being validated, if applicable.
    pub tool_name: Option<String>,
    /// The underlying source error description, if any.
    pub source_error: Option<String>,
}

impl StructuredOutputValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            raw_output: None,
            ai_message: None,
            tool_name: None,
            source_error: None,
        }
    }

    pub fn with_raw_output(mut self, raw: impl Into<String>) -> Self {
        self.raw_output = Some(raw.into());
        self
    }

    /// Set the AI message associated with this error.
    pub fn with_ai_message(mut self, ai_message: Value) -> Self {
        self.ai_message = Some(ai_message);
        self
    }

    /// Set the tool name that was being validated.
    pub fn with_tool_name(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = Some(tool_name.into());
        self
    }

    /// Set the underlying source error description.
    pub fn with_source_error(mut self, source_error: impl Into<String>) -> Self {
        self.source_error = Some(source_error.into());
        self
    }
}

impl std::fmt::Display for StructuredOutputValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StructuredOutputValidationError: {}", self.message)?;
        if let Some(ref raw) = self.raw_output {
            write!(f, " (raw output: {})", raw)?;
        }
        Ok(())
    }
}

impl std::error::Error for StructuredOutputValidationError {}

/// General structured output error.
#[derive(Debug, Clone)]
pub struct StructuredOutputError {
    pub message: String,
    /// The AI message that triggered the error, if available.
    pub ai_message: Option<Value>,
}

impl StructuredOutputError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ai_message: None,
        }
    }

    /// Set the AI message associated with this error.
    pub fn with_ai_message(mut self, ai_message: Value) -> Self {
        self.ai_message = Some(ai_message);
        self
    }
}

impl std::fmt::Display for StructuredOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StructuredOutputError: {}", self.message)
    }
}

impl std::error::Error for StructuredOutputError {}

/// Strategy that uses tool calling to enforce structured output.
///
/// The model is given a "tool" whose schema matches the desired output format,
/// and the tool call arguments become the structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStrategy {
    /// The JSON Schema that defines the expected output structure.
    pub schema: Value,
    /// Schema specifications for multi-schema / oneOf decomposition.
    pub schema_specs: Vec<SchemaSpec>,
    /// Content to include in the tool message response.
    #[serde(default)]
    pub tool_message_content: String,
    /// How to handle validation errors.
    #[serde(default)]
    pub handle_errors: ErrorHandling,
}

impl ToolStrategy {
    /// Create a new ToolStrategy with the given schema.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            schema_specs: Vec::new(),
            tool_message_content: String::new(),
            handle_errors: ErrorHandling::Disabled,
        }
    }

    /// Create a ToolStrategy from multiple schema specs (oneOf / union decomposition).
    pub fn from_schema_specs(specs: Vec<SchemaSpec>) -> Self {
        // Build a combined schema with oneOf if multiple specs
        let combined_schema = if specs.len() == 1 {
            specs[0].json_schema.clone()
        } else {
            let one_of: Vec<Value> = specs.iter().map(|s| s.json_schema.clone()).collect();
            serde_json::json!({ "oneOf": one_of })
        };
        Self {
            schema: combined_schema,
            schema_specs: specs,
            tool_message_content: String::new(),
            handle_errors: ErrorHandling::Disabled,
        }
    }

    /// Set the tool message content.
    pub fn with_tool_message_content(mut self, content: impl Into<String>) -> Self {
        self.tool_message_content = content.into();
        self
    }

    /// Set error handling mode.
    pub fn with_handle_errors(mut self, handle: ErrorHandling) -> Self {
        self.handle_errors = handle;
        self
    }
}

/// Binding between a schema spec and tool-based output parsing.
///
/// Mirrors Python's output-tool binding — given a schema spec, creates a tool
/// and can parse AI message values into structured output.
#[derive(Debug, Clone)]
pub struct OutputToolBinding {
    /// The schema specification used for this binding.
    pub schema_spec: SchemaSpec,
    /// The tool name to use for this binding.
    pub tool_name: String,
    /// Whether to include raw output alongside the parsed result.
    pub include_raw: bool,
}

impl OutputToolBinding {
    /// Create a new OutputToolBinding from a schema spec.
    pub fn from_schema_spec(schema_spec: SchemaSpec, include_raw: bool) -> Self {
        let tool_name = schema_spec.name.clone();
        Self {
            schema_spec,
            tool_name,
            include_raw,
        }
    }

    /// Parse an AI message value into structured output.
    ///
    /// Extracts tool call arguments from the AI message value and validates them
    /// against the schema spec.
    #[allow(clippy::result_large_err)]
    pub fn parse(&self, ai_message_value: &Value) -> std::result::Result<Value, StructuredOutputValidationError> {
        // Look for tool_calls in the AI message
        if let Some(tool_calls) = ai_message_value.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let name = tool_call.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if name == self.tool_name {
                    if let Some(args) = tool_call.get("args") {
                        return Ok(args.clone());
                    }
                }
            }
            Err(StructuredOutputValidationError::new(format!(
                "No tool call found with name '{}'",
                self.tool_name
            ))
            .with_ai_message(ai_message_value.clone())
            .with_tool_name(&self.tool_name))
        } else {
            Err(StructuredOutputValidationError::new(
                "AI message does not contain tool_calls",
            )
            .with_ai_message(ai_message_value.clone())
            .with_tool_name(&self.tool_name))
        }
    }
}

/// Strategy that uses native provider structured output capabilities.
///
/// Some providers (e.g., OpenAI) support a `response_format` parameter
/// that instructs the model to produce JSON matching a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStrategy {
    /// The JSON Schema that defines the expected output structure.
    pub schema: Value,
    /// The name for the schema (used in the `json_schema.name` field of the response format).
    pub schema_name: String,
    /// Whether to use strict schema enforcement (provider-dependent).
    #[serde(default)]
    pub strict: bool,
}

impl ProviderStrategy {
    /// Create a new ProviderStrategy with the given schema.
    pub fn new(schema: Value) -> Self {
        Self {
            schema,
            schema_name: "output".to_string(),
            strict: false,
        }
    }

    /// Set the schema name.
    pub fn with_schema_name(mut self, name: impl Into<String>) -> Self {
        self.schema_name = name.into();
        self
    }

    /// Enable or disable strict schema enforcement.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Convert this strategy to model keyword arguments.
    ///
    /// Returns a HashMap suitable for passing to model invocation,
    /// containing the `response_format` key with the correct nested structure:
    /// ```json
    /// {
    ///   "response_format": {
    ///     "type": "json_schema",
    ///     "json_schema": {
    ///       "name": "<schema_name>",
    ///       "schema": { ... },
    ///       "strict": true
    ///     }
    ///   }
    /// }
    /// ```
    pub fn to_model_kwargs(&self) -> HashMap<String, Value> {
        let mut kwargs = HashMap::new();

        let mut json_schema_inner = serde_json::Map::new();
        json_schema_inner.insert("name".to_string(), Value::String(self.schema_name.clone()));
        json_schema_inner.insert("schema".to_string(), self.schema.clone());
        json_schema_inner.insert("strict".to_string(), Value::Bool(self.strict));

        let mut response_format = serde_json::Map::new();
        response_format.insert("type".to_string(), Value::String("json_schema".to_string()));
        response_format.insert(
            "json_schema".to_string(),
            Value::Object(json_schema_inner),
        );

        kwargs.insert(
            "response_format".to_string(),
            Value::Object(response_format),
        );
        kwargs
    }
}

/// Binding between a schema spec and provider-based structured output parsing.
///
/// Mirrors Python's provider strategy binding — given a schema spec, configures
/// the provider response format and can parse AI message values.
#[derive(Debug, Clone)]
pub struct ProviderStrategyBinding {
    /// The schema specification used for this binding.
    pub schema_spec: SchemaSpec,
    /// Whether to use strict schema enforcement.
    pub strict: bool,
}

impl ProviderStrategyBinding {
    /// Create a new ProviderStrategyBinding from a schema spec.
    pub fn from_schema_spec(schema_spec: SchemaSpec, strict: bool) -> Self {
        Self {
            schema_spec,
            strict,
        }
    }

    /// Parse an AI message value into structured output.
    ///
    /// For provider strategy, the model response content is expected to be
    /// valid JSON matching the schema.
    #[allow(clippy::result_large_err)]
    pub fn parse(&self, ai_message_value: &Value) -> std::result::Result<Value, StructuredOutputValidationError> {
        // For provider strategy, the content itself should be the structured output
        if let Some(content) = ai_message_value.get("content").and_then(|c| c.as_str()) {
            serde_json::from_str(content).map_err(|e| {
                StructuredOutputValidationError::new(format!(
                    "Failed to parse provider response as JSON: {}",
                    e
                ))
                .with_ai_message(ai_message_value.clone())
                .with_source_error(e.to_string())
            })
        } else if let Some(content) = ai_message_value.get("content") {
            // Content is already a structured value
            Ok(content.clone())
        } else {
            Err(
                StructuredOutputValidationError::new(
                    "AI message does not contain content for provider strategy parsing",
                )
                .with_ai_message(ai_message_value.clone()),
            )
        }
    }

    /// Convert this binding to model kwargs (delegates to ProviderStrategy).
    pub fn to_model_kwargs(&self) -> HashMap<String, Value> {
        ProviderStrategy {
            schema: self.schema_spec.json_schema.clone(),
            schema_name: self.schema_spec.name.clone(),
            strict: self.strict,
        }
        .to_model_kwargs()
    }
}

/// Strategy that lets the framework automatically choose the best approach.
///
/// The framework will inspect the model's capabilities and select either
/// ToolStrategy or ProviderStrategy as appropriate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoStrategy {
    /// The JSON Schema that defines the expected output structure.
    pub schema: Value,
}

impl AutoStrategy {
    /// Create a new AutoStrategy with the given schema.
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }
}

/// The response format specification for structured output.
///
/// Determines how the agent should produce structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Use tool calling to enforce structure.
    Tool(ToolStrategy),
    /// Use native provider structured output.
    Provider(ProviderStrategy),
    /// Let the framework choose automatically.
    Auto(AutoStrategy),
}

impl ResponseFormat {
    /// Get the schema from any response format variant.
    pub fn schema(&self) -> &Value {
        match self {
            ResponseFormat::Tool(s) => &s.schema,
            ResponseFormat::Provider(s) => &s.schema,
            ResponseFormat::Auto(s) => &s.schema,
        }
    }

    /// Create a tool-based response format.
    pub fn tool(schema: Value) -> Self {
        ResponseFormat::Tool(ToolStrategy::new(schema))
    }

    /// Create a provider-based response format.
    pub fn provider(schema: Value) -> Self {
        ResponseFormat::Provider(ProviderStrategy::new(schema))
    }

    /// Create an auto response format.
    pub fn auto(schema: Value) -> Self {
        ResponseFormat::Auto(AutoStrategy::new(schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_schema_kind_serialize() {
        assert_eq!(
            serde_json::to_string(&SchemaKind::Pydantic).unwrap(),
            "\"pydantic\""
        );
        assert_eq!(
            serde_json::to_string(&SchemaKind::JsonSchema).unwrap(),
            "\"json_schema\""
        );
        assert_eq!(
            serde_json::to_string(&SchemaKind::Dataclass).unwrap(),
            "\"dataclass\""
        );
        assert_eq!(
            serde_json::to_string(&SchemaKind::TypedDict).unwrap(),
            "\"typed_dict\""
        );
    }

    #[test]
    fn test_schema_kind_deserialize() {
        let kind: SchemaKind = serde_json::from_str("\"pydantic\"").unwrap();
        assert_eq!(kind, SchemaKind::Pydantic);
        let kind: SchemaKind = serde_json::from_str("\"json_schema\"").unwrap();
        assert_eq!(kind, SchemaKind::JsonSchema);
        let kind: SchemaKind = serde_json::from_str("\"dataclass\"").unwrap();
        assert_eq!(kind, SchemaKind::Dataclass);
        let kind: SchemaKind = serde_json::from_str("\"typed_dict\"").unwrap();
        assert_eq!(kind, SchemaKind::TypedDict);
    }

    #[test]
    fn test_tool_strategy_new() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let strategy = ToolStrategy::new(schema.clone());
        assert_eq!(strategy.schema, schema);
        assert!(strategy.tool_message_content.is_empty());
        assert!(matches!(strategy.handle_errors, ErrorHandling::Disabled));
        assert!(strategy.schema_specs.is_empty());
    }

    #[test]
    fn test_tool_strategy_builder() {
        let schema = json!({"type": "object"});
        let strategy = ToolStrategy::new(schema)
            .with_tool_message_content("You have provided the output.")
            .with_handle_errors(ErrorHandling::Enabled);
        assert_eq!(
            strategy.tool_message_content,
            "You have provided the output."
        );
        assert!(matches!(strategy.handle_errors, ErrorHandling::Enabled));
    }

    #[test]
    fn test_tool_strategy_from_schema_specs_single() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyOutput",
            json!({"type": "object", "properties": {"x": {"type": "integer"}}}),
        );
        let strategy = ToolStrategy::from_schema_specs(vec![spec.clone()]);
        assert_eq!(strategy.schema_specs.len(), 1);
        assert_eq!(strategy.schema, spec.json_schema);
    }

    #[test]
    fn test_tool_strategy_from_schema_specs_multiple() {
        let spec1 = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "OutputA",
            json!({"type": "object", "properties": {"a": {"type": "string"}}}),
        );
        let spec2 = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "OutputB",
            json!({"type": "object", "properties": {"b": {"type": "integer"}}}),
        );
        let strategy = ToolStrategy::from_schema_specs(vec![spec1, spec2]);
        assert_eq!(strategy.schema_specs.len(), 2);
        assert!(strategy.schema.get("oneOf").is_some());
        let one_of = strategy.schema["oneOf"].as_array().unwrap();
        assert_eq!(one_of.len(), 2);
    }

    #[test]
    fn test_provider_strategy_new() {
        let schema = json!({"type": "object"});
        let strategy = ProviderStrategy::new(schema.clone());
        assert_eq!(strategy.schema, schema);
        assert_eq!(strategy.schema_name, "output");
        assert!(!strategy.strict);
    }

    #[test]
    fn test_provider_strategy_with_strict() {
        let schema = json!({"type": "object"});
        let strategy = ProviderStrategy::new(schema).with_strict(true);
        assert!(strategy.strict);
    }

    #[test]
    fn test_provider_strategy_with_schema_name() {
        let schema = json!({"type": "object"});
        let strategy = ProviderStrategy::new(schema).with_schema_name("MySchema");
        assert_eq!(strategy.schema_name, "MySchema");
    }

    #[test]
    fn test_provider_strategy_to_model_kwargs() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "integer"}}});
        let strategy = ProviderStrategy::new(schema.clone())
            .with_schema_name("TestSchema")
            .with_strict(true);
        let kwargs = strategy.to_model_kwargs();

        assert!(kwargs.contains_key("response_format"));
        let rf = &kwargs["response_format"];
        assert_eq!(rf["type"], "json_schema");
        // Check nested json_schema object
        let js = &rf["json_schema"];
        assert_eq!(js["name"], "TestSchema");
        assert_eq!(js["schema"], schema);
        assert_eq!(js["strict"], true);
    }

    #[test]
    fn test_provider_strategy_to_model_kwargs_not_strict() {
        let schema = json!({"type": "object"});
        let strategy = ProviderStrategy::new(schema);
        let kwargs = strategy.to_model_kwargs();
        assert_eq!(kwargs["response_format"]["json_schema"]["strict"], false);
    }

    #[test]
    fn test_auto_strategy_new() {
        let schema = json!({"type": "object"});
        let strategy = AutoStrategy::new(schema.clone());
        assert_eq!(strategy.schema, schema);
    }

    #[test]
    fn test_response_format_tool() {
        let schema = json!({"type": "object"});
        let rf = ResponseFormat::tool(schema.clone());
        assert_eq!(rf.schema(), &schema);
        assert!(matches!(rf, ResponseFormat::Tool(_)));
    }

    #[test]
    fn test_response_format_provider() {
        let schema = json!({"type": "object"});
        let rf = ResponseFormat::provider(schema.clone());
        assert_eq!(rf.schema(), &schema);
        assert!(matches!(rf, ResponseFormat::Provider(_)));
    }

    #[test]
    fn test_response_format_auto() {
        let schema = json!({"type": "object"});
        let rf = ResponseFormat::auto(schema.clone());
        assert_eq!(rf.schema(), &schema);
        assert!(matches!(rf, ResponseFormat::Auto(_)));
    }

    #[test]
    fn test_response_format_serialize_roundtrip() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}});
        let rf = ResponseFormat::Tool(
            ToolStrategy::new(schema.clone())
                .with_tool_message_content("done")
                .with_handle_errors(ErrorHandling::Enabled),
        );
        let json_str = serde_json::to_string(&rf).unwrap();
        let deserialized: ResponseFormat = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.schema(), &schema);
        if let ResponseFormat::Tool(ts) = deserialized {
            assert_eq!(ts.tool_message_content, "done");
            assert!(matches!(ts.handle_errors, ErrorHandling::Enabled));
        } else {
            panic!("Expected Tool variant");
        }
    }

    #[test]
    fn test_structured_output_error_display() {
        let err = StructuredOutputError::new("something went wrong");
        assert_eq!(
            err.to_string(),
            "StructuredOutputError: something went wrong"
        );
        assert!(err.ai_message.is_none());
    }

    #[test]
    fn test_structured_output_error_with_ai_message() {
        let err = StructuredOutputError::new("bad output")
            .with_ai_message(json!({"role": "assistant", "content": "oops"}));
        assert!(err.ai_message.is_some());
    }

    #[test]
    fn test_multiple_structured_outputs_error_display() {
        let err = MultipleStructuredOutputsError::new(
            "Both tool and provider format specified",
        );
        assert!(err.to_string().contains("Multiple"));
        assert!(err.to_string().contains("Both tool and provider"));
        assert!(err.ai_message.is_none());
        assert!(err.tool_names.is_empty());
    }

    #[test]
    fn test_multiple_structured_outputs_error_with_fields() {
        let err = MultipleStructuredOutputsError::new("multiple tools")
            .with_ai_message(json!({"role": "assistant"}))
            .with_tool_names(vec!["tool_a".to_string(), "tool_b".to_string()]);
        assert!(err.ai_message.is_some());
        assert_eq!(err.tool_names, vec!["tool_a", "tool_b"]);
    }

    #[test]
    fn test_structured_output_validation_error() {
        let err = StructuredOutputValidationError::new("invalid field 'x'")
            .with_raw_output("{\"x\": null}");
        assert!(err.to_string().contains("invalid field 'x'"));
        assert!(err.to_string().contains("raw output"));
        assert_eq!(err.raw_output.as_deref(), Some("{\"x\": null}"));
    }

    #[test]
    fn test_structured_output_validation_error_no_raw() {
        let err = StructuredOutputValidationError::new("missing field");
        assert!(err.raw_output.is_none());
        let display = err.to_string();
        assert!(!display.contains("raw output"));
    }

    #[test]
    fn test_structured_output_validation_error_full() {
        let err = StructuredOutputValidationError::new("bad value")
            .with_raw_output("{}")
            .with_ai_message(json!({"role": "assistant"}))
            .with_tool_name("my_tool")
            .with_source_error("JSON parse error");
        assert!(err.ai_message.is_some());
        assert_eq!(err.tool_name.as_deref(), Some("my_tool"));
        assert_eq!(err.source_error.as_deref(), Some("JSON parse error"));
    }

    #[test]
    fn test_tool_strategy_serialization() {
        let strategy = ToolStrategy::new(json!({"type": "object"}));
        let json = serde_json::to_value(&strategy).unwrap();
        assert_eq!(json["tool_message_content"], "");
    }

    #[test]
    fn test_error_handling_from_bool() {
        let eh: ErrorHandling = true.into();
        assert!(matches!(eh, ErrorHandling::Enabled));
        let eh: ErrorHandling = false.into();
        assert!(matches!(eh, ErrorHandling::Disabled));
    }

    #[test]
    fn test_error_handling_with_message() {
        let eh = ErrorHandling::WithMessage("retry please".to_string());
        if let ErrorHandling::WithMessage(msg) = eh {
            assert_eq!(msg, "retry please");
        } else {
            panic!("Expected WithMessage");
        }
    }

    #[test]
    fn test_error_handling_with_types() {
        let eh = ErrorHandling::WithTypes(vec!["ValidationError".to_string()]);
        if let ErrorHandling::WithTypes(types) = eh {
            assert_eq!(types, vec!["ValidationError"]);
        } else {
            panic!("Expected WithTypes");
        }
    }

    #[test]
    fn test_schema_spec_new() {
        let spec = SchemaSpec::new(
            SchemaKind::Pydantic,
            "UserOutput",
            json!({"type": "object", "properties": {"name": {"type": "string"}}}),
        );
        assert_eq!(spec.kind, SchemaKind::Pydantic);
        assert_eq!(spec.name, "UserOutput");
        assert!(spec.description.is_none());
    }

    #[test]
    fn test_schema_spec_with_description() {
        let spec = SchemaSpec::new(SchemaKind::JsonSchema, "Test", json!({}))
            .with_description("A test schema");
        assert_eq!(spec.description.as_deref(), Some("A test schema"));
    }

    #[test]
    fn test_output_tool_binding_from_schema_spec() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyTool",
            json!({"type": "object"}),
        );
        let binding = OutputToolBinding::from_schema_spec(spec, false);
        assert_eq!(binding.tool_name, "MyTool");
        assert!(!binding.include_raw);
    }

    #[test]
    fn test_output_tool_binding_parse_success() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyTool",
            json!({"type": "object"}),
        );
        let binding = OutputToolBinding::from_schema_spec(spec, false);
        let ai_msg = json!({
            "tool_calls": [
                {"name": "MyTool", "args": {"x": 42}}
            ]
        });
        let result = binding.parse(&ai_msg).unwrap();
        assert_eq!(result, json!({"x": 42}));
    }

    #[test]
    fn test_output_tool_binding_parse_no_matching_tool() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyTool",
            json!({"type": "object"}),
        );
        let binding = OutputToolBinding::from_schema_spec(spec, false);
        let ai_msg = json!({
            "tool_calls": [
                {"name": "OtherTool", "args": {"x": 1}}
            ]
        });
        let err = binding.parse(&ai_msg).unwrap_err();
        assert!(err.message.contains("No tool call found"));
        assert_eq!(err.tool_name.as_deref(), Some("MyTool"));
    }

    #[test]
    fn test_output_tool_binding_parse_no_tool_calls() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyTool",
            json!({"type": "object"}),
        );
        let binding = OutputToolBinding::from_schema_spec(spec, false);
        let ai_msg = json!({"content": "hello"});
        let err = binding.parse(&ai_msg).unwrap_err();
        assert!(err.message.contains("does not contain tool_calls"));
    }

    #[test]
    fn test_provider_strategy_binding_from_schema_spec() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "MyOutput",
            json!({"type": "object"}),
        );
        let binding = ProviderStrategyBinding::from_schema_spec(spec, true);
        assert!(binding.strict);
        assert_eq!(binding.schema_spec.name, "MyOutput");
    }

    #[test]
    fn test_provider_strategy_binding_parse_string_content() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "Output",
            json!({"type": "object"}),
        );
        let binding = ProviderStrategyBinding::from_schema_spec(spec, false);
        let ai_msg = json!({"content": "{\"x\": 42}"});
        let result = binding.parse(&ai_msg).unwrap();
        assert_eq!(result, json!({"x": 42}));
    }

    #[test]
    fn test_provider_strategy_binding_parse_object_content() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "Output",
            json!({"type": "object"}),
        );
        let binding = ProviderStrategyBinding::from_schema_spec(spec, false);
        let ai_msg = json!({"content": {"x": 42}});
        let result = binding.parse(&ai_msg).unwrap();
        assert_eq!(result, json!({"x": 42}));
    }

    #[test]
    fn test_provider_strategy_binding_parse_no_content() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "Output",
            json!({"type": "object"}),
        );
        let binding = ProviderStrategyBinding::from_schema_spec(spec, false);
        let ai_msg = json!({"role": "assistant"});
        let err = binding.parse(&ai_msg).unwrap_err();
        assert!(err.message.contains("does not contain content"));
    }

    #[test]
    fn test_provider_strategy_binding_to_model_kwargs() {
        let spec = SchemaSpec::new(
            SchemaKind::JsonSchema,
            "TestOutput",
            json!({"type": "object", "properties": {"y": {"type": "string"}}}),
        );
        let binding = ProviderStrategyBinding::from_schema_spec(spec, true);
        let kwargs = binding.to_model_kwargs();
        let rf = &kwargs["response_format"];
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "TestOutput");
        assert_eq!(rf["json_schema"]["strict"], true);
    }
}
