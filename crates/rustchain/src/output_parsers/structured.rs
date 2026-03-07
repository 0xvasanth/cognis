//! Structured output parser with JSON schema validation.
//!
//! Equivalent to Python's `PydanticOutputParser` but validates against a
//! JSON schema instead of a Pydantic model.

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::output_parsers::OutputParser;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

/// An output parser that parses JSON and validates it against a JSON schema.
///
/// Generates format instructions from the schema and returns structured error
/// messages for invalid or missing fields.
///
/// Builds on the existing `JsonOutputParser` for raw JSON extraction and adds
/// schema-level validation including required fields and type checking.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::output_parsers::StructuredOutputParser;
/// use serde_json::json;
///
/// let parser = StructuredOutputParser::builder()
///     .type_name("Person")
///     .schema(json!({
///         "type": "object",
///         "properties": {
///             "name": {"type": "string"},
///             "age": {"type": "integer"}
///         },
///         "required": ["name", "age"]
///     }))
///     .build();
/// ```
pub struct StructuredOutputParser {
    /// JSON schema describing the expected output structure.
    schema: Value,
    /// Name of the type being parsed (for error messages).
    type_name: String,
}

/// Builder for [`StructuredOutputParser`].
pub struct StructuredOutputParserBuilder {
    schema: Option<Value>,
    type_name: String,
}

impl StructuredOutputParserBuilder {
    /// Set the JSON schema.
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the type name (used in error messages).
    pub fn type_name(mut self, name: impl Into<String>) -> Self {
        self.type_name = name.into();
        self
    }

    /// Build the [`StructuredOutputParser`].
    ///
    /// # Panics
    ///
    /// Panics if `schema` has not been set.
    pub fn build(self) -> StructuredOutputParser {
        StructuredOutputParser {
            schema: self.schema.expect("schema is required"),
            type_name: self.type_name,
        }
    }
}

impl StructuredOutputParser {
    /// Create a new builder.
    pub fn builder() -> StructuredOutputParserBuilder {
        StructuredOutputParserBuilder {
            schema: None,
            type_name: "Output".to_string(),
        }
    }

    /// Create directly from a type name and schema.
    pub fn new(type_name: impl Into<String>, schema: Value) -> Self {
        Self {
            schema,
            type_name: type_name.into(),
        }
    }

    /// Parse JSON from text, stripping markdown code fences if present.
    fn parse_json(&self, text: &str) -> Result<Value> {
        let trimmed = text.trim();

        let json_str = if trimmed.starts_with("```") {
            let after_fence = if let Some(rest) = trimmed.strip_prefix("```json") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix("```JSON") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix("```") {
                rest
            } else {
                trimmed
            };

            after_fence
                .trim()
                .strip_suffix("```")
                .unwrap_or(after_fence)
                .trim()
        } else {
            trimmed
        };

        serde_json::from_str(json_str).map_err(|e| RustChainError::OutputParserError {
            message: format!("Failed to parse JSON for {}: {}", self.type_name, e),
            observation: Some(json_str.to_string()),
            llm_output: Some(text.to_string()),
        })
    }

    /// Validate parsed JSON against the schema.
    fn validate(&self, value: &Value) -> Result<()> {
        // Check that it's an object if schema expects one
        if let Some(schema_type) = self.schema.get("type").and_then(|t| t.as_str()) {
            if schema_type == "object" && !value.is_object() {
                return Err(RustChainError::OutputParserError {
                    message: format!(
                        "Expected JSON object for {}, got {}",
                        self.type_name,
                        value_type_name(value)
                    ),
                    observation: Some(value.to_string()),
                    llm_output: None,
                });
            }
        }

        // Check required fields
        if let Some(required) = self.schema.get("required").and_then(|r| r.as_array()) {
            if let Value::Object(map) = value {
                let mut missing: Vec<&str> = Vec::new();
                for req in required {
                    if let Some(field) = req.as_str() {
                        if !map.contains_key(field) {
                            missing.push(field);
                        }
                    }
                }
                if !missing.is_empty() {
                    return Err(RustChainError::OutputParserError {
                        message: format!(
                            "Missing required field(s) in {} output: {}",
                            self.type_name,
                            missing.join(", ")
                        ),
                        observation: Some(value.to_string()),
                        llm_output: None,
                    });
                }
            }
        }

        // Validate field types
        if let (Some(Value::Object(props)), Some(Value::Object(obj))) =
            (self.schema.get("properties"), Some(value))
        {
            let mut type_errors: Vec<String> = Vec::new();

            for (field_name, field_schema) in props {
                if let Some(field_value) = obj.get(field_name) {
                    if let Some(expected_type) = field_schema.get("type").and_then(|t| t.as_str()) {
                        if !check_json_type(field_value, expected_type) {
                            type_errors.push(format!(
                                "field '{}': expected {}, got {}",
                                field_name,
                                expected_type,
                                value_type_name(field_value)
                            ));
                        }
                    }
                }
            }

            if !type_errors.is_empty() {
                return Err(RustChainError::OutputParserError {
                    message: format!(
                        "Type validation errors in {} output: {}",
                        self.type_name,
                        type_errors.join("; ")
                    ),
                    observation: Some(value.to_string()),
                    llm_output: None,
                });
            }
        }

        Ok(())
    }
}

/// Check whether a JSON value matches an expected JSON schema type.
fn check_json_type(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true, // unknown type, don't reject
    }
}

/// Human-readable name for a JSON value type.
fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

impl OutputParser for StructuredOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let value = self.parse_json(text)?;
        self.validate(&value)?;
        Ok(value)
    }

    fn get_format_instructions(&self) -> Option<String> {
        let mut display_schema = self.schema.clone();
        // Remove title and top-level type from display (matching Python PydanticOutputParser)
        if let Value::Object(ref mut map) = display_schema {
            map.remove("title");
        }
        let schema_str = serde_json::to_string_pretty(&display_schema).unwrap_or_default();

        Some(format!(
            "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n\
             As an example, for the schema {{\"properties\": {{\"foo\": {{\"title\": \"Foo\", \
             \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {{\"type\": \"string\"}}}}}}, \
             \"required\": [\"foo\"]}}\n\
             the object {{\"foo\": [\"bar\", \"baz\"]}} is a well-formatted instance of the schema. \
             The object {{\"properties\": {{\"foo\": [\"bar\", \"baz\"]}}}} is not well-formatted.\n\n\
             Here is the output schema:\n```\n{}\n```",
            schema_str
        ))
    }

    fn parser_type(&self) -> &str {
        "structured_output_parser"
    }
}

#[async_trait]
impl Runnable for StructuredOutputParser {
    fn name(&self) -> &str {
        "StructuredOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
