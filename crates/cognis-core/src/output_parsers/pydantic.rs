//! Schema-based output parser that validates JSON against a schema.
//!
//! Mirrors Python `langchain_core.output_parsers.pydantic`.
//! Since Rust doesn't have Pydantic, we validate against a JSON schema structure.

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};
use crate::outputs::Generation;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;
use super::json::JsonOutputParser;

/// Output parser that parses JSON and validates required fields against a schema.
///
/// This is the Rust equivalent of `PydanticOutputParser` — instead of validating
/// against a Pydantic model, it validates against a JSON schema object.
pub struct SchemaOutputParser {
    /// JSON schema describing the expected output structure.
    pub schema: Value,
    /// The name of the type being parsed (for error messages).
    pub type_name: String,
    json_parser: JsonOutputParser,
}

impl SchemaOutputParser {
    pub fn new(type_name: impl Into<String>, schema: Value) -> Self {
        let json_parser = JsonOutputParser::with_schema(schema.clone());
        Self {
            schema,
            type_name: type_name.into(),
            json_parser,
        }
    }

    /// Validate that required fields are present in the parsed JSON.
    fn validate_required(&self, obj: &Value) -> Result<()> {
        if let Some(required) = self.schema.get("required").and_then(|r| r.as_array()) {
            if let Value::Object(map) = obj {
                for req in required {
                    if let Some(field) = req.as_str() {
                        if !map.contains_key(field) {
                            return Err(CognisError::OutputParserError {
                                message: format!(
                                    "Missing required field '{}' in {} output",
                                    field, self.type_name
                                ),
                                observation: Some(obj.to_string()),
                                llm_output: None,
                            });
                        }
                    }
                }
            } else {
                return Err(CognisError::OutputParserError {
                    message: format!(
                        "Expected JSON object for {}, got {}",
                        self.type_name,
                        match obj {
                            Value::Array(_) => "array",
                            Value::String(_) => "string",
                            Value::Number(_) => "number",
                            Value::Bool(_) => "boolean",
                            Value::Null => "null",
                            _ => "unknown",
                        }
                    ),
                    observation: Some(obj.to_string()),
                    llm_output: None,
                });
            }
        }
        Ok(())
    }
}

impl OutputParser for SchemaOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let obj = self.json_parser.parse(text)?;
        self.validate_required(&obj)?;
        Ok(obj)
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        let obj = self.json_parser.parse_result(result, partial)?;
        if !partial {
            self.validate_required(&obj)?;
        }
        Ok(obj)
    }

    fn get_format_instructions(&self) -> Option<String> {
        let mut schema = self.schema.clone();
        // Remove title and type from display schema (matching Python behavior)
        if let Value::Object(ref mut map) = schema {
            map.remove("title");
            map.remove("type");
        }
        let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
        Some(format!(
            "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n\
             As an example, for the schema {{\"properties\": {{\"foo\": {{\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {{\"type\": \"string\"}}}}}}, \"required\": [\"foo\"]}}\n\
             the object {{\"foo\": [\"bar\", \"baz\"]}} is a well-formatted instance of the schema. \
             The object {{\"properties\": {{\"foo\": [\"bar\", \"baz\"]}}}} is not well-formatted.\n\n\
             Here is the output schema:\n```\n{}\n```",
            schema_str
        ))
    }

    fn parser_type(&self) -> &str {
        "schema_output_parser"
    }
}

#[async_trait]
impl Runnable for SchemaOutputParser {
    fn name(&self) -> &str {
        "SchemaOutputParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}
