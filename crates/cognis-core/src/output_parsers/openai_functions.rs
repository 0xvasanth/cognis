//! Output parsers for OpenAI-style function calls (legacy format).
//!
//! Mirrors Python `langchain_core.output_parsers.openai_functions`.
//!
//! These parsers handle the older `function_call` format in `additional_kwargs`,
//! as opposed to the newer `tool_calls` format handled by `openai_tools`.

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{CognisError, Result};
use crate::outputs::Generation;
use crate::runnables::base::Runnable;
use crate::runnables::config::RunnableConfig;

use super::base::OutputParser;

// ---------------------------------------------------------------------------
// Helper: extract function_call from generation info / text
// ---------------------------------------------------------------------------

/// Extract the `function_call` object from a generation.
///
/// Looks in `generation_info.function_call`, then
/// `generation_info.message.additional_kwargs.function_call`, and finally
/// tries parsing the generation text as JSON with a `function_call` key.
fn extract_function_call(gen: &Generation) -> Result<Value> {
    // Try generation_info first
    if let Some(info) = &gen.generation_info {
        // Direct function_call key
        if let Some(fc) = info.get("function_call") {
            return Ok(fc.clone());
        }
        // Nested under message.additional_kwargs
        if let Some(fc) = info
            .get("message")
            .and_then(|m| m.get("additional_kwargs"))
            .and_then(|ak| ak.get("function_call"))
        {
            return Ok(fc.clone());
        }
        // Nested under additional_kwargs directly
        if let Some(fc) = info
            .get("additional_kwargs")
            .and_then(|ak| ak.get("function_call"))
        {
            return Ok(fc.clone());
        }
    }

    // Try parsing the text as JSON
    if let Ok(parsed) = serde_json::from_str::<Value>(&gen.text) {
        if let Some(fc) = parsed.get("function_call") {
            return Ok(fc.clone());
        }
        // If the text itself looks like a function_call object (has "name" and "arguments")
        if parsed.get("name").is_some() && parsed.get("arguments").is_some() {
            return Ok(parsed);
        }
    }

    Err(CognisError::OutputParserError {
        message: "Could not parse function call from generation".into(),
        observation: Some(gen.text.clone()),
        llm_output: None,
    })
}

// ---------------------------------------------------------------------------
// OutputFunctionsParser
// ---------------------------------------------------------------------------

/// Extracts a function call from an AIMessage's `additional_kwargs`.
///
/// When `args_only` is true (the default), returns just the `arguments` string.
/// Otherwise returns the full `function_call` object (`{"name": ..., "arguments": ...}`).
pub struct OutputFunctionsParser {
    /// Whether to only return the arguments string (default: true).
    pub args_only: bool,
}

impl OutputFunctionsParser {
    pub fn new() -> Self {
        Self { args_only: true }
    }

    pub fn with_args_only(mut self, args_only: bool) -> Self {
        self.args_only = args_only;
        self
    }
}

impl Default for OutputFunctionsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for OutputFunctionsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let parsed: Value =
            serde_json::from_str(text).map_err(|e| CognisError::OutputParserError {
                message: format!("Failed to parse function call JSON: {}", e),
                observation: Some(text.to_string()),
                llm_output: None,
            })?;

        if self.args_only {
            parsed
                .get("arguments")
                .cloned()
                .ok_or_else(|| CognisError::OutputParserError {
                    message: "Function call missing 'arguments' key".into(),
                    observation: Some(text.to_string()),
                    llm_output: None,
                })
        } else {
            Ok(parsed)
        }
    }

    fn parse_result(&self, result: &[Generation], _partial: bool) -> Result<Value> {
        if result.is_empty() {
            return Err(CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }

        let gen = &result[0];
        let func_call = extract_function_call(gen)?;

        if self.args_only {
            func_call
                .get("arguments")
                .cloned()
                .ok_or_else(|| CognisError::OutputParserError {
                    message: "Function call missing 'arguments' key".into(),
                    observation: Some(func_call.to_string()),
                    llm_output: None,
                })
        } else {
            Ok(func_call)
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "output_functions"
    }
}

#[async_trait]
impl Runnable for OutputFunctionsParser {
    fn name(&self) -> &str {
        "OutputFunctionsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        match &input {
            Value::String(s) => self.parse(s),
            other => {
                // Try to find function_call in the value directly
                if let Some(fc) = other.get("function_call") {
                    if self.args_only {
                        fc.get("arguments")
                            .cloned()
                            .ok_or_else(|| CognisError::OutputParserError {
                                message: "Function call missing 'arguments' key".into(),
                                observation: Some(other.to_string()),
                                llm_output: None,
                            })
                    } else {
                        Ok(fc.clone())
                    }
                } else {
                    self.parse(&other.to_string())
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JsonOutputFunctionsParser
// ---------------------------------------------------------------------------

/// Parses the function call arguments as a JSON object.
///
/// When `args_only` is true (the default), returns the parsed arguments JSON.
/// Otherwise returns the full function call with `arguments` parsed from a
/// string into a JSON value.
pub struct JsonOutputFunctionsParser {
    /// Whether to only return the parsed arguments (default: true).
    pub args_only: bool,
    /// Whether to use strict JSON parsing (default: false).
    pub strict: bool,
}

impl JsonOutputFunctionsParser {
    pub fn new() -> Self {
        Self {
            args_only: true,
            strict: false,
        }
    }

    pub fn with_args_only(mut self, args_only: bool) -> Self {
        self.args_only = args_only;
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Parse the arguments string into a JSON value.
    fn parse_arguments(&self, args_str: &str) -> Result<Value> {
        serde_json::from_str(args_str).map_err(|e| CognisError::OutputParserError {
            message: format!("Could not parse function call data: {}", e),
            observation: Some(args_str.to_string()),
            llm_output: None,
        })
    }

    /// Given a function_call Value, return the parsed result based on args_only.
    fn process_function_call(&self, func_call: &Value, partial: bool) -> Result<Value> {
        let args_val = func_call.get("arguments");

        let args_str = match args_val {
            Some(Value::String(s)) => s.as_str(),
            Some(other) => {
                // Arguments already parsed as JSON
                if self.args_only {
                    return Ok(other.clone());
                } else {
                    let mut result = func_call.clone();
                    if let Value::Object(ref mut map) = result {
                        map.insert("arguments".to_string(), other.clone());
                    }
                    return Ok(result);
                }
            }
            None => {
                if partial {
                    return Ok(Value::Null);
                }
                return Err(CognisError::OutputParserError {
                    message: "Function call missing 'arguments' key".into(),
                    observation: Some(func_call.to_string()),
                    llm_output: None,
                });
            }
        };

        if partial {
            // For partial parsing, try to parse what we can; return Null on failure
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => {
                    if self.args_only {
                        Ok(parsed)
                    } else {
                        let mut result = func_call.clone();
                        if let Value::Object(ref mut map) = result {
                            map.insert("arguments".to_string(), parsed);
                        }
                        Ok(result)
                    }
                }
                Err(_) => Ok(Value::Null),
            }
        } else {
            let parsed = self.parse_arguments(args_str)?;
            if self.args_only {
                Ok(parsed)
            } else {
                let mut result = func_call.clone();
                if let Value::Object(ref mut map) = result {
                    map.insert("arguments".to_string(), parsed);
                }
                Ok(result)
            }
        }
    }
}

impl Default for JsonOutputFunctionsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for JsonOutputFunctionsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let parsed: Value =
            serde_json::from_str(text).map_err(|e| CognisError::OutputParserError {
                message: format!("Failed to parse function call JSON: {}", e),
                observation: Some(text.to_string()),
                llm_output: None,
            })?;

        self.process_function_call(&parsed, false)
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        if result.is_empty() {
            return Err(CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }

        if result.len() != 1 {
            return Err(CognisError::OutputParserError {
                message: format!("Expected exactly one result, but got {}", result.len()),
                observation: None,
                llm_output: None,
            });
        }

        let gen = &result[0];

        let func_call = match extract_function_call(gen) {
            Ok(fc) => fc,
            Err(_) if partial => return Ok(Value::Null),
            Err(e) => return Err(e),
        };

        self.process_function_call(&func_call, partial)
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "json_functions"
    }
}

#[async_trait]
impl Runnable for JsonOutputFunctionsParser {
    fn name(&self) -> &str {
        "JsonOutputFunctionsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        match &input {
            Value::String(s) => self.parse(s),
            other => {
                if let Some(fc) = other.get("function_call") {
                    self.process_function_call(fc, false)
                } else {
                    self.parse(&other.to_string())
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JsonKeyOutputFunctionsParser
// ---------------------------------------------------------------------------

/// Parses function call arguments as JSON and extracts a specific key.
///
/// Delegates to `JsonOutputFunctionsParser` (with `args_only = true`) and then
/// extracts the value at `key_name` from the resulting JSON object.
pub struct JsonKeyOutputFunctionsParser {
    /// The key to extract from the parsed arguments.
    pub key_name: String,
    /// Whether to use strict JSON parsing.
    pub strict: bool,
}

impl JsonKeyOutputFunctionsParser {
    pub fn new(key_name: impl Into<String>) -> Self {
        Self {
            key_name: key_name.into(),
            strict: false,
        }
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Internal helper: create the underlying JSON parser.
    fn json_parser(&self) -> JsonOutputFunctionsParser {
        JsonOutputFunctionsParser {
            args_only: true,
            strict: self.strict,
        }
    }
}

impl OutputParser for JsonKeyOutputFunctionsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let parsed = self.json_parser().parse(text)?;
        parsed
            .get(&self.key_name)
            .cloned()
            .ok_or_else(|| CognisError::OutputParserError {
                message: format!("Key '{}' not found in parsed output", self.key_name),
                observation: Some(parsed.to_string()),
                llm_output: None,
            })
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        let parsed = self.json_parser().parse_result(result, partial)?;
        if partial && parsed.is_null() {
            return Ok(Value::Null);
        }
        if partial {
            // For partial, use .get() which returns None gracefully
            Ok(parsed.get(&self.key_name).cloned().unwrap_or(Value::Null))
        } else {
            parsed
                .get(&self.key_name)
                .cloned()
                .ok_or_else(|| CognisError::OutputParserError {
                    message: format!("Key '{}' not found in parsed output", self.key_name),
                    observation: Some(parsed.to_string()),
                    llm_output: None,
                })
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "json_key_functions"
    }
}

#[async_trait]
impl Runnable for JsonKeyOutputFunctionsParser {
    fn name(&self) -> &str {
        "JsonKeyOutputFunctionsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}

// ---------------------------------------------------------------------------
// SchemaOutputFunctionsParser (Pydantic equivalent)
// ---------------------------------------------------------------------------

/// Parses function call output and validates it against a JSON schema.
///
/// This is the Rust equivalent of Python's `PydanticOutputFunctionsParser`.
/// Since Rust doesn't have Pydantic, we validate against a JSON schema object
/// (checking required fields are present).
///
/// When `args_only` is true, expects a single schema and parses the arguments
/// directly. When false, expects a map of function name to schema and selects
/// the appropriate schema based on the function name.
pub struct SchemaOutputFunctionsParser {
    /// A single schema (used when `args_only` is true) or ignored when using `schemas`.
    pub schema: Option<Value>,
    /// A map from function name to JSON schema (used when `args_only` is false).
    pub schemas: Option<std::collections::HashMap<String, Value>>,
    /// Whether to only parse the arguments (default: determined by schema type).
    pub args_only: bool,
}

impl SchemaOutputFunctionsParser {
    /// Create a parser for a single schema (args_only = true).
    pub fn from_single_schema(schema: Value) -> Self {
        Self {
            schema: Some(schema),
            schemas: None,
            args_only: true,
        }
    }

    /// Create a parser for multiple schemas keyed by function name (args_only = false).
    pub fn from_named_schemas(schemas: std::collections::HashMap<String, Value>) -> Self {
        Self {
            schema: None,
            schemas: Some(schemas),
            args_only: false,
        }
    }

    /// Validate that required fields from the schema are present.
    fn validate_required(schema: &Value, obj: &Value) -> Result<()> {
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            if let Value::Object(map) = obj {
                for req in required {
                    if let Some(field) = req.as_str() {
                        if !map.contains_key(field) {
                            return Err(CognisError::OutputParserError {
                                message: format!(
                                    "Missing required field '{}' in function output",
                                    field
                                ),
                                observation: Some(obj.to_string()),
                                llm_output: None,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl OutputParser for SchemaOutputFunctionsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // Delegate to the underlying OutputFunctionsParser then validate
        let base = OutputFunctionsParser {
            args_only: self.args_only,
        };
        let raw = base.parse(text)?;

        if self.args_only {
            let args_str = raw.as_str().ok_or_else(|| CognisError::OutputParserError {
                message: "Expected arguments to be a string".into(),
                observation: Some(raw.to_string()),
                llm_output: None,
            })?;
            let parsed: Value =
                serde_json::from_str(args_str).map_err(|e| CognisError::OutputParserError {
                    message: format!("Failed to parse arguments JSON: {}", e),
                    observation: Some(args_str.to_string()),
                    llm_output: None,
                })?;
            if let Some(ref schema) = self.schema {
                Self::validate_required(schema, &parsed)?;
            }
            Ok(parsed)
        } else {
            let fn_name = raw
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let args_str = raw
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let parsed: Value =
                serde_json::from_str(args_str).map_err(|e| CognisError::OutputParserError {
                    message: format!("Failed to parse arguments JSON: {}", e),
                    observation: Some(args_str.to_string()),
                    llm_output: None,
                })?;
            if let Some(ref schemas) = self.schemas {
                if let Some(schema) = schemas.get(fn_name) {
                    Self::validate_required(schema, &parsed)?;
                }
            }
            Ok(parsed)
        }
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        if result.is_empty() {
            return Err(CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }

        let gen = &result[0];
        let func_call = extract_function_call(gen)?;

        let base_parser = OutputFunctionsParser {
            args_only: self.args_only,
        };
        let raw = if self.args_only {
            func_call
                .get("arguments")
                .cloned()
                .ok_or_else(|| CognisError::OutputParserError {
                    message: "Function call missing 'arguments' key".into(),
                    observation: Some(func_call.to_string()),
                    llm_output: None,
                })?
        } else {
            // Keep the base_parser reference alive to suppress the warning
            let _ = &base_parser;
            func_call.clone()
        };

        if self.args_only {
            let args_str = raw.as_str().ok_or_else(|| CognisError::OutputParserError {
                message: "Expected arguments to be a string".into(),
                observation: Some(raw.to_string()),
                llm_output: None,
            })?;
            let parsed: Value =
                serde_json::from_str(args_str).map_err(|e| CognisError::OutputParserError {
                    message: format!("Failed to parse arguments JSON: {}", e),
                    observation: Some(args_str.to_string()),
                    llm_output: None,
                })?;
            if !partial {
                if let Some(ref schema) = self.schema {
                    Self::validate_required(schema, &parsed)?;
                }
            }
            Ok(parsed)
        } else {
            let fn_name = raw
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let args_str = raw
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let parsed: Value =
                serde_json::from_str(args_str).map_err(|e| CognisError::OutputParserError {
                    message: format!("Failed to parse arguments JSON: {}", e),
                    observation: Some(args_str.to_string()),
                    llm_output: None,
                })?;
            if !partial {
                if let Some(ref schemas) = self.schemas {
                    if let Some(schema) = schemas.get(fn_name) {
                        Self::validate_required(schema, &parsed)?;
                    }
                }
            }
            Ok(parsed)
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    fn parser_type(&self) -> &str {
        "schema_functions"
    }
}

#[async_trait]
impl Runnable for SchemaOutputFunctionsParser {
    fn name(&self) -> &str {
        "SchemaOutputFunctionsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}

// ---------------------------------------------------------------------------
// SchemaAttrOutputFunctionsParser (PydanticAttr equivalent)
// ---------------------------------------------------------------------------

/// Parses function call output via schema validation and extracts a specific attribute.
///
/// This is the Rust equivalent of Python's `PydanticAttrOutputFunctionsParser`.
pub struct SchemaAttrOutputFunctionsParser {
    /// The underlying schema parser.
    pub inner: SchemaOutputFunctionsParser,
    /// The attribute name to extract from the parsed output.
    pub attr_name: String,
}

impl SchemaAttrOutputFunctionsParser {
    pub fn new(inner: SchemaOutputFunctionsParser, attr_name: impl Into<String>) -> Self {
        Self {
            inner,
            attr_name: attr_name.into(),
        }
    }
}

impl OutputParser for SchemaAttrOutputFunctionsParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let parsed = self.inner.parse(text)?;
        parsed
            .get(&self.attr_name)
            .cloned()
            .ok_or_else(|| CognisError::OutputParserError {
                message: format!("Attribute '{}' not found in parsed output", self.attr_name),
                observation: Some(parsed.to_string()),
                llm_output: None,
            })
    }

    fn parse_result(&self, result: &[Generation], partial: bool) -> Result<Value> {
        let parsed = self.inner.parse_result(result, partial)?;
        parsed
            .get(&self.attr_name)
            .cloned()
            .ok_or_else(|| CognisError::OutputParserError {
                message: format!("Attribute '{}' not found in parsed output", self.attr_name),
                observation: Some(parsed.to_string()),
                llm_output: None,
            })
    }

    fn get_format_instructions(&self) -> Option<String> {
        self.inner.get_format_instructions()
    }

    fn parser_type(&self) -> &str {
        "schema_attr_functions"
    }
}

#[async_trait]
impl Runnable for SchemaAttrOutputFunctionsParser {
    fn name(&self) -> &str {
        "SchemaAttrOutputFunctionsParser"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let text = match &input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        self.parse(&text)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    /// Helper to create a Generation with function_call in generation_info.
    fn make_function_call_generation(name: &str, arguments: &str) -> Generation {
        let mut info = HashMap::new();
        info.insert(
            "function_call".to_string(),
            json!({
                "name": name,
                "arguments": arguments
            }),
        );
        Generation {
            text: String::new(),
            generation_info: Some(info),
        }
    }

    /// Helper to create a Generation with function_call nested in additional_kwargs.
    fn make_additional_kwargs_generation(name: &str, arguments: &str) -> Generation {
        let mut info = HashMap::new();
        info.insert(
            "additional_kwargs".to_string(),
            json!({
                "function_call": {
                    "name": name,
                    "arguments": arguments
                }
            }),
        );
        Generation {
            text: String::new(),
            generation_info: Some(info),
        }
    }

    // -----------------------------------------------------------------------
    // OutputFunctionsParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_functions_parser_args_only() {
        let parser = OutputFunctionsParser::new();
        assert!(parser.args_only);

        let gen = make_function_call_generation("get_weather", r#"{"city": "Paris"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        // args_only returns the arguments string
        assert_eq!(result.as_str().unwrap(), r#"{"city": "Paris"}"#);
    }

    #[test]
    fn test_output_functions_parser_full_call() {
        let parser = OutputFunctionsParser::new().with_args_only(false);

        let gen = make_function_call_generation("get_weather", r#"{"city": "Paris"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result.get("name").unwrap().as_str().unwrap(), "get_weather");
        assert_eq!(
            result.get("arguments").unwrap().as_str().unwrap(),
            r#"{"city": "Paris"}"#
        );
    }

    #[test]
    fn test_output_functions_parser_empty_result() {
        let parser = OutputFunctionsParser::new();
        let result = parser.parse_result(&[], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_output_functions_parser_no_function_call() {
        let parser = OutputFunctionsParser::new();
        let gen = Generation::new("just some text");
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_output_functions_parser_additional_kwargs() {
        let parser = OutputFunctionsParser::new();
        let gen = make_additional_kwargs_generation("search", r#"{"q": "rust"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"q": "rust"}"#);
    }

    #[test]
    fn test_output_functions_parser_parse_text() {
        let parser = OutputFunctionsParser::new();
        let input = r#"{"name": "test_fn", "arguments": "{\"key\": \"value\"}"}"#;
        let result = parser.parse(input).unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"key": "value"}"#);
    }

    #[test]
    fn test_output_functions_parser_parse_text_full() {
        let parser = OutputFunctionsParser::new().with_args_only(false);
        let input = r#"{"name": "test_fn", "arguments": "{\"key\": \"value\"}"}"#;
        let result = parser.parse(input).unwrap();
        assert_eq!(result.get("name").unwrap().as_str().unwrap(), "test_fn");
    }

    // -----------------------------------------------------------------------
    // JsonOutputFunctionsParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_output_functions_parser_args_only() {
        let parser = JsonOutputFunctionsParser::new();
        let gen = make_function_call_generation("get_weather", r#"{"city": "Berlin"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result, json!({"city": "Berlin"}));
    }

    #[test]
    fn test_json_output_functions_parser_full_call() {
        let parser = JsonOutputFunctionsParser::new().with_args_only(false);
        let gen = make_function_call_generation("get_weather", r#"{"city": "Berlin"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result.get("name").unwrap().as_str().unwrap(), "get_weather");
        assert_eq!(result.get("arguments").unwrap(), &json!({"city": "Berlin"}));
    }

    #[test]
    fn test_json_output_functions_parser_invalid_json() {
        let parser = JsonOutputFunctionsParser::new();
        let gen = make_function_call_generation("bad_fn", r#"not json at all"#);
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_output_functions_parser_partial_no_function_call() {
        let parser = JsonOutputFunctionsParser::new();
        let gen = Generation::new("incomplete");
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_output_functions_parser_partial_bad_json() {
        let parser = JsonOutputFunctionsParser::new();
        let gen = make_function_call_generation("fn", r#"{"incomplete": "#);
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_output_functions_parser_partial_good_json() {
        let parser = JsonOutputFunctionsParser::new();
        let gen = make_function_call_generation("fn", r#"{"key": "value"}"#);
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_json_output_functions_parser_multiple_results_error() {
        let parser = JsonOutputFunctionsParser::new();
        let gen1 = make_function_call_generation("fn1", r#"{"a": 1}"#);
        let gen2 = make_function_call_generation("fn2", r#"{"b": 2}"#);
        let result = parser.parse_result(&[gen1, gen2], false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Expected exactly one result"));
    }

    #[test]
    fn test_json_output_functions_parser_type() {
        let parser = JsonOutputFunctionsParser::new();
        assert_eq!(parser.parser_type(), "json_functions");
    }

    // -----------------------------------------------------------------------
    // JsonKeyOutputFunctionsParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_key_output_functions_parser() {
        let parser = JsonKeyOutputFunctionsParser::new("city");
        let gen = make_function_call_generation("get_weather", r#"{"city": "Tokyo", "unit": "C"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result, json!("Tokyo"));
    }

    #[test]
    fn test_json_key_output_functions_parser_missing_key() {
        let parser = JsonKeyOutputFunctionsParser::new("nonexistent");
        let gen = make_function_call_generation("fn", r#"{"city": "Tokyo"}"#);
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("nonexistent"));
    }

    #[test]
    fn test_json_key_output_functions_parser_partial_null() {
        let parser = JsonKeyOutputFunctionsParser::new("city");
        let gen = Generation::new("incomplete");
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_key_output_functions_parser_partial_missing_key() {
        let parser = JsonKeyOutputFunctionsParser::new("missing");
        let gen = make_function_call_generation("fn", r#"{"other": "value"}"#);
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_key_output_functions_parser_parse() {
        let parser = JsonKeyOutputFunctionsParser::new("name");
        let input = r#"{"name": "test_fn", "arguments": "{\"name\": \"Alice\", \"age\": 30}"}"#;
        let result = parser.parse(input).unwrap();
        assert_eq!(result, json!("Alice"));
    }

    // -----------------------------------------------------------------------
    // SchemaOutputFunctionsParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_output_functions_parser_single() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let parser = SchemaOutputFunctionsParser::from_single_schema(schema);

        let gen = make_function_call_generation("create_user", r#"{"name": "Alice", "age": 30}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn test_schema_output_functions_parser_missing_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let parser = SchemaOutputFunctionsParser::from_single_schema(schema);

        let gen = make_function_call_generation("create_user", r#"{"name": "Alice"}"#);
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("age"));
    }

    #[test]
    fn test_schema_output_functions_parser_partial_skips_validation() {
        let schema = json!({
            "type": "object",
            "required": ["name", "age"]
        });
        let parser = SchemaOutputFunctionsParser::from_single_schema(schema);

        let gen = make_function_call_generation("fn", r#"{"name": "Alice"}"#);
        let result = parser.parse_result(&[gen], true).unwrap();
        assert_eq!(result, json!({"name": "Alice"}));
    }

    #[test]
    fn test_schema_output_functions_parser_named_schemas() {
        let mut schemas = std::collections::HashMap::new();
        schemas.insert(
            "cookie".to_string(),
            json!({
                "type": "object",
                "required": ["flavor"]
            }),
        );
        schemas.insert(
            "dog".to_string(),
            json!({
                "type": "object",
                "required": ["breed"]
            }),
        );
        let parser = SchemaOutputFunctionsParser::from_named_schemas(schemas);

        let gen = make_function_call_generation("cookie", r#"{"flavor": "chocolate"}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result, json!({"flavor": "chocolate"}));
    }

    #[test]
    fn test_schema_output_functions_parser_named_schemas_validation_fail() {
        let mut schemas = std::collections::HashMap::new();
        schemas.insert(
            "cookie".to_string(),
            json!({
                "type": "object",
                "required": ["flavor"]
            }),
        );
        let parser = SchemaOutputFunctionsParser::from_named_schemas(schemas);

        // Missing required "flavor" field
        let gen = make_function_call_generation("cookie", r#"{"size": "large"}"#);
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // SchemaAttrOutputFunctionsParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_attr_output_functions_parser() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "count": {"type": "integer"}
            },
            "required": ["items"]
        });
        let inner = SchemaOutputFunctionsParser::from_single_schema(schema);
        let parser = SchemaAttrOutputFunctionsParser::new(inner, "items");

        let gen =
            make_function_call_generation("list_items", r#"{"items": ["a", "b"], "count": 2}"#);
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result, json!(["a", "b"]));
    }

    #[test]
    fn test_schema_attr_output_functions_parser_missing_attr() {
        let inner = SchemaOutputFunctionsParser::from_single_schema(json!({}));
        let parser = SchemaAttrOutputFunctionsParser::new(inner, "nonexistent");

        let gen = make_function_call_generation("fn", r#"{"key": "value"}"#);
        let result = parser.parse_result(&[gen], false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Runnable impls
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_output_functions_parser_runnable() {
        let parser = OutputFunctionsParser::new();
        let input = json!({"name": "test_fn", "arguments": "{\"key\": \"value\"}"});
        // parse as string
        let result = parser
            .invoke(Value::String(serde_json::to_string(&input).unwrap()), None)
            .await
            .unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"key": "value"}"#);
    }

    #[tokio::test]
    async fn test_output_functions_parser_runnable_object_with_function_call() {
        let parser = OutputFunctionsParser::new();
        let input = json!({
            "function_call": {
                "name": "my_fn",
                "arguments": "{\"x\": 1}"
            }
        });
        let result = parser.invoke(input, None).await.unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"x": 1}"#);
    }

    #[tokio::test]
    async fn test_json_output_functions_parser_runnable() {
        let parser = JsonOutputFunctionsParser::new();
        let input = json!({"name": "fn", "arguments": "{\"city\": \"London\"}"});
        let result = parser
            .invoke(Value::String(serde_json::to_string(&input).unwrap()), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"city": "London"}));
    }

    #[tokio::test]
    async fn test_json_output_functions_parser_runnable_object() {
        let parser = JsonOutputFunctionsParser::new();
        let input = json!({
            "function_call": {
                "name": "fn",
                "arguments": "{\"city\": \"London\"}"
            }
        });
        let result = parser.invoke(input, None).await.unwrap();
        assert_eq!(result, json!({"city": "London"}));
    }

    #[tokio::test]
    async fn test_json_key_output_functions_parser_runnable() {
        let parser = JsonKeyOutputFunctionsParser::new("greeting");
        let input =
            json!({"name": "hello", "arguments": "{\"greeting\": \"hi\", \"lang\": \"en\"}"});
        let result = parser
            .invoke(Value::String(serde_json::to_string(&input).unwrap()), None)
            .await
            .unwrap();
        assert_eq!(result, json!("hi"));
    }

    #[tokio::test]
    async fn test_schema_output_functions_parser_runnable() {
        let schema = json!({"required": ["x"]});
        let parser = SchemaOutputFunctionsParser::from_single_schema(schema);
        let input = json!({"name": "fn", "arguments": "{\"x\": 42}"});
        let result = parser
            .invoke(Value::String(serde_json::to_string(&input).unwrap()), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"x": 42}));
    }

    #[tokio::test]
    async fn test_schema_attr_output_functions_parser_runnable() {
        let inner = SchemaOutputFunctionsParser::from_single_schema(json!({}));
        let parser = SchemaAttrOutputFunctionsParser::new(inner, "value");
        let input = json!({"name": "fn", "arguments": "{\"value\": 99}"});
        let result = parser
            .invoke(Value::String(serde_json::to_string(&input).unwrap()), None)
            .await
            .unwrap();
        assert_eq!(result, json!(99));
    }

    // -----------------------------------------------------------------------
    // Default / name tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_impls() {
        let p1 = OutputFunctionsParser::default();
        assert!(p1.args_only);

        let p2 = JsonOutputFunctionsParser::default();
        assert!(p2.args_only);
        assert!(!p2.strict);
    }

    #[test]
    fn test_parser_names() {
        assert_eq!(OutputFunctionsParser::new().name(), "OutputFunctionsParser");
        assert_eq!(
            JsonOutputFunctionsParser::new().name(),
            "JsonOutputFunctionsParser"
        );
        assert_eq!(
            JsonKeyOutputFunctionsParser::new("k").name(),
            "JsonKeyOutputFunctionsParser"
        );
        assert_eq!(
            SchemaOutputFunctionsParser::from_single_schema(json!({})).name(),
            "SchemaOutputFunctionsParser"
        );
        let inner = SchemaOutputFunctionsParser::from_single_schema(json!({}));
        assert_eq!(
            SchemaAttrOutputFunctionsParser::new(inner, "a").name(),
            "SchemaAttrOutputFunctionsParser"
        );
    }

    #[test]
    fn test_format_instructions_are_none() {
        assert!(OutputFunctionsParser::new()
            .get_format_instructions()
            .is_none());
        assert!(JsonOutputFunctionsParser::new()
            .get_format_instructions()
            .is_none());
        assert!(JsonKeyOutputFunctionsParser::new("k")
            .get_format_instructions()
            .is_none());
    }

    // -----------------------------------------------------------------------
    // Edge case: generation text contains function_call JSON
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_function_call_from_text() {
        let parser = OutputFunctionsParser::new();
        let gen = Generation {
            text: r#"{"function_call": {"name": "fn", "arguments": "{\"v\": 1}"}}"#.to_string(),
            generation_info: None,
        };
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"v": 1}"#);
    }

    #[test]
    fn test_extract_function_call_bare_object_in_text() {
        let parser = OutputFunctionsParser::new();
        let gen = Generation {
            text: r#"{"name": "fn", "arguments": "{\"v\": 2}"}"#.to_string(),
            generation_info: None,
        };
        let result = parser.parse_result(&[gen], false).unwrap();
        assert_eq!(result.as_str().unwrap(), r#"{"v": 2}"#);
    }
}
