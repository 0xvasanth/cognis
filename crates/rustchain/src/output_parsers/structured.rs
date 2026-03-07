//! Structured output parsers that extract typed data from LLM text output.
//!
//! This module provides a suite of output parsers for common output formats:
//!
//! - [`JsonOutputParser`] -- extracts JSON from text (handles code fences and surrounding text)
//! - [`MarkdownListParser`] -- parses markdown lists into a JSON array of strings
//! - [`KeyValueParser`] -- parses `key: value` lines into a JSON object
//! - [`RegexParser`] -- extracts named capture groups into a JSON object
//! - [`CommaSeparatedListParser`] -- parses CSV-style values into a JSON array
//! - [`BooleanParser`] -- parses yes/no/true/false/1/0 into a JSON boolean
//! - [`EnumParser`] -- validates output is one of allowed values
//! - [`CombiningParser`] -- tries multiple parsers in sequence, returns first success
//! - [`StructuredOutputParser`] -- validates parsed JSON against a JSON schema

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::output_parsers::OutputParser;
use rustchain_core::runnables::base::Runnable;
use rustchain_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parser_error(message: impl Into<String>, text: &str) -> RustChainError {
    RustChainError::OutputParserError {
        message: message.into(),
        observation: Some(text.to_string()),
        llm_output: Some(text.to_string()),
    }
}

/// Check whether a JSON value matches an expected JSON schema type string.
fn check_json_type(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
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

/// Strip markdown code fences from text if present.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if trimmed.starts_with("```") {
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
    }
}

/// Find the first balanced JSON block (`{...}` or `[...]`) in text.
fn find_json_block(text: &str) -> Option<&str> {
    let stripped = strip_code_fences(text);

    // Try parsing the whole stripped text first
    if serde_json::from_str::<Value>(stripped).is_ok() {
        return Some(stripped);
    }

    // Find first `{` or `[` and match its closing bracket
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let Some(start) = stripped.find(open) {
            let mut depth = 0i32;
            let mut in_string = false;
            let mut escape = false;
            for (i, ch) in stripped[start..].char_indices() {
                if escape {
                    escape = false;
                    continue;
                }
                if ch == '\\' && in_string {
                    escape = true;
                    continue;
                }
                if ch == '"' {
                    in_string = !in_string;
                    continue;
                }
                if in_string {
                    continue;
                }
                if ch == open {
                    depth += 1;
                } else if ch == close {
                    depth -= 1;
                    if depth == 0 {
                        let end = start + i + ch.len_utf8();
                        return Some(&stripped[start..end]);
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// JsonOutputParser
// ---------------------------------------------------------------------------

/// Extracts JSON from LLM text output.
///
/// Handles markdown code fences, surrounding prose, and finds the first valid
/// `{...}` or `[...]` block in the text.
pub struct JsonOutputParser;

impl JsonOutputParser {
    /// Create a new JSON output parser.
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for JsonOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let block = find_json_block(text)
            .ok_or_else(|| parser_error("No JSON object or array found in text", text))?;
        serde_json::from_str(block)
            .map_err(|e| parser_error(format!("Failed to parse JSON: {}", e), text))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a JSON object or array. \
             You may optionally wrap it in ```json code fences."
                .to_string(),
        )
    }

    fn parser_type(&self) -> &str {
        "json_output_parser"
    }
}

// ---------------------------------------------------------------------------
// MarkdownListParser
// ---------------------------------------------------------------------------

/// Parses markdown lists (ordered and unordered) into a JSON array of strings.
///
/// Recognises `- item`, `* item`, and `1. item` patterns.
pub struct MarkdownListParser;

impl MarkdownListParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MarkdownListParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for MarkdownListParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let re = Regex::new(r"(?m)^\s*(?:[-*]|\d+\.)\s+(.+)$").unwrap();
        let items: Vec<Value> = re
            .captures_iter(text)
            .map(|cap| Value::String(cap[1].trim().to_string()))
            .collect();
        if items.is_empty() {
            return Err(parser_error("No markdown list items found", text));
        }
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a markdown list. For example:\n\
             - item one\n\
             - item two\n\
             - item three"
                .to_string(),
        )
    }

    fn parser_type(&self) -> &str {
        "markdown_list_parser"
    }
}

// ---------------------------------------------------------------------------
// KeyValueParser
// ---------------------------------------------------------------------------

/// Parses `key: value` lines into a JSON object.
///
/// The separator is configurable (default `:`).
pub struct KeyValueParser {
    separator: String,
}

impl KeyValueParser {
    /// Create a new parser with the default separator `:`.
    pub fn new() -> Self {
        Self {
            separator: ":".to_string(),
        }
    }

    /// Create a parser with a custom separator.
    pub fn with_separator(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
        }
    }
}

impl Default for KeyValueParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for KeyValueParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let mut map = serde_json::Map::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(idx) = trimmed.find(&self.separator) {
                let key = trimmed[..idx].trim().to_string();
                let value = trimmed[idx + self.separator.len()..].trim().to_string();
                if !key.is_empty() {
                    map.insert(key, Value::String(value));
                }
            }
        }
        if map.is_empty() {
            return Err(parser_error(
                format!("No key-value pairs found (separator: '{}')", self.separator),
                text,
            ));
        }
        Ok(Value::Object(map))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(format!(
            "Your response should be key-value pairs, one per line, \
             separated by '{}'. For example:\n\
             name{} Alice\n\
             age{} 30",
            self.separator, self.separator, self.separator,
        ))
    }

    fn parser_type(&self) -> &str {
        "key_value_parser"
    }
}

// ---------------------------------------------------------------------------
// RegexParser
// ---------------------------------------------------------------------------

/// Extracts named capture groups from text using a regex pattern.
///
/// Returns a JSON object with group names as keys.
pub struct RegexParser {
    pattern: Regex,
    pattern_str: String,
}

impl RegexParser {
    /// Create a new regex parser from a pattern string.
    ///
    /// The pattern should contain named capture groups, e.g.
    /// `(?P<name>\w+) is (?P<age>\d+)`.
    pub fn new(pattern: &str) -> std::result::Result<Self, regex::Error> {
        let re = Regex::new(pattern)?;
        Ok(Self {
            pattern: re,
            pattern_str: pattern.to_string(),
        })
    }
}

impl OutputParser for RegexParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let caps = self
            .pattern
            .captures(text)
            .ok_or_else(|| parser_error("Regex pattern did not match", text))?;

        let mut map = serde_json::Map::new();
        for name in self.pattern.capture_names().flatten() {
            if let Some(m) = caps.name(name) {
                map.insert(name.to_string(), Value::String(m.as_str().to_string()));
            }
        }
        if map.is_empty() {
            return Err(parser_error(
                "Regex matched but no named groups were captured",
                text,
            ));
        }
        Ok(Value::Object(map))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(format!(
            "Your response should match the pattern: {}",
            self.pattern_str
        ))
    }

    fn parser_type(&self) -> &str {
        "regex_parser"
    }
}

// ---------------------------------------------------------------------------
// CommaSeparatedListParser
// ---------------------------------------------------------------------------

/// Parses comma-separated values into a JSON array of strings.
///
/// Trims whitespace from each value.
pub struct CommaSeparatedListParser;

impl CommaSeparatedListParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommaSeparatedListParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for CommaSeparatedListParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(parser_error("Empty input", text));
        }
        let items: Vec<Value> = trimmed
            .split(',')
            .map(|s| Value::String(s.trim().to_string()))
            .filter(|v| v.as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .collect();
        if items.is_empty() {
            return Err(parser_error(
                "No items found after splitting by comma",
                text,
            ));
        }
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a comma-separated list of values, \
             e.g.: foo, bar, baz"
                .to_string(),
        )
    }

    fn parser_type(&self) -> &str {
        "comma_separated_list_parser"
    }
}

// ---------------------------------------------------------------------------
// BooleanParser
// ---------------------------------------------------------------------------

/// Parses yes/no/true/false/1/0 into a JSON boolean (case-insensitive).
pub struct BooleanParser;

impl BooleanParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BooleanParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for BooleanParser {
    fn parse(&self, text: &str) -> Result<Value> {
        match text.trim().to_lowercase().as_str() {
            "true" | "yes" | "1" => Ok(Value::Bool(true)),
            "false" | "no" | "0" => Ok(Value::Bool(false)),
            _ => Err(parser_error(
                format!(
                    "Could not parse '{}' as boolean. Expected one of: true, false, yes, no, 1, 0",
                    text.trim()
                ),
                text,
            )),
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Your response should be a single boolean value: \
             true/false, yes/no, or 1/0."
                .to_string(),
        )
    }

    fn parser_type(&self) -> &str {
        "boolean_parser"
    }
}

// ---------------------------------------------------------------------------
// EnumParser
// ---------------------------------------------------------------------------

/// Validates that the output is one of a set of allowed values (case-insensitive).
pub struct EnumParser {
    /// Allowed values (stored in lowercase for comparison).
    allowed: Vec<String>,
    /// Original allowed values for display.
    display_values: Vec<String>,
}

impl EnumParser {
    /// Create a new enum parser with allowed values.
    pub fn new(allowed: Vec<String>) -> Self {
        let lower: Vec<String> = allowed.iter().map(|s| s.to_lowercase()).collect();
        Self {
            allowed: lower,
            display_values: allowed,
        }
    }
}

impl OutputParser for EnumParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let trimmed = text.trim().to_lowercase();
        if let Some(idx) = self.allowed.iter().position(|a| *a == trimmed) {
            Ok(Value::String(self.display_values[idx].clone()))
        } else {
            Err(parser_error(
                format!(
                    "'{}' is not one of the allowed values: [{}]",
                    text.trim(),
                    self.display_values.join(", ")
                ),
                text,
            ))
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(format!(
            "Your response should be one of the following values: {}",
            self.display_values.join(", ")
        ))
    }

    fn parser_type(&self) -> &str {
        "enum_parser"
    }
}

// ---------------------------------------------------------------------------
// CombiningParser
// ---------------------------------------------------------------------------

/// Tries multiple parsers in sequence, returning the first successful result.
pub struct CombiningParser {
    parsers: Vec<Box<dyn OutputParser>>,
}

impl CombiningParser {
    /// Create a combining parser from a list of parsers.
    pub fn new(parsers: Vec<Box<dyn OutputParser>>) -> Self {
        Self { parsers }
    }
}

impl OutputParser for CombiningParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let mut last_err = None;
        for parser in &self.parsers {
            match parser.parse(text) {
                Ok(v) => return Ok(v),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| parser_error("No parsers configured", text)))
    }

    fn get_format_instructions(&self) -> Option<String> {
        let instructions: Vec<String> = self
            .parsers
            .iter()
            .filter_map(|p| p.get_format_instructions())
            .collect();
        if instructions.is_empty() {
            None
        } else {
            Some(instructions.join("\n\nOR\n\n"))
        }
    }

    fn parser_type(&self) -> &str {
        "combining_parser"
    }
}

// ---------------------------------------------------------------------------
// StructuredOutputParser
// ---------------------------------------------------------------------------

/// An output parser that parses JSON and validates it against a JSON schema.
///
/// Generates format instructions from the schema and returns structured error
/// messages for invalid or missing fields.
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

    /// Parse JSON from text using the shared JSON extraction logic.
    fn parse_json(&self, text: &str) -> Result<Value> {
        let block = find_json_block(text).ok_or_else(|| RustChainError::OutputParserError {
            message: format!("No JSON found in {} output", self.type_name),
            observation: Some(text.to_string()),
            llm_output: Some(text.to_string()),
        })?;
        serde_json::from_str(block).map_err(|e| RustChainError::OutputParserError {
            message: format!("Failed to parse JSON for {}: {}", self.type_name, e),
            observation: Some(block.to_string()),
            llm_output: Some(text.to_string()),
        })
    }

    /// Validate parsed JSON against the schema.
    fn validate(&self, value: &Value) -> Result<()> {
        // Check top-level type
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
                let missing: Vec<&str> = required
                    .iter()
                    .filter_map(|r| r.as_str())
                    .filter(|field| !map.contains_key(*field))
                    .collect();
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

impl OutputParser for StructuredOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let value = self.parse_json(text)?;
        self.validate(&value)?;
        Ok(value)
    }

    fn get_format_instructions(&self) -> Option<String> {
        let mut display_schema = self.schema.clone();
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- JsonOutputParser ---------------------------------------------------

    #[test]
    fn json_parser_clean_object() {
        let parser = JsonOutputParser::new();
        let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn json_parser_clean_array() {
        let parser = JsonOutputParser::new();
        let result = parser.parse(r#"[1, 2, 3]"#).unwrap();
        assert_eq!(result, json!([1, 2, 3]));
    }

    #[test]
    fn json_parser_code_fence() {
        let parser = JsonOutputParser::new();
        let text = "```json\n{\"key\": \"value\"}\n```";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn json_parser_code_fence_no_lang() {
        let parser = JsonOutputParser::new();
        let text = "```\n{\"key\": \"value\"}\n```";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn json_parser_surrounding_text() {
        let parser = JsonOutputParser::new();
        let text = "Here is the result:\n{\"answer\": 42}\nHope that helps!";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"answer": 42}));
    }

    #[test]
    fn json_parser_no_json() {
        let parser = JsonOutputParser::new();
        assert!(parser.parse("no json here").is_err());
    }

    #[test]
    fn json_parser_empty_input() {
        let parser = JsonOutputParser::new();
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn json_parser_nested_object() {
        let parser = JsonOutputParser::new();
        let text = r#"Result: {"outer": {"inner": [1, 2]}}"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"outer": {"inner": [1, 2]}}));
    }

    #[test]
    fn json_parser_format_instructions() {
        let parser = JsonOutputParser::new();
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("JSON"));
    }

    // -- MarkdownListParser -------------------------------------------------

    #[test]
    fn markdown_list_unordered_dash() {
        let parser = MarkdownListParser::new();
        let text = "- apple\n- banana\n- cherry";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["apple", "banana", "cherry"]));
    }

    #[test]
    fn markdown_list_unordered_star() {
        let parser = MarkdownListParser::new();
        let text = "* one\n* two\n* three";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["one", "two", "three"]));
    }

    #[test]
    fn markdown_list_ordered() {
        let parser = MarkdownListParser::new();
        let text = "1. first\n2. second\n3. third";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["first", "second", "third"]));
    }

    #[test]
    fn markdown_list_mixed_with_prose() {
        let parser = MarkdownListParser::new();
        let text = "Here are the items:\n- alpha\n- beta\nEnd of list.";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["alpha", "beta"]));
    }

    #[test]
    fn markdown_list_empty() {
        let parser = MarkdownListParser::new();
        assert!(parser.parse("no list here").is_err());
    }

    #[test]
    fn markdown_list_format_instructions() {
        let parser = MarkdownListParser::new();
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("markdown list"));
    }

    // -- KeyValueParser -----------------------------------------------------

    #[test]
    fn kv_parser_default_separator() {
        let parser = KeyValueParser::new();
        let text = "name: Alice\nage: 30\ncity: Wonderland";
        let result = parser.parse(text).unwrap();
        assert_eq!(
            result,
            json!({"name": "Alice", "age": "30", "city": "Wonderland"})
        );
    }

    #[test]
    fn kv_parser_custom_separator() {
        let parser = KeyValueParser::with_separator("=");
        let text = "host=localhost\nport=8080";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"host": "localhost", "port": "8080"}));
    }

    #[test]
    fn kv_parser_whitespace_handling() {
        let parser = KeyValueParser::new();
        let text = "  key  :  value  \n\n  other  :  data  ";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"key": "value", "other": "data"}));
    }

    #[test]
    fn kv_parser_no_pairs() {
        let parser = KeyValueParser::new();
        assert!(parser.parse("no pairs here").is_err());
    }

    #[test]
    fn kv_parser_format_instructions() {
        let parser = KeyValueParser::with_separator("=");
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("="));
    }

    // -- RegexParser --------------------------------------------------------

    #[test]
    fn regex_parser_named_groups() {
        let parser = RegexParser::new(r"(?P<name>\w+) is (?P<age>\d+) years old").unwrap();
        let result = parser.parse("Alice is 30 years old").unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": "30"}));
    }

    #[test]
    fn regex_parser_no_match() {
        let parser = RegexParser::new(r"(?P<name>\w+) is (?P<age>\d+)").unwrap();
        assert!(parser.parse("nothing matches here!").is_err());
    }

    #[test]
    fn regex_parser_format_instructions() {
        let parser = RegexParser::new(r"(?P<x>\d+)").unwrap();
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("pattern"));
    }

    #[test]
    fn regex_parser_partial_text() {
        let parser = RegexParser::new(r"Answer: (?P<answer>\w+)").unwrap();
        let result = parser
            .parse("The model says Answer: yes and then continues")
            .unwrap();
        assert_eq!(result, json!({"answer": "yes"}));
    }

    // -- CommaSeparatedListParser -------------------------------------------

    #[test]
    fn csv_parser_basic() {
        let parser = CommaSeparatedListParser::new();
        let result = parser.parse("apple, banana, cherry").unwrap();
        assert_eq!(result, json!(["apple", "banana", "cherry"]));
    }

    #[test]
    fn csv_parser_whitespace_trimming() {
        let parser = CommaSeparatedListParser::new();
        let result = parser.parse("  one ,  two , three  ").unwrap();
        assert_eq!(result, json!(["one", "two", "three"]));
    }

    #[test]
    fn csv_parser_single_item() {
        let parser = CommaSeparatedListParser::new();
        let result = parser.parse("solo").unwrap();
        assert_eq!(result, json!(["solo"]));
    }

    #[test]
    fn csv_parser_empty() {
        let parser = CommaSeparatedListParser::new();
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn csv_parser_format_instructions() {
        let parser = CommaSeparatedListParser::new();
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("comma"));
    }

    // -- BooleanParser ------------------------------------------------------

    #[test]
    fn boolean_parser_true_variants() {
        let parser = BooleanParser::new();
        assert_eq!(parser.parse("true").unwrap(), json!(true));
        assert_eq!(parser.parse("True").unwrap(), json!(true));
        assert_eq!(parser.parse("TRUE").unwrap(), json!(true));
        assert_eq!(parser.parse("yes").unwrap(), json!(true));
        assert_eq!(parser.parse("Yes").unwrap(), json!(true));
        assert_eq!(parser.parse("1").unwrap(), json!(true));
    }

    #[test]
    fn boolean_parser_false_variants() {
        let parser = BooleanParser::new();
        assert_eq!(parser.parse("false").unwrap(), json!(false));
        assert_eq!(parser.parse("False").unwrap(), json!(false));
        assert_eq!(parser.parse("no").unwrap(), json!(false));
        assert_eq!(parser.parse("NO").unwrap(), json!(false));
        assert_eq!(parser.parse("0").unwrap(), json!(false));
    }

    #[test]
    fn boolean_parser_whitespace() {
        let parser = BooleanParser::new();
        assert_eq!(parser.parse("  true  ").unwrap(), json!(true));
        assert_eq!(parser.parse("\nfalse\n").unwrap(), json!(false));
    }

    #[test]
    fn boolean_parser_invalid() {
        let parser = BooleanParser::new();
        assert!(parser.parse("maybe").is_err());
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn boolean_parser_format_instructions() {
        let parser = BooleanParser::new();
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("true/false"));
    }

    // -- EnumParser ---------------------------------------------------------

    #[test]
    fn enum_parser_match() {
        let parser = EnumParser::new(vec!["Red".into(), "Green".into(), "Blue".into()]);
        assert_eq!(parser.parse("red").unwrap(), json!("Red"));
        assert_eq!(parser.parse("GREEN").unwrap(), json!("Green"));
        assert_eq!(parser.parse("Blue").unwrap(), json!("Blue"));
    }

    #[test]
    fn enum_parser_no_match() {
        let parser = EnumParser::new(vec!["Red".into(), "Green".into(), "Blue".into()]);
        assert!(parser.parse("Yellow").is_err());
    }

    #[test]
    fn enum_parser_whitespace() {
        let parser = EnumParser::new(vec!["yes".into(), "no".into()]);
        assert_eq!(parser.parse("  yes  ").unwrap(), json!("yes"));
    }

    #[test]
    fn enum_parser_format_instructions() {
        let parser = EnumParser::new(vec!["A".into(), "B".into()]);
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("A"));
        assert!(instructions.contains("B"));
    }

    // -- CombiningParser ----------------------------------------------------

    #[test]
    fn combining_parser_first_succeeds() {
        let parsers: Vec<Box<dyn OutputParser>> = vec![
            Box::new(JsonOutputParser::new()),
            Box::new(CommaSeparatedListParser::new()),
        ];
        let parser = CombiningParser::new(parsers);
        let result = parser.parse(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn combining_parser_fallback() {
        let parsers: Vec<Box<dyn OutputParser>> = vec![
            Box::new(JsonOutputParser::new()),
            Box::new(CommaSeparatedListParser::new()),
        ];
        let parser = CombiningParser::new(parsers);
        let result = parser.parse("apple, banana, cherry").unwrap();
        assert_eq!(result, json!(["apple", "banana", "cherry"]));
    }

    #[test]
    fn combining_parser_all_fail() {
        let parsers: Vec<Box<dyn OutputParser>> = vec![
            Box::new(JsonOutputParser::new()),
            Box::new(BooleanParser::new()),
        ];
        let parser = CombiningParser::new(parsers);
        assert!(parser.parse("not json or bool").is_err());
    }

    #[test]
    fn combining_parser_format_instructions() {
        let parsers: Vec<Box<dyn OutputParser>> = vec![
            Box::new(JsonOutputParser::new()),
            Box::new(BooleanParser::new()),
        ];
        let parser = CombiningParser::new(parsers);
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("OR"));
    }

    // -- StructuredOutputParser ---------------------------------------------

    #[test]
    fn structured_parser_valid() {
        let parser = StructuredOutputParser::builder()
            .type_name("Person")
            .schema(json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }))
            .build();
        let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn structured_parser_missing_required() {
        let parser = StructuredOutputParser::new(
            "Person",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }),
        );
        let err = parser.parse(r#"{"name": "Alice"}"#).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Missing required"));
        assert!(msg.contains("age"));
    }

    #[test]
    fn structured_parser_wrong_type() {
        let parser = StructuredOutputParser::new(
            "Data",
            json!({
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                },
                "required": ["count"]
            }),
        );
        let err = parser.parse(r#"{"count": "not a number"}"#).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Type validation"));
    }

    #[test]
    fn structured_parser_not_object() {
        let parser = StructuredOutputParser::new(
            "Data",
            json!({
                "type": "object",
                "properties": {}
            }),
        );
        let err = parser.parse("[1, 2, 3]").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Expected JSON object"));
    }

    #[test]
    fn structured_parser_code_fence() {
        let parser = StructuredOutputParser::new(
            "T",
            json!({
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                },
                "required": ["x"]
            }),
        );
        let text = "```json\n{\"x\": 3.14}\n```";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"x": 3.14}));
    }

    #[test]
    fn structured_parser_format_instructions() {
        let parser = StructuredOutputParser::new(
            "T",
            json!({
                "type": "object",
                "properties": {
                    "foo": {"type": "string"}
                }
            }),
        );
        let instructions = parser.get_format_instructions().unwrap();
        assert!(instructions.contains("JSON schema"));
    }

    #[test]
    fn structured_parser_surrounding_text() {
        let parser = StructuredOutputParser::new(
            "T",
            json!({
                "type": "object",
                "properties": {
                    "val": {"type": "string"}
                },
                "required": ["val"]
            }),
        );
        let text = "Here is the output:\n{\"val\": \"hello\"}\nDone.";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"val": "hello"}));
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn json_parser_string_with_braces() {
        let parser = JsonOutputParser::new();
        let text = r#"{"msg": "hello {world}"}"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"msg": "hello {world}"}));
    }

    #[test]
    fn kv_parser_value_with_separator() {
        let parser = KeyValueParser::new();
        // The value itself contains ':'
        let text = "url: https://example.com";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"url": "https://example.com"}));
    }

    #[test]
    fn csv_parser_trailing_comma() {
        let parser = CommaSeparatedListParser::new();
        let result = parser.parse("a, b, c,").unwrap();
        assert_eq!(result, json!(["a", "b", "c"]));
    }
}
