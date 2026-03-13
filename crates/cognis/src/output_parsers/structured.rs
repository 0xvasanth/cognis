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

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use cognis_core::error::{Result, CognisError};
use cognis_core::output_parsers::OutputParser;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parser_error(message: impl Into<String>, text: &str) -> CognisError {
    CognisError::OutputParserError {
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
/// use cognis::output_parsers::StructuredOutputParser;
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
        let block = find_json_block(text).ok_or_else(|| CognisError::OutputParserError {
            message: format!("No JSON found in {} output", self.type_name),
            observation: Some(text.to_string()),
            llm_output: Some(text.to_string()),
        })?;
        serde_json::from_str(block).map_err(|e| CognisError::OutputParserError {
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
                return Err(CognisError::OutputParserError {
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
                    return Err(CognisError::OutputParserError {
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
                return Err(CognisError::OutputParserError {
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

// ---------------------------------------------------------------------------
// OutputFormat
// ---------------------------------------------------------------------------

/// Represents the format of parsed output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// JSON format.
    Json,
    /// YAML format.
    Yaml,
    /// CSV format.
    Csv,
    /// Markdown format.
    Markdown,
    /// Plain text format.
    PlainText,
    /// Custom format with a name.
    Custom(String),
}

impl OutputFormat {
    /// Returns the name of this format.
    pub fn name(&self) -> &str {
        match self {
            OutputFormat::Json => "json",
            OutputFormat::Yaml => "yaml",
            OutputFormat::Csv => "csv",
            OutputFormat::Markdown => "markdown",
            OutputFormat::PlainText => "plain_text",
            OutputFormat::Custom(name) => name.as_str(),
        }
    }

    /// Converts this format to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "format": self.name()
        })
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ParseResult<T>
// ---------------------------------------------------------------------------

/// A result wrapper that includes parsing metadata alongside the parsed value.
#[derive(Debug, Clone)]
pub struct ParseResult<T> {
    /// The parsed value.
    pub value: T,
    /// The raw input string that was parsed.
    pub raw_input: String,
    /// The format of the parsed output.
    pub format: OutputFormat,
    /// Time taken to parse in microseconds.
    pub parse_duration_us: u64,
    /// Any warnings generated during parsing.
    pub warnings: Vec<String>,
}

impl<T> ParseResult<T> {
    /// Create a new parse result with zero duration and no warnings.
    pub fn new(value: T, raw_input: impl Into<String>, format: OutputFormat) -> Self {
        Self {
            value,
            raw_input: raw_input.into(),
            format,
            parse_duration_us: 0,
            warnings: Vec::new(),
        }
    }

    /// Add a warning message to this parse result.
    pub fn with_warning(mut self, msg: impl Into<String>) -> Self {
        self.warnings.push(msg.into());
        self
    }

    /// Returns true if this parse result has any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl<T: Serialize> ParseResult<T> {
    /// Converts this parse result to a JSON value including metadata.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "value": serde_json::to_value(&self.value).unwrap_or(Value::Null),
            "raw_input": self.raw_input,
            "format": self.format.name(),
            "parse_duration_us": self.parse_duration_us,
            "warnings": self.warnings,
        })
    }
}

// ---------------------------------------------------------------------------
// JsonType
// ---------------------------------------------------------------------------

/// Represents a JSON type for schema validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonType {
    /// JSON string type.
    String,
    /// JSON number type.
    Number,
    /// JSON boolean type.
    Bool,
    /// JSON array type.
    Array,
    /// JSON object type.
    Object,
    /// JSON null type.
    Null,
}

impl JsonType {
    /// Returns true if the given JSON value matches this type.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            JsonType::String => value.is_string(),
            JsonType::Number => value.is_number(),
            JsonType::Bool => value.is_boolean(),
            JsonType::Array => value.is_array(),
            JsonType::Object => value.is_object(),
            JsonType::Null => value.is_null(),
        }
    }
}

impl fmt::Display for JsonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonType::String => write!(f, "string"),
            JsonType::Number => write!(f, "number"),
            JsonType::Bool => write!(f, "boolean"),
            JsonType::Array => write!(f, "array"),
            JsonType::Object => write!(f, "object"),
            JsonType::Null => write!(f, "null"),
        }
    }
}

// ---------------------------------------------------------------------------
// ViolationType
// ---------------------------------------------------------------------------

/// Describes the kind of schema violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    /// A required field is missing.
    MissingField,
    /// A field has the wrong type.
    WrongType {
        /// The expected JSON type.
        expected: JsonType,
        /// The actual type name found.
        actual: String,
    },
    /// A field has an invalid value.
    InvalidValue(String),
}

impl fmt::Display for ViolationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationType::MissingField => write!(f, "missing required field"),
            ViolationType::WrongType { expected, actual } => {
                write!(f, "expected type {}, got {}", expected, actual)
            }
            ViolationType::InvalidValue(reason) => write!(f, "invalid value: {}", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaViolation
// ---------------------------------------------------------------------------

/// Represents a single schema violation found during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaViolation {
    /// The field that violated the schema.
    pub field: String,
    /// The type of violation.
    pub violation: ViolationType,
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "field '{}': {}", self.field, self.violation)
    }
}

// ---------------------------------------------------------------------------
// SchemaEnforcer
// ---------------------------------------------------------------------------

/// Validates parsed JSON output against an expected schema defined
/// programmatically via required fields and type constraints.
pub struct SchemaEnforcer {
    required_fields: Vec<String>,
    type_constraints: HashMap<String, JsonType>,
    optional_defaults: HashMap<String, Value>,
}

impl SchemaEnforcer {
    /// Create a new empty schema enforcer.
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
            type_constraints: HashMap::new(),
            optional_defaults: HashMap::new(),
        }
    }

    /// Mark a field as required.
    pub fn require_field(mut self, name: impl Into<String>) -> Self {
        self.required_fields.push(name.into());
        self
    }

    /// Require a field to have a specific JSON type.
    pub fn require_type(mut self, field: impl Into<String>, expected_type: JsonType) -> Self {
        self.type_constraints.insert(field.into(), expected_type);
        self
    }

    /// Mark a field as optional with a default value that will be inserted if
    /// the field is missing.
    pub fn optional_field(mut self, name: impl Into<String>, default: Value) -> Self {
        self.optional_defaults.insert(name.into(), default);
        self
    }

    /// Validate a JSON value against this schema.
    ///
    /// Returns the (potentially modified) value with defaults applied on
    /// success, or a list of violations on failure.
    pub fn validate(&self, value: &Value) -> std::result::Result<Value, Vec<SchemaViolation>> {
        let mut violations = Vec::new();

        let obj = match value.as_object() {
            Some(obj) => obj.clone(),
            None => {
                violations.push(SchemaViolation {
                    field: "<root>".to_string(),
                    violation: ViolationType::WrongType {
                        expected: JsonType::Object,
                        actual: json_value_type_name(value).to_string(),
                    },
                });
                return Err(violations);
            }
        };

        // Check required fields
        for field in &self.required_fields {
            if !obj.contains_key(field) {
                violations.push(SchemaViolation {
                    field: field.clone(),
                    violation: ViolationType::MissingField,
                });
            }
        }

        // Check type constraints
        for (field, expected_type) in &self.type_constraints {
            if let Some(val) = obj.get(field) {
                if !expected_type.matches(val) {
                    violations.push(SchemaViolation {
                        field: field.clone(),
                        violation: ViolationType::WrongType {
                            expected: expected_type.clone(),
                            actual: json_value_type_name(val).to_string(),
                        },
                    });
                }
            }
        }

        if !violations.is_empty() {
            return Err(violations);
        }

        // Apply defaults for missing optional fields
        let mut result = obj;
        for (field, default) in &self.optional_defaults {
            if !result.contains_key(field) {
                result.insert(field.clone(), default.clone());
            }
        }

        Ok(Value::Object(result))
    }
}

impl Default for SchemaEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns a human-readable type name for a JSON value.
fn json_value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// StructuredParser
// ---------------------------------------------------------------------------

/// A general-purpose parser that extracts structured data from LLM text output.
///
/// Provides methods for parsing JSON, key-value pairs, lists, and markdown
/// tables, returning [`ParseResult`] wrappers with metadata.
pub struct StructuredParser;

impl StructuredParser {
    /// Create a new structured parser.
    pub fn new() -> Self {
        Self
    }

    /// Parse JSON from the input string, deserializing into the given type.
    pub fn parse_json<T: DeserializeOwned>(&self, input: &str) -> Result<ParseResult<Value>> {
        let start = Instant::now();
        let stripped = strip_code_fences(input);
        let value: Value = serde_json::from_str(stripped)
            .map_err(|e| parser_error(format!("Failed to parse JSON: {}", e), input))?;
        let duration = start.elapsed().as_micros() as u64;
        let mut result = ParseResult::new(value, input, OutputFormat::Json);
        result.parse_duration_us = duration;
        Ok(result)
    }

    /// Extract JSON from a markdown code block (```json ... ```).
    ///
    /// If no code block is found, attempts to find a raw JSON object/array.
    pub fn parse_json_block(&self, input: &str) -> Result<Value> {
        let block = find_json_block(input)
            .ok_or_else(|| parser_error("No JSON block found in input", input))?;
        serde_json::from_str(block)
            .map_err(|e| parser_error(format!("Failed to parse JSON block: {}", e), input))
    }

    /// Parse "key: value" lines into a hash map.
    pub fn parse_key_value(&self, input: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(idx) = trimmed.find(':') {
                let key = trimmed[..idx].trim().to_string();
                let value = trimmed[idx + 1..].trim().to_string();
                if !key.is_empty() {
                    map.insert(key, value);
                }
            }
        }
        map
    }

    /// Parse numbered or bulleted lists into a vector of strings.
    ///
    /// Recognises `- item`, `* item`, and `1. item` patterns.
    pub fn parse_list(&self, input: &str) -> Vec<String> {
        let re = Regex::new(r"(?m)^\s*(?:[-*]|\d+\.)\s+(.+)$").unwrap();
        re.captures_iter(input)
            .map(|cap| cap[1].trim().to_string())
            .collect()
    }

    /// Parse a markdown table into a vector of rows (each row is a vector of
    /// cell strings).
    ///
    /// The header row is included as the first element. Separator rows
    /// (e.g. `| --- | --- |`) are excluded.
    pub fn parse_table(&self, input: &str) -> Vec<Vec<String>> {
        let mut rows = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
                continue;
            }
            // Skip separator rows like | --- | --- |
            let inner = &trimmed[1..trimmed.len() - 1];
            let is_separator = inner.split('|').all(|cell| {
                let c = cell.trim();
                c.chars().all(|ch| ch == '-' || ch == ':') && !c.is_empty()
            });
            if is_separator {
                continue;
            }
            let cells: Vec<String> = inner
                .split('|')
                .map(|cell| cell.trim().to_string())
                .collect();
            rows.push(cells);
        }
        rows
    }
}

impl Default for StructuredParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OutputRepairer
// ---------------------------------------------------------------------------

/// Attempts to repair malformed LLM output, particularly broken JSON.
pub struct OutputRepairer;

impl OutputRepairer {
    /// Create a new output repairer.
    pub fn new() -> Self {
        Self
    }

    /// Attempt to repair common JSON issues in the input string.
    ///
    /// Fixes:
    /// - Trailing commas before `}` or `]`
    /// - Single quotes used instead of double quotes
    /// - Unquoted keys in objects
    /// - Strips markdown code fences
    pub fn repair_json(&self, input: &str) -> Result<String> {
        let stripped = strip_code_fences(input).to_string();

        // If it already parses, return as-is
        if serde_json::from_str::<Value>(&stripped).is_ok() {
            return Ok(stripped);
        }

        let mut repaired = stripped;

        // Replace single quotes with double quotes (naive but handles common cases)
        repaired = self.replace_single_quotes(&repaired);

        // Fix unquoted keys: word followed by colon
        let unquoted_key_re = Regex::new(r"(?m)([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:").unwrap();
        repaired = unquoted_key_re
            .replace_all(&repaired, |caps: &regex::Captures| {
                format!("{}\"{}\":", &caps[1], &caps[2])
            })
            .to_string();

        // Remove trailing commas before } or ]
        let trailing_comma_re = Regex::new(r",\s*([}\]])").unwrap();
        repaired = trailing_comma_re.replace_all(&repaired, "$1").to_string();

        // Validate the repaired JSON
        serde_json::from_str::<Value>(&repaired)
            .map_err(|e| parser_error(format!("JSON repair failed: {}", e), input))?;

        Ok(repaired)
    }

    /// Replace single quotes with double quotes in a JSON-like string.
    ///
    /// This is a simplified approach that handles common LLM output patterns.
    fn replace_single_quotes(&self, input: &str) -> String {
        let mut result = String::with_capacity(input.len());
        let chars = input.chars().peekable();
        let mut in_double_string = false;
        let mut in_single_string = false;
        let mut prev_was_escape = false;

        for ch in chars {
            if prev_was_escape {
                result.push(ch);
                prev_was_escape = false;
                continue;
            }
            if ch == '\\' && (in_double_string || in_single_string) {
                result.push(ch);
                prev_was_escape = true;
                continue;
            }
            if ch == '"' && !in_single_string {
                in_double_string = !in_double_string;
                result.push(ch);
            } else if ch == '\'' && !in_double_string {
                in_single_string = !in_single_string;
                result.push('"');
            } else {
                result.push(ch);
            }
        }
        result
    }

    /// Extract text between a start and end delimiter.
    pub fn extract_between(&self, input: &str, start: &str, end: &str) -> Option<String> {
        let start_idx = input.find(start)?;
        let after_start = start_idx + start.len();
        let rest = &input[after_start..];
        let end_idx = rest.find(end)?;
        Some(rest[..end_idx].to_string())
    }

    /// Strip markdown code fences from input text.
    pub fn strip_markdown_fences(&self, input: &str) -> String {
        strip_code_fences(input).to_string()
    }
}

impl Default for OutputRepairer {
    fn default() -> Self {
        Self::new()
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

    // -- OutputFormat -------------------------------------------------------

    #[test]
    fn output_format_name_json() {
        assert_eq!(OutputFormat::Json.name(), "json");
    }

    #[test]
    fn output_format_name_yaml() {
        assert_eq!(OutputFormat::Yaml.name(), "yaml");
    }

    #[test]
    fn output_format_name_csv() {
        assert_eq!(OutputFormat::Csv.name(), "csv");
    }

    #[test]
    fn output_format_name_markdown() {
        assert_eq!(OutputFormat::Markdown.name(), "markdown");
    }

    #[test]
    fn output_format_name_plain_text() {
        assert_eq!(OutputFormat::PlainText.name(), "plain_text");
    }

    #[test]
    fn output_format_name_custom() {
        let f = OutputFormat::Custom("protobuf".to_string());
        assert_eq!(f.name(), "protobuf");
    }

    #[test]
    fn output_format_display() {
        assert_eq!(format!("{}", OutputFormat::Json), "json");
        assert_eq!(
            format!("{}", OutputFormat::Custom("xml".to_string())),
            "xml"
        );
    }

    #[test]
    fn output_format_to_json() {
        let v = OutputFormat::Json.to_json();
        assert_eq!(v["format"], "json");
    }

    #[test]
    fn output_format_equality() {
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_ne!(OutputFormat::Json, OutputFormat::Yaml);
    }

    // -- ParseResult --------------------------------------------------------

    #[test]
    fn parse_result_new() {
        let pr = ParseResult::new(42, "input", OutputFormat::Json);
        assert_eq!(pr.value, 42);
        assert_eq!(pr.raw_input, "input");
        assert_eq!(pr.format, OutputFormat::Json);
        assert_eq!(pr.parse_duration_us, 0);
        assert!(pr.warnings.is_empty());
    }

    #[test]
    fn parse_result_with_warning() {
        let pr = ParseResult::new("val", "raw", OutputFormat::PlainText)
            .with_warning("first warning")
            .with_warning("second warning");
        assert!(pr.has_warnings());
        assert_eq!(pr.warnings.len(), 2);
        assert_eq!(pr.warnings[0], "first warning");
    }

    #[test]
    fn parse_result_no_warnings() {
        let pr = ParseResult::new(1, "", OutputFormat::Json);
        assert!(!pr.has_warnings());
    }

    #[test]
    fn parse_result_to_json() {
        let pr =
            ParseResult::new(json!({"a": 1}), "raw text", OutputFormat::Json).with_warning("w1");
        let j = pr.to_json();
        assert_eq!(j["value"]["a"], 1);
        assert_eq!(j["raw_input"], "raw text");
        assert_eq!(j["format"], "json");
        assert_eq!(j["warnings"][0], "w1");
    }

    #[test]
    fn parse_result_to_json_primitive() {
        let pr = ParseResult::new(42_i32, "42", OutputFormat::PlainText);
        let j = pr.to_json();
        assert_eq!(j["value"], 42);
    }

    // -- JsonType -----------------------------------------------------------

    #[test]
    fn json_type_matches_string() {
        assert!(JsonType::String.matches(&json!("hello")));
        assert!(!JsonType::String.matches(&json!(42)));
    }

    #[test]
    fn json_type_matches_number() {
        assert!(JsonType::Number.matches(&json!(3.14)));
        assert!(JsonType::Number.matches(&json!(42)));
        assert!(!JsonType::Number.matches(&json!("42")));
    }

    #[test]
    fn json_type_matches_bool() {
        assert!(JsonType::Bool.matches(&json!(true)));
        assert!(JsonType::Bool.matches(&json!(false)));
        assert!(!JsonType::Bool.matches(&json!(1)));
    }

    #[test]
    fn json_type_matches_array() {
        assert!(JsonType::Array.matches(&json!([1, 2, 3])));
        assert!(!JsonType::Array.matches(&json!({"a": 1})));
    }

    #[test]
    fn json_type_matches_object() {
        assert!(JsonType::Object.matches(&json!({"key": "val"})));
        assert!(!JsonType::Object.matches(&json!([1])));
    }

    #[test]
    fn json_type_matches_null() {
        assert!(JsonType::Null.matches(&json!(null)));
        assert!(!JsonType::Null.matches(&json!(0)));
    }

    #[test]
    fn json_type_display() {
        assert_eq!(format!("{}", JsonType::String), "string");
        assert_eq!(format!("{}", JsonType::Number), "number");
        assert_eq!(format!("{}", JsonType::Bool), "boolean");
        assert_eq!(format!("{}", JsonType::Array), "array");
        assert_eq!(format!("{}", JsonType::Object), "object");
        assert_eq!(format!("{}", JsonType::Null), "null");
    }

    // -- ViolationType / SchemaViolation ------------------------------------

    #[test]
    fn violation_type_display_missing() {
        assert_eq!(
            format!("{}", ViolationType::MissingField),
            "missing required field"
        );
    }

    #[test]
    fn violation_type_display_wrong_type() {
        let v = ViolationType::WrongType {
            expected: JsonType::String,
            actual: "number".to_string(),
        };
        assert_eq!(format!("{}", v), "expected type string, got number");
    }

    #[test]
    fn violation_type_display_invalid_value() {
        let v = ViolationType::InvalidValue("must be positive".to_string());
        assert_eq!(format!("{}", v), "invalid value: must be positive");
    }

    #[test]
    fn schema_violation_display() {
        let sv = SchemaViolation {
            field: "name".to_string(),
            violation: ViolationType::MissingField,
        };
        assert_eq!(format!("{}", sv), "field 'name': missing required field");
    }

    // -- SchemaEnforcer -----------------------------------------------------

    #[test]
    fn schema_enforcer_valid_object() {
        let enforcer = SchemaEnforcer::new()
            .require_field("name")
            .require_field("age");
        let val = json!({"name": "Alice", "age": 30});
        let result = enforcer.validate(&val);
        assert!(result.is_ok());
    }

    #[test]
    fn schema_enforcer_missing_field() {
        let enforcer = SchemaEnforcer::new()
            .require_field("name")
            .require_field("age");
        let val = json!({"name": "Alice"});
        let err = enforcer.validate(&val).unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].field, "age");
        assert_eq!(err[0].violation, ViolationType::MissingField);
    }

    #[test]
    fn schema_enforcer_wrong_type() {
        let enforcer = SchemaEnforcer::new()
            .require_field("count")
            .require_type("count", JsonType::Number);
        let val = json!({"count": "not a number"});
        let err = enforcer.validate(&val).unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].field, "count");
        matches!(&err[0].violation, ViolationType::WrongType { .. });
    }

    #[test]
    fn schema_enforcer_optional_default_applied() {
        let enforcer = SchemaEnforcer::new().optional_field("status", json!("active"));
        let val = json!({"name": "Alice"});
        let result = enforcer.validate(&val).unwrap();
        assert_eq!(result["status"], "active");
        assert_eq!(result["name"], "Alice");
    }

    #[test]
    fn schema_enforcer_optional_default_not_overwritten() {
        let enforcer = SchemaEnforcer::new().optional_field("status", json!("active"));
        let val = json!({"status": "inactive"});
        let result = enforcer.validate(&val).unwrap();
        assert_eq!(result["status"], "inactive");
    }

    #[test]
    fn schema_enforcer_not_an_object() {
        let enforcer = SchemaEnforcer::new().require_field("x");
        let val = json!([1, 2, 3]);
        let err = enforcer.validate(&val).unwrap_err();
        assert_eq!(err[0].field, "<root>");
    }

    #[test]
    fn schema_enforcer_multiple_violations() {
        let enforcer = SchemaEnforcer::new()
            .require_field("a")
            .require_field("b")
            .require_field("c");
        let val = json!({"a": 1});
        let err = enforcer.validate(&val).unwrap_err();
        assert_eq!(err.len(), 2);
    }

    #[test]
    fn schema_enforcer_type_constraint_correct() {
        let enforcer = SchemaEnforcer::new().require_type("tags", JsonType::Array);
        let val = json!({"tags": ["a", "b"]});
        assert!(enforcer.validate(&val).is_ok());
    }

    // -- StructuredParser ---------------------------------------------------

    #[test]
    fn structured_parser_parse_json_valid() {
        let parser = StructuredParser::new();
        let result = parser.parse_json::<Value>(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result.value, json!({"key": "value"}));
        assert_eq!(result.format, OutputFormat::Json);
    }

    #[test]
    fn structured_parser_parse_json_invalid() {
        let parser = StructuredParser::new();
        assert!(parser.parse_json::<Value>("not json").is_err());
    }

    #[test]
    fn structured_parser_parse_json_code_fence() {
        let parser = StructuredParser::new();
        let input = "```json\n{\"x\": 1}\n```";
        let result = parser.parse_json::<Value>(input).unwrap();
        assert_eq!(result.value, json!({"x": 1}));
    }

    #[test]
    fn structured_parser_parse_json_block() {
        let parser = StructuredParser::new();
        let input = "Here is the data:\n```json\n{\"a\": 1}\n```\nDone.";
        let result = parser.parse_json_block(input).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn structured_parser_parse_json_block_no_fences() {
        let parser = StructuredParser::new();
        let input = "Result: {\"b\": 2}";
        let result = parser.parse_json_block(input).unwrap();
        assert_eq!(result, json!({"b": 2}));
    }

    #[test]
    fn structured_parser_parse_key_value() {
        let parser = StructuredParser::new();
        let input = "name: Alice\nage: 30\ncity: London";
        let map = parser.parse_key_value(input);
        assert_eq!(map.get("name").unwrap(), "Alice");
        assert_eq!(map.get("age").unwrap(), "30");
        assert_eq!(map.get("city").unwrap(), "London");
    }

    #[test]
    fn structured_parser_parse_key_value_empty_lines() {
        let parser = StructuredParser::new();
        let input = "\nname: Bob\n\nrole: admin\n";
        let map = parser.parse_key_value(input);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn structured_parser_parse_list_bulleted() {
        let parser = StructuredParser::new();
        let input = "- first\n- second\n- third";
        let list = parser.parse_list(input);
        assert_eq!(list, vec!["first", "second", "third"]);
    }

    #[test]
    fn structured_parser_parse_list_numbered() {
        let parser = StructuredParser::new();
        let input = "1. alpha\n2. beta\n3. gamma";
        let list = parser.parse_list(input);
        assert_eq!(list, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn structured_parser_parse_list_starred() {
        let parser = StructuredParser::new();
        let input = "* one\n* two";
        let list = parser.parse_list(input);
        assert_eq!(list, vec!["one", "two"]);
    }

    #[test]
    fn structured_parser_parse_list_empty() {
        let parser = StructuredParser::new();
        let list = parser.parse_list("no list here");
        assert!(list.is_empty());
    }

    #[test]
    fn structured_parser_parse_table() {
        let parser = StructuredParser::new();
        let input = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |";
        let table = parser.parse_table(input);
        assert_eq!(table.len(), 3); // header + 2 data rows
        assert_eq!(table[0], vec!["Name", "Age"]);
        // Note: separator row is excluded, but first data row index is 1
        assert_eq!(table[1], vec!["Alice", "30"]);
        assert_eq!(table[2], vec!["Bob", "25"]);
    }

    #[test]
    fn structured_parser_parse_table_no_rows() {
        let parser = StructuredParser::new();
        let table = parser.parse_table("no table here");
        assert!(table.is_empty());
    }

    #[test]
    fn structured_parser_parse_table_with_surrounding_text() {
        let parser = StructuredParser::new();
        let input = "Here is the table:\n| X | Y |\n| - | - |\n| 1 | 2 |\nEnd.";
        let table = parser.parse_table(input);
        assert_eq!(table.len(), 2);
        assert_eq!(table[0], vec!["X", "Y"]);
        assert_eq!(table[1], vec!["1", "2"]);
    }

    // -- OutputRepairer -----------------------------------------------------

    #[test]
    fn repairer_valid_json_unchanged() {
        let repairer = OutputRepairer::new();
        let input = r#"{"key": "value"}"#;
        let result = repairer.repair_json(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn repairer_trailing_comma() {
        let repairer = OutputRepairer::new();
        let input = r#"{"a": 1, "b": 2,}"#;
        let result = repairer.repair_json(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn repairer_single_quotes() {
        let repairer = OutputRepairer::new();
        let input = "{'name': 'Alice', 'age': 30}";
        let result = repairer.repair_json(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert_eq!(parsed["age"], 30);
    }

    #[test]
    fn repairer_unquoted_keys() {
        let repairer = OutputRepairer::new();
        let input = r#"{name: "Alice", age: 30}"#;
        let result = repairer.repair_json(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["name"], "Alice");
    }

    #[test]
    fn repairer_code_fences() {
        let repairer = OutputRepairer::new();
        let input = "```json\n{\"x\": 1}\n```";
        let result = repairer.repair_json(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["x"], 1);
    }

    #[test]
    fn repairer_unfixable_json() {
        let repairer = OutputRepairer::new();
        assert!(repairer.repair_json("completely not json at all").is_err());
    }

    #[test]
    fn repairer_extract_between() {
        let repairer = OutputRepairer::new();
        let result = repairer
            .extract_between("start<<hello>>end", "<<", ">>")
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn repairer_extract_between_not_found() {
        let repairer = OutputRepairer::new();
        assert!(repairer
            .extract_between("no delimiters", "<<", ">>")
            .is_none());
    }

    #[test]
    fn repairer_extract_between_missing_end() {
        let repairer = OutputRepairer::new();
        assert!(repairer
            .extract_between("<<start only", "<<", ">>")
            .is_none());
    }

    #[test]
    fn repairer_strip_markdown_fences() {
        let repairer = OutputRepairer::new();
        let result = repairer.strip_markdown_fences("```json\n{\"a\": 1}\n```");
        assert_eq!(result, r#"{"a": 1}"#);
    }

    #[test]
    fn repairer_strip_markdown_fences_no_fences() {
        let repairer = OutputRepairer::new();
        let result = repairer.strip_markdown_fences("plain text");
        assert_eq!(result, "plain text");
    }

    #[test]
    fn repairer_trailing_comma_in_array() {
        let repairer = OutputRepairer::new();
        let input = r#"[1, 2, 3,]"#;
        let result = repairer.repair_json(input).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed, json!([1, 2, 3]));
    }

    #[test]
    fn structured_parser_parse_duration_populated() {
        let parser = StructuredParser::new();
        let result = parser.parse_json::<Value>(r#"{"x": 1}"#).unwrap();
        // Duration should be >= 0 (it's measured)
        assert!(result.parse_duration_us < 1_000_000); // less than 1 second
    }
}
