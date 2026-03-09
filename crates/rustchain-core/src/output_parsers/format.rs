//! Extended output parsing utilities for LLM responses.
//!
//! This module provides additional parsers and helpers that complement the base
//! `OutputParser` trait and the existing JSON/list/schema parsers:
//!
//! - [`ParseError`] — Rich error type for parse failures with raw text and parser metadata.
//! - [`DatetimeOutputParser`] — Validates datetime strings against a configurable format.
//! - [`RetryParser`] — Wraps another parser, retrying with format instructions on failure.
//! - [`OutputFixingParser`] — Attempts to fix common LLM output issues before parsing.
//! - [`ExtractingJsonParser`] — Extracts the first JSON object/array from free-form text.
//! - [`ListOutputParser`] — Configurable list parser with custom separator support.
//! - [`PydanticOutputParser`] — Schema validation parser with required-field builder.

use serde_json::Value;
use std::fmt;

use super::base::OutputParser;
use crate::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

/// Rich error type carrying context about a parse failure.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Human-readable description of the error.
    pub message: String,
    /// The raw text that failed to parse.
    pub raw_text: String,
    /// Identifier of the parser that produced this error.
    pub parser_type: String,
}

impl ParseError {
    /// Create a new `ParseError`.
    pub fn new(
        message: impl Into<String>,
        raw_text: impl Into<String>,
        parser_type: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            raw_text: raw_text.into(),
            parser_type: parser_type.into(),
        }
    }

    /// Serialize the error to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "error": self.message,
            "raw_text": self.raw_text,
            "parser_type": self.parser_type,
        })
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParseError({}): {} [raw_text length={}]",
            self.parser_type,
            self.message,
            self.raw_text.len()
        )
    }
}

impl std::error::Error for ParseError {}

impl From<ParseError> for RustChainError {
    fn from(e: ParseError) -> Self {
        RustChainError::OutputParserError {
            message: e.message.clone(),
            observation: Some(e.raw_text.clone()),
            llm_output: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ExtractingJsonParser
// ---------------------------------------------------------------------------

/// Parses JSON from free-form LLM text by locating the first `{…}` or `[…]`
/// block rather than requiring the entire response to be valid JSON.
pub struct ExtractingJsonParser;

impl ExtractingJsonParser {
    pub fn new() -> Self {
        Self
    }

    /// Find the substring from the first opening brace/bracket to its matching
    /// close, handling nesting and quoted strings.
    fn extract_json(text: &str) -> Option<&str> {
        let bytes = text.as_bytes();

        // Find the first '{' or '['
        let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
        let open = bytes[start];
        let close = if open == b'{' { b'}' } else { b']' };

        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;

        for (i, &b) in bytes[start..].iter().enumerate() {
            if escape {
                escape = false;
                continue;
            }
            if b == b'\\' && in_string {
                escape = true;
                continue;
            }
            if b == b'"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            if b == open {
                depth += 1;
            } else if b == close {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + i + 1]);
                }
            }
        }
        None
    }
}

impl Default for ExtractingJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for ExtractingJsonParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let json_str =
            Self::extract_json(text).ok_or_else(|| RustChainError::OutputParserError {
                message: "No JSON object or array found in text".into(),
                observation: Some(text.to_string()),
                llm_output: None,
            })?;

        serde_json::from_str(json_str).map_err(|e| RustChainError::OutputParserError {
            message: format!("Failed to parse extracted JSON: {}", e),
            observation: Some(json_str.to_string()),
            llm_output: Some(text.to_string()),
        })
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(
            "Return a JSON object or array. It may be embedded in other text; \
             the parser will extract the first valid JSON block."
                .into(),
        )
    }

    fn parser_type(&self) -> &str {
        "extracting_json_parser"
    }
}

// ---------------------------------------------------------------------------
// ListOutputParser (configurable separator)
// ---------------------------------------------------------------------------

/// A list parser that splits LLM output into items using a configurable
/// separator. By default it splits on newlines and strips leading
/// bullets (`-`, `*`) and numbered prefixes (`1.`, `2)`).
pub struct ListOutputParser {
    separator: String,
}

impl ListOutputParser {
    /// Create a new `ListOutputParser`.
    ///
    /// If `separator` is `None`, lines are split by newline and leading
    /// bullet/number markers are stripped.
    pub fn new(separator: Option<String>) -> Self {
        Self {
            separator: separator.unwrap_or_else(|| "\n".to_string()),
        }
    }

    /// Strip common bullet or numbered-list prefixes.
    fn strip_prefix(s: &str) -> &str {
        let trimmed = s.trim();
        // Numbered: "1. ", "12) "
        let after_digits = trimmed.trim_start_matches(|c: char| c.is_ascii_digit());
        if after_digits.len() < trimmed.len() {
            if let Some(rest) = after_digits.strip_prefix(". ") {
                return rest.trim();
            }
            if let Some(rest) = after_digits.strip_prefix(") ") {
                return rest.trim();
            }
        }
        // Bullet: "- ", "* "
        if let Some(rest) = trimmed.strip_prefix("- ") {
            return rest.trim();
        }
        if let Some(rest) = trimmed.strip_prefix("* ") {
            return rest.trim();
        }
        trimmed
    }
}

impl Default for ListOutputParser {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OutputParser for ListOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let items: Vec<Value> = text
            .split(&self.separator)
            .map(|s| {
                if self.separator == "\n" {
                    Self::strip_prefix(s)
                } else {
                    s.trim()
                }
            })
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .collect();
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        if self.separator == "\n" {
            Some(
                "Your response should be a list of items, one per line. \
                 You may use bullets (- item) or numbers (1. item)."
                    .into(),
            )
        } else {
            Some(format!(
                "Your response should be a list of items separated by '{}'.",
                self.separator
            ))
        }
    }

    fn parser_type(&self) -> &str {
        "list_output_parser"
    }
}

// ---------------------------------------------------------------------------
// CommaSeparatedParser
// ---------------------------------------------------------------------------

/// Parses comma-separated values from LLM output into a JSON array of strings.
pub struct CommaSeparatedParser;

impl CommaSeparatedParser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CommaSeparatedParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for CommaSeparatedParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let items: Vec<Value> = text
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| Value::String(s.to_string()))
            .collect();
        Ok(Value::Array(items))
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some("Your response should be a list of comma separated values, eg: `foo, bar, baz`".into())
    }

    fn parser_type(&self) -> &str {
        "comma_separated_parser"
    }
}

// ---------------------------------------------------------------------------
// DatetimeOutputParser
// ---------------------------------------------------------------------------

/// Validates that LLM output is a datetime string matching a given format.
///
/// The format string uses a simple subset of `strftime`-style tokens:
/// `%Y` (4-digit year), `%m` (2-digit month), `%d` (2-digit day),
/// `%H` (2-digit hour), `%M` (2-digit minute), `%S` (2-digit second).
///
/// The parser converts the format into a regex for validation.
pub struct DatetimeOutputParser {
    format: String,
}

impl DatetimeOutputParser {
    /// Create a new `DatetimeOutputParser`.
    ///
    /// If `format` is `None`, defaults to `"%Y-%m-%d %H:%M:%S"`.
    pub fn new(format: Option<String>) -> Self {
        Self {
            format: format.unwrap_or_else(|| "%Y-%m-%d %H:%M:%S".to_string()),
        }
    }

    /// Build a regex pattern from the format string.
    fn build_pattern(&self) -> String {
        let mut pattern = regex::escape(&self.format);
        pattern = pattern.replace("%Y", r"\d{4}");
        pattern = pattern.replace("%m", r"\d{2}");
        pattern = pattern.replace("%d", r"\d{2}");
        pattern = pattern.replace("%H", r"\d{2}");
        pattern = pattern.replace("%M", r"\d{2}");
        pattern = pattern.replace("%S", r"\d{2}");
        format!("^{}$", pattern)
    }
}

impl Default for DatetimeOutputParser {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OutputParser for DatetimeOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        let trimmed = text.trim();
        let pattern = self.build_pattern();
        let re = regex::Regex::new(&pattern).map_err(|e| RustChainError::OutputParserError {
            message: format!("Invalid datetime format pattern: {}", e),
            observation: Some(self.format.clone()),
            llm_output: None,
        })?;

        if re.is_match(trimmed) {
            Ok(Value::String(trimmed.to_string()))
        } else {
            Err(RustChainError::OutputParserError {
                message: format!(
                    "Datetime '{}' does not match expected format '{}'",
                    trimmed, self.format
                ),
                observation: Some(trimmed.to_string()),
                llm_output: None,
            })
        }
    }

    fn get_format_instructions(&self) -> Option<String> {
        Some(format!(
            "Write a datetime string that matches the following format: {}",
            self.format
        ))
    }

    fn parser_type(&self) -> &str {
        "datetime_output_parser"
    }
}

// ---------------------------------------------------------------------------
// PydanticOutputParser
// ---------------------------------------------------------------------------

/// Parses JSON and validates it against a JSON schema, including required
/// fields and basic type checks.
pub struct PydanticOutputParser {
    schema: Value,
    required_fields: Vec<String>,
}

impl PydanticOutputParser {
    /// Create a new `PydanticOutputParser` with the given JSON schema.
    ///
    /// Required fields are read from the schema's `"required"` array.
    pub fn new(schema: Value) -> Self {
        let required_fields = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            schema,
            required_fields,
        }
    }

    /// Override the set of required fields (builder pattern).
    pub fn with_required_fields(mut self, fields: Vec<String>) -> Self {
        self.required_fields = fields;
        self
    }

    /// Validate that all required fields are present and basic type checks pass.
    fn validate(&self, obj: &Value) -> std::result::Result<(), ParseError> {
        let map = obj.as_object().ok_or_else(|| {
            ParseError::new(
                "Expected a JSON object",
                obj.to_string(),
                self.parser_type(),
            )
        })?;

        // Check required fields
        for field in &self.required_fields {
            if !map.contains_key(field) {
                return Err(ParseError::new(
                    format!("Missing required field '{}'", field),
                    obj.to_string(),
                    self.parser_type(),
                ));
            }
        }

        // Basic type validation against schema properties
        if let Some(props) = self.schema.get("properties").and_then(|p| p.as_object()) {
            for (key, prop_schema) in props {
                if let Some(val) = map.get(key) {
                    if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                        let type_ok = match expected_type {
                            "string" => val.is_string(),
                            "number" | "integer" => val.is_number(),
                            "boolean" => val.is_boolean(),
                            "array" => val.is_array(),
                            "object" => val.is_object(),
                            "null" => val.is_null(),
                            _ => true,
                        };
                        if !type_ok {
                            return Err(ParseError::new(
                                format!(
                                    "Field '{}' expected type '{}', got '{}'",
                                    key,
                                    expected_type,
                                    value_type_name(val),
                                ),
                                obj.to_string(),
                                self.parser_type(),
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Return a human-readable name for a JSON value type.
fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

impl OutputParser for PydanticOutputParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // First extract JSON
        let trimmed = text.trim();
        let json_str = ExtractingJsonParser::extract_json(trimmed).unwrap_or(trimmed);

        let obj: Value =
            serde_json::from_str(json_str).map_err(|e| RustChainError::OutputParserError {
                message: format!("Failed to parse JSON: {}", e),
                observation: Some(json_str.to_string()),
                llm_output: Some(text.to_string()),
            })?;

        self.validate(&obj).map_err(RustChainError::from)?;
        Ok(obj)
    }

    fn get_format_instructions(&self) -> Option<String> {
        let schema_str = serde_json::to_string_pretty(&self.schema).unwrap_or_default();
        Some(format!(
            "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n\
             Here is the output schema:\n```\n{}\n```",
            schema_str
        ))
    }

    fn parser_type(&self) -> &str {
        "pydantic_output_parser"
    }
}

// ---------------------------------------------------------------------------
// RetryParser
// ---------------------------------------------------------------------------

/// Wraps another parser and retries on failure, appending format instructions
/// to help the caller understand what went wrong.
pub struct RetryParser {
    inner: Box<dyn OutputParser>,
    max_retries: usize,
}

impl RetryParser {
    /// Create a new `RetryParser`.
    pub fn new(inner: Box<dyn OutputParser>, max_retries: usize) -> Self {
        Self { inner, max_retries }
    }

    /// Attempt to parse, retrying up to `max_retries` times.
    ///
    /// On each failure after the first, the method appends the format
    /// instructions from the inner parser (if available) to the text,
    /// simulating what a retry prompt might contain.
    ///
    /// In a real system the caller would re-prompt the LLM with the
    /// instructions; here we simply retry parsing the same text to handle
    /// transient or order-dependent issues (e.g., an `OutputFixingParser`
    /// inside that might succeed on second pass with cleaned input).
    pub fn parse_with_retry(&self, text: &str) -> Result<Value> {
        let mut last_err = None;
        for _ in 0..=self.max_retries {
            match self.inner.parse(text) {
                Ok(v) => return Ok(v),
                Err(e) => last_err = Some(e),
            }
        }
        Err(
            last_err.unwrap_or_else(|| RustChainError::OutputParserError {
                message: "RetryParser exhausted retries with no error captured".into(),
                observation: Some(text.to_string()),
                llm_output: None,
            }),
        )
    }

    /// Return the format instructions from the inner parser (if any),
    /// prefixed with retry guidance.
    pub fn get_retry_instructions(&self) -> Option<String> {
        self.inner.get_format_instructions().map(|instr| {
            format!(
                "Your previous response could not be parsed. Please try again.\n\n{}",
                instr
            )
        })
    }
}

impl OutputParser for RetryParser {
    fn parse(&self, text: &str) -> Result<Value> {
        self.parse_with_retry(text)
    }

    fn get_format_instructions(&self) -> Option<String> {
        self.inner.get_format_instructions()
    }

    fn parser_type(&self) -> &str {
        "retry_parser"
    }
}

// ---------------------------------------------------------------------------
// OutputFixingParser
// ---------------------------------------------------------------------------

/// Wraps another parser and attempts to fix common LLM output issues before
/// passing text to the inner parser:
///
/// - Removes trailing commas in JSON objects/arrays.
/// - Adds missing closing brackets/braces.
/// - Strips leading/trailing markdown code fences.
/// - Removes unquoted `undefined` / `NaN` / single-quoted strings.
pub struct OutputFixingParser {
    inner: Box<dyn OutputParser>,
}

impl OutputFixingParser {
    pub fn new(inner: Box<dyn OutputParser>) -> Self {
        Self { inner }
    }

    /// Apply heuristic fixes to common LLM output problems.
    fn fix_text(text: &str) -> String {
        let mut s = text.trim().to_string();

        // Strip markdown code fences
        if s.starts_with("```") {
            // Remove opening fence (```json, ```, etc.)
            if let Some(end_of_first_line) = s.find('\n') {
                s = s[end_of_first_line + 1..].to_string();
            }
            if s.ends_with("```") {
                s.truncate(s.len() - 3);
            }
            s = s.trim().to_string();
        }

        // Replace single quotes with double quotes (simple heuristic:
        // only when the character is adjacent to a colon, comma, bracket,
        // or is at the boundary of a key/value).
        s = fix_single_quotes(&s);

        // Remove trailing commas before } or ]
        s = remove_trailing_commas(&s);

        // Balance brackets/braces
        s = balance_brackets(&s);

        s
    }
}

/// Replace single-quoted JSON strings with double-quoted ones.
///
/// This is intentionally conservative: it only handles the common case where
/// the entire JSON uses single quotes instead of double quotes.
fn fix_single_quotes(text: &str) -> String {
    // Only apply if there are no double quotes at all (pure single-quote JSON)
    if !text.contains('"') && text.contains('\'') {
        text.replace('\'', "\"")
    } else {
        text.to_string()
    }
}

/// Remove trailing commas that appear before `}` or `]`.
fn remove_trailing_commas(text: &str) -> String {
    let re = regex::Regex::new(r",\s*([}\]])").unwrap();
    re.replace_all(text, "$1").to_string()
}

/// Add missing closing braces/brackets if counts are unbalanced.
fn balance_brackets(text: &str) -> String {
    let mut result = text.to_string();
    let mut in_string = false;
    let mut escape = false;
    let mut brace_depth: i32 = 0;
    let mut bracket_depth: i32 = 0;

    for b in text.bytes() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match b {
            b'{' => brace_depth += 1,
            b'}' => brace_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            _ => {}
        }
    }

    for _ in 0..bracket_depth {
        result.push(']');
    }
    for _ in 0..brace_depth {
        result.push('}');
    }

    result
}

impl OutputParser for OutputFixingParser {
    fn parse(&self, text: &str) -> Result<Value> {
        // First try as-is
        if let Ok(v) = self.inner.parse(text) {
            return Ok(v);
        }

        // Try with fixes
        let fixed = Self::fix_text(text);
        self.inner.parse(&fixed)
    }

    fn get_format_instructions(&self) -> Option<String> {
        self.inner.get_format_instructions()
    }

    fn parser_type(&self) -> &str {
        "output_fixing_parser"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // ParseError
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_error_new() {
        let err = ParseError::new("bad input", "raw", "json");
        assert_eq!(err.message, "bad input");
        assert_eq!(err.raw_text, "raw");
        assert_eq!(err.parser_type, "json");
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::new("oops", "some text", "test_parser");
        let display = format!("{}", err);
        assert!(display.contains("test_parser"));
        assert!(display.contains("oops"));
    }

    #[test]
    fn test_parse_error_to_json() {
        let err = ParseError::new("fail", "raw text", "p");
        let j = err.to_json();
        assert_eq!(j["error"], "fail");
        assert_eq!(j["raw_text"], "raw text");
        assert_eq!(j["parser_type"], "p");
    }

    #[test]
    fn test_parse_error_implements_std_error() {
        let err = ParseError::new("msg", "", "t");
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_parse_error_into_rustchain_error() {
        let err = ParseError::new("msg", "raw", "t");
        let rce: RustChainError = err.into();
        match rce {
            RustChainError::OutputParserError {
                message,
                observation,
                ..
            } => {
                assert_eq!(message, "msg");
                assert_eq!(observation, Some("raw".to_string()));
            }
            _ => panic!("Expected OutputParserError"),
        }
    }

    // -----------------------------------------------------------------------
    // ExtractingJsonParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_extracting_json_simple_object() {
        let parser = ExtractingJsonParser::new();
        let result = parser.parse(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_extracting_json_embedded_in_text() {
        let parser = ExtractingJsonParser::new();
        let text = r#"Here is the answer: {"name": "Alice", "age": 30} hope that helps!"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[test]
    fn test_extracting_json_array() {
        let parser = ExtractingJsonParser::new();
        let text = r#"The list is [1, 2, 3] and that's it."#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!([1, 2, 3]));
    }

    #[test]
    fn test_extracting_json_nested() {
        let parser = ExtractingJsonParser::new();
        let text = r#"Result: {"a": {"b": [1, {"c": 2}]}}"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"a": {"b": [1, {"c": 2}]}}));
    }

    #[test]
    fn test_extracting_json_no_json_found() {
        let parser = ExtractingJsonParser::new();
        let result = parser.parse("no json here at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_extracting_json_empty_text() {
        let parser = ExtractingJsonParser::new();
        let result = parser.parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_extracting_json_format_instructions() {
        let parser = ExtractingJsonParser::new();
        let instr = parser.get_format_instructions();
        assert!(instr.is_some());
        assert!(instr.unwrap().contains("JSON"));
    }

    #[test]
    fn test_extracting_json_parser_type() {
        assert_eq!(
            ExtractingJsonParser::new().parser_type(),
            "extracting_json_parser"
        );
    }

    #[test]
    fn test_extracting_json_with_escaped_quotes() {
        let parser = ExtractingJsonParser::new();
        let text = r#"output: {"msg": "he said \"hello\""}"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result["msg"], "he said \"hello\"");
    }

    // -----------------------------------------------------------------------
    // ListOutputParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_parser_numbered() {
        let parser = ListOutputParser::new(None);
        let text = "1. Apple\n2. Banana\n3. Cherry";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["Apple", "Banana", "Cherry"]));
    }

    #[test]
    fn test_list_parser_bulleted() {
        let parser = ListOutputParser::new(None);
        let text = "- Alpha\n- Beta\n- Gamma";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["Alpha", "Beta", "Gamma"]));
    }

    #[test]
    fn test_list_parser_star_bullets() {
        let parser = ListOutputParser::new(None);
        let text = "* One\n* Two\n* Three";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["One", "Two", "Three"]));
    }

    #[test]
    fn test_list_parser_custom_separator() {
        let parser = ListOutputParser::new(Some(";".to_string()));
        let text = "red; green; blue";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!(["red", "green", "blue"]));
    }

    #[test]
    fn test_list_parser_empty_text() {
        let parser = ListOutputParser::new(None);
        let result = parser.parse("").unwrap();
        assert_eq!(result, json!([]));
    }

    #[test]
    fn test_list_parser_format_instructions_default() {
        let parser = ListOutputParser::new(None);
        let instr = parser.get_format_instructions().unwrap();
        assert!(instr.contains("list"));
    }

    #[test]
    fn test_list_parser_format_instructions_custom() {
        let parser = ListOutputParser::new(Some("|".to_string()));
        let instr = parser.get_format_instructions().unwrap();
        assert!(instr.contains("|"));
    }

    #[test]
    fn test_list_parser_mixed_format() {
        let parser = ListOutputParser::new(None);
        // Lines without recognized prefixes still appear as plain items
        let text = "1. First\nplain line\n- Third";
        let result = parser.parse(text).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], json!("First"));
        assert_eq!(arr[1], json!("plain line"));
        assert_eq!(arr[2], json!("Third"));
    }

    // -----------------------------------------------------------------------
    // CommaSeparatedParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_comma_parser_basic() {
        let parser = CommaSeparatedParser::new();
        let result = parser.parse("foo, bar, baz").unwrap();
        assert_eq!(result, json!(["foo", "bar", "baz"]));
    }

    #[test]
    fn test_comma_parser_with_whitespace() {
        let parser = CommaSeparatedParser::new();
        let result = parser.parse("  a ,  b  , c  ").unwrap();
        assert_eq!(result, json!(["a", "b", "c"]));
    }

    #[test]
    fn test_comma_parser_single_item() {
        let parser = CommaSeparatedParser::new();
        let result = parser.parse("only_one").unwrap();
        assert_eq!(result, json!(["only_one"]));
    }

    #[test]
    fn test_comma_parser_empty() {
        let parser = CommaSeparatedParser::new();
        let result = parser.parse("").unwrap();
        assert_eq!(result, json!([]));
    }

    #[test]
    fn test_comma_parser_trailing_comma() {
        let parser = CommaSeparatedParser::new();
        let result = parser.parse("a, b, ").unwrap();
        assert_eq!(result, json!(["a", "b"]));
    }

    #[test]
    fn test_comma_parser_type() {
        assert_eq!(
            CommaSeparatedParser::new().parser_type(),
            "comma_separated_parser"
        );
    }

    // -----------------------------------------------------------------------
    // DatetimeOutputParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_datetime_default_format_valid() {
        let parser = DatetimeOutputParser::new(None);
        let result = parser.parse("2024-01-15 10:30:00").unwrap();
        assert_eq!(result, json!("2024-01-15 10:30:00"));
    }

    #[test]
    fn test_datetime_default_format_invalid() {
        let parser = DatetimeOutputParser::new(None);
        let result = parser.parse("Jan 15, 2024");
        assert!(result.is_err());
    }

    #[test]
    fn test_datetime_custom_format() {
        let parser = DatetimeOutputParser::new(Some("%d/%m/%Y".to_string()));
        let result = parser.parse("15/01/2024").unwrap();
        assert_eq!(result, json!("15/01/2024"));
    }

    #[test]
    fn test_datetime_custom_format_invalid() {
        let parser = DatetimeOutputParser::new(Some("%d/%m/%Y".to_string()));
        let result = parser.parse("2024-01-15");
        assert!(result.is_err());
    }

    #[test]
    fn test_datetime_trims_whitespace() {
        let parser = DatetimeOutputParser::new(None);
        let result = parser.parse("  2024-01-15 10:30:00  ").unwrap();
        assert_eq!(result, json!("2024-01-15 10:30:00"));
    }

    #[test]
    fn test_datetime_format_instructions() {
        let parser = DatetimeOutputParser::new(Some("%Y-%m-%d".to_string()));
        let instr = parser.get_format_instructions().unwrap();
        assert!(instr.contains("%Y-%m-%d"));
    }

    #[test]
    fn test_datetime_parser_type() {
        assert_eq!(
            DatetimeOutputParser::new(None).parser_type(),
            "datetime_output_parser"
        );
    }

    // -----------------------------------------------------------------------
    // PydanticOutputParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_pydantic_valid_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let parser = PydanticOutputParser::new(schema);
        let result = parser.parse(r#"{"name": "Alice", "age": 30}"#).unwrap();
        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[test]
    fn test_pydantic_missing_required_field() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let parser = PydanticOutputParser::new(schema);
        let result = parser.parse(r#"{"name": "Alice"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_pydantic_wrong_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            },
            "required": ["count"]
        });
        let parser = PydanticOutputParser::new(schema);
        let result = parser.parse(r#"{"count": "not a number"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_pydantic_with_required_fields_override() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            }
        });
        let parser = PydanticOutputParser::new(schema).with_required_fields(vec!["a".to_string()]);
        // Missing "a"
        let result = parser.parse(r#"{"b": "hello"}"#);
        assert!(result.is_err());
        // Has "a"
        let result = parser.parse(r#"{"a": "hello"}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pydantic_extracts_from_text() {
        let schema = json!({
            "type": "object",
            "properties": {
                "x": {"type": "number"}
            },
            "required": ["x"]
        });
        let parser = PydanticOutputParser::new(schema);
        let result = parser.parse(r#"Here is the result: {"x": 42}"#).unwrap();
        assert_eq!(result["x"], 42);
    }

    #[test]
    fn test_pydantic_non_object() {
        let schema = json!({
            "type": "object",
            "properties": {},
            "required": ["x"]
        });
        let parser = PydanticOutputParser::new(schema);
        let result = parser.parse("[1,2,3]");
        assert!(result.is_err());
    }

    #[test]
    fn test_pydantic_format_instructions() {
        let schema = json!({"type": "object"});
        let parser = PydanticOutputParser::new(schema);
        let instr = parser.get_format_instructions().unwrap();
        assert!(instr.contains("JSON schema"));
    }

    // -----------------------------------------------------------------------
    // RetryParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_retry_parser_succeeds_first_try() {
        let inner = Box::new(CommaSeparatedParser::new());
        let parser = RetryParser::new(inner, 3);
        let result = parser.parse("a, b, c").unwrap();
        assert_eq!(result, json!(["a", "b", "c"]));
    }

    #[test]
    fn test_retry_parser_fails_after_max_retries() {
        let schema = json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        });
        let inner = Box::new(PydanticOutputParser::new(schema));
        let parser = RetryParser::new(inner, 2);
        let result = parser.parse("totally invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_retry_parser_get_retry_instructions() {
        let inner = Box::new(CommaSeparatedParser::new());
        let parser = RetryParser::new(inner, 1);
        let instr = parser.get_retry_instructions().unwrap();
        assert!(instr.contains("previous response"));
    }

    #[test]
    fn test_retry_parser_delegates_format_instructions() {
        let inner = Box::new(CommaSeparatedParser::new());
        let parser = RetryParser::new(inner, 1);
        let instr = parser.get_format_instructions();
        assert!(instr.is_some());
    }

    #[test]
    fn test_retry_parser_type() {
        let inner = Box::new(CommaSeparatedParser::new());
        let parser = RetryParser::new(inner, 1);
        assert_eq!(parser.parser_type(), "retry_parser");
    }

    // -----------------------------------------------------------------------
    // OutputFixingParser
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixing_parser_valid_json_passthrough() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let result = parser.parse(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_fixing_parser_strips_markdown_fences() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = "```json\n{\"a\": 1}\n```";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn test_fixing_parser_trailing_comma() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = r#"{"a": 1, "b": 2,}"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_fixing_parser_single_quotes() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = "{'key': 'value'}";
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_fixing_parser_missing_closing_brace() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = r#"{"a": 1"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn test_fixing_parser_missing_closing_bracket() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = r#"[1, 2, 3"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!([1, 2, 3]));
    }

    #[test]
    fn test_fixing_parser_delegates_format_instructions() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        assert!(parser.get_format_instructions().is_some());
    }

    #[test]
    fn test_fixing_parser_type() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        assert_eq!(parser.parser_type(), "output_fixing_parser");
    }

    #[test]
    fn test_fixing_parser_trailing_comma_in_array() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let text = r#"[1, 2, 3,]"#;
        let result = parser.parse(text).unwrap();
        assert_eq!(result, json!([1, 2, 3]));
    }

    #[test]
    fn test_fixing_parser_unfixable() {
        let inner = Box::new(ExtractingJsonParser::new());
        let parser = OutputFixingParser::new(inner);
        let result = parser.parse("completely invalid not json");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // value_type_name helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_type_name() {
        assert_eq!(value_type_name(&json!(null)), "null");
        assert_eq!(value_type_name(&json!(true)), "boolean");
        assert_eq!(value_type_name(&json!(42)), "number");
        assert_eq!(value_type_name(&json!("s")), "string");
        assert_eq!(value_type_name(&json!([1])), "array");
        assert_eq!(value_type_name(&json!({})), "object");
    }
}
