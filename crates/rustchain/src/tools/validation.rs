//! Tool call validation and auto-correction.
//!
//! Provides [`ToolCallValidator`] for validating tool call arguments against
//! JSON schemas, [`ToolCallCorrector`] for auto-correcting common mistakes,
//! and [`ValidatedToolExecutor`] for wrapping a [`BaseTool`] with validation.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::messages::ToolCall;
use rustchain_core::tools::base::{BaseTool, ToolSchema};
use rustchain_core::tools::types::{ToolInput, ToolOutput};

// ---------------------------------------------------------------------------
// Strictness mode
// ---------------------------------------------------------------------------

/// Controls how strict validation and correction behave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrictnessMode {
    /// Every deviation is an error.
    Strict,
    /// Unexpected fields are silently ignored (but other errors still reported).
    Lenient,
    /// Attempt to fix problems automatically and populate `corrected_args`.
    AutoCorrect,
}

impl Default for StrictnessMode {
    fn default() -> Self {
        Self::Strict
    }
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

/// A single issue found during validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationError {
    /// A required field is missing from the arguments.
    MissingRequiredField { field: String },
    /// An unexpected field is present that is not in the schema.
    UnexpectedField { field: String },
    /// The value type does not match the schema expectation.
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    /// The value is the right type but otherwise invalid.
    InvalidValue { field: String, reason: String },
    /// The raw arguments could not be parsed as valid JSON.
    InvalidJson { message: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRequiredField { field } => {
                write!(f, "missing required field: {}", field)
            }
            Self::UnexpectedField { field } => write!(f, "unexpected field: {}", field),
            Self::TypeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "type mismatch for '{}': expected {}, got {}",
                field, expected, actual
            ),
            Self::InvalidValue { field, reason } => {
                write!(f, "invalid value for '{}': {}", field, reason)
            }
            Self::InvalidJson { message } => write!(f, "invalid JSON: {}", message),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// The outcome of validating a single tool call.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the tool call is valid (no errors).
    pub is_valid: bool,
    /// All issues found.
    pub errors: Vec<ValidationError>,
    /// Auto-corrected arguments (populated only in `AutoCorrect` mode).
    pub corrected_args: Option<Value>,
    /// The original arguments that were validated.
    pub original_args: Value,
}

// ---------------------------------------------------------------------------
// ToolCallValidator
// ---------------------------------------------------------------------------

/// Validates tool call arguments against a JSON-Schema-like `ToolSchema`.
#[derive(Debug, Clone)]
pub struct ToolCallValidator {
    /// The strictness mode to use.
    pub mode: StrictnessMode,
}

impl Default for ToolCallValidator {
    fn default() -> Self {
        Self {
            mode: StrictnessMode::Strict,
        }
    }
}

impl ToolCallValidator {
    /// Create a new validator with the given strictness mode.
    pub fn new(mode: StrictnessMode) -> Self {
        Self { mode }
    }

    /// Validate a single tool call against a schema.
    pub fn validate(&self, tool_call: &ToolCall, schema: &ToolSchema) -> ValidationResult {
        let args_value = serde_json::to_value(&tool_call.args).unwrap_or(Value::Object(Default::default()));
        self.validate_value(&args_value, schema)
    }

    /// Validate a raw JSON value against a schema.
    pub fn validate_value(&self, args: &Value, schema: &ToolSchema) -> ValidationResult {
        let mut errors = Vec::new();

        let params = match schema.parameters.as_ref() {
            Some(p) => p,
            None => {
                // No schema defined — anything goes.
                return ValidationResult {
                    is_valid: true,
                    errors: vec![],
                    corrected_args: None,
                    original_args: args.clone(),
                };
            }
        };

        let args_obj = match args.as_object() {
            Some(o) => o,
            None => {
                errors.push(ValidationError::InvalidJson {
                    message: "arguments must be a JSON object".into(),
                });
                return ValidationResult {
                    is_valid: false,
                    errors,
                    corrected_args: None,
                    original_args: args.clone(),
                };
            }
        };

        let properties = params
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let required_fields: Vec<String> = params
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Check for missing required fields.
        for field in &required_fields {
            if !args_obj.contains_key(field) {
                errors.push(ValidationError::MissingRequiredField {
                    field: field.clone(),
                });
            }
        }

        // Check for unexpected fields.
        for key in args_obj.keys() {
            if !properties.contains_key(key) {
                if self.mode == StrictnessMode::Strict {
                    errors.push(ValidationError::UnexpectedField {
                        field: key.clone(),
                    });
                } else if self.mode == StrictnessMode::AutoCorrect {
                    errors.push(ValidationError::UnexpectedField {
                        field: key.clone(),
                    });
                }
                // In Lenient mode we silently skip unexpected fields.
            }
        }

        // Type-check each present field.
        for (key, value) in args_obj.iter() {
            if let Some(prop_schema) = properties.get(key) {
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    let actual_type = json_type_name(value);
                    if !type_matches(value, expected_type) {
                        errors.push(ValidationError::TypeMismatch {
                            field: key.clone(),
                            expected: expected_type.to_string(),
                            actual: actual_type.to_string(),
                        });
                    }
                }
            }
        }

        let corrected_args = if self.mode == StrictnessMode::AutoCorrect && !errors.is_empty() {
            Some(ToolCallCorrector::correct(args, schema))
        } else {
            None
        };

        let is_valid = errors.is_empty();
        ValidationResult {
            is_valid,
            errors,
            corrected_args,
            original_args: args.clone(),
        }
    }

    /// Validate a batch of tool calls, each matched to its schema by name.
    pub fn validate_batch(
        &self,
        tool_calls: &[ToolCall],
        schemas: &HashMap<String, ToolSchema>,
    ) -> Vec<ValidationResult> {
        tool_calls
            .iter()
            .map(|tc| {
                if let Some(schema) = schemas.get(&tc.name) {
                    self.validate(tc, schema)
                } else {
                    let args_value = serde_json::to_value(&tc.args)
                        .unwrap_or(Value::Object(Default::default()));
                    ValidationResult {
                        is_valid: false,
                        errors: vec![ValidationError::InvalidValue {
                            field: "name".into(),
                            reason: format!("unknown tool: {}", tc.name),
                        }],
                        corrected_args: None,
                        original_args: args_value,
                    }
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ToolCallCorrector
// ---------------------------------------------------------------------------

/// Auto-correction strategies for malformed tool call arguments.
pub struct ToolCallCorrector;

impl ToolCallCorrector {
    /// Attempt to fix malformed JSON text and return a parsed `Value`.
    ///
    /// Handles:
    /// - trailing commas before `}` or `]`
    /// - single-quoted strings (replaced with double-quotes)
    pub fn fix_json(raw: &str) -> std::result::Result<Value, String> {
        // First try as-is.
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            return Ok(v);
        }

        let mut fixed = raw.to_string();

        // Remove trailing commas before closing brace/bracket.
        let re_trailing = regex::Regex::new(r",\s*([}\]])").unwrap();
        fixed = re_trailing.replace_all(&fixed, "$1").to_string();

        // Replace single quotes around keys/values with double quotes (simple heuristic).
        // This intentionally only handles the outermost single-quoted tokens.
        let re_single = regex::Regex::new(r"'([^']*)'").unwrap();
        fixed = re_single.replace_all(&fixed, "\"$1\"").to_string();

        serde_json::from_str::<Value>(&fixed).map_err(|e| e.to_string())
    }

    /// Apply auto-correction to an arguments `Value` using the schema.
    pub fn correct(args: &Value, schema: &ToolSchema) -> Value {
        let params = match schema.parameters.as_ref() {
            Some(p) => p,
            None => return args.clone(),
        };

        let properties = params
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let required_fields: Vec<String> = params
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut obj = match args.as_object() {
            Some(o) => o.clone(),
            None => return args.clone(),
        };

        // 1. Remove unexpected fields.
        let known_keys: Vec<String> = properties.keys().cloned().collect();
        obj.retain(|k, _| known_keys.contains(k));

        // 2. Type coercion.
        for (key, prop_schema) in &properties {
            if let Some(value) = obj.get(key).cloned() {
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    if !type_matches(&value, expected_type) {
                        if let Some(coerced) = coerce_value(&value, expected_type) {
                            obj.insert(key.clone(), coerced);
                        }
                    }
                }
            }
        }

        // 3. Trim whitespace from string values.
        for (key, prop_schema) in &properties {
            if let Some(Value::String(s)) = obj.get(key) {
                let is_string_type = prop_schema
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "string")
                    .unwrap_or(false);
                if is_string_type {
                    let trimmed = s.trim().to_string();
                    if trimmed != *s {
                        obj.insert(key.clone(), Value::String(trimmed));
                    }
                }
            }
        }

        // 4. Fill missing optional fields with defaults from schema.
        for (key, prop_schema) in &properties {
            if !obj.contains_key(key) && !required_fields.contains(key) {
                if let Some(default_val) = prop_schema.get("default") {
                    obj.insert(key.clone(), default_val.clone());
                }
            }
        }

        Value::Object(obj)
    }
}

// ---------------------------------------------------------------------------
// ValidatedToolExecutor
// ---------------------------------------------------------------------------

/// Wraps a `BaseTool`, validating (and optionally auto-correcting) arguments
/// before forwarding to the inner tool.
pub struct ValidatedToolExecutor {
    inner: Arc<dyn BaseTool>,
    schema: ToolSchema,
    validator: ToolCallValidator,
    /// Maximum number of correction attempts (default 1).
    pub max_correction_attempts: usize,
}

impl ValidatedToolExecutor {
    /// Create a new validated executor.
    pub fn new(inner: Arc<dyn BaseTool>, schema: ToolSchema, mode: StrictnessMode) -> Self {
        Self {
            inner,
            schema,
            validator: ToolCallValidator::new(mode),
            max_correction_attempts: 1,
        }
    }

    /// Set the maximum number of correction attempts.
    pub fn with_max_correction_attempts(mut self, max: usize) -> Self {
        self.max_correction_attempts = max;
        self
    }

    /// Validate and execute a tool call.
    pub async fn execute(&self, tool_call: &ToolCall) -> Result<Value> {
        let args_value = serde_json::to_value(&tool_call.args)
            .unwrap_or(Value::Object(Default::default()));

        let result = self.validator.validate_value(&args_value, &self.schema);

        if result.is_valid {
            return self
                .inner
                .run(ToolInput::Structured(tool_call.args.clone()), tool_call.id.as_deref())
                .await;
        }

        // If not auto-correct mode, fail immediately.
        if self.validator.mode != StrictnessMode::AutoCorrect {
            let msg = result
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(RustChainError::ToolValidationError(msg));
        }

        // Auto-correct loop.
        let mut current_args = args_value.clone();
        for _ in 0..self.max_correction_attempts {
            let corrected = ToolCallCorrector::correct(&current_args, &self.schema);
            let re_result = self.validator.validate_value(&corrected, &self.schema);
            if re_result.is_valid {
                let map: HashMap<String, Value> = corrected
                    .as_object()
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                return self
                    .inner
                    .run(ToolInput::Structured(map), tool_call.id.as_deref())
                    .await;
            }
            current_args = corrected;
        }

        let final_result = self.validator.validate_value(&current_args, &self.schema);
        let msg = final_result
            .errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        Err(RustChainError::ToolValidationError(msg))
    }
}

#[async_trait]
impl BaseTool for ValidatedToolExecutor {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn args_schema(&self) -> Option<Value> {
        self.schema.parameters.clone()
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        // Build a ToolCall from the input so we can validate it.
        let args_map = match &input {
            ToolInput::Structured(m) => m.clone(),
            ToolInput::ToolCall(tc) => tc.args.clone(),
            ToolInput::Text(s) => {
                let parsed: HashMap<String, Value> =
                    serde_json::from_str(s).unwrap_or_default();
                parsed
            }
        };
        let tc = ToolCall {
            name: self.inner.name().to_string(),
            args: args_map,
            id: None,
        };

        let value = self.execute(&tc).await?;
        Ok(ToolOutput::Content(value))
    }
}

// ---------------------------------------------------------------------------
// ToolSchemaBuilder (convenience re-export / extension)
// ---------------------------------------------------------------------------

/// Convenience builder that mirrors the task requirements.
///
/// This delegates to `rustchain_core::tools::schema_gen::ToolSchemaBuilder`
/// but adds the `.required_param()` / `.optional_param()` shorthand API.
pub struct ValidationSchemaBuilder {
    name: String,
    description: String,
    properties: Vec<(String, String, String, bool)>, // (name, type, desc, required)
    defaults: HashMap<String, Value>,
}

impl ValidationSchemaBuilder {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            properties: Vec::new(),
            defaults: HashMap::new(),
        }
    }

    /// Set the tool name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the tool description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add a required parameter.
    pub fn required_param(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.properties
            .push((name.into(), param_type.into(), description.into(), true));
        self
    }

    /// Add an optional parameter.
    pub fn optional_param(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.properties
            .push((name.into(), param_type.into(), description.into(), false));
        self
    }

    /// Add an optional parameter with a default value.
    pub fn optional_param_with_default(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        default: Value,
    ) -> Self {
        let n: String = name.into();
        self.defaults.insert(n.clone(), default);
        self.properties
            .push((n, param_type.into(), description.into(), false));
        self
    }

    /// Build the `ToolSchema`.
    pub fn build(self) -> ToolSchema {
        let mut props = serde_json::Map::new();
        let mut required: Vec<Value> = Vec::new();

        for (name, param_type, desc, is_required) in &self.properties {
            let mut prop = json!({
                "type": param_type,
                "description": desc,
            });
            if let Some(default_val) = self.defaults.get(name) {
                prop["default"] = default_val.clone();
            }
            props.insert(name.clone(), prop);
            if *is_required {
                required.push(Value::String(name.clone()));
            }
        }

        let mut parameters = json!({
            "type": "object",
            "properties": Value::Object(props),
        });
        if !required.is_empty() {
            parameters["required"] = Value::Array(required);
        }

        ToolSchema {
            name: self.name,
            description: self.description,
            parameters: Some(parameters),
            extras: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return a JSON Schema type name for a `serde_json::Value`.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_f64() && n.as_i64().is_none() && n.as_u64().is_none() {
                "number"
            } else {
                "integer"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Check whether a value matches an expected JSON Schema type string.
fn type_matches(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true, // unknown type — allow
    }
}

/// Attempt to coerce a value to the expected type.
fn coerce_value(value: &Value, expected: &str) -> Option<Value> {
    match expected {
        "number" | "integer" => {
            if let Value::String(s) = value {
                if let Ok(n) = s.trim().parse::<i64>() {
                    return Some(Value::Number(n.into()));
                }
                if let Ok(n) = s.trim().parse::<f64>() {
                    return serde_json::Number::from_f64(n).map(Value::Number);
                }
            }
            None
        }
        "boolean" => {
            if let Value::String(s) = value {
                match s.trim().to_lowercase().as_str() {
                    "true" | "1" | "yes" => return Some(Value::Bool(true)),
                    "false" | "0" | "no" => return Some(Value::Bool(false)),
                    _ => {}
                }
            }
            None
        }
        "string" => {
            // Coerce non-string scalars to strings.
            match value {
                Value::Number(n) => Some(Value::String(n.to_string())),
                Value::Bool(b) => Some(Value::String(b.to_string())),
                _ => None,
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::ToolCall;
    use rustchain_core::tools::base::ToolSchema;
    use serde_json::json;
    use std::collections::HashMap;

    /// Helper: build a simple schema with a required "query" (string)
    /// and optional "limit" (integer).
    fn search_schema() -> ToolSchema {
        ValidationSchemaBuilder::new()
            .name("search")
            .description("Search tool")
            .required_param("query", "string", "The search query")
            .optional_param("limit", "integer", "Max results")
            .build()
    }

    fn make_tool_call(args: Value) -> ToolCall {
        let map: HashMap<String, Value> = args
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        ToolCall {
            name: "search".into(),
            args: map,
            id: Some("tc_1".into()),
        }
    }

    // 1. Valid tool call passes validation
    #[test]
    fn test_valid_tool_call_passes() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": "rust lang"}));
        let result = v.validate(&tc, &search_schema());
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.corrected_args.is_none());
    }

    // 2. Missing required field detected
    #[test]
    fn test_missing_required_field() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"limit": 5}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredField { field } if field == "query"
        )));
    }

    // 3. Unexpected field detected
    #[test]
    fn test_unexpected_field_strict() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": "test", "unknown_field": true}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::UnexpectedField { field } if field == "unknown_field"
        )));
    }

    // 4. Type mismatch detected
    #[test]
    fn test_type_mismatch() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": 42}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::TypeMismatch { field, expected, .. }
                if field == "query" && expected == "string"
        )));
    }

    // 5. Auto-correct type coercion: string -> number
    #[test]
    fn test_autocorrect_string_to_number() {
        let v = ToolCallValidator::new(StrictnessMode::AutoCorrect);
        let tc = make_tool_call(json!({"query": "test", "limit": "42"}));
        let result = v.validate(&tc, &search_schema());
        // The original has a type mismatch so is_valid is false.
        assert!(!result.is_valid);
        let corrected = result.corrected_args.unwrap();
        assert_eq!(corrected["limit"], json!(42));
        assert_eq!(corrected["query"], json!("test"));
    }

    // 6. Auto-correct type coercion: string -> bool
    #[test]
    fn test_autocorrect_string_to_bool() {
        let schema = ValidationSchemaBuilder::new()
            .name("toggle")
            .description("Toggle tool")
            .required_param("enabled", "boolean", "Whether enabled")
            .build();

        let v = ToolCallValidator::new(StrictnessMode::AutoCorrect);
        let tc = make_tool_call(json!({"enabled": "true"}));
        let result = v.validate(&tc, &schema);
        let corrected = result.corrected_args.unwrap();
        assert_eq!(corrected["enabled"], json!(true));
    }

    // 7. Auto-correct trailing comma in JSON
    #[test]
    fn test_fix_json_trailing_comma() {
        let raw = r#"{"query": "test", "limit": 5,}"#;
        let fixed = ToolCallCorrector::fix_json(raw).unwrap();
        assert_eq!(fixed["query"], json!("test"));
        assert_eq!(fixed["limit"], json!(5));
    }

    // 8. Auto-correct removes unexpected fields
    #[test]
    fn test_autocorrect_removes_unexpected_fields() {
        let v = ToolCallValidator::new(StrictnessMode::AutoCorrect);
        let tc = make_tool_call(json!({"query": "test", "bogus": "value"}));
        let result = v.validate(&tc, &search_schema());
        let corrected = result.corrected_args.unwrap();
        assert!(corrected.get("bogus").is_none());
        assert_eq!(corrected["query"], json!("test"));
    }

    // 9. Batch validation
    #[test]
    fn test_batch_validation() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let mut schemas = HashMap::new();
        schemas.insert("search".into(), search_schema());

        let calls = vec![
            make_tool_call(json!({"query": "good"})),
            make_tool_call(json!({"limit": 5})), // missing query
        ];

        let results = v.validate_batch(&calls, &schemas);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_valid);
        assert!(!results[1].is_valid);
    }

    // 10. ValidatedToolExecutor with valid args
    #[tokio::test]
    async fn test_validated_executor_valid_args() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::Strict);

        let tc = make_tool_call(json!({"query": "hello"}));
        let result = executor.execute(&tc).await;
        assert!(result.is_ok());
    }

    // 11. ValidatedToolExecutor with auto-correction
    #[tokio::test]
    async fn test_validated_executor_autocorrect() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::AutoCorrect);

        // Pass "limit" as string — should be auto-corrected to integer.
        let tc = make_tool_call(json!({"query": "hello", "limit": "10", "extra": true}));
        let result = executor.execute(&tc).await;
        assert!(result.is_ok());
    }

    // 12. ValidatedToolExecutor fails after max attempts
    #[tokio::test]
    async fn test_validated_executor_fails_after_max_attempts() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::AutoCorrect)
            .with_max_correction_attempts(2);

        // Missing required field cannot be auto-corrected.
        let tc = make_tool_call(json!({"limit": 5}));
        let result = executor.execute(&tc).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing required field"));
    }

    // 13. ToolSchemaBuilder basic usage
    #[test]
    fn test_schema_builder_basic() {
        let schema = ValidationSchemaBuilder::new()
            .name("my_tool")
            .description("Does things")
            .required_param("input", "string", "The input")
            .optional_param("verbose", "boolean", "Verbose output")
            .build();

        assert_eq!(schema.name, "my_tool");
        assert_eq!(schema.description, "Does things");
        let params = schema.parameters.unwrap();
        assert_eq!(params["properties"]["input"]["type"], "string");
        assert_eq!(params["properties"]["verbose"]["type"], "boolean");
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "input");
    }

    // 14. Strict vs lenient mode: unexpected field
    #[test]
    fn test_strict_vs_lenient_unexpected_field() {
        let strict = ToolCallValidator::new(StrictnessMode::Strict);
        let lenient = ToolCallValidator::new(StrictnessMode::Lenient);
        let tc = make_tool_call(json!({"query": "test", "extra": 1}));
        let schema = search_schema();

        let strict_result = strict.validate(&tc, &schema);
        let lenient_result = lenient.validate(&tc, &schema);

        // Strict reports the unexpected field.
        assert!(!strict_result.is_valid);
        assert!(strict_result.errors.iter().any(|e| matches!(
            e,
            ValidationError::UnexpectedField { .. }
        )));

        // Lenient does not report unexpected fields.
        assert!(lenient_result.is_valid);
        assert!(!lenient_result.errors.iter().any(|e| matches!(
            e,
            ValidationError::UnexpectedField { .. }
        )));
    }

    // 15. Multiple errors in one validation
    #[test]
    fn test_multiple_errors() {
        let schema = ValidationSchemaBuilder::new()
            .name("multi")
            .description("Multi-field")
            .required_param("a", "string", "Field A")
            .required_param("b", "integer", "Field B")
            .build();

        let v = ToolCallValidator::new(StrictnessMode::Strict);
        // Missing "a", wrong type for "b", unexpected "c".
        let tc = make_tool_call(json!({"b": "not_a_number", "c": true}));
        let result = v.validate(&tc, &schema);
        assert!(!result.is_valid);

        let has_missing = result.errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredField { field } if field == "a"
        ));
        let has_type_mismatch = result.errors.iter().any(|e| matches!(
            e,
            ValidationError::TypeMismatch { field, .. } if field == "b"
        ));
        let has_unexpected = result.errors.iter().any(|e| matches!(
            e,
            ValidationError::UnexpectedField { field } if field == "c"
        ));

        assert!(has_missing, "should detect missing field 'a'");
        assert!(has_type_mismatch, "should detect type mismatch on 'b'");
        assert!(has_unexpected, "should detect unexpected field 'c'");
    }

    // 16. Whitespace trimming
    #[test]
    fn test_whitespace_trimming() {
        let schema = search_schema();
        let args = json!({"query": "  hello world  "});
        let corrected = ToolCallCorrector::correct(&args, &schema);
        assert_eq!(corrected["query"], json!("hello world"));
    }

    // 17. Fix JSON single quotes
    #[test]
    fn test_fix_json_single_quotes() {
        let raw = "{'query': 'test', 'limit': 5}";
        let fixed = ToolCallCorrector::fix_json(raw).unwrap();
        assert_eq!(fixed["query"], json!("test"));
        assert_eq!(fixed["limit"], json!(5));
    }

    // 18. Schema with defaults fills missing optional fields
    #[test]
    fn test_fill_defaults() {
        let schema = ValidationSchemaBuilder::new()
            .name("search")
            .description("Search")
            .required_param("query", "string", "Query")
            .optional_param_with_default("limit", "integer", "Limit", json!(10))
            .build();

        let args = json!({"query": "test"});
        let corrected = ToolCallCorrector::correct(&args, &schema);
        assert_eq!(corrected["query"], json!("test"));
        assert_eq!(corrected["limit"], json!(10));
    }

    // 19. Batch validation with unknown tool name
    #[test]
    fn test_batch_unknown_tool() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let schemas = HashMap::new(); // empty

        let tc = ToolCall {
            name: "nonexistent".into(),
            args: HashMap::new(),
            id: None,
        };
        let results = v.validate_batch(&[tc], &schemas);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_valid);
    }

    // ---------- Test helper tool ----------

    /// A trivial tool that echoes its input as JSON.
    struct EchoTool;

    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "search"
        }
        fn description(&self) -> &str {
            "Echo tool for testing"
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            let val = match input {
                ToolInput::Text(s) => Value::String(s),
                ToolInput::Structured(m) => serde_json::to_value(m).unwrap(),
                ToolInput::ToolCall(tc) => serde_json::to_value(tc.args).unwrap(),
            };
            Ok(ToolOutput::Content(val))
        }
    }
}
