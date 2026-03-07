//! Tool input/output validation framework with schema support.
//!
//! Provides a comprehensive validation system for tool inputs and outputs:
//!
//! - [`ValidationRule`] — individual validation constraints (required, type, length, pattern, etc.)
//! - [`JsonType`] — JSON type discriminator for type-checking values
//! - [`FieldValidator`] — builder-pattern validator for a single field
//! - [`ToolValidator`] — validates tool inputs against multiple field validators
//! - [`OutputValidator`] — validates tool outputs against expected shape
//! - [`ValidatedTool`] — wraps a tool with input/output validation
//!
//! Also retains the original validation/correction API:
//! - [`ToolCallValidator`], [`ToolCallCorrector`], [`ValidatedToolExecutor`]
//! - [`StrictnessMode`], [`ValidationSchemaBuilder`]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::messages::ToolCall;
use rustchain_core::tools::base::{BaseTool, ToolSchema};
use rustchain_core::tools::types::{ToolInput, ToolOutput};

// ===========================================================================
// New validation framework types
// ===========================================================================

// ---------------------------------------------------------------------------
// JsonType
// ---------------------------------------------------------------------------

/// Represents JSON value types for type-checking validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonType {
    /// JSON string
    String,
    /// JSON number (float or integer)
    Number,
    /// JSON integer only
    Integer,
    /// JSON boolean
    Boolean,
    /// JSON array
    Array,
    /// JSON object
    Object,
    /// JSON null
    Null,
}

impl JsonType {
    /// Returns `true` if the given JSON value matches this type.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            JsonType::String => value.is_string(),
            JsonType::Number => value.is_number(),
            JsonType::Integer => value.is_i64() || value.is_u64(),
            JsonType::Boolean => value.is_boolean(),
            JsonType::Array => value.is_array(),
            JsonType::Object => value.is_object(),
            JsonType::Null => value.is_null(),
        }
    }

    /// Returns the JSON Schema type string for this type.
    pub fn as_schema_str(&self) -> &'static str {
        match self {
            JsonType::String => "string",
            JsonType::Number => "number",
            JsonType::Integer => "integer",
            JsonType::Boolean => "boolean",
            JsonType::Array => "array",
            JsonType::Object => "object",
            JsonType::Null => "null",
        }
    }

    /// Parse a JSON Schema type string into a `JsonType`.
    pub fn from_schema_str(s: &str) -> Option<Self> {
        match s {
            "string" => Some(JsonType::String),
            "number" => Some(JsonType::Number),
            "integer" => Some(JsonType::Integer),
            "boolean" => Some(JsonType::Boolean),
            "array" => Some(JsonType::Array),
            "object" => Some(JsonType::Object),
            "null" => Some(JsonType::Null),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationRule
// ---------------------------------------------------------------------------

/// A single validation constraint that can be applied to a field value.
pub enum ValidationRule {
    /// The field must be present (not missing and not null).
    Required,
    /// The field must match the specified JSON type.
    Type(JsonType),
    /// String values must have at least this many characters.
    MinLength(usize),
    /// String values must have at most this many characters.
    MaxLength(usize),
    /// String values must match this regex pattern.
    Pattern(String),
    /// Numeric values must fall within the given range.
    Range { min: Option<f64>, max: Option<f64> },
    /// The value must be one of the specified values.
    OneOf(Vec<Value>),
    /// A custom validation function.
    Custom {
        name: String,
        validator: Arc<dyn Fn(&Value) -> bool + Send + Sync>,
    },
}

impl std::fmt::Debug for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Required => write!(f, "Required"),
            Self::Type(t) => write!(f, "Type({:?})", t),
            Self::MinLength(n) => write!(f, "MinLength({})", n),
            Self::MaxLength(n) => write!(f, "MaxLength({})", n),
            Self::Pattern(p) => write!(f, "Pattern({})", p),
            Self::Range { min, max } => write!(f, "Range({:?}, {:?})", min, max),
            Self::OneOf(vals) => write!(f, "OneOf({:?})", vals),
            Self::Custom { name, .. } => write!(f, "Custom({})", name),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationError (new struct-based)
// ---------------------------------------------------------------------------

/// A single validation error for a specific field and rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputValidationError {
    /// The field that failed validation.
    pub field: String,
    /// The name of the rule that was violated.
    pub rule: String,
    /// A human-readable error message.
    pub message: String,
}

impl InputValidationError {
    /// Create a new validation error.
    pub fn new(
        field: impl Into<String>,
        rule: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            rule: rule.into(),
            message: message.into(),
        }
    }

    /// Convert this error to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "field": self.field,
            "rule": self.rule,
            "message": self.message,
        })
    }
}

impl std::fmt::Display for InputValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.field, self.message, self.rule)
    }
}

// ---------------------------------------------------------------------------
// InputValidationResult
// ---------------------------------------------------------------------------

/// Aggregated result of validating one or more fields.
#[derive(Debug, Clone, Default)]
pub struct InputValidationResult {
    /// All validation errors found.
    errors: Vec<InputValidationError>,
}

impl InputValidationResult {
    /// Create an empty (valid) result.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Add an error to this result.
    pub fn add_error(&mut self, error: InputValidationError) {
        self.errors.push(error);
    }

    /// Returns `true` if no validation errors were found.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns a reference to all validation errors.
    pub fn errors(&self) -> &[InputValidationError] {
        &self.errors
    }

    /// Returns all error messages as strings.
    pub fn error_messages(&self) -> Vec<String> {
        self.errors.iter().map(|e| e.message.clone()).collect()
    }

    /// Convert the result to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "is_valid": self.is_valid(),
            "errors": self.errors.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// FieldValidator
// ---------------------------------------------------------------------------

/// Validates a single field against a set of rules, using builder pattern.
#[derive(Debug)]
pub struct FieldValidator {
    /// The name of the field to validate.
    pub field: String,
    /// The validation rules for this field.
    pub rules: Vec<ValidationRule>,
    /// Optional custom error message override.
    pub error_message: Option<String>,
}

impl FieldValidator {
    /// Create a new field validator for the given field name.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            rules: Vec::new(),
            error_message: None,
        }
    }

    /// Add a `Required` rule.
    pub fn required(mut self) -> Self {
        self.rules.push(ValidationRule::Required);
        self
    }

    /// Add a `Type` rule.
    pub fn typed(mut self, json_type: JsonType) -> Self {
        self.rules.push(ValidationRule::Type(json_type));
        self
    }

    /// Add a `MinLength` rule.
    pub fn min_length(mut self, n: usize) -> Self {
        self.rules.push(ValidationRule::MinLength(n));
        self
    }

    /// Add a `MaxLength` rule.
    pub fn max_length(mut self, n: usize) -> Self {
        self.rules.push(ValidationRule::MaxLength(n));
        self
    }

    /// Add a `Pattern` rule.
    pub fn pattern(mut self, regex: impl Into<String>) -> Self {
        self.rules.push(ValidationRule::Pattern(regex.into()));
        self
    }

    /// Add a `Range` rule.
    pub fn range(mut self, min: Option<f64>, max: Option<f64>) -> Self {
        self.rules.push(ValidationRule::Range { min, max });
        self
    }

    /// Add a `OneOf` rule.
    pub fn one_of(mut self, values: Vec<Value>) -> Self {
        self.rules.push(ValidationRule::OneOf(values));
        self
    }

    /// Add a `Custom` validation rule.
    pub fn custom(
        mut self,
        name: impl Into<String>,
        validator: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.rules.push(ValidationRule::Custom {
            name: name.into(),
            validator: Arc::new(validator),
        });
        self
    }

    /// Set a custom error message that overrides the default messages.
    pub fn with_error_message(mut self, msg: impl Into<String>) -> Self {
        self.error_message = Some(msg.into());
        self
    }

    /// Validate the given input value (expected to be a JSON object) for this field.
    pub fn validate(&self, input: &Value) -> Result<()> {
        let obj = input.as_object();

        for rule in &self.rules {
            match rule {
                ValidationRule::Required => {
                    let present = obj
                        .map(|o| o.get(&self.field).map(|v| !v.is_null()).unwrap_or(false))
                        .unwrap_or(false);
                    if !present {
                        return Err(self.make_error(
                            "required",
                            format!("field '{}' is required", self.field),
                        ));
                    }
                }
                ValidationRule::Type(json_type) => {
                    if let Some(value) = obj.and_then(|o| o.get(&self.field)) {
                        if !value.is_null() && !json_type.matches(value) {
                            return Err(self.make_error(
                                "type",
                                format!(
                                    "field '{}' expected type {}, got {}",
                                    self.field,
                                    json_type.as_schema_str(),
                                    value_type_name(value)
                                ),
                            ));
                        }
                    }
                }
                ValidationRule::MinLength(n) => {
                    if let Some(Value::String(s)) = obj.and_then(|o| o.get(&self.field)) {
                        if s.len() < *n {
                            return Err(self.make_error(
                                "min_length",
                                format!(
                                    "field '{}' must be at least {} characters, got {}",
                                    self.field,
                                    n,
                                    s.len()
                                ),
                            ));
                        }
                    }
                }
                ValidationRule::MaxLength(n) => {
                    if let Some(Value::String(s)) = obj.and_then(|o| o.get(&self.field)) {
                        if s.len() > *n {
                            return Err(self.make_error(
                                "max_length",
                                format!(
                                    "field '{}' must be at most {} characters, got {}",
                                    self.field,
                                    n,
                                    s.len()
                                ),
                            ));
                        }
                    }
                }
                ValidationRule::Pattern(pat) => {
                    if let Some(Value::String(s)) = obj.and_then(|o| o.get(&self.field)) {
                        let re = regex::Regex::new(pat).map_err(|e| {
                            RustChainError::ToolValidationError(format!(
                                "invalid regex pattern '{}': {}",
                                pat, e
                            ))
                        })?;
                        if !re.is_match(s) {
                            return Err(self.make_error(
                                "pattern",
                                format!("field '{}' does not match pattern '{}'", self.field, pat),
                            ));
                        }
                    }
                }
                ValidationRule::Range { min, max } => {
                    if let Some(value) = obj.and_then(|o| o.get(&self.field)) {
                        if let Some(num) = value.as_f64() {
                            if let Some(min_val) = min {
                                if num < *min_val {
                                    return Err(self.make_error(
                                        "range",
                                        format!(
                                            "field '{}' value {} is below minimum {}",
                                            self.field, num, min_val
                                        ),
                                    ));
                                }
                            }
                            if let Some(max_val) = max {
                                if num > *max_val {
                                    return Err(self.make_error(
                                        "range",
                                        format!(
                                            "field '{}' value {} exceeds maximum {}",
                                            self.field, num, max_val
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                }
                ValidationRule::OneOf(allowed) => {
                    if let Some(value) = obj.and_then(|o| o.get(&self.field)) {
                        if !allowed.contains(value) {
                            return Err(self.make_error(
                                "one_of",
                                format!(
                                    "field '{}' value is not one of the allowed values",
                                    self.field
                                ),
                            ));
                        }
                    }
                }
                ValidationRule::Custom { name, validator } => {
                    if let Some(value) = obj.and_then(|o| o.get(&self.field)) {
                        if !validator(value) {
                            return Err(self.make_error(
                                name,
                                format!(
                                    "field '{}' failed custom validation '{}'",
                                    self.field, name
                                ),
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn make_error(&self, rule: &str, default_msg: String) -> RustChainError {
        let msg = self.error_message.clone().unwrap_or(default_msg);
        RustChainError::ToolValidationError(format!("[{}:{}] {}", self.field, rule, msg))
    }
}

// ---------------------------------------------------------------------------
// ToolValidator
// ---------------------------------------------------------------------------

/// Validates tool inputs against multiple field validators.
#[derive(Debug, Default)]
pub struct ToolValidator {
    /// Field validators to apply.
    fields: Vec<FieldValidator>,
}

impl ToolValidator {
    /// Create a new empty tool validator.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a field validator.
    pub fn add_field(&mut self, validator: FieldValidator) {
        self.fields.push(validator);
    }

    /// Validate input against all field validators, collecting all errors.
    pub fn validate(&self, input: &Value) -> InputValidationResult {
        let mut result = InputValidationResult::new();

        for field_validator in &self.fields {
            if let Err(e) = field_validator.validate(input) {
                let msg = e.to_string();
                result.add_error(InputValidationError::new(
                    &field_validator.field,
                    rule_name_from_error(&msg),
                    msg,
                ));
            }
        }

        result
    }

    /// Create a `ToolValidator` from a JSON Schema value.
    ///
    /// Reads `properties`, `required`, and per-field constraints like
    /// `type`, `minLength`, `maxLength`, `pattern`, `minimum`, `maximum`, `enum`.
    pub fn from_schema(schema: &Value) -> Self {
        let mut tv = Self::new();

        let properties = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let required_fields: Vec<String> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for (field_name, prop_schema) in &properties {
            let mut fv = FieldValidator::new(field_name);

            if required_fields.contains(field_name) {
                fv = fv.required();
            }

            if let Some(type_str) = prop_schema.get("type").and_then(|t| t.as_str()) {
                if let Some(jt) = JsonType::from_schema_str(type_str) {
                    fv = fv.typed(jt);
                }
            }

            if let Some(min_len) = prop_schema.get("minLength").and_then(|v| v.as_u64()) {
                fv = fv.min_length(min_len as usize);
            }

            if let Some(max_len) = prop_schema.get("maxLength").and_then(|v| v.as_u64()) {
                fv = fv.max_length(max_len as usize);
            }

            if let Some(pattern) = prop_schema.get("pattern").and_then(|v| v.as_str()) {
                fv = fv.pattern(pattern);
            }

            let min_val = prop_schema.get("minimum").and_then(|v| v.as_f64());
            let max_val = prop_schema.get("maximum").and_then(|v| v.as_f64());
            if min_val.is_some() || max_val.is_some() {
                fv = fv.range(min_val, max_val);
            }

            if let Some(enum_vals) = prop_schema.get("enum").and_then(|v| v.as_array()) {
                fv = fv.one_of(enum_vals.clone());
            }

            tv.add_field(fv);
        }

        tv
    }
}

// ---------------------------------------------------------------------------
// OutputValidator
// ---------------------------------------------------------------------------

/// Validates tool output values against expected shape.
#[derive(Debug, Default)]
pub struct OutputValidator {
    /// Expected output type.
    expected_type: Option<JsonType>,
    /// Expected fields in the output (if object).
    expected_fields: Vec<String>,
    /// Whether the output must be non-empty.
    expect_non_empty: bool,
}

impl OutputValidator {
    /// Create a new empty output validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Expect the output to be of a specific JSON type.
    pub fn expect_type(mut self, json_type: JsonType) -> Self {
        self.expected_type = Some(json_type);
        self
    }

    /// Expect the output to contain the specified fields (assumes object).
    pub fn expect_fields(mut self, fields: Vec<String>) -> Self {
        self.expected_fields = fields;
        self
    }

    /// Expect the output to be non-empty (non-null, non-empty string/array/object).
    pub fn expect_non_empty(mut self) -> Self {
        self.expect_non_empty = true;
        self
    }

    /// Validate a tool output value.
    pub fn validate(&self, output: &Value) -> InputValidationResult {
        let mut result = InputValidationResult::new();

        if let Some(ref expected) = self.expected_type {
            if !expected.matches(output) {
                result.add_error(InputValidationError::new(
                    "output",
                    "type",
                    format!(
                        "expected output type {}, got {}",
                        expected.as_schema_str(),
                        value_type_name(output)
                    ),
                ));
            }
        }

        if !self.expected_fields.is_empty() {
            if let Some(obj) = output.as_object() {
                for field in &self.expected_fields {
                    if !obj.contains_key(field) {
                        result.add_error(InputValidationError::new(
                            field,
                            "expected_field",
                            format!("expected field '{}' missing from output", field),
                        ));
                    }
                }
            } else {
                result.add_error(InputValidationError::new(
                    "output",
                    "type",
                    "expected output to be an object for field checking",
                ));
            }
        }

        if self.expect_non_empty && is_empty_value(output) {
            result.add_error(InputValidationError::new(
                "output",
                "non_empty",
                "expected non-empty output",
            ));
        }

        result
    }
}

// ---------------------------------------------------------------------------
// ValidatedTool
// ---------------------------------------------------------------------------

/// Wraps a tool with input and output validation.
#[derive(Debug)]
pub struct ValidatedTool {
    /// Name of the tool being validated.
    pub tool_name: String,
    /// Input validator.
    pub input_validator: ToolValidator,
    /// Output validator.
    pub output_validator: OutputValidator,
}

impl ValidatedTool {
    /// Create a new validated tool wrapper.
    pub fn new(
        tool_name: impl Into<String>,
        input_validator: ToolValidator,
        output_validator: OutputValidator,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            input_validator,
            output_validator,
        }
    }

    /// Validate tool input.
    pub fn validate_input(&self, input: &Value) -> InputValidationResult {
        self.input_validator.validate(input)
    }

    /// Validate tool output.
    pub fn validate_output(&self, output: &Value) -> InputValidationResult {
        self.output_validator.validate(output)
    }
}

// ===========================================================================
// Original validation types (preserved for backward compatibility)
// ===========================================================================

// ---------------------------------------------------------------------------
// Strictness mode
// ---------------------------------------------------------------------------

/// Controls how strict validation and correction behave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StrictnessMode {
    /// Every deviation is an error.
    #[default]
    Strict,
    /// Unexpected fields are silently ignored (but other errors still reported).
    Lenient,
    /// Attempt to fix problems automatically and populate `corrected_args`.
    AutoCorrect,
}

// ---------------------------------------------------------------------------
// Validation errors (original enum form)
// ---------------------------------------------------------------------------

/// A single issue found during validation (original enum-based error).
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
// Validation result (original)
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
        let args_value =
            serde_json::to_value(&tool_call.args).unwrap_or(Value::Object(Default::default()));
        self.validate_value(&args_value, schema)
    }

    /// Validate a raw JSON value against a schema.
    pub fn validate_value(&self, args: &Value, schema: &ToolSchema) -> ValidationResult {
        let mut errors = Vec::new();

        let params = match schema.parameters.as_ref() {
            Some(p) => p,
            None => {
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

        for field in &required_fields {
            if !args_obj.contains_key(field) {
                errors.push(ValidationError::MissingRequiredField {
                    field: field.clone(),
                });
            }
        }

        for key in args_obj.keys() {
            if !properties.contains_key(key)
                && (self.mode == StrictnessMode::Strict || self.mode == StrictnessMode::AutoCorrect)
            {
                errors.push(ValidationError::UnexpectedField { field: key.clone() });
            }
        }

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
                    let args_value =
                        serde_json::to_value(&tc.args).unwrap_or(Value::Object(Default::default()));
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
    pub fn fix_json(raw: &str) -> std::result::Result<Value, String> {
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            return Ok(v);
        }

        let mut fixed = raw.to_string();

        let re_trailing = regex::Regex::new(r",\s*([}\]])").unwrap();
        fixed = re_trailing.replace_all(&fixed, "$1").to_string();

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

        let known_keys: Vec<String> = properties.keys().cloned().collect();
        obj.retain(|k, _| known_keys.contains(k));

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
        let args_value =
            serde_json::to_value(&tool_call.args).unwrap_or(Value::Object(Default::default()));

        let result = self.validator.validate_value(&args_value, &self.schema);

        if result.is_valid {
            return self
                .inner
                .run(
                    ToolInput::Structured(tool_call.args.clone()),
                    tool_call.id.as_deref(),
                )
                .await;
        }

        if self.validator.mode != StrictnessMode::AutoCorrect {
            let msg = result
                .errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(RustChainError::ToolValidationError(msg));
        }

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
        let args_map = match &input {
            ToolInput::Structured(m) => m.clone(),
            ToolInput::ToolCall(tc) => tc.args.clone(),
            ToolInput::Text(s) => {
                let parsed: HashMap<String, Value> = serde_json::from_str(s).unwrap_or_default();
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
// ValidationSchemaBuilder
// ---------------------------------------------------------------------------

/// Convenience builder that mirrors the task requirements.
#[derive(Default)]
pub struct ValidationSchemaBuilder {
    name: String,
    description: String,
    properties: Vec<(String, String, String, bool)>,
    defaults: HashMap<String, Value>,
}

impl ValidationSchemaBuilder {
    pub fn new() -> Self {
        Self::default()
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

// ===========================================================================
// Internal helpers
// ===========================================================================

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

/// Human-readable type name for error messages.
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
        _ => true,
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
        "string" => match value {
            Value::Number(n) => Some(Value::String(n.to_string())),
            Value::Bool(b) => Some(Value::String(b.to_string())),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a value is considered "empty".
fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// Extract a rule name from an error message (best effort).
fn rule_name_from_error(msg: &str) -> String {
    if let Some(start) = msg.find(':') {
        if let Some(end) = msg[start + 1..].find(']') {
            return msg[start + 1..start + 1 + end].to_string();
        }
    }
    "unknown".to_string()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::ToolCall;
    use rustchain_core::tools::base::ToolSchema;
    use serde_json::json;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // JsonType matching tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_type_matches_string() {
        assert!(JsonType::String.matches(&json!("hello")));
        assert!(!JsonType::String.matches(&json!(42)));
    }

    #[test]
    fn test_json_type_matches_number() {
        assert!(JsonType::Number.matches(&json!(3.14)));
        assert!(JsonType::Number.matches(&json!(42)));
        assert!(!JsonType::Number.matches(&json!("42")));
    }

    #[test]
    fn test_json_type_matches_integer() {
        assert!(JsonType::Integer.matches(&json!(42)));
        assert!(!JsonType::Integer.matches(&json!(3.14)));
        assert!(!JsonType::Integer.matches(&json!("42")));
    }

    #[test]
    fn test_json_type_matches_boolean() {
        assert!(JsonType::Boolean.matches(&json!(true)));
        assert!(JsonType::Boolean.matches(&json!(false)));
        assert!(!JsonType::Boolean.matches(&json!("true")));
    }

    #[test]
    fn test_json_type_matches_array() {
        assert!(JsonType::Array.matches(&json!([1, 2, 3])));
        assert!(!JsonType::Array.matches(&json!({"a": 1})));
    }

    #[test]
    fn test_json_type_matches_object() {
        assert!(JsonType::Object.matches(&json!({"key": "value"})));
        assert!(!JsonType::Object.matches(&json!([1])));
    }

    #[test]
    fn test_json_type_matches_null() {
        assert!(JsonType::Null.matches(&json!(null)));
        assert!(!JsonType::Null.matches(&json!(0)));
    }

    // -----------------------------------------------------------------------
    // ValidationRule tests via FieldValidator
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_required_present() {
        let fv = FieldValidator::new("name").required();
        assert!(fv.validate(&json!({"name": "Alice"})).is_ok());
    }

    #[test]
    fn test_rule_required_missing() {
        let fv = FieldValidator::new("name").required();
        assert!(fv.validate(&json!({})).is_err());
    }

    #[test]
    fn test_rule_required_null() {
        let fv = FieldValidator::new("name").required();
        assert!(fv.validate(&json!({"name": null})).is_err());
    }

    #[test]
    fn test_rule_type_match() {
        let fv = FieldValidator::new("age").typed(JsonType::Number);
        assert!(fv.validate(&json!({"age": 25})).is_ok());
    }

    #[test]
    fn test_rule_type_mismatch() {
        let fv = FieldValidator::new("age").typed(JsonType::Number);
        assert!(fv.validate(&json!({"age": "twenty-five"})).is_err());
    }

    #[test]
    fn test_rule_min_length() {
        let fv = FieldValidator::new("query").min_length(3);
        assert!(fv.validate(&json!({"query": "ab"})).is_err());
        assert!(fv.validate(&json!({"query": "abc"})).is_ok());
    }

    #[test]
    fn test_rule_max_length() {
        let fv = FieldValidator::new("query").max_length(5);
        assert!(fv.validate(&json!({"query": "hello"})).is_ok());
        assert!(fv.validate(&json!({"query": "hello!"})).is_err());
    }

    #[test]
    fn test_rule_pattern() {
        let fv = FieldValidator::new("email").pattern(r"^[^@]+@[^@]+\.[^@]+$");
        assert!(fv.validate(&json!({"email": "test@example.com"})).is_ok());
        assert!(fv.validate(&json!({"email": "invalid"})).is_err());
    }

    #[test]
    fn test_rule_range_within() {
        let fv = FieldValidator::new("score").range(Some(0.0), Some(100.0));
        assert!(fv.validate(&json!({"score": 50})).is_ok());
    }

    #[test]
    fn test_rule_range_below_min() {
        let fv = FieldValidator::new("score").range(Some(0.0), Some(100.0));
        assert!(fv.validate(&json!({"score": -1})).is_err());
    }

    #[test]
    fn test_rule_range_above_max() {
        let fv = FieldValidator::new("score").range(Some(0.0), Some(100.0));
        assert!(fv.validate(&json!({"score": 101})).is_err());
    }

    #[test]
    fn test_rule_one_of_valid() {
        let fv =
            FieldValidator::new("color").one_of(vec![json!("red"), json!("green"), json!("blue")]);
        assert!(fv.validate(&json!({"color": "red"})).is_ok());
    }

    #[test]
    fn test_rule_one_of_invalid() {
        let fv =
            FieldValidator::new("color").one_of(vec![json!("red"), json!("green"), json!("blue")]);
        assert!(fv.validate(&json!({"color": "yellow"})).is_err());
    }

    #[test]
    fn test_rule_custom() {
        let fv = FieldValidator::new("value").custom("even_number", |v| {
            v.as_i64().map(|n| n % 2 == 0).unwrap_or(false)
        });
        assert!(fv.validate(&json!({"value": 4})).is_ok());
        assert!(fv.validate(&json!({"value": 3})).is_err());
    }

    // -----------------------------------------------------------------------
    // FieldValidator builder and combined rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_validator_builder_chaining() {
        let fv = FieldValidator::new("query")
            .required()
            .typed(JsonType::String)
            .min_length(1)
            .max_length(100);
        assert!(fv.validate(&json!({"query": "hello"})).is_ok());
    }

    #[test]
    fn test_field_validator_custom_error_message() {
        let fv = FieldValidator::new("name")
            .required()
            .with_error_message("Name is mandatory");
        let err = fv.validate(&json!({})).unwrap_err();
        assert!(err.to_string().contains("Name is mandatory"));
    }

    // -----------------------------------------------------------------------
    // ToolValidator with multiple fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_validator_all_valid() {
        let mut tv = ToolValidator::new();
        tv.add_field(
            FieldValidator::new("query")
                .required()
                .typed(JsonType::String),
        );
        tv.add_field(FieldValidator::new("limit").typed(JsonType::Number));

        let result = tv.validate(&json!({"query": "test", "limit": 10}));
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
    }

    #[test]
    fn test_tool_validator_multiple_errors() {
        let mut tv = ToolValidator::new();
        tv.add_field(
            FieldValidator::new("query")
                .required()
                .typed(JsonType::String),
        );
        tv.add_field(
            FieldValidator::new("limit")
                .required()
                .typed(JsonType::Integer),
        );

        let result = tv.validate(&json!({}));
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 2);
    }

    #[test]
    fn test_tool_validator_partial_errors() {
        let mut tv = ToolValidator::new();
        tv.add_field(FieldValidator::new("query").required());
        tv.add_field(FieldValidator::new("limit").typed(JsonType::Number));

        let result = tv.validate(&json!({"limit": "not_number"}));
        // query is required but missing -> error
        // limit has wrong type -> error
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 2);
    }

    // -----------------------------------------------------------------------
    // from_schema
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_schema_required_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        let tv = ToolValidator::from_schema(&schema);
        // name is required, age is not
        let result = tv.validate(&json!({"age": 25}));
        assert!(!result.is_valid());

        let result = tv.validate(&json!({"name": "Alice", "age": 25}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_from_schema_type_checking() {
        let schema = json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        });

        let tv = ToolValidator::from_schema(&schema);
        let result = tv.validate(&json!({"count": "not_integer"}));
        assert!(!result.is_valid());
    }

    #[test]
    fn test_from_schema_min_max_length() {
        let schema = json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "minLength": 2, "maxLength": 5}
            },
            "required": ["code"]
        });

        let tv = ToolValidator::from_schema(&schema);
        assert!(!tv.validate(&json!({"code": "a"})).is_valid());
        assert!(tv.validate(&json!({"code": "ab"})).is_valid());
        assert!(tv.validate(&json!({"code": "abcde"})).is_valid());
        assert!(!tv.validate(&json!({"code": "abcdef"})).is_valid());
    }

    #[test]
    fn test_from_schema_enum() {
        let schema = json!({
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            }
        });

        let tv = ToolValidator::from_schema(&schema);
        assert!(tv.validate(&json!({"status": "active"})).is_valid());
        assert!(!tv.validate(&json!({"status": "unknown"})).is_valid());
    }

    #[test]
    fn test_from_schema_range() {
        let schema = json!({
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100}
            }
        });

        let tv = ToolValidator::from_schema(&schema);
        assert!(tv.validate(&json!({"score": 50})).is_valid());
        assert!(!tv.validate(&json!({"score": -1})).is_valid());
        assert!(!tv.validate(&json!({"score": 101})).is_valid());
    }

    // -----------------------------------------------------------------------
    // OutputValidator
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_validator_type_check() {
        let ov = OutputValidator::new().expect_type(JsonType::Object);
        assert!(ov.validate(&json!({"result": "ok"})).is_valid());
        assert!(!ov.validate(&json!("just a string")).is_valid());
    }

    #[test]
    fn test_output_validator_expected_fields() {
        let ov = OutputValidator::new()
            .expect_type(JsonType::Object)
            .expect_fields(vec!["status".into(), "data".into()]);

        assert!(ov.validate(&json!({"status": "ok", "data": []})).is_valid());
        assert!(!ov.validate(&json!({"status": "ok"})).is_valid());
    }

    #[test]
    fn test_output_validator_non_empty() {
        let ov = OutputValidator::new().expect_non_empty();
        assert!(!ov.validate(&json!(null)).is_valid());
        assert!(!ov.validate(&json!("")).is_valid());
        assert!(!ov.validate(&json!([])).is_valid());
        assert!(!ov.validate(&json!({})).is_valid());
        assert!(ov.validate(&json!("hello")).is_valid());
        assert!(ov.validate(&json!(42)).is_valid());
    }

    // -----------------------------------------------------------------------
    // ValidatedTool wrapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_validated_tool_input_valid() {
        let mut tv = ToolValidator::new();
        tv.add_field(
            FieldValidator::new("query")
                .required()
                .typed(JsonType::String),
        );

        let ov = OutputValidator::new().expect_type(JsonType::Object);

        let vt = ValidatedTool::new("search", tv, ov);
        let result = vt.validate_input(&json!({"query": "hello"}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_validated_tool_input_invalid() {
        let mut tv = ToolValidator::new();
        tv.add_field(FieldValidator::new("query").required());

        let ov = OutputValidator::new();
        let vt = ValidatedTool::new("search", tv, ov);
        let result = vt.validate_input(&json!({}));
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validated_tool_output_valid() {
        let tv = ToolValidator::new();
        let ov = OutputValidator::new()
            .expect_type(JsonType::Object)
            .expect_fields(vec!["result".into()]);

        let vt = ValidatedTool::new("search", tv, ov);
        let result = vt.validate_output(&json!({"result": "found"}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_validated_tool_output_invalid() {
        let tv = ToolValidator::new();
        let ov = OutputValidator::new().expect_type(JsonType::Array);

        let vt = ValidatedTool::new("search", tv, ov);
        let result = vt.validate_output(&json!({"result": "found"}));
        assert!(!result.is_valid());
    }

    // -----------------------------------------------------------------------
    // InputValidationResult aggregation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validation_result_new_is_valid() {
        let result = InputValidationResult::new();
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
        assert!(result.error_messages().is_empty());
    }

    #[test]
    fn test_validation_result_add_errors() {
        let mut result = InputValidationResult::new();
        result.add_error(InputValidationError::new(
            "field1",
            "required",
            "missing field1",
        ));
        result.add_error(InputValidationError::new("field2", "type", "wrong type"));

        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 2);
        assert_eq!(result.error_messages().len(), 2);
    }

    #[test]
    fn test_validation_result_to_json() {
        let mut result = InputValidationResult::new();
        result.add_error(InputValidationError::new(
            "name",
            "required",
            "name is required",
        ));

        let j = result.to_json();
        assert_eq!(j["is_valid"], json!(false));
        assert!(j["errors"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn test_input_validation_error_to_json() {
        let err = InputValidationError::new("email", "pattern", "invalid email");
        let j = err.to_json();
        assert_eq!(j["field"], json!("email"));
        assert_eq!(j["rule"], json!("pattern"));
        assert_eq!(j["message"], json!("invalid email"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_input_object() {
        let mut tv = ToolValidator::new();
        tv.add_field(FieldValidator::new("a").required());
        tv.add_field(FieldValidator::new("b").required());

        let result = tv.validate(&json!({}));
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 2);
    }

    #[test]
    fn test_field_validator_on_non_object_input() {
        let fv = FieldValidator::new("x").required();
        // Non-object input: field is not present
        assert!(fv.validate(&json!("not an object")).is_err());
    }

    #[test]
    fn test_type_skip_when_field_missing_and_not_required() {
        let fv = FieldValidator::new("opt").typed(JsonType::String);
        // Field not present -- type check is skipped (no error)
        assert!(fv.validate(&json!({})).is_ok());
    }

    #[test]
    fn test_from_schema_pattern() {
        let schema = json!({
            "type": "object",
            "properties": {
                "zip": {"type": "string", "pattern": "^\\d{5}$"}
            },
            "required": ["zip"]
        });

        let tv = ToolValidator::from_schema(&schema);
        assert!(tv.validate(&json!({"zip": "12345"})).is_valid());
        assert!(!tv.validate(&json!({"zip": "1234"})).is_valid());
        assert!(!tv.validate(&json!({"zip": "abcde"})).is_valid());
    }

    // -----------------------------------------------------------------------
    // Original validation tests (preserved)
    // -----------------------------------------------------------------------

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

    #[test]
    fn test_valid_tool_call_passes() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": "rust lang"}));
        let result = v.validate(&tc, &search_schema());
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.corrected_args.is_none());
    }

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

    #[test]
    fn test_unexpected_field_strict() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": "test", "unknown_field": true}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
    }

    #[test]
    fn test_type_mismatch() {
        let v = ToolCallValidator::new(StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": 42}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
    }

    #[test]
    fn test_autocorrect_string_to_number() {
        let v = ToolCallValidator::new(StrictnessMode::AutoCorrect);
        let tc = make_tool_call(json!({"query": "test", "limit": "42"}));
        let result = v.validate(&tc, &search_schema());
        assert!(!result.is_valid);
        let corrected = result.corrected_args.unwrap();
        assert_eq!(corrected["limit"], json!(42));
    }

    #[test]
    fn test_fix_json_trailing_comma() {
        let raw = r#"{"query": "test", "limit": 5,}"#;
        let fixed = ToolCallCorrector::fix_json(raw).unwrap();
        assert_eq!(fixed["query"], json!("test"));
    }

    #[test]
    fn test_fix_json_single_quotes() {
        let raw = "{'query': 'test', 'limit': 5}";
        let fixed = ToolCallCorrector::fix_json(raw).unwrap();
        assert_eq!(fixed["query"], json!("test"));
    }

    #[test]
    fn test_whitespace_trimming() {
        let schema = search_schema();
        let args = json!({"query": "  hello world  "});
        let corrected = ToolCallCorrector::correct(&args, &schema);
        assert_eq!(corrected["query"], json!("hello world"));
    }

    #[test]
    fn test_schema_builder_basic() {
        let schema = ValidationSchemaBuilder::new()
            .name("my_tool")
            .description("Does things")
            .required_param("input", "string", "The input")
            .optional_param("verbose", "boolean", "Verbose output")
            .build();
        assert_eq!(schema.name, "my_tool");
        let params = schema.parameters.unwrap();
        assert_eq!(params["properties"]["input"]["type"], "string");
    }

    #[tokio::test]
    async fn test_validated_executor_valid_args() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::Strict);
        let tc = make_tool_call(json!({"query": "hello"}));
        let result = executor.execute(&tc).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validated_executor_autocorrect() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::AutoCorrect);
        let tc = make_tool_call(json!({"query": "hello", "limit": "10", "extra": true}));
        let result = executor.execute(&tc).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validated_executor_fails_after_max_attempts() {
        let tool = Arc::new(EchoTool);
        let schema = search_schema();
        let executor = ValidatedToolExecutor::new(tool, schema, StrictnessMode::AutoCorrect)
            .with_max_correction_attempts(2);
        let tc = make_tool_call(json!({"limit": 5}));
        let result = executor.execute(&tc).await;
        assert!(result.is_err());
    }

    // ---------- Test helper tool ----------

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
