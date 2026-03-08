//! JSON Schema-based tool parameter validation.
//!
//! This module provides types for defining, validating, and exporting tool parameter
//! schemas in both OpenAI function-calling and Anthropic tool_use formats. It mirrors
//! LangChain's tool schema generation capabilities.

use std::collections::HashMap;
use std::fmt;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// PropertySchema
// ---------------------------------------------------------------------------

/// Describes a single property within a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// The JSON Schema type (e.g. "string", "integer", "number", "boolean", "array", "object").
    pub type_name: String,
    /// Human-readable description of the property.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Allowed enum values (typically for string enums).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Minimum value for numeric types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value for numeric types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Minimum length for string types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,
    /// Maximum length for string types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    /// Regex pattern for string types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Schema for array items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<PropertySchema>>,
    /// Default value for this property.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<Value>,
}

impl PropertySchema {
    // -- Static constructors --------------------------------------------------

    /// Create a string property schema.
    pub fn string() -> Self {
        Self::new("string")
    }

    /// Create an integer property schema.
    pub fn integer() -> Self {
        Self::new("integer")
    }

    /// Create a number (float) property schema.
    pub fn number() -> Self {
        Self::new("number")
    }

    /// Create a boolean property schema.
    pub fn boolean() -> Self {
        Self::new("boolean")
    }

    /// Create an array property schema with the given items schema.
    pub fn array(items: PropertySchema) -> Self {
        Self {
            items: Some(Box::new(items)),
            ..Self::new("array")
        }
    }

    /// Create an object property schema.
    pub fn object() -> Self {
        Self::new("object")
    }

    /// Create a string-enum property schema with the given allowed values.
    pub fn enum_type(values: Vec<String>) -> Self {
        Self {
            enum_values: Some(values),
            ..Self::new("string")
        }
    }

    // -- Builder methods ------------------------------------------------------

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the enum values.
    pub fn with_enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }

    /// Set the minimum value constraint.
    pub fn with_minimum(mut self, min: f64) -> Self {
        self.minimum = Some(min);
        self
    }

    /// Set the maximum value constraint.
    pub fn with_maximum(mut self, max: f64) -> Self {
        self.maximum = Some(max);
        self
    }

    /// Set the minimum string length constraint.
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Set the maximum string length constraint.
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Set the regex pattern constraint.
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// Set the items schema (for array types).
    pub fn with_items(mut self, items: PropertySchema) -> Self {
        self.items = Some(Box::new(items));
        self
    }

    /// Set the default value.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }

    // -- Conversion -----------------------------------------------------------

    /// Convert this property schema to a JSON Schema value.
    pub fn to_json(&self) -> Value {
        let mut schema = json!({ "type": self.type_name });

        if let Some(ref desc) = self.description {
            schema["description"] = Value::String(desc.clone());
        }
        if let Some(ref vals) = self.enum_values {
            schema["enum"] = json!(vals);
        }
        if let Some(min) = self.minimum {
            schema["minimum"] = json!(min);
        }
        if let Some(max) = self.maximum {
            schema["maximum"] = json!(max);
        }
        if let Some(min_len) = self.min_length {
            schema["minLength"] = json!(min_len);
        }
        if let Some(max_len) = self.max_length {
            schema["maxLength"] = json!(max_len);
        }
        if let Some(ref pat) = self.pattern {
            schema["pattern"] = Value::String(pat.clone());
        }
        if let Some(ref items) = self.items {
            schema["items"] = items.to_json();
        }
        if let Some(ref default) = self.default_value {
            schema["default"] = default.clone();
        }

        schema
    }

    // -- Private helpers ------------------------------------------------------

    fn new(type_name: &str) -> Self {
        Self {
            type_name: type_name.to_string(),
            description: None,
            enum_values: None,
            minimum: None,
            maximum: None,
            min_length: None,
            max_length: None,
            pattern: None,
            items: None,
            default_value: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaObject
// ---------------------------------------------------------------------------

/// Represents a JSON Schema object with properties, required fields, and
/// additionalProperties control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaObject {
    /// The JSON Schema type (always "object" by default).
    pub type_name: String,
    /// Property definitions keyed by name.
    pub properties: HashMap<String, PropertySchema>,
    /// Names of required properties.
    pub required: Vec<String>,
    /// Whether additional properties are allowed.
    pub additional_properties: bool,
}

impl Default for SchemaObject {
    fn default() -> Self {
        Self {
            type_name: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            additional_properties: false,
        }
    }
}

impl SchemaObject {
    /// Create a new empty schema object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a property.
    pub fn with_property(mut self, name: impl Into<String>, schema: PropertySchema) -> Self {
        self.properties.insert(name.into(), schema);
        self
    }

    /// Mark a field as required.
    pub fn with_required(mut self, name: impl Into<String>) -> Self {
        self.required.push(name.into());
        self
    }

    /// Set whether additional properties are allowed.
    pub fn with_additional_properties(mut self, allow: bool) -> Self {
        self.additional_properties = allow;
        self
    }

    /// Convert to a JSON Schema value.
    pub fn to_json(&self) -> Value {
        let mut props = serde_json::Map::new();
        for (name, prop) in &self.properties {
            props.insert(name.clone(), prop.to_json());
        }

        let mut schema = json!({
            "type": self.type_name,
            "properties": Value::Object(props),
            "additionalProperties": self.additional_properties,
        });

        if !self.required.is_empty() {
            schema["required"] = json!(self.required);
        }

        schema
    }
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

/// Describes a single validation failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationError {
    /// The field that failed validation.
    pub field: String,
    /// Human-readable error message.
    pub message: String,
    /// What was expected.
    pub expected: String,
    /// What was actually received.
    pub actual: String,
}

impl ValidationError {
    /// Create a new validation error.
    pub fn new(
        field: impl Into<String>,
        message: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Convert to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "field": self.field,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        })
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Validation error on field '{}': {} (expected: {}, actual: {})",
            self.field, self.message, self.expected, self.actual
        )
    }
}

// ---------------------------------------------------------------------------
// ToolSchema (new, richer version — distinct from base::ToolSchema)
// ---------------------------------------------------------------------------

/// A tool's parameter schema with JSON Schema validation, supporting export to
/// OpenAI function-calling and Anthropic tool_use formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// The parameters schema object.
    pub parameters: SchemaObject,
    /// Whether the schema enforces strict mode (e.g. OpenAI strict mode).
    #[serde(default)]
    pub strict: bool,
}

impl ToolSchema {
    /// Create a new tool schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: SchemaObject::new(),
            strict: false,
        }
    }

    /// Set the parameters schema.
    pub fn with_parameters(mut self, parameters: SchemaObject) -> Self {
        self.parameters = parameters;
        self
    }

    /// Enable or disable strict mode.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Export in OpenAI function-calling format.
    ///
    /// ```json
    /// {
    ///   "type": "function",
    ///   "function": {
    ///     "name": "...",
    ///     "description": "...",
    ///     "parameters": { ... },
    ///     "strict": false
    ///   }
    /// }
    /// ```
    pub fn to_json(&self) -> Value {
        let mut func = json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_json(),
        });

        if self.strict {
            func["strict"] = Value::Bool(true);
        }

        json!({
            "type": "function",
            "function": func,
        })
    }

    /// Export in Anthropic tool_use format.
    ///
    /// ```json
    /// {
    ///   "name": "...",
    ///   "description": "...",
    ///   "input_schema": { ... }
    /// }
    /// ```
    pub fn to_anthropic_json(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.to_json(),
        })
    }

    /// Validate an input value against this schema's parameters.
    pub fn validate_input(&self, input: &Value) -> std::result::Result<(), Vec<ValidationError>> {
        SchemaValidator::validate(&self.parameters, input)
    }
}

// ---------------------------------------------------------------------------
// SchemaValidator
// ---------------------------------------------------------------------------

/// Validates JSON values against a [`SchemaObject`].
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate a value against the given schema object.
    ///
    /// Returns `Ok(())` if valid, or `Err(errors)` with all validation errors.
    pub fn validate(
        schema: &SchemaObject,
        value: &Value,
    ) -> std::result::Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        let obj = match value.as_object() {
            Some(o) => o,
            None => {
                errors.push(ValidationError::new(
                    "<root>",
                    "Expected an object",
                    "object",
                    value_type_name(value),
                ));
                return Err(errors);
            }
        };

        // Check required fields.
        for req in &schema.required {
            if !obj.contains_key(req) {
                errors.push(ValidationError::new(
                    req,
                    format!("Required field '{}' is missing", req),
                    "present",
                    "missing",
                ));
            }
        }

        // Check unknown fields when additional_properties is false.
        if !schema.additional_properties {
            for key in obj.keys() {
                if !schema.properties.contains_key(key) {
                    errors.push(ValidationError::new(
                        key,
                        format!("Unknown field '{}'", key),
                        "known property",
                        "unknown property",
                    ));
                }
            }
        }

        // Validate each present property.
        for (name, prop_schema) in &schema.properties {
            if let Some(val) = obj.get(name) {
                Self::validate_property(name, prop_schema, val, &mut errors);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_property(
        field: &str,
        schema: &PropertySchema,
        value: &Value,
        errors: &mut Vec<ValidationError>,
    ) {
        // Type check.
        if !Self::type_matches(&schema.type_name, value) {
            errors.push(ValidationError::new(
                field,
                format!("Type mismatch for field '{}'", field),
                &schema.type_name,
                value_type_name(value),
            ));
            return; // Skip further checks if the type is wrong.
        }

        // Enum check.
        if let Some(ref allowed) = schema.enum_values {
            if let Some(s) = value.as_str() {
                if !allowed.iter().any(|v| v == s) {
                    errors.push(ValidationError::new(
                        field,
                        format!("Value '{}' is not one of the allowed enum values", s),
                        format!("one of {:?}", allowed),
                        s.to_string(),
                    ));
                }
            }
        }

        // Numeric range checks.
        if let Some(num) = value.as_f64() {
            if let Some(min) = schema.minimum {
                if num < min {
                    errors.push(ValidationError::new(
                        field,
                        format!("Value {} is less than minimum {}", num, min),
                        format!(">= {}", min),
                        num.to_string(),
                    ));
                }
            }
            if let Some(max) = schema.maximum {
                if num > max {
                    errors.push(ValidationError::new(
                        field,
                        format!("Value {} exceeds maximum {}", num, max),
                        format!("<= {}", max),
                        num.to_string(),
                    ));
                }
            }
        }

        // String length checks.
        if let Some(s) = value.as_str() {
            if let Some(min_len) = schema.min_length {
                if s.len() < min_len {
                    errors.push(ValidationError::new(
                        field,
                        format!(
                            "String length {} is less than minimum {}",
                            s.len(),
                            min_len
                        ),
                        format!("length >= {}", min_len),
                        format!("length {}", s.len()),
                    ));
                }
            }
            if let Some(max_len) = schema.max_length {
                if s.len() > max_len {
                    errors.push(ValidationError::new(
                        field,
                        format!(
                            "String length {} exceeds maximum {}",
                            s.len(),
                            max_len
                        ),
                        format!("length <= {}", max_len),
                        format!("length {}", s.len()),
                    ));
                }
            }

            // Pattern check.
            if let Some(ref pat) = schema.pattern {
                if let Ok(re) = Regex::new(pat) {
                    if !re.is_match(s) {
                        errors.push(ValidationError::new(
                            field,
                            format!("Value '{}' does not match pattern '{}'", s, pat),
                            format!("matches /{}/", pat),
                            s.to_string(),
                        ));
                    }
                }
            }
        }

        // Array item validation.
        if let (Some(arr), Some(ref items_schema)) = (value.as_array(), &schema.items) {
            for (i, item) in arr.iter().enumerate() {
                let item_field = format!("{}[{}]", field, i);
                Self::validate_property(&item_field, items_schema, item, errors);
            }
        }
    }

    /// Check if a value matches the expected JSON Schema type name.
    fn type_matches(type_name: &str, value: &Value) -> bool {
        match type_name {
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true, // Unknown type, allow.
        }
    }
}

/// Return the JSON type name for a value.
fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// ToolSchemaGenerator
// ---------------------------------------------------------------------------

/// Declarative builder for constructing a [`ToolSchema`] from individual
/// parameter definitions.
pub struct ToolSchemaGenerator {
    name: String,
    description: String,
    properties: Vec<(String, PropertySchema, bool)>,
}

impl ToolSchemaGenerator {
    /// Create a new generator.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            properties: Vec::new(),
        }
    }

    /// Add a string parameter.
    pub fn add_string_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::string().with_description(description),
            required,
        ));
        self
    }

    /// Add an integer parameter.
    pub fn add_integer_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::integer().with_description(description),
            required,
        ));
        self
    }

    /// Add a number (float) parameter.
    pub fn add_number_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::number().with_description(description),
            required,
        ));
        self
    }

    /// Add a boolean parameter.
    pub fn add_boolean_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::boolean().with_description(description),
            required,
        ));
        self
    }

    /// Add a string-enum parameter.
    pub fn add_enum_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::enum_type(values).with_description(description),
            required,
        ));
        self
    }

    /// Add an array parameter with the given items schema.
    pub fn add_array_param(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        items: PropertySchema,
        required: bool,
    ) -> Self {
        self.properties.push((
            name.into(),
            PropertySchema::array(items).with_description(description),
            required,
        ));
        self
    }

    /// Build the final [`ToolSchema`].
    pub fn build(self) -> ToolSchema {
        let mut schema_obj = SchemaObject::new();
        for (name, prop, required) in self.properties {
            schema_obj.properties.insert(name.clone(), prop);
            if required {
                schema_obj.required.push(name);
            }
        }

        ToolSchema {
            name: self.name,
            description: self.description,
            parameters: schema_obj,
            strict: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaRegistry
// ---------------------------------------------------------------------------

/// A registry that stores tool schemas and supports validation of tool calls.
#[derive(Debug, Clone, Default)]
pub struct SchemaRegistry {
    schemas: HashMap<String, ToolSchema>,
}

impl SchemaRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool schema.
    pub fn register(&mut self, schema: ToolSchema) {
        self.schemas.insert(schema.name.clone(), schema);
    }

    /// Look up a schema by tool name.
    pub fn get(&self, name: &str) -> Option<&ToolSchema> {
        self.schemas.get(name)
    }

    /// Validate a tool call by name and input.
    pub fn validate_call(&self, name: &str, input: &Value) -> Result<()> {
        let schema = self.schemas.get(name).ok_or_else(|| {
            RustChainError::ToolValidationError(format!("Unknown tool '{}'", name))
        })?;

        schema.validate_input(input).map_err(|errs| {
            let messages: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            RustChainError::ToolValidationError(messages.join("; "))
        })
    }

    /// Return references to all registered schemas.
    pub fn all_schemas(&self) -> Vec<&ToolSchema> {
        self.schemas.values().collect()
    }

    /// Export all schemas as a JSON array (OpenAI format).
    pub fn to_json(&self) -> Value {
        let arr: Vec<Value> = self.schemas.values().map(|s| s.to_json()).collect();
        Value::Array(arr)
    }

    /// Return the number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- PropertySchema static constructors -----------------------------------

    #[test]
    fn test_property_string() {
        let p = PropertySchema::string();
        assert_eq!(p.type_name, "string");
        assert!(p.description.is_none());
    }

    #[test]
    fn test_property_integer() {
        let p = PropertySchema::integer();
        assert_eq!(p.type_name, "integer");
    }

    #[test]
    fn test_property_number() {
        let p = PropertySchema::number();
        assert_eq!(p.type_name, "number");
    }

    #[test]
    fn test_property_boolean() {
        let p = PropertySchema::boolean();
        assert_eq!(p.type_name, "boolean");
    }

    #[test]
    fn test_property_array() {
        let p = PropertySchema::array(PropertySchema::string());
        assert_eq!(p.type_name, "array");
        assert!(p.items.is_some());
        assert_eq!(p.items.as_ref().unwrap().type_name, "string");
    }

    #[test]
    fn test_property_object() {
        let p = PropertySchema::object();
        assert_eq!(p.type_name, "object");
    }

    #[test]
    fn test_property_enum_type() {
        let p = PropertySchema::enum_type(vec!["a".into(), "b".into()]);
        assert_eq!(p.type_name, "string");
        assert_eq!(p.enum_values.as_ref().unwrap(), &["a", "b"]);
    }

    // -- PropertySchema builder -----------------------------------------------

    #[test]
    fn test_property_builder_chain() {
        let p = PropertySchema::string()
            .with_description("A name")
            .with_min_length(1)
            .with_max_length(100)
            .with_pattern("^[a-z]+$")
            .with_default(json!("default"));

        assert_eq!(p.description.as_deref(), Some("A name"));
        assert_eq!(p.min_length, Some(1));
        assert_eq!(p.max_length, Some(100));
        assert_eq!(p.pattern.as_deref(), Some("^[a-z]+$"));
        assert_eq!(p.default_value, Some(json!("default")));
    }

    #[test]
    fn test_property_number_constraints() {
        let p = PropertySchema::number()
            .with_minimum(0.0)
            .with_maximum(100.0);

        assert_eq!(p.minimum, Some(0.0));
        assert_eq!(p.maximum, Some(100.0));
    }

    #[test]
    fn test_property_to_json() {
        let p = PropertySchema::string()
            .with_description("user name")
            .with_min_length(1);

        let j = p.to_json();
        assert_eq!(j["type"], "string");
        assert_eq!(j["description"], "user name");
        assert_eq!(j["minLength"], 1);
    }

    #[test]
    fn test_property_to_json_enum() {
        let p = PropertySchema::enum_type(vec!["red".into(), "green".into()]);
        let j = p.to_json();
        assert_eq!(j["enum"], json!(["red", "green"]));
    }

    #[test]
    fn test_property_to_json_array_items() {
        let p = PropertySchema::array(PropertySchema::integer());
        let j = p.to_json();
        assert_eq!(j["type"], "array");
        assert_eq!(j["items"]["type"], "integer");
    }

    // -- SchemaObject ---------------------------------------------------------

    #[test]
    fn test_schema_object_default() {
        let s = SchemaObject::new();
        assert_eq!(s.type_name, "object");
        assert!(s.properties.is_empty());
        assert!(s.required.is_empty());
        assert!(!s.additional_properties);
    }

    #[test]
    fn test_schema_object_builder() {
        let s = SchemaObject::new()
            .with_property("name", PropertySchema::string().with_description("Name"))
            .with_property("age", PropertySchema::integer())
            .with_required("name")
            .with_additional_properties(true);

        assert_eq!(s.properties.len(), 2);
        assert_eq!(s.required, vec!["name"]);
        assert!(s.additional_properties);
    }

    #[test]
    fn test_schema_object_to_json() {
        let s = SchemaObject::new()
            .with_property("query", PropertySchema::string())
            .with_required("query");

        let j = s.to_json();
        assert_eq!(j["type"], "object");
        assert_eq!(j["properties"]["query"]["type"], "string");
        assert_eq!(j["required"], json!(["query"]));
        assert_eq!(j["additionalProperties"], false);
    }

    #[test]
    fn test_schema_object_no_required() {
        let s = SchemaObject::new()
            .with_property("opt", PropertySchema::string());

        let j = s.to_json();
        assert!(j.get("required").is_none());
    }

    // -- ToolSchema to_json (OpenAI format) -----------------------------------

    #[test]
    fn test_tool_schema_to_openai_json() {
        let schema = ToolSchema::new("search", "Search for docs")
            .with_parameters(
                SchemaObject::new()
                    .with_property("query", PropertySchema::string())
                    .with_required("query"),
            );

        let j = schema.to_json();
        assert_eq!(j["type"], "function");
        assert_eq!(j["function"]["name"], "search");
        assert_eq!(j["function"]["description"], "Search for docs");
        assert_eq!(j["function"]["parameters"]["type"], "object");
        assert_eq!(j["function"]["parameters"]["properties"]["query"]["type"], "string");
        // strict should not be present when false
        assert!(j["function"].get("strict").is_none());
    }

    #[test]
    fn test_tool_schema_strict_mode() {
        let schema = ToolSchema::new("test", "Test").with_strict(true);
        let j = schema.to_json();
        assert_eq!(j["function"]["strict"], true);
    }

    // -- ToolSchema to_anthropic_json -----------------------------------------

    #[test]
    fn test_tool_schema_to_anthropic_json() {
        let schema = ToolSchema::new("calculator", "Do math")
            .with_parameters(
                SchemaObject::new()
                    .with_property("expression", PropertySchema::string())
                    .with_required("expression"),
            );

        let j = schema.to_anthropic_json();
        assert_eq!(j["name"], "calculator");
        assert_eq!(j["description"], "Do math");
        assert_eq!(j["input_schema"]["type"], "object");
        assert_eq!(j["input_schema"]["properties"]["expression"]["type"], "string");
        // Should NOT have "type": "function" wrapper
        assert!(j.get("type").is_none());
    }

    // -- SchemaValidator: type checking ---------------------------------------

    #[test]
    fn test_validate_correct_types() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string())
            .with_property("age", PropertySchema::integer())
            .with_property("score", PropertySchema::number())
            .with_property("active", PropertySchema::boolean());

        let input = json!({ "name": "Alice", "age": 30, "score": 9.5, "active": true });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_validate_type_mismatch() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string())
            .with_required("name");

        let input = json!({ "name": 42 });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].field, "name");
        assert!(errs[0].expected.contains("string"));
    }

    #[test]
    fn test_validate_non_object_input() {
        let schema = SchemaObject::new();
        let input = json!("not an object");
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "<root>");
    }

    // -- SchemaValidator: required fields -------------------------------------

    #[test]
    fn test_validate_missing_required_field() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string())
            .with_required("name");

        let input = json!({});
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert!(errs.iter().any(|e| e.field == "name" && e.actual == "missing"));
    }

    #[test]
    fn test_validate_all_optional_empty_input() {
        let schema = SchemaObject::new()
            .with_property("opt", PropertySchema::string())
            .with_additional_properties(true);

        let input = json!({});
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    // -- SchemaValidator: enum ------------------------------------------------

    #[test]
    fn test_validate_enum_valid() {
        let schema = SchemaObject::new()
            .with_property(
                "color",
                PropertySchema::enum_type(vec!["red".into(), "green".into(), "blue".into()]),
            );

        let input = json!({ "color": "red" });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_validate_enum_invalid() {
        let schema = SchemaObject::new()
            .with_property(
                "color",
                PropertySchema::enum_type(vec!["red".into(), "green".into()]),
            );

        let input = json!({ "color": "yellow" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "color");
    }

    // -- SchemaValidator: numeric ranges --------------------------------------

    #[test]
    fn test_validate_minimum() {
        let schema = SchemaObject::new()
            .with_property("age", PropertySchema::integer().with_minimum(0.0));

        let input = json!({ "age": -1 });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "age");
    }

    #[test]
    fn test_validate_maximum() {
        let schema = SchemaObject::new()
            .with_property("pct", PropertySchema::number().with_maximum(100.0));

        let input = json!({ "pct": 101.5 });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "pct");
    }

    #[test]
    fn test_validate_range_ok() {
        let schema = SchemaObject::new()
            .with_property(
                "temp",
                PropertySchema::number().with_minimum(-273.15).with_maximum(1000.0),
            );

        let input = json!({ "temp": 22.5 });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    // -- SchemaValidator: string lengths --------------------------------------

    #[test]
    fn test_validate_min_length() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string().with_min_length(3));

        let input = json!({ "name": "ab" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "name");
    }

    #[test]
    fn test_validate_max_length() {
        let schema = SchemaObject::new()
            .with_property("code", PropertySchema::string().with_max_length(5));

        let input = json!({ "code": "abcdef" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "code");
    }

    // -- SchemaValidator: pattern ---------------------------------------------

    #[test]
    fn test_validate_pattern_match() {
        let schema = SchemaObject::new()
            .with_property("email", PropertySchema::string().with_pattern(r"^.+@.+\..+$"));

        let input = json!({ "email": "a@b.com" });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_validate_pattern_no_match() {
        let schema = SchemaObject::new()
            .with_property("email", PropertySchema::string().with_pattern(r"^.+@.+\..+$"));

        let input = json!({ "email": "invalid" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs[0].field, "email");
    }

    // -- SchemaValidator: unknown fields --------------------------------------

    #[test]
    fn test_validate_unknown_field_rejected() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string());

        let input = json!({ "name": "Alice", "extra": 42 });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert!(errs.iter().any(|e| e.field == "extra"));
    }

    #[test]
    fn test_validate_unknown_field_allowed() {
        let schema = SchemaObject::new()
            .with_property("name", PropertySchema::string())
            .with_additional_properties(true);

        let input = json!({ "name": "Alice", "extra": 42 });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    // -- SchemaValidator: nested arrays ---------------------------------------

    #[test]
    fn test_validate_array_items() {
        let schema = SchemaObject::new()
            .with_property("tags", PropertySchema::array(PropertySchema::string()));

        let input = json!({ "tags": ["a", "b", "c"] });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_validate_array_items_wrong_type() {
        let schema = SchemaObject::new()
            .with_property("nums", PropertySchema::array(PropertySchema::integer()));

        let input = json!({ "nums": [1, "two", 3] });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].field, "nums[1]");
    }

    // -- SchemaValidator: empty schema ----------------------------------------

    #[test]
    fn test_validate_empty_schema_empty_input() {
        let schema = SchemaObject::new().with_additional_properties(true);
        let input = json!({});
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_validate_empty_schema_rejects_extra_fields() {
        let schema = SchemaObject::new(); // additional_properties = false
        let input = json!({ "foo": "bar" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert!(!errs.is_empty());
    }

    // -- ToolSchemaGenerator --------------------------------------------------

    #[test]
    fn test_generator_build_simple() {
        let schema = ToolSchemaGenerator::new("search", "Search docs")
            .add_string_param("query", "The query", true)
            .add_integer_param("limit", "Max results", false)
            .build();

        assert_eq!(schema.name, "search");
        assert_eq!(schema.description, "Search docs");
        assert!(schema.parameters.properties.contains_key("query"));
        assert!(schema.parameters.properties.contains_key("limit"));
        assert!(schema.parameters.required.contains(&"query".to_string()));
        assert!(!schema.parameters.required.contains(&"limit".to_string()));
    }

    #[test]
    fn test_generator_all_param_types() {
        let schema = ToolSchemaGenerator::new("test", "Test tool")
            .add_string_param("s", "string param", true)
            .add_integer_param("i", "integer param", true)
            .add_number_param("n", "number param", false)
            .add_boolean_param("b", "boolean param", false)
            .add_enum_param("e", "enum param", vec!["x".into(), "y".into()], true)
            .add_array_param("a", "array param", PropertySchema::string(), false)
            .build();

        assert_eq!(schema.parameters.properties.len(), 6);
        assert_eq!(schema.parameters.required.len(), 3);
    }

    #[test]
    fn test_generator_builds_valid_openai_json() {
        let schema = ToolSchemaGenerator::new("calc", "Calculator")
            .add_string_param("expression", "Math expression", true)
            .build();

        let j = schema.to_json();
        assert_eq!(j["type"], "function");
        assert_eq!(j["function"]["name"], "calc");
    }

    #[test]
    fn test_generator_builds_valid_anthropic_json() {
        let schema = ToolSchemaGenerator::new("calc", "Calculator")
            .add_string_param("expression", "Math expression", true)
            .build();

        let j = schema.to_anthropic_json();
        assert_eq!(j["name"], "calc");
        assert!(j.get("input_schema").is_some());
    }

    // -- SchemaRegistry -------------------------------------------------------

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = SchemaRegistry::new();
        let schema = ToolSchemaGenerator::new("search", "Search")
            .add_string_param("q", "query", true)
            .build();

        reg.register(schema);
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert!(reg.get("search").is_some());
        assert!(reg.get("unknown").is_none());
    }

    #[test]
    fn test_registry_validate_call_ok() {
        let mut reg = SchemaRegistry::new();
        reg.register(
            ToolSchemaGenerator::new("greet", "Greet")
                .add_string_param("name", "Name", true)
                .build(),
        );

        let input = json!({ "name": "Alice" });
        assert!(reg.validate_call("greet", &input).is_ok());
    }

    #[test]
    fn test_registry_validate_call_missing_field() {
        let mut reg = SchemaRegistry::new();
        reg.register(
            ToolSchemaGenerator::new("greet", "Greet")
                .add_string_param("name", "Name", true)
                .build(),
        );

        let input = json!({});
        assert!(reg.validate_call("greet", &input).is_err());
    }

    #[test]
    fn test_registry_validate_call_unknown_tool() {
        let reg = SchemaRegistry::new();
        let result = reg.validate_call("nope", &json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_all_schemas() {
        let mut reg = SchemaRegistry::new();
        reg.register(ToolSchemaGenerator::new("a", "A").build());
        reg.register(ToolSchemaGenerator::new("b", "B").build());

        let all = reg.all_schemas();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_registry_to_json() {
        let mut reg = SchemaRegistry::new();
        reg.register(ToolSchemaGenerator::new("a", "A").build());

        let j = reg.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
    }

    // -- ValidationError formatting -------------------------------------------

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::new("name", "is required", "present", "missing");
        let s = err.to_string();
        assert!(s.contains("name"));
        assert!(s.contains("is required"));
    }

    #[test]
    fn test_validation_error_to_json() {
        let err = ValidationError::new("age", "too low", ">= 0", "-1");
        let j = err.to_json();
        assert_eq!(j["field"], "age");
        assert_eq!(j["message"], "too low");
        assert_eq!(j["expected"], ">= 0");
        assert_eq!(j["actual"], "-1");
    }

    // -- ToolSchema validate_input convenience --------------------------------

    #[test]
    fn test_tool_schema_validate_input_ok() {
        let schema = ToolSchemaGenerator::new("test", "Test")
            .add_string_param("x", "X", true)
            .build();

        assert!(schema.validate_input(&json!({ "x": "hello" })).is_ok());
    }

    #[test]
    fn test_tool_schema_validate_input_err() {
        let schema = ToolSchemaGenerator::new("test", "Test")
            .add_string_param("x", "X", true)
            .build();

        let result = schema.validate_input(&json!({}));
        assert!(result.is_err());
        assert!(!result.unwrap_err().is_empty());
    }

    // -- Multiple validation errors at once -----------------------------------

    #[test]
    fn test_multiple_validation_errors() {
        let schema = SchemaObject::new()
            .with_property("a", PropertySchema::string())
            .with_property("b", PropertySchema::integer())
            .with_required("a")
            .with_required("b");

        let input = json!({ "a": 123, "b": "not int" });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        // Both fields have type mismatches
        assert_eq!(errs.len(), 2);
    }

    // -- Nested array with constraints ----------------------------------------

    #[test]
    fn test_nested_array_of_arrays() {
        let inner = PropertySchema::array(PropertySchema::integer());
        let schema = SchemaObject::new()
            .with_property("matrix", PropertySchema::array(inner));

        let input = json!({ "matrix": [[1, 2], [3, 4]] });
        assert!(SchemaValidator::validate(&schema, &input).is_ok());
    }

    #[test]
    fn test_nested_array_of_arrays_invalid() {
        let inner = PropertySchema::array(PropertySchema::integer());
        let schema = SchemaObject::new()
            .with_property("matrix", PropertySchema::array(inner));

        let input = json!({ "matrix": [[1, "bad"], [3, 4]] });
        let errs = SchemaValidator::validate(&schema, &input).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].field, "matrix[0][1]");
    }
}
