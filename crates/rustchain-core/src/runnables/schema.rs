use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Data payload for a standard stream event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventData {
    /// The input to the runnable, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// The output of the runnable, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// A streaming chunk, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<Value>,
    /// An error message, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A stream event emitted during runnable execution.
///
/// Events follow the LangChain v2 streaming event protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamEvent {
    /// A standard event with structured `EventData`.
    Standard {
        event: String,
        name: String,
        run_id: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: HashMap<String, Value>,
        #[serde(default)]
        parent_ids: Vec<String>,
        data: EventData,
    },
    /// A custom event with arbitrary JSON data.
    Custom {
        event: String,
        name: String,
        run_id: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: HashMap<String, Value>,
        #[serde(default)]
        parent_ids: Vec<String>,
        data: Value,
    },
}

// ---------------------------------------------------------------------------
// Schema validation types for runnable pipelines
// ---------------------------------------------------------------------------

/// Represents a JSON type constraint for schema validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaType {
    /// JSON string value.
    String,
    /// JSON number value (float or integer).
    Number,
    /// JSON integer value.
    Integer,
    /// JSON boolean value.
    Boolean,
    /// JSON array where each element matches the inner type.
    Array(Box<SchemaType>),
    /// JSON object (any key/value structure).
    Object,
    /// Matches any JSON value.
    Any,
    /// Matches null or the inner type.
    Nullable(Box<SchemaType>),
}

impl SchemaType {
    /// Returns `true` if `value` matches this schema type.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            SchemaType::String => value.is_string(),
            SchemaType::Number => value.is_number(),
            SchemaType::Integer => value.is_i64() || value.is_u64(),
            SchemaType::Boolean => value.is_boolean(),
            SchemaType::Array(inner) => {
                if let Some(arr) = value.as_array() {
                    arr.iter().all(|item| inner.matches(item))
                } else {
                    false
                }
            }
            SchemaType::Object => value.is_object(),
            SchemaType::Any => true,
            SchemaType::Nullable(inner) => value.is_null() || inner.matches(value),
        }
    }

    /// Returns a human-readable name for this type.
    pub fn type_name(&self) -> String {
        match self {
            SchemaType::String => "string".to_string(),
            SchemaType::Number => "number".to_string(),
            SchemaType::Integer => "integer".to_string(),
            SchemaType::Boolean => "boolean".to_string(),
            SchemaType::Array(inner) => format!("array<{}>", inner.type_name()),
            SchemaType::Object => "object".to_string(),
            SchemaType::Any => "any".to_string(),
            SchemaType::Nullable(inner) => format!("nullable<{}>", inner.type_name()),
        }
    }

    /// Converts this type to a JSON Schema type representation.
    fn to_json_schema_type(&self) -> Value {
        match self {
            SchemaType::String => json!({"type": "string"}),
            SchemaType::Number => json!({"type": "number"}),
            SchemaType::Integer => json!({"type": "integer"}),
            SchemaType::Boolean => json!({"type": "boolean"}),
            SchemaType::Array(inner) => {
                json!({"type": "array", "items": inner.to_json_schema_type()})
            }
            SchemaType::Object => json!({"type": "object"}),
            SchemaType::Any => json!({}),
            SchemaType::Nullable(inner) => {
                let mut schema = inner.to_json_schema_type();
                if let Some(obj) = schema.as_object_mut() {
                    if let Some(t) = obj.get("type").cloned() {
                        obj.insert("type".to_string(), json!([t, "null"]));
                    } else {
                        obj.insert("type".to_string(), json!("null"));
                    }
                }
                schema
            }
        }
    }
}

/// A single field definition within a [`RunnableSchema`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaField {
    /// The name of the field.
    pub name: String,
    /// The expected type of the field.
    pub field_type: SchemaType,
    /// Whether this field must be present.
    pub required: bool,
    /// An optional human-readable description.
    pub description: Option<String>,
    /// An optional default value applied when the field is missing.
    pub default: Option<Value>,
}

impl SchemaField {
    /// Creates a new required field with the given name and type.
    pub fn new(name: impl Into<String>, field_type: SchemaType) -> Self {
        Self {
            name: name.into(),
            field_type,
            required: true,
            description: None,
            default: None,
        }
    }

    /// Mark this field as required (the default).
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Mark this field as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Attach a description to this field.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set a default value for this field.
    pub fn with_default(mut self, val: Value) -> Self {
        self.default = Some(val);
        self
    }
}

/// Describes the expected structure of a JSON value used as input or output
/// for a runnable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnableSchema {
    fields: Vec<SchemaField>,
}

impl RunnableSchema {
    /// Creates an empty schema.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Adds a pre-built [`SchemaField`] to this schema.
    pub fn add_field(mut self, field: SchemaField) -> Self {
        self.fields.push(field);
        self
    }

    /// Shorthand builder: adds a required field with the given name and type.
    pub fn field(self, name: impl Into<String>, field_type: SchemaType) -> Self {
        self.add_field(SchemaField::new(name, field_type))
    }

    /// Returns a reference to the field with the given name, if it exists.
    pub fn get_field(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Returns all required fields.
    pub fn required_fields(&self) -> Vec<&SchemaField> {
        self.fields.iter().filter(|f| f.required).collect()
    }

    /// Returns all optional fields.
    pub fn optional_fields(&self) -> Vec<&SchemaField> {
        self.fields.iter().filter(|f| !f.required).collect()
    }

    /// Validates a JSON value against this schema.
    pub fn validate(&self, value: &Value) -> SchemaValidationResult {
        let mut errors = Vec::new();

        let obj = match value.as_object() {
            Some(o) => o,
            None => {
                if self.fields.is_empty() {
                    return SchemaValidationResult {
                        valid: true,
                        errors: vec![],
                    };
                }
                errors.push(SchemaError {
                    field: "<root>".to_string(),
                    expected: "object".to_string(),
                    actual: value_type_name(value),
                    message: format!(
                        "Expected an object but got {}",
                        value_type_name(value)
                    ),
                });
                return SchemaValidationResult {
                    valid: false,
                    errors,
                };
            }
        };

        for field in &self.fields {
            match obj.get(&field.name) {
                None => {
                    if field.required && field.default.is_none() {
                        errors.push(SchemaError {
                            field: field.name.clone(),
                            expected: field.field_type.type_name(),
                            actual: "missing".to_string(),
                            message: format!(
                                "Required field '{}' is missing",
                                field.name
                            ),
                        });
                    }
                }
                Some(val) => {
                    if !field.field_type.matches(val) {
                        errors.push(SchemaError {
                            field: field.name.clone(),
                            expected: field.field_type.type_name(),
                            actual: value_type_name(val),
                            message: format!(
                                "Field '{}' expected type '{}' but got '{}'",
                                field.name,
                                field.field_type.type_name(),
                                value_type_name(val),
                            ),
                        });
                    }
                }
            }
        }

        SchemaValidationResult {
            valid: errors.is_empty(),
            errors,
        }
    }

    /// Fills missing optional fields with their default values in-place.
    pub fn apply_defaults(&self, value: &mut Value) {
        if let Some(obj) = value.as_object_mut() {
            for field in &self.fields {
                if !obj.contains_key(&field.name) {
                    if let Some(default) = &field.default {
                        obj.insert(field.name.clone(), default.clone());
                    }
                }
            }
        }
    }

    /// Converts this schema to a JSON Schema (draft-compatible) representation.
    pub fn to_json_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for field in &self.fields {
            let mut field_schema = field.field_type.to_json_schema_type();
            if let Some(desc) = &field.description {
                if let Some(obj) = field_schema.as_object_mut() {
                    obj.insert("description".to_string(), json!(desc));
                }
            }
            if let Some(default) = &field.default {
                if let Some(obj) = field_schema.as_object_mut() {
                    obj.insert("default".to_string(), default.clone());
                }
            }
            properties.insert(field.name.clone(), field_schema);
            if field.required {
                required.push(json!(field.name));
            }
        }

        let mut schema = json!({
            "type": "object",
            "properties": properties,
        });
        if !required.is_empty() {
            schema
                .as_object_mut()
                .unwrap()
                .insert("required".to_string(), json!(required));
        }
        schema
    }

    /// Returns the number of fields in this schema.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Returns true if the schema has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Result of a schema validation.
#[derive(Debug, Clone)]
pub struct SchemaValidationResult {
    /// Whether the value is valid.
    pub valid: bool,
    /// All validation errors found.
    pub errors: Vec<SchemaError>,
}

impl SchemaValidationResult {
    /// Returns `true` if validation passed.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Returns a slice of the errors.
    pub fn errors(&self) -> &[SchemaError] {
        &self.errors
    }

    /// Returns human-readable error messages.
    pub fn error_messages(&self) -> Vec<String> {
        self.errors.iter().map(|e| e.message.clone()).collect()
    }
}

/// A single schema validation error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaError {
    /// The field that failed validation.
    pub field: String,
    /// The expected type or constraint.
    pub expected: String,
    /// The actual type or value encountered.
    pub actual: String,
    /// A human-readable error message.
    pub message: String,
}

impl SchemaError {
    /// Serialises this error to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        })
    }
}

/// Pairs an input schema and an output schema into a single contract
/// for a runnable.
#[derive(Debug, Clone)]
pub struct SchemaContract {
    input: RunnableSchema,
    output: RunnableSchema,
}

impl SchemaContract {
    /// Creates a new contract with the given input and output schemas.
    pub fn new(input: RunnableSchema, output: RunnableSchema) -> Self {
        Self { input, output }
    }

    /// Validates a value against the input schema.
    pub fn validate_input(&self, value: &Value) -> SchemaValidationResult {
        self.input.validate(value)
    }

    /// Validates a value against the output schema.
    pub fn validate_output(&self, value: &Value) -> SchemaValidationResult {
        self.output.validate(value)
    }

    /// Returns a reference to the input schema.
    pub fn input_schema(&self) -> &RunnableSchema {
        &self.input
    }

    /// Returns a reference to the output schema.
    pub fn output_schema(&self) -> &RunnableSchema {
        &self.output
    }

    /// Serialises the full contract to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "input": self.input.to_json_schema(),
            "output": self.output.to_json_schema(),
        })
    }
}

/// Utility for inferring a [`RunnableSchema`] from sample JSON data.
pub struct SchemaInference;

impl SchemaInference {
    /// Infers a schema from a single JSON value.
    ///
    /// All top-level keys in the object become required fields.
    pub fn infer(value: &Value) -> RunnableSchema {
        let mut schema = RunnableSchema::new();
        if let Some(obj) = value.as_object() {
            for (key, val) in obj {
                schema = schema.add_field(
                    SchemaField::new(key.clone(), Self::infer_type(val)).required(),
                );
            }
        }
        schema
    }

    /// Infers a merged schema from multiple sample values.
    ///
    /// A field is marked required only if it appears in every sample.
    /// Fields that appear in some but not all samples are optional.
    pub fn infer_from_samples(values: &[Value]) -> RunnableSchema {
        if values.is_empty() {
            return RunnableSchema::new();
        }
        if values.len() == 1 {
            return Self::infer(&values[0]);
        }

        // Collect all field names and their types across samples.
        let mut field_types: HashMap<String, SchemaType> = HashMap::new();
        let mut field_counts: HashMap<String, usize> = HashMap::new();
        let total = values.len();

        for value in values {
            if let Some(obj) = value.as_object() {
                let mut seen = HashSet::new();
                for (key, val) in obj {
                    seen.insert(key.clone());
                    let inferred = Self::infer_type(val);
                    field_types
                        .entry(key.clone())
                        .and_modify(|existing| {
                            *existing = Self::merge_types(existing, &inferred);
                        })
                        .or_insert(inferred);
                }
                for key in seen {
                    *field_counts.entry(key).or_insert(0) += 1;
                }
            }
        }

        let mut schema = RunnableSchema::new();
        for (name, schema_type) in field_types {
            let count = field_counts.get(&name).copied().unwrap_or(0);
            let mut field = SchemaField::new(name, schema_type);
            if count < total {
                field = field.optional();
            }
            schema = schema.add_field(field);
        }
        schema
    }

    /// Infers a [`SchemaType`] from a single JSON value.
    fn infer_type(value: &Value) -> SchemaType {
        match value {
            Value::Null => SchemaType::Nullable(Box::new(SchemaType::Any)),
            Value::Bool(_) => SchemaType::Boolean,
            Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    SchemaType::Integer
                } else {
                    SchemaType::Number
                }
            }
            Value::String(_) => SchemaType::String,
            Value::Array(arr) => {
                if arr.is_empty() {
                    SchemaType::Array(Box::new(SchemaType::Any))
                } else {
                    let mut elem_type = Self::infer_type(&arr[0]);
                    for item in &arr[1..] {
                        elem_type = Self::merge_types(&elem_type, &Self::infer_type(item));
                    }
                    SchemaType::Array(Box::new(elem_type))
                }
            }
            Value::Object(_) => SchemaType::Object,
        }
    }

    /// Merges two schema types into the most general compatible type.
    fn merge_types(a: &SchemaType, b: &SchemaType) -> SchemaType {
        if a == b {
            return a.clone();
        }
        // Integer is a subset of Number.
        match (a, b) {
            (SchemaType::Integer, SchemaType::Number)
            | (SchemaType::Number, SchemaType::Integer) => SchemaType::Number,
            (SchemaType::Nullable(inner_a), SchemaType::Nullable(inner_b)) => {
                SchemaType::Nullable(Box::new(Self::merge_types(inner_a, inner_b)))
            }
            (SchemaType::Nullable(inner), other) | (other, SchemaType::Nullable(inner)) => {
                SchemaType::Nullable(Box::new(Self::merge_types(inner, other)))
            }
            _ => SchemaType::Any,
        }
    }
}

/// Returns a human-readable type name for a JSON value.
fn value_type_name(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(_) => "boolean".to_string(),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer".to_string()
            } else {
                "number".to_string()
            }
        }
        Value::String(_) => "string".to_string(),
        Value::Array(_) => "array".to_string(),
        Value::Object(_) => "object".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // SchemaType::matches
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_type_matches_string() {
        assert!(SchemaType::String.matches(&json!("hello")));
        assert!(!SchemaType::String.matches(&json!(42)));
    }

    #[test]
    fn test_schema_type_matches_number() {
        assert!(SchemaType::Number.matches(&json!(3.14)));
        assert!(SchemaType::Number.matches(&json!(42)));
        assert!(!SchemaType::Number.matches(&json!("nope")));
    }

    #[test]
    fn test_schema_type_matches_integer() {
        assert!(SchemaType::Integer.matches(&json!(42)));
        assert!(!SchemaType::Integer.matches(&json!(3.14)));
        assert!(!SchemaType::Integer.matches(&json!("nope")));
    }

    #[test]
    fn test_schema_type_matches_boolean() {
        assert!(SchemaType::Boolean.matches(&json!(true)));
        assert!(SchemaType::Boolean.matches(&json!(false)));
        assert!(!SchemaType::Boolean.matches(&json!(1)));
    }

    #[test]
    fn test_schema_type_matches_array() {
        let t = SchemaType::Array(Box::new(SchemaType::Integer));
        assert!(t.matches(&json!([1, 2, 3])));
        assert!(!t.matches(&json!([1, "two", 3])));
        assert!(!t.matches(&json!("not array")));
    }

    #[test]
    fn test_schema_type_matches_empty_array() {
        let t = SchemaType::Array(Box::new(SchemaType::String));
        assert!(t.matches(&json!([])));
    }

    #[test]
    fn test_schema_type_matches_nested_array() {
        let t = SchemaType::Array(Box::new(SchemaType::Array(Box::new(SchemaType::Integer))));
        assert!(t.matches(&json!([[1, 2], [3]])));
        assert!(!t.matches(&json!([[1, "x"]])));
    }

    #[test]
    fn test_schema_type_matches_object() {
        assert!(SchemaType::Object.matches(&json!({"a": 1})));
        assert!(SchemaType::Object.matches(&json!({})));
        assert!(!SchemaType::Object.matches(&json!("nope")));
    }

    #[test]
    fn test_schema_type_matches_any() {
        assert!(SchemaType::Any.matches(&json!(null)));
        assert!(SchemaType::Any.matches(&json!(42)));
        assert!(SchemaType::Any.matches(&json!("hi")));
        assert!(SchemaType::Any.matches(&json!({"x": 1})));
    }

    #[test]
    fn test_schema_type_matches_nullable() {
        let t = SchemaType::Nullable(Box::new(SchemaType::String));
        assert!(t.matches(&json!(null)));
        assert!(t.matches(&json!("hello")));
        assert!(!t.matches(&json!(42)));
    }

    #[test]
    fn test_schema_type_matches_nullable_nested() {
        let t = SchemaType::Nullable(Box::new(SchemaType::Array(Box::new(SchemaType::Integer))));
        assert!(t.matches(&json!(null)));
        assert!(t.matches(&json!([1, 2])));
        assert!(!t.matches(&json!([1, "x"])));
    }

    // -----------------------------------------------------------------------
    // SchemaType::type_name
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_type_names() {
        assert_eq!(SchemaType::String.type_name(), "string");
        assert_eq!(SchemaType::Number.type_name(), "number");
        assert_eq!(SchemaType::Integer.type_name(), "integer");
        assert_eq!(SchemaType::Boolean.type_name(), "boolean");
        assert_eq!(
            SchemaType::Array(Box::new(SchemaType::String)).type_name(),
            "array<string>"
        );
        assert_eq!(SchemaType::Object.type_name(), "object");
        assert_eq!(SchemaType::Any.type_name(), "any");
        assert_eq!(
            SchemaType::Nullable(Box::new(SchemaType::Integer)).type_name(),
            "nullable<integer>"
        );
    }

    // -----------------------------------------------------------------------
    // SchemaField builder
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_field_new_defaults() {
        let f = SchemaField::new("age", SchemaType::Integer);
        assert_eq!(f.name, "age");
        assert!(f.required);
        assert!(f.description.is_none());
        assert!(f.default.is_none());
    }

    #[test]
    fn test_schema_field_builder_chain() {
        let f = SchemaField::new("name", SchemaType::String)
            .optional()
            .with_description("The user name")
            .with_default(json!("anonymous"));

        assert!(!f.required);
        assert_eq!(f.description.as_deref(), Some("The user name"));
        assert_eq!(f.default, Some(json!("anonymous")));
    }

    #[test]
    fn test_schema_field_required_toggle() {
        let f = SchemaField::new("x", SchemaType::Any).optional().required();
        assert!(f.required);
    }

    // -----------------------------------------------------------------------
    // RunnableSchema validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_valid_object() {
        let schema = RunnableSchema::new()
            .field("name", SchemaType::String)
            .field("age", SchemaType::Integer);

        let result = schema.validate(&json!({"name": "Alice", "age": 30}));
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
    }

    #[test]
    fn test_schema_missing_required_field() {
        let schema = RunnableSchema::new()
            .field("name", SchemaType::String)
            .field("age", SchemaType::Integer);

        let result = schema.validate(&json!({"name": "Alice"}));
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
        assert!(result.error_messages()[0].contains("age"));
    }

    #[test]
    fn test_schema_wrong_type() {
        let schema = RunnableSchema::new().field("count", SchemaType::Integer);

        let result = schema.validate(&json!({"count": "not a number"}));
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
        assert!(result.error_messages()[0].contains("count"));
    }

    #[test]
    fn test_schema_optional_field_missing_is_ok() {
        let schema = RunnableSchema::new()
            .add_field(SchemaField::new("name", SchemaType::String))
            .add_field(SchemaField::new("bio", SchemaType::String).optional());

        let result = schema.validate(&json!({"name": "Bob"}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_schema_non_object_value_with_fields() {
        let schema = RunnableSchema::new().field("x", SchemaType::Integer);
        let result = schema.validate(&json!(42));
        assert!(!result.is_valid());
        assert!(result.error_messages()[0].contains("object"));
    }

    #[test]
    fn test_schema_empty_validates_anything() {
        let schema = RunnableSchema::new();
        assert!(schema.validate(&json!(42)).is_valid());
        assert!(schema.validate(&json!(null)).is_valid());
        assert!(schema.validate(&json!({"a": 1})).is_valid());
    }

    // -----------------------------------------------------------------------
    // apply_defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_defaults_fills_missing() {
        let schema = RunnableSchema::new()
            .add_field(
                SchemaField::new("color", SchemaType::String)
                    .optional()
                    .with_default(json!("red")),
            )
            .add_field(SchemaField::new("size", SchemaType::Integer));

        let mut val = json!({"size": 10});
        schema.apply_defaults(&mut val);
        assert_eq!(val, json!({"size": 10, "color": "red"}));
    }

    #[test]
    fn test_apply_defaults_does_not_overwrite_existing() {
        let schema = RunnableSchema::new().add_field(
            SchemaField::new("color", SchemaType::String)
                .optional()
                .with_default(json!("red")),
        );

        let mut val = json!({"color": "blue"});
        schema.apply_defaults(&mut val);
        assert_eq!(val["color"], json!("blue"));
    }

    #[test]
    fn test_apply_defaults_on_non_object_is_noop() {
        let schema = RunnableSchema::new().add_field(
            SchemaField::new("x", SchemaType::Integer)
                .optional()
                .with_default(json!(0)),
        );

        let mut val = json!(42);
        schema.apply_defaults(&mut val);
        assert_eq!(val, json!(42));
    }

    // -----------------------------------------------------------------------
    // to_json_schema
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_json_schema_basic() {
        let schema = RunnableSchema::new()
            .add_field(
                SchemaField::new("name", SchemaType::String)
                    .with_description("User name"),
            )
            .add_field(SchemaField::new("scores", SchemaType::Array(Box::new(SchemaType::Integer))));

        let js = schema.to_json_schema();
        assert_eq!(js["type"], "object");
        assert_eq!(js["properties"]["name"]["type"], "string");
        assert_eq!(js["properties"]["name"]["description"], "User name");
        assert_eq!(js["properties"]["scores"]["type"], "array");
        assert_eq!(js["properties"]["scores"]["items"]["type"], "integer");
        assert!(js["required"].as_array().unwrap().contains(&json!("name")));
        assert!(js["required"].as_array().unwrap().contains(&json!("scores")));
    }

    #[test]
    fn test_to_json_schema_no_required() {
        let schema = RunnableSchema::new()
            .add_field(SchemaField::new("opt", SchemaType::String).optional());

        let js = schema.to_json_schema();
        assert!(js.get("required").is_none());
    }

    #[test]
    fn test_to_json_schema_nullable() {
        let schema = RunnableSchema::new().add_field(SchemaField::new(
            "maybe",
            SchemaType::Nullable(Box::new(SchemaType::String)),
        ));

        let js = schema.to_json_schema();
        let maybe_type = &js["properties"]["maybe"]["type"];
        let arr = maybe_type.as_array().unwrap();
        assert!(arr.contains(&json!("string")));
        assert!(arr.contains(&json!("null")));
    }

    #[test]
    fn test_to_json_schema_with_default() {
        let schema = RunnableSchema::new().add_field(
            SchemaField::new("level", SchemaType::Integer)
                .optional()
                .with_default(json!(1)),
        );

        let js = schema.to_json_schema();
        assert_eq!(js["properties"]["level"]["default"], json!(1));
    }

    // -----------------------------------------------------------------------
    // SchemaContract
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_contract_validate_input() {
        let contract = SchemaContract::new(
            RunnableSchema::new().field("prompt", SchemaType::String),
            RunnableSchema::new().field("response", SchemaType::String),
        );

        let result = contract.validate_input(&json!({"prompt": "hello"}));
        assert!(result.is_valid());

        let result = contract.validate_input(&json!({"prompt": 42}));
        assert!(!result.is_valid());
    }

    #[test]
    fn test_schema_contract_validate_output() {
        let contract = SchemaContract::new(
            RunnableSchema::new().field("prompt", SchemaType::String),
            RunnableSchema::new().field("response", SchemaType::String),
        );

        let result = contract.validate_output(&json!({"response": "hi"}));
        assert!(result.is_valid());

        let result = contract.validate_output(&json!({}));
        assert!(!result.is_valid());
    }

    #[test]
    fn test_schema_contract_to_json() {
        let contract = SchemaContract::new(
            RunnableSchema::new().field("q", SchemaType::String),
            RunnableSchema::new().field("a", SchemaType::String),
        );

        let j = contract.to_json();
        assert!(j["input"]["properties"]["q"].is_object());
        assert!(j["output"]["properties"]["a"].is_object());
    }

    #[test]
    fn test_schema_contract_accessors() {
        let contract = SchemaContract::new(
            RunnableSchema::new().field("in1", SchemaType::String),
            RunnableSchema::new().field("out1", SchemaType::Integer),
        );

        assert!(contract.input_schema().get_field("in1").is_some());
        assert!(contract.output_schema().get_field("out1").is_some());
    }

    // -----------------------------------------------------------------------
    // SchemaError
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_error_to_json() {
        let err = SchemaError {
            field: "age".to_string(),
            expected: "integer".to_string(),
            actual: "string".to_string(),
            message: "wrong type".to_string(),
        };
        let j = err.to_json();
        assert_eq!(j["field"], "age");
        assert_eq!(j["expected"], "integer");
        assert_eq!(j["actual"], "string");
        assert_eq!(j["message"], "wrong type");
    }

    // -----------------------------------------------------------------------
    // SchemaInference
    // -----------------------------------------------------------------------

    #[test]
    fn test_infer_from_single_value() {
        let schema = SchemaInference::infer(&json!({
            "name": "Alice",
            "age": 30,
            "active": true
        }));

        assert_eq!(schema.len(), 3);
        assert!(schema.get_field("name").is_some());
        assert!(schema.get_field("age").is_some());
        assert!(schema.get_field("active").is_some());

        // All fields should be required
        assert_eq!(schema.required_fields().len(), 3);
    }

    #[test]
    fn test_infer_type_detection() {
        let schema = SchemaInference::infer(&json!({
            "s": "hello",
            "i": 42,
            "f": 3.14,
            "b": false,
            "a": [1, 2],
            "o": {"nested": true},
            "n": null
        }));

        assert_eq!(
            schema.get_field("s").unwrap().field_type,
            SchemaType::String
        );
        assert_eq!(
            schema.get_field("i").unwrap().field_type,
            SchemaType::Integer
        );
        assert_eq!(
            schema.get_field("f").unwrap().field_type,
            SchemaType::Number
        );
        assert_eq!(
            schema.get_field("b").unwrap().field_type,
            SchemaType::Boolean
        );
        assert_eq!(
            schema.get_field("a").unwrap().field_type,
            SchemaType::Array(Box::new(SchemaType::Integer))
        );
        assert_eq!(
            schema.get_field("o").unwrap().field_type,
            SchemaType::Object
        );
        assert_eq!(
            schema.get_field("n").unwrap().field_type,
            SchemaType::Nullable(Box::new(SchemaType::Any))
        );
    }

    #[test]
    fn test_infer_from_multiple_samples_required_vs_optional() {
        let samples = vec![
            json!({"name": "Alice", "age": 30}),
            json!({"name": "Bob"}),
            json!({"name": "Carol", "age": 25}),
        ];

        let schema = SchemaInference::infer_from_samples(&samples);

        // "name" appears in all 3 -> required
        let name_field = schema.get_field("name").unwrap();
        assert!(name_field.required);

        // "age" appears in 2/3 -> optional
        let age_field = schema.get_field("age").unwrap();
        assert!(!age_field.required);
    }

    #[test]
    fn test_infer_from_samples_type_merging() {
        let samples = vec![
            json!({"val": 42}),
            json!({"val": 3.14}),
        ];

        let schema = SchemaInference::infer_from_samples(&samples);
        // Integer + Number -> Number
        assert_eq!(
            schema.get_field("val").unwrap().field_type,
            SchemaType::Number
        );
    }

    #[test]
    fn test_infer_from_empty_samples() {
        let schema = SchemaInference::infer_from_samples(&[]);
        assert!(schema.is_empty());
    }

    #[test]
    fn test_infer_from_single_sample() {
        let schema = SchemaInference::infer_from_samples(&[json!({"x": 1})]);
        assert_eq!(schema.len(), 1);
        assert!(schema.get_field("x").unwrap().required);
    }

    #[test]
    fn test_infer_non_object_value() {
        let schema = SchemaInference::infer(&json!(42));
        assert!(schema.is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_all_optional_fields() {
        let schema = RunnableSchema::new()
            .add_field(SchemaField::new("a", SchemaType::String).optional())
            .add_field(SchemaField::new("b", SchemaType::Integer).optional());

        let result = schema.validate(&json!({}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_schema_extra_fields_are_allowed() {
        let schema = RunnableSchema::new().field("name", SchemaType::String);
        let result = schema.validate(&json!({"name": "Alice", "extra": 123}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_schema_required_with_default_allows_missing() {
        let schema = RunnableSchema::new().add_field(
            SchemaField::new("x", SchemaType::Integer).with_default(json!(0)),
        );

        // Field is required but has a default, so missing is OK
        let result = schema.validate(&json!({}));
        assert!(result.is_valid());
    }

    #[test]
    fn test_schema_multiple_errors() {
        let schema = RunnableSchema::new()
            .field("a", SchemaType::String)
            .field("b", SchemaType::Integer)
            .field("c", SchemaType::Boolean);

        let result = schema.validate(&json!({}));
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 3);
    }

    #[test]
    fn test_runnable_schema_len_and_is_empty() {
        let empty = RunnableSchema::new();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = RunnableSchema::new().field("x", SchemaType::Any);
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 1);
    }

    #[test]
    fn test_validation_result_error_messages() {
        let schema = RunnableSchema::new()
            .field("a", SchemaType::String)
            .field("b", SchemaType::Integer);

        let result = schema.validate(&json!({"a": 1, "b": "x"}));
        assert!(!result.is_valid());
        let msgs = result.error_messages();
        assert_eq!(msgs.len(), 2);
        assert!(msgs.iter().any(|m| m.contains("a")));
        assert!(msgs.iter().any(|m| m.contains("b")));
    }
}
