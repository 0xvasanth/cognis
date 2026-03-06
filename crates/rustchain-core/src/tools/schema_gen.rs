use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::base::ToolSchema;

/// JSON Schema primitive types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JsonSchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

impl JsonSchemaType {
    /// Returns the JSON Schema type string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        }
    }
}

/// Describes a single property in a JSON Schema object.
#[derive(Debug, Clone)]
pub struct PropertySchema {
    /// Property name.
    pub name: String,
    /// Human-readable description of the property.
    pub description: Option<String>,
    /// The JSON Schema type of this property.
    pub schema_type: JsonSchemaType,
    /// Whether this property is required.
    pub required: bool,
    /// Allowed enum values (for string enums).
    pub enum_values: Option<Vec<String>>,
    /// Schema for array items (only relevant when `schema_type` is `Array`).
    pub items: Option<Box<PropertySchema>>,
    /// Default value for this property.
    pub default_value: Option<Value>,
}

impl PropertySchema {
    /// Create a new property schema.
    pub fn new(name: impl Into<String>, schema_type: JsonSchemaType, required: bool) -> Self {
        Self {
            name: name.into(),
            description: None,
            schema_type,
            required,
            enum_values: None,
            items: None,
            default_value: None,
        }
    }

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

    /// Convert this property to its JSON Schema representation (without the name wrapper).
    fn to_schema_value(&self) -> Value {
        let mut schema = json!({
            "type": self.schema_type.as_str()
        });

        if let Some(ref desc) = self.description {
            schema["description"] = Value::String(desc.clone());
        }

        if let Some(ref values) = self.enum_values {
            schema["enum"] = json!(values);
        }

        if let Some(ref items) = self.items {
            schema["items"] = items.to_schema_value();
        }

        if let Some(ref default) = self.default_value {
            schema["default"] = default.clone();
        }

        schema
    }
}

/// Builder for constructing a `ToolSchema` with a well-formed JSON Schema
/// `parameters` object.
#[derive(Debug, Clone)]
pub struct ToolSchemaBuilder {
    name: String,
    description: String,
    properties: Vec<PropertySchema>,
}

impl ToolSchemaBuilder {
    /// Create a new builder.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            properties: Vec::new(),
        }
    }

    /// Add an arbitrary property.
    pub fn add_property(mut self, prop: PropertySchema) -> Self {
        self.properties.push(prop);
        self
    }

    /// Add a string property.
    pub fn add_string_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let prop = PropertySchema::new(name, JsonSchemaType::String, required)
            .with_description(description);
        self.add_property(prop)
    }

    /// Add a number property.
    pub fn add_number_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let prop = PropertySchema::new(name, JsonSchemaType::Number, required)
            .with_description(description);
        self.add_property(prop)
    }

    /// Add an integer property.
    pub fn add_integer_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let prop = PropertySchema::new(name, JsonSchemaType::Integer, required)
            .with_description(description);
        self.add_property(prop)
    }

    /// Add a boolean property.
    pub fn add_boolean_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let prop = PropertySchema::new(name, JsonSchemaType::Boolean, required)
            .with_description(description);
        self.add_property(prop)
    }

    /// Add an array property with a specified items type.
    pub fn add_array_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        items_type: JsonSchemaType,
        required: bool,
    ) -> Self {
        let items = PropertySchema::new("items", items_type, false);
        let prop = PropertySchema::new(name, JsonSchemaType::Array, required)
            .with_description(description)
            .with_items(items);
        self.add_property(prop)
    }

    /// Add a string enum property.
    pub fn add_enum_property(
        self,
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
        required: bool,
    ) -> Self {
        let prop = PropertySchema::new(name, JsonSchemaType::String, required)
            .with_description(description)
            .with_enum_values(values);
        self.add_property(prop)
    }

    /// Build the final `ToolSchema`.
    pub fn build(self) -> ToolSchema {
        let parameters = json_schema_from_properties(&self.properties);
        ToolSchema {
            name: self.name,
            description: self.description,
            parameters: Some(parameters),
            extras: None,
        }
    }
}

/// Convert a slice of `PropertySchema` into a JSON Schema object value
/// with `"type": "object"`, `"properties"`, and `"required"` fields.
pub fn json_schema_from_properties(properties: &[PropertySchema]) -> Value {
    let mut props = serde_json::Map::new();
    let mut required: Vec<Value> = Vec::new();

    for prop in properties {
        props.insert(prop.name.clone(), prop.to_schema_value());
        if prop.required {
            required.push(Value::String(prop.name.clone()));
        }
    }

    let mut schema = json!({
        "type": "object",
        "properties": Value::Object(props),
    });

    if !required.is_empty() {
        schema["required"] = Value::Array(required);
    }

    schema
}

/// Convenience function for quickly defining a `ToolSchema` from a list of
/// parameter tuples.
///
/// Each tuple is `(name, description, type, required)`.
///
/// # Example
///
/// ```ignore
/// use rustchain_core::tools::schema_gen::{quick_tool_schema, JsonSchemaType};
///
/// let schema = quick_tool_schema(
///     "search",
///     "Search for documents",
///     &[
///         ("query", "The search query", JsonSchemaType::String, true),
///         ("limit", "Max results to return", JsonSchemaType::Integer, false),
///     ],
/// );
/// ```
pub fn quick_tool_schema(
    name: impl Into<String>,
    description: impl Into<String>,
    params: &[(&str, &str, JsonSchemaType, bool)],
) -> ToolSchema {
    let properties: Vec<PropertySchema> = params
        .iter()
        .map(|(name, desc, schema_type, required)| {
            PropertySchema::new(*name, schema_type.clone(), *required)
                .with_description(*desc)
        })
        .collect();

    let parameters = json_schema_from_properties(&properties);

    ToolSchema {
        name: name.into(),
        description: description.into(),
        parameters: Some(parameters),
        extras: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_schema_with_string_property() {
        let schema = ToolSchemaBuilder::new("test", "A test tool")
            .add_string_property("name", "The user name", true)
            .build();

        assert_eq!(schema.name, "test");
        let params = schema.parameters.unwrap();
        let props = &params["properties"];
        assert_eq!(props["name"]["type"], "string");
        assert_eq!(props["name"]["description"], "The user name");
    }

    #[test]
    fn test_build_schema_with_multiple_property_types() {
        let schema = ToolSchemaBuilder::new("multi", "Multi-type tool")
            .add_string_property("name", "A name", true)
            .add_number_property("score", "A score", true)
            .add_integer_property("count", "A count", false)
            .add_boolean_property("active", "Is active", false)
            .build();

        let params = schema.parameters.unwrap();
        let props = &params["properties"];
        assert_eq!(props["name"]["type"], "string");
        assert_eq!(props["score"]["type"], "number");
        assert_eq!(props["count"]["type"], "integer");
        assert_eq!(props["active"]["type"], "boolean");
    }

    #[test]
    fn test_required_vs_optional_properties() {
        let schema = ToolSchemaBuilder::new("test", "Test")
            .add_string_property("required_field", "Required", true)
            .add_string_property("optional_field", "Optional", false)
            .build();

        let params = schema.parameters.unwrap();
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "required_field");
        // The optional field should still be in properties
        assert!(params["properties"]["optional_field"].is_object());
    }

    #[test]
    fn test_array_property_with_items() {
        let schema = ToolSchemaBuilder::new("test", "Test")
            .add_array_property("tags", "List of tags", JsonSchemaType::String, true)
            .build();

        let params = schema.parameters.unwrap();
        let tags = &params["properties"]["tags"];
        assert_eq!(tags["type"], "array");
        assert_eq!(tags["items"]["type"], "string");
    }

    #[test]
    fn test_enum_property() {
        let schema = ToolSchemaBuilder::new("test", "Test")
            .add_enum_property(
                "color",
                "Pick a color",
                vec!["red".into(), "green".into(), "blue".into()],
                true,
            )
            .build();

        let params = schema.parameters.unwrap();
        let color = &params["properties"]["color"];
        assert_eq!(color["type"], "string");
        let enum_vals = color["enum"].as_array().unwrap();
        assert_eq!(enum_vals.len(), 3);
        assert_eq!(enum_vals[0], "red");
        assert_eq!(enum_vals[1], "green");
        assert_eq!(enum_vals[2], "blue");
    }

    #[test]
    fn test_build_generates_valid_json_schema_structure() {
        let schema = ToolSchemaBuilder::new("search", "Search tool")
            .add_string_property("query", "Search query", true)
            .add_integer_property("limit", "Max results", false)
            .build();

        let params = schema.parameters.unwrap();
        // Must have "type": "object" at top level
        assert_eq!(params["type"], "object");
        // Must have "properties" object
        assert!(params["properties"].is_object());
        // Required array should contain only required fields
        let required = params["required"].as_array().unwrap();
        assert!(required.contains(&json!("query")));
        assert!(!required.contains(&json!("limit")));
    }

    #[test]
    fn test_quick_tool_schema() {
        let schema = quick_tool_schema(
            "search",
            "Search for documents",
            &[
                ("query", "The search query", JsonSchemaType::String, true),
                ("limit", "Max results", JsonSchemaType::Integer, false),
            ],
        );

        assert_eq!(schema.name, "search");
        assert_eq!(schema.description, "Search for documents");
        let params = schema.parameters.unwrap();
        assert_eq!(params["properties"]["query"]["type"], "string");
        assert_eq!(params["properties"]["limit"]["type"], "integer");
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "query");
    }

    #[test]
    fn test_generated_schema_has_object_type_wrapper() {
        let schema = ToolSchemaBuilder::new("tool", "A tool")
            .add_string_property("input", "The input", true)
            .build();

        let params = schema.parameters.unwrap();
        assert_eq!(params["type"], "object");
        assert!(params["properties"].is_object());
    }

    #[test]
    fn test_property_with_default_value() {
        let prop = PropertySchema::new("limit", JsonSchemaType::Integer, false)
            .with_description("Max results")
            .with_default(json!(10));

        let schema = ToolSchemaBuilder::new("test", "Test")
            .add_property(prop)
            .build();

        let params = schema.parameters.unwrap();
        assert_eq!(params["properties"]["limit"]["default"], 10);
    }

    #[test]
    fn test_no_required_array_when_all_optional() {
        let schema = ToolSchemaBuilder::new("test", "Test")
            .add_string_property("a", "Optional a", false)
            .add_string_property("b", "Optional b", false)
            .build();

        let params = schema.parameters.unwrap();
        assert_eq!(params["type"], "object");
        // "required" key should not be present
        assert!(params.get("required").is_none());
    }
}
