use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// ParamType
// ---------------------------------------------------------------------------

/// Describes the expected type of a configuration parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamType {
    /// A JSON string value.
    String,
    /// A JSON integer value.
    Integer,
    /// A JSON floating-point value.
    Float,
    /// A JSON boolean value.
    Boolean,
    /// A JSON array value.
    Array,
    /// A JSON object value.
    Object,
    /// Matches any JSON value.
    Any,
}

impl ParamType {
    /// Returns a human-readable name for this parameter type.
    pub fn type_name(&self) -> &str {
        match self {
            ParamType::String => "string",
            ParamType::Integer => "integer",
            ParamType::Float => "float",
            ParamType::Boolean => "boolean",
            ParamType::Array => "array",
            ParamType::Object => "object",
            ParamType::Any => "any",
        }
    }

    /// Returns `true` if the given JSON value matches this parameter type.
    pub fn validate_value(&self, value: &Value) -> bool {
        match self {
            ParamType::String => value.is_string(),
            ParamType::Integer => value.is_i64() || value.is_u64(),
            ParamType::Float => value.is_f64(),
            ParamType::Boolean => value.is_boolean(),
            ParamType::Array => value.is_array(),
            ParamType::Object => value.is_object(),
            ParamType::Any => true,
        }
    }
}

// ---------------------------------------------------------------------------
// ParamSpec
// ---------------------------------------------------------------------------

/// Specification for a single configuration parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    /// The name of the parameter.
    pub name: String,
    /// The expected type of the parameter.
    pub param_type: ParamType,
    /// A human-readable description of the parameter.
    pub description: String,
    /// An optional default value for the parameter.
    pub default_value: Option<Value>,
}

impl ParamSpec {
    /// Creates a new `ParamSpec` with the given name and type.
    pub fn new(name: impl Into<String>, param_type: ParamType) -> Self {
        Self {
            name: name.into(),
            param_type,
            description: String::new(),
            default_value: None,
        }
    }

    /// Sets the description for this parameter spec.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the default value for this parameter spec.
    pub fn with_default(mut self, value: Value) -> Self {
        self.default_value = Some(value);
        self
    }
}

// ---------------------------------------------------------------------------
// SerializableConfig
// ---------------------------------------------------------------------------

/// A serializable configuration for a runnable, containing named parameters,
/// version information, and arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableConfig {
    /// The name of this configuration.
    pub name: String,
    /// The type name of the runnable this configuration applies to.
    pub type_name: String,
    /// The parameter values.
    pub parameters: HashMap<String, Value>,
    /// The version string for this configuration.
    pub version: String,
    /// Arbitrary metadata associated with this configuration.
    pub metadata: HashMap<String, Value>,
}

impl SerializableConfig {
    /// Creates a new builder for `SerializableConfig`.
    pub fn builder() -> SerializableConfigBuilder {
        SerializableConfigBuilder::default()
    }

    /// Serializes this config to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "type_name": self.type_name,
            "parameters": self.parameters,
            "version": self.version,
            "metadata": self.metadata,
        })
    }

    /// Deserializes a `SerializableConfig` from a JSON `Value`.
    pub fn from_json(value: &Value) -> Result<Self> {
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RustChainError::Other("missing or invalid 'name' field".into()))?
            .to_string();

        let type_name = value
            .get("type_name")
            .and_then(Value::as_str)
            .ok_or_else(|| RustChainError::Other("missing or invalid 'type_name' field".into()))?
            .to_string();

        let version = value
            .get("version")
            .and_then(Value::as_str)
            .ok_or_else(|| RustChainError::Other("missing or invalid 'version' field".into()))?
            .to_string();

        let parameters = value
            .get("parameters")
            .and_then(Value::as_object)
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<String, Value>>()
            })
            .unwrap_or_default();

        let metadata = value
            .get("metadata")
            .and_then(Value::as_object)
            .map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<HashMap<String, Value>>()
            })
            .unwrap_or_default();

        Ok(Self {
            name,
            type_name,
            parameters,
            version,
            metadata,
        })
    }

    /// Serializes this config to a compact JSON string.
    pub fn to_compact_json(&self) -> String {
        serde_json::to_string(&self.to_json()).unwrap_or_default()
    }
}

/// Builder for [`SerializableConfig`].
#[derive(Debug, Default)]
pub struct SerializableConfigBuilder {
    name: String,
    type_name: String,
    parameters: HashMap<String, Value>,
    version: String,
    metadata: HashMap<String, Value>,
}

impl SerializableConfigBuilder {
    /// Sets the name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the type name.
    pub fn type_name(mut self, type_name: impl Into<String>) -> Self {
        self.type_name = type_name.into();
        self
    }

    /// Sets all parameters at once.
    pub fn parameters(mut self, parameters: HashMap<String, Value>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Adds a single parameter.
    pub fn parameter(mut self, key: impl Into<String>, value: Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Sets the version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Sets all metadata at once.
    pub fn metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Adds a single metadata entry.
    pub fn meta(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Builds the [`SerializableConfig`].
    pub fn build(self) -> SerializableConfig {
        SerializableConfig {
            name: self.name,
            type_name: self.type_name,
            parameters: self.parameters,
            version: self.version,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigSchema
// ---------------------------------------------------------------------------

/// Describes the expected structure of a [`SerializableConfig`]'s parameters,
/// including required and optional parameter specifications.
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    /// Parameter specifications that must be present.
    pub required_params: Vec<ParamSpec>,
    /// Parameter specifications that may be present.
    pub optional_params: Vec<ParamSpec>,
}

impl ConfigSchema {
    /// Creates a new empty `ConfigSchema`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a required parameter specification.
    pub fn add_required(mut self, spec: ParamSpec) -> Self {
        self.required_params.push(spec);
        self
    }

    /// Adds an optional parameter specification.
    pub fn add_optional(mut self, spec: ParamSpec) -> Self {
        self.optional_params.push(spec);
        self
    }

    /// Validates a [`SerializableConfig`] against this schema.
    ///
    /// Returns `Ok(())` if valid, or `Err` with a list of validation error messages.
    pub fn validate(&self, config: &SerializableConfig) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check required parameters are present and correctly typed.
        for spec in &self.required_params {
            match config.parameters.get(&spec.name) {
                None => {
                    errors.push(format!("missing required parameter '{}'", spec.name));
                }
                Some(value) => {
                    if !spec.param_type.validate_value(value) {
                        errors.push(format!(
                            "parameter '{}' expected type '{}' but got incompatible value",
                            spec.name,
                            spec.param_type.type_name(),
                        ));
                    }
                }
            }
        }

        // Check optional parameters that are present have correct types.
        for spec in &self.optional_params {
            if let Some(value) = config.parameters.get(&spec.name) {
                if !spec.param_type.validate_value(value) {
                    errors.push(format!(
                        "parameter '{}' expected type '{}' but got incompatible value",
                        spec.name,
                        spec.param_type.type_name(),
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Converts this schema to a JSON representation.
    pub fn to_json(&self) -> Value {
        let required: Vec<Value> = self
            .required_params
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "type": s.param_type.type_name(),
                    "description": s.description,
                    "required": true,
                    "default": s.default_value,
                })
            })
            .collect();

        let optional: Vec<Value> = self
            .optional_params
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "type": s.param_type.type_name(),
                    "description": s.description,
                    "required": false,
                    "default": s.default_value,
                })
            })
            .collect();

        json!({
            "required_params": required,
            "optional_params": optional,
        })
    }
}

// ---------------------------------------------------------------------------
// ConfigDiff
// ---------------------------------------------------------------------------

/// Represents the differences between two [`SerializableConfig`] instances.
#[derive(Debug, Clone)]
pub struct ConfigDiff {
    /// Parameters present in `b` but not in `a`.
    pub added: HashMap<String, Value>,
    /// Parameters present in `a` but not in `b`.
    pub removed: HashMap<String, Value>,
    /// Parameters present in both but with different values: `(old, new)`.
    pub changed: HashMap<String, (Value, Value)>,
    /// Parameter names present in both with identical values.
    pub unchanged: Vec<String>,
}

impl ConfigDiff {
    /// Computes the diff between two configs.
    pub fn diff(a: &SerializableConfig, b: &SerializableConfig) -> Self {
        let mut added = HashMap::new();
        let mut removed = HashMap::new();
        let mut changed = HashMap::new();
        let mut unchanged = Vec::new();

        // Find removed and changed/unchanged.
        for (key, val_a) in &a.parameters {
            match b.parameters.get(key) {
                None => {
                    removed.insert(key.clone(), val_a.clone());
                }
                Some(val_b) => {
                    if val_a == val_b {
                        unchanged.push(key.clone());
                    } else {
                        changed.insert(key.clone(), (val_a.clone(), val_b.clone()));
                    }
                }
            }
        }

        // Find added.
        for (key, val_b) in &b.parameters {
            if !a.parameters.contains_key(key) {
                added.insert(key.clone(), val_b.clone());
            }
        }

        unchanged.sort();

        Self {
            added,
            removed,
            changed,
            unchanged,
        }
    }

    /// Returns `true` if there are any differences between the two configs.
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.changed.is_empty()
    }

    /// Converts this diff to a JSON representation.
    pub fn to_json(&self) -> Value {
        let changed_json: HashMap<String, Value> = self
            .changed
            .iter()
            .map(|(k, (old, new))| {
                (
                    k.clone(),
                    json!({
                        "old": old,
                        "new": new,
                    }),
                )
            })
            .collect();

        json!({
            "added": self.added,
            "removed": self.removed,
            "changed": changed_json,
            "unchanged": self.unchanged,
        })
    }
}

// ---------------------------------------------------------------------------
// ConfigMigration
// ---------------------------------------------------------------------------

/// Describes a migration from one config version to another, including
/// parameter renames, new defaults, and removals.
#[derive(Debug, Clone)]
pub struct ConfigMigration {
    /// The source version.
    pub from_version: String,
    /// The target version.
    pub to_version: String,
    /// Renames: `(old_key, new_key)`.
    renames: Vec<(String, String)>,
    /// New default values to add if not present: `(key, value)`.
    defaults: Vec<(String, Value)>,
    /// Keys to remove.
    removals: Vec<String>,
}

impl ConfigMigration {
    /// Creates a new migration between two versions.
    pub fn new(from_version: impl Into<String>, to_version: impl Into<String>) -> Self {
        Self {
            from_version: from_version.into(),
            to_version: to_version.into(),
            renames: Vec::new(),
            defaults: Vec::new(),
            removals: Vec::new(),
        }
    }

    /// Adds a parameter rename rule.
    pub fn add_rename(
        mut self,
        old_key: impl Into<String>,
        new_key: impl Into<String>,
    ) -> Self {
        self.renames.push((old_key.into(), new_key.into()));
        self
    }

    /// Adds a default value for a parameter (applied if not already present).
    pub fn add_default(mut self, key: impl Into<String>, value: Value) -> Self {
        self.defaults.push((key.into(), value));
        self
    }

    /// Adds a parameter removal rule.
    pub fn add_removal(mut self, key: impl Into<String>) -> Self {
        self.removals.push(key.into());
        self
    }

    /// Applies this migration to the given config in place.
    ///
    /// Verifies the config's version matches `from_version`, applies all
    /// transformations, and updates the version to `to_version`.
    pub fn migrate(&self, config: &mut SerializableConfig) -> Result<()> {
        if config.version != self.from_version {
            return Err(RustChainError::Other(format!(
                "config version '{}' does not match migration source version '{}'",
                config.version, self.from_version,
            )));
        }

        // Apply renames.
        for (old_key, new_key) in &self.renames {
            if let Some(value) = config.parameters.remove(old_key) {
                config.parameters.insert(new_key.clone(), value);
            }
        }

        // Apply defaults.
        for (key, value) in &self.defaults {
            config
                .parameters
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }

        // Apply removals.
        for key in &self.removals {
            config.parameters.remove(key);
        }

        // Update version.
        config.version = self.to_version.clone();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConfigRegistry
// ---------------------------------------------------------------------------

/// A registry of named [`ConfigSchema`] instances, allowing validation of
/// configs against registered schemas.
#[derive(Debug, Default)]
pub struct ConfigRegistry {
    schemas: HashMap<String, ConfigSchema>,
}

impl ConfigRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a schema under the given name.
    pub fn register(&mut self, name: impl Into<String>, schema: ConfigSchema) {
        self.schemas.insert(name.into(), schema);
    }

    /// Returns a reference to the schema registered under the given name.
    pub fn get(&self, name: &str) -> Option<&ConfigSchema> {
        self.schemas.get(name)
    }

    /// Validates a config against the schema registered under the given name.
    pub fn validate(&self, name: &str, config: &SerializableConfig) -> Result<()> {
        let schema = self.schemas.get(name).ok_or_else(|| {
            RustChainError::Other(format!("no schema registered for '{}'", name))
        })?;

        schema.validate(config).map_err(|errors| {
            RustChainError::Other(format!("validation errors: {}", errors.join("; ")))
        })
    }

    /// Returns the names of all registered schemas.
    pub fn names(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Returns `true` if no schemas are registered.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // ParamType
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_type_type_name() {
        assert_eq!(ParamType::String.type_name(), "string");
        assert_eq!(ParamType::Integer.type_name(), "integer");
        assert_eq!(ParamType::Float.type_name(), "float");
        assert_eq!(ParamType::Boolean.type_name(), "boolean");
        assert_eq!(ParamType::Array.type_name(), "array");
        assert_eq!(ParamType::Object.type_name(), "object");
        assert_eq!(ParamType::Any.type_name(), "any");
    }

    #[test]
    fn test_param_type_validate_string() {
        assert!(ParamType::String.validate_value(&json!("hello")));
        assert!(!ParamType::String.validate_value(&json!(42)));
    }

    #[test]
    fn test_param_type_validate_integer() {
        assert!(ParamType::Integer.validate_value(&json!(42)));
        assert!(!ParamType::Integer.validate_value(&json!("not int")));
    }

    #[test]
    fn test_param_type_validate_float() {
        assert!(ParamType::Float.validate_value(&json!(3.14)));
        assert!(ParamType::Float.validate_value(&json!(0.0)));
        assert!(!ParamType::Float.validate_value(&json!("not float")));
    }

    #[test]
    fn test_param_type_validate_boolean() {
        assert!(ParamType::Boolean.validate_value(&json!(true)));
        assert!(ParamType::Boolean.validate_value(&json!(false)));
        assert!(!ParamType::Boolean.validate_value(&json!(1)));
    }

    #[test]
    fn test_param_type_validate_array() {
        assert!(ParamType::Array.validate_value(&json!([1, 2, 3])));
        assert!(ParamType::Array.validate_value(&json!([])));
        assert!(!ParamType::Array.validate_value(&json!("not array")));
    }

    #[test]
    fn test_param_type_validate_object() {
        assert!(ParamType::Object.validate_value(&json!({"a": 1})));
        assert!(ParamType::Object.validate_value(&json!({})));
        assert!(!ParamType::Object.validate_value(&json!("not object")));
    }

    #[test]
    fn test_param_type_validate_any() {
        assert!(ParamType::Any.validate_value(&json!(null)));
        assert!(ParamType::Any.validate_value(&json!(42)));
        assert!(ParamType::Any.validate_value(&json!("hello")));
        assert!(ParamType::Any.validate_value(&json!({"a": 1})));
    }

    // -----------------------------------------------------------------------
    // ParamSpec
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_spec_builder() {
        let spec = ParamSpec::new("temperature", ParamType::Float)
            .with_description("Sampling temperature")
            .with_default(json!(0.7));

        assert_eq!(spec.name, "temperature");
        assert_eq!(spec.param_type, ParamType::Float);
        assert_eq!(spec.description, "Sampling temperature");
        assert_eq!(spec.default_value, Some(json!(0.7)));
    }

    #[test]
    fn test_param_spec_defaults() {
        let spec = ParamSpec::new("model", ParamType::String);
        assert_eq!(spec.name, "model");
        assert!(spec.description.is_empty());
        assert!(spec.default_value.is_none());
    }

    // -----------------------------------------------------------------------
    // SerializableConfig building
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_builder_basic() {
        let config = SerializableConfig::builder()
            .name("my-config")
            .type_name("ChatModel")
            .version("1.0.0")
            .parameter("model", json!("gpt-4"))
            .parameter("temperature", json!(0.7))
            .meta("author", json!("test"))
            .build();

        assert_eq!(config.name, "my-config");
        assert_eq!(config.type_name, "ChatModel");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.parameters.get("model"), Some(&json!("gpt-4")));
        assert_eq!(config.parameters.get("temperature"), Some(&json!(0.7)));
        assert_eq!(config.metadata.get("author"), Some(&json!("test")));
    }

    #[test]
    fn test_config_builder_with_maps() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), json!("value"));
        let mut meta = HashMap::new();
        meta.insert("env".to_string(), json!("prod"));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("Tool")
            .version("2.0.0")
            .parameters(params)
            .metadata(meta)
            .build();

        assert_eq!(config.parameters.len(), 1);
        assert_eq!(config.metadata.len(), 1);
    }

    #[test]
    fn test_config_builder_empty() {
        let config = SerializableConfig::builder().build();
        assert!(config.name.is_empty());
        assert!(config.type_name.is_empty());
        assert!(config.version.is_empty());
        assert!(config.parameters.is_empty());
        assert!(config.metadata.is_empty());
    }

    // -----------------------------------------------------------------------
    // SerializableConfig JSON round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_to_json() {
        let config = SerializableConfig::builder()
            .name("test")
            .type_name("LLM")
            .version("1.0.0")
            .parameter("model", json!("claude"))
            .build();

        let j = config.to_json();
        assert_eq!(j["name"], "test");
        assert_eq!(j["type_name"], "LLM");
        assert_eq!(j["version"], "1.0.0");
        assert_eq!(j["parameters"]["model"], "claude");
    }

    #[test]
    fn test_config_json_round_trip() {
        let config = SerializableConfig::builder()
            .name("roundtrip")
            .type_name("Agent")
            .version("2.1.0")
            .parameter("max_tokens", json!(1024))
            .parameter("stop", json!([".", "!"]))
            .meta("created", json!("2024-01-01"))
            .build();

        let json_val = config.to_json();
        let restored = SerializableConfig::from_json(&json_val).unwrap();

        assert_eq!(restored.name, config.name);
        assert_eq!(restored.type_name, config.type_name);
        assert_eq!(restored.version, config.version);
        assert_eq!(restored.parameters, config.parameters);
        assert_eq!(restored.metadata, config.metadata);
    }

    #[test]
    fn test_config_from_json_missing_name() {
        let val = json!({"type_name": "X", "version": "1"});
        assert!(SerializableConfig::from_json(&val).is_err());
    }

    #[test]
    fn test_config_from_json_missing_type_name() {
        let val = json!({"name": "X", "version": "1"});
        assert!(SerializableConfig::from_json(&val).is_err());
    }

    #[test]
    fn test_config_from_json_missing_version() {
        let val = json!({"name": "X", "type_name": "Y"});
        assert!(SerializableConfig::from_json(&val).is_err());
    }

    #[test]
    fn test_config_to_compact_json() {
        let config = SerializableConfig::builder()
            .name("c")
            .type_name("T")
            .version("1")
            .build();

        let compact = config.to_compact_json();
        assert!(compact.contains("\"name\":\"c\""));
        assert!(!compact.contains('\n'));
    }

    #[test]
    fn test_config_from_json_missing_optional_maps() {
        let val = json!({"name": "X", "type_name": "Y", "version": "1"});
        let config = SerializableConfig::from_json(&val).unwrap();
        assert!(config.parameters.is_empty());
        assert!(config.metadata.is_empty());
    }

    // -----------------------------------------------------------------------
    // ConfigSchema validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_validate_success() {
        let schema = ConfigSchema::new()
            .add_required(ParamSpec::new("model", ParamType::String))
            .add_optional(ParamSpec::new("temperature", ParamType::Float));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("LLM")
            .version("1")
            .parameter("model", json!("gpt-4"))
            .parameter("temperature", json!(0.5))
            .build();

        assert!(schema.validate(&config).is_ok());
    }

    #[test]
    fn test_schema_validate_missing_required() {
        let schema = ConfigSchema::new()
            .add_required(ParamSpec::new("model", ParamType::String))
            .add_required(ParamSpec::new("api_key", ParamType::String));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("LLM")
            .version("1")
            .parameter("model", json!("gpt-4"))
            .build();

        let result = schema.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("api_key"));
    }

    #[test]
    fn test_schema_validate_wrong_type_required() {
        let schema =
            ConfigSchema::new().add_required(ParamSpec::new("count", ParamType::Integer));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("count", json!("not a number"))
            .build();

        let result = schema.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].contains("count"));
        assert!(errors[0].contains("integer"));
    }

    #[test]
    fn test_schema_validate_wrong_type_optional() {
        let schema =
            ConfigSchema::new().add_optional(ParamSpec::new("debug", ParamType::Boolean));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("debug", json!("yes"))
            .build();

        let result = schema.validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_validate_optional_missing_ok() {
        let schema =
            ConfigSchema::new().add_optional(ParamSpec::new("debug", ParamType::Boolean));

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        assert!(schema.validate(&config).is_ok());
    }

    #[test]
    fn test_schema_to_json() {
        let schema = ConfigSchema::new()
            .add_required(
                ParamSpec::new("model", ParamType::String).with_description("The model name"),
            )
            .add_optional(ParamSpec::new("temp", ParamType::Float).with_default(json!(0.7)));

        let j = schema.to_json();
        assert_eq!(j["required_params"].as_array().unwrap().len(), 1);
        assert_eq!(j["optional_params"].as_array().unwrap().len(), 1);
        assert_eq!(j["required_params"][0]["name"], "model");
        assert_eq!(j["optional_params"][0]["default"], 0.7);
    }

    #[test]
    fn test_schema_empty_validates_any_config() {
        let schema = ConfigSchema::new();
        let config = SerializableConfig::builder()
            .name("any")
            .type_name("Any")
            .version("1")
            .parameter("whatever", json!(42))
            .build();

        assert!(schema.validate(&config).is_ok());
    }

    // -----------------------------------------------------------------------
    // ConfigDiff
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_added() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("new_key", json!("new_val"))
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added.get("new_key"), Some(&json!("new_val")));
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_diff_removed() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("old_key", json!("old_val"))
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed.get("old_key"), Some(&json!("old_val")));
    }

    #[test]
    fn test_diff_changed() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("key", json!(1))
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("key", json!(2))
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert_eq!(diff.changed.len(), 1);
        let (old, new) = diff.changed.get("key").unwrap();
        assert_eq!(old, &json!(1));
        assert_eq!(new, &json!(2));
    }

    #[test]
    fn test_diff_unchanged() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("key", json!("same"))
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("key", json!("same"))
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(!diff.has_changes());
        assert_eq!(diff.unchanged, vec!["key"]);
    }

    #[test]
    fn test_diff_no_changes_empty() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(!diff.has_changes());
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
        assert!(diff.unchanged.is_empty());
    }

    #[test]
    fn test_diff_mixed() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("kept", json!("same"))
            .parameter("modified", json!(1))
            .parameter("gone", json!("bye"))
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("kept", json!("same"))
            .parameter("modified", json!(2))
            .parameter("fresh", json!("hi"))
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert_eq!(diff.unchanged, vec!["kept"]);
        assert_eq!(diff.added.get("fresh"), Some(&json!("hi")));
        assert_eq!(diff.removed.get("gone"), Some(&json!("bye")));
        assert!(diff.changed.contains_key("modified"));
    }

    #[test]
    fn test_diff_to_json() {
        let a = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("x", json!(1))
            .build();

        let b = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .parameter("x", json!(2))
            .parameter("y", json!(3))
            .build();

        let diff = ConfigDiff::diff(&a, &b);
        let j = diff.to_json();
        assert!(j["added"]["y"].is_number());
        assert!(j["changed"]["x"].is_object());
        assert_eq!(j["changed"]["x"]["old"], 1);
        assert_eq!(j["changed"]["x"]["new"], 2);
    }

    // -----------------------------------------------------------------------
    // ConfigMigration
    // -----------------------------------------------------------------------

    #[test]
    fn test_migration_rename() {
        let migration = ConfigMigration::new("1.0", "2.0").add_rename("old_name", "new_name");

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .parameter("old_name", json!("value"))
            .build();

        migration.migrate(&mut config).unwrap();
        assert_eq!(config.version, "2.0");
        assert!(config.parameters.get("old_name").is_none());
        assert_eq!(config.parameters.get("new_name"), Some(&json!("value")));
    }

    #[test]
    fn test_migration_add_default() {
        let migration = ConfigMigration::new("1.0", "2.0")
            .add_default("new_param", json!("default_value"));

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .build();

        migration.migrate(&mut config).unwrap();
        assert_eq!(
            config.parameters.get("new_param"),
            Some(&json!("default_value"))
        );
    }

    #[test]
    fn test_migration_default_does_not_overwrite() {
        let migration =
            ConfigMigration::new("1.0", "2.0").add_default("param", json!("default"));

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .parameter("param", json!("existing"))
            .build();

        migration.migrate(&mut config).unwrap();
        assert_eq!(config.parameters.get("param"), Some(&json!("existing")));
    }

    #[test]
    fn test_migration_removal() {
        let migration = ConfigMigration::new("1.0", "2.0").add_removal("deprecated");

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .parameter("deprecated", json!("gone"))
            .parameter("kept", json!("stay"))
            .build();

        migration.migrate(&mut config).unwrap();
        assert!(config.parameters.get("deprecated").is_none());
        assert_eq!(config.parameters.get("kept"), Some(&json!("stay")));
    }

    #[test]
    fn test_migration_wrong_version() {
        let migration = ConfigMigration::new("1.0", "2.0");

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("3.0")
            .build();

        assert!(migration.migrate(&mut config).is_err());
    }

    #[test]
    fn test_migration_combined() {
        let migration = ConfigMigration::new("1.0", "2.0")
            .add_rename("old_key", "new_key")
            .add_default("added_param", json!(true))
            .add_removal("removed_param");

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .parameter("old_key", json!("val"))
            .parameter("removed_param", json!("bye"))
            .parameter("untouched", json!(42))
            .build();

        migration.migrate(&mut config).unwrap();
        assert_eq!(config.version, "2.0");
        assert_eq!(config.parameters.get("new_key"), Some(&json!("val")));
        assert!(config.parameters.get("old_key").is_none());
        assert_eq!(config.parameters.get("added_param"), Some(&json!(true)));
        assert!(config.parameters.get("removed_param").is_none());
        assert_eq!(config.parameters.get("untouched"), Some(&json!(42)));
    }

    #[test]
    fn test_migration_empty_config() {
        let migration = ConfigMigration::new("1.0", "2.0")
            .add_rename("a", "b")
            .add_default("c", json!(1))
            .add_removal("d");

        let mut config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1.0")
            .build();

        migration.migrate(&mut config).unwrap();
        assert_eq!(config.version, "2.0");
        assert_eq!(config.parameters.get("c"), Some(&json!(1)));
        assert!(config.parameters.get("a").is_none());
        assert!(config.parameters.get("b").is_none());
        assert!(config.parameters.get("d").is_none());
    }

    // -----------------------------------------------------------------------
    // ConfigRegistry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ConfigRegistry::new();
        let schema =
            ConfigSchema::new().add_required(ParamSpec::new("model", ParamType::String));

        registry.register("llm", schema);
        assert!(registry.get("llm").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_validate_success() {
        let mut registry = ConfigRegistry::new();
        let schema =
            ConfigSchema::new().add_required(ParamSpec::new("model", ParamType::String));
        registry.register("llm", schema);

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("LLM")
            .version("1")
            .parameter("model", json!("gpt-4"))
            .build();

        assert!(registry.validate("llm", &config).is_ok());
    }

    #[test]
    fn test_registry_validate_failure() {
        let mut registry = ConfigRegistry::new();
        let schema =
            ConfigSchema::new().add_required(ParamSpec::new("model", ParamType::String));
        registry.register("llm", schema);

        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("LLM")
            .version("1")
            .build();

        assert!(registry.validate("llm", &config).is_err());
    }

    #[test]
    fn test_registry_validate_unknown_schema() {
        let registry = ConfigRegistry::new();
        let config = SerializableConfig::builder()
            .name("cfg")
            .type_name("T")
            .version("1")
            .build();

        assert!(registry.validate("unknown", &config).is_err());
    }

    #[test]
    fn test_registry_names() {
        let mut registry = ConfigRegistry::new();
        registry.register("a", ConfigSchema::new());
        registry.register("b", ConfigSchema::new());

        let mut names = registry.names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_registry_len_and_empty() {
        let mut registry = ConfigRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        registry.register("x", ConfigSchema::new());
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }
}
