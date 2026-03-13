//! Configurable runnables — runnables whose behavior can be adjusted at
//! invocation time via key-value configuration fields.
//!
//! This module provides a schema-driven configuration system that lets callers
//! override parameters (model name, temperature, prompt variants, etc.) without
//! creating new runnable instances. Field definitions carry type information and
//! are validated before the inner runnable executes.
//!
//! - [`ConfigurableFieldType`] — supported value types for configuration fields
//! - [`ConfigurableField`] — a single named, typed, optionally-required field
//! - [`ConfigurableSpec`] — a collection of field definitions forming a config schema
//! - [`ConfigurableRunnable`] — wraps an inner [`Runnable`] with a [`ConfigurableSpec`]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

// ---------------------------------------------------------------------------
// ConfigurableFieldType
// ---------------------------------------------------------------------------

/// The type of a configurable field, used for validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigurableFieldType {
    /// A string value.
    String,
    /// An integer value.
    Integer,
    /// A floating-point value.
    Float,
    /// A boolean value.
    Boolean,
    /// One of a fixed set of allowed string values.
    Enum(Vec<std::string::String>),
    /// Any valid JSON value.
    Json,
}

impl ConfigurableFieldType {
    /// Check whether `value` is compatible with this type.
    pub fn validate(&self, value: &Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Integer => value.is_i64() || value.is_u64(),
            Self::Float => value.is_f64() || value.is_i64() || value.is_u64(),
            Self::Boolean => value.is_boolean(),
            Self::Enum(variants) => value
                .as_str()
                .is_some_and(|s| variants.contains(&s.to_owned())),
            Self::Json => true, // anything goes
        }
    }

    /// Returns a sensible default value for this type.
    pub fn default_value(&self) -> Value {
        match self {
            Self::String => Value::String(std::string::String::new()),
            Self::Integer => Value::Number(0.into()),
            Self::Float => serde_json::json!(0.0),
            Self::Boolean => Value::Bool(false),
            Self::Enum(variants) => {
                if let Some(first) = variants.first() {
                    Value::String(first.clone())
                } else {
                    Value::Null
                }
            }
            Self::Json => Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigurableField
// ---------------------------------------------------------------------------

/// Describes a single configurable field with metadata, type information,
/// and optional default value.
#[derive(Debug, Clone)]
pub struct ConfigurableField {
    /// Unique identifier for the field, used as the key in `RunnableConfig::configurable`.
    pub id: String,
    /// Human-readable name for the field.
    pub name: String,
    /// Description of what this field controls.
    pub description: Option<String>,
    /// The expected type for values of this field.
    pub field_type: ConfigurableFieldType,
    /// An optional default value.
    pub default: Option<Value>,
    /// Whether this field must be provided in the configuration.
    pub required: bool,
}

impl ConfigurableField {
    /// Create a new `ConfigurableField` with the given id, name, and type.
    ///
    /// By default the field is not required and has no default value or description.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        field_type: ConfigurableFieldType,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            field_type,
            default: None,
            required: false,
        }
    }

    /// Set the description for this field.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the default value for this field.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Set whether this field is required.
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Validate a value against this field's type and required constraint.
    pub fn validate_value(&self, value: &Value) -> Result<()> {
        if !self.field_type.validate(value) {
            return Err(CognisError::TypeMismatch {
                expected: format!("{:?}", self.field_type),
                got: format!("{}", value),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConfigurableSpec
// ---------------------------------------------------------------------------

/// A collection of `ConfigurableField` definitions that together form a
/// configuration schema for a runnable.
#[derive(Debug, Clone)]
pub struct ConfigurableSpec {
    fields: Vec<ConfigurableField>,
}

impl ConfigurableSpec {
    /// Create an empty spec.
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a field definition.
    pub fn add_field(mut self, field: ConfigurableField) -> Self {
        self.fields.push(field);
        self
    }

    /// Look up a field definition by id.
    pub fn get_field(&self, id: &str) -> Option<&ConfigurableField> {
        self.fields.iter().find(|f| f.id == id)
    }

    /// Return all field definitions.
    pub fn fields(&self) -> &[ConfigurableField] {
        &self.fields
    }

    /// Return only the required field definitions.
    pub fn required_fields(&self) -> Vec<&ConfigurableField> {
        self.fields.iter().filter(|f| f.required).collect()
    }

    /// Validate a configuration object against this spec.
    ///
    /// Checks that all required fields are present and that each provided
    /// value matches the expected type.
    pub fn validate(&self, config: &Value) -> Result<()> {
        let obj = config
            .as_object()
            .ok_or_else(|| CognisError::TypeMismatch {
                expected: "object".into(),
                got: format!("{}", config),
            })?;

        for field in &self.fields {
            if field.required && !obj.contains_key(&field.id) {
                return Err(CognisError::InvalidKey(format!(
                    "Required configurable field '{}' is missing",
                    field.id
                )));
            }
            if let Some(val) = obj.get(&field.id) {
                field.validate_value(val)?;
            }
        }
        Ok(())
    }

    /// Fill in missing fields with their defaults (if any) in-place.
    pub fn apply_defaults(&self, config: &mut Value) {
        let obj = match config.as_object_mut() {
            Some(o) => o,
            None => return,
        };

        for field in &self.fields {
            if !obj.contains_key(&field.id) {
                if let Some(ref default) = field.default {
                    obj.insert(field.id.clone(), default.clone());
                }
            }
        }
    }
}

impl Default for ConfigurableSpec {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RunnableConfigurableFields
// ---------------------------------------------------------------------------

/// A runnable that selects between alternative implementations based on configuration.
///
/// When invoked, it checks `RunnableConfig::configurable` for field values that match
/// registered alternatives. If no match is found, it falls back to the default runnable.
///
/// Optionally carries a `ConfigurableSpec` that defines the schema of accepted
/// configuration values and can validate / apply defaults.
pub struct RunnableConfigurableFields {
    default: Arc<dyn Runnable>,
    fields: Vec<ConfigurableField>,
    alternatives: HashMap<String, HashMap<String, Arc<dyn Runnable>>>,
    spec: Option<ConfigurableSpec>,
    pre_config: Option<Value>,
}

impl RunnableConfigurableFields {
    /// Create a new `RunnableConfigurableFields` with a default runnable and configurable fields.
    pub fn new(default: Arc<dyn Runnable>, fields: Vec<ConfigurableField>) -> Self {
        Self {
            default,
            fields,
            alternatives: HashMap::new(),
            spec: None,
            pre_config: None,
        }
    }

    /// Create a new `RunnableConfigurableFields` with a default runnable and a spec.
    pub fn with_spec(default: Arc<dyn Runnable>, spec: ConfigurableSpec) -> Self {
        let fields = spec.fields().to_vec();
        Self {
            default,
            fields,
            alternatives: HashMap::new(),
            spec: Some(spec),
            pre_config: None,
        }
    }

    /// Register alternative runnables for a given field.
    pub fn with_alternatives(
        mut self,
        field_id: &str,
        alternatives: HashMap<String, Arc<dyn Runnable>>,
    ) -> Self {
        self.alternatives.insert(field_id.into(), alternatives);
        self
    }

    /// Validate and apply a configuration against the spec (if any).
    ///
    /// Returns a new `Value` with defaults applied.
    pub fn configure(&self, mut config: Value) -> Result<Value> {
        if let Some(ref spec) = self.spec {
            spec.apply_defaults(&mut config);
            spec.validate(&config)?;
        }
        Ok(config)
    }

    /// Return the spec (if any).
    pub fn get_spec(&self) -> Option<&ConfigurableSpec> {
        self.spec.as_ref()
    }

    /// Return a pre-configured instance that will merge the given config into
    /// every invocation.
    pub fn with_config(mut self, config: Value) -> Self {
        self.pre_config = Some(config);
        self
    }

    fn resolve(&self, config: Option<&RunnableConfig>) -> Arc<dyn Runnable> {
        if let Some(config) = config {
            for field in &self.fields {
                if let Some(val) = config.configurable.get(&field.id) {
                    if let Some(key) = val.as_str() {
                        if let Some(alts) = self.alternatives.get(&field.id) {
                            if let Some(alt) = alts.get(key) {
                                return alt.clone();
                            }
                        }
                    }
                }
            }
        }
        self.default.clone()
    }

    /// Build the effective config by merging pre_config into the provided config.
    fn effective_config(&self, config: Option<&RunnableConfig>) -> Option<RunnableConfig> {
        match (&self.pre_config, config) {
            (Some(pre), Some(cfg)) => {
                let mut merged = cfg.clone();
                if let Some(obj) = pre.as_object() {
                    for (k, v) in obj {
                        merged
                            .configurable
                            .entry(k.clone())
                            .or_insert_with(|| v.clone());
                    }
                }
                Some(merged)
            }
            (Some(pre), None) => {
                let mut cfg = RunnableConfig::default();
                if let Some(obj) = pre.as_object() {
                    for (k, v) in obj {
                        cfg.configurable.insert(k.clone(), v.clone());
                    }
                }
                Some(cfg)
            }
            _ => config.cloned(),
        }
    }
}

#[async_trait]
impl Runnable for RunnableConfigurableFields {
    fn name(&self) -> &str {
        "RunnableConfigurableFields"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let eff = self.effective_config(config);
        let eff_ref = eff.as_ref().or(config);
        self.resolve(eff_ref).invoke(input, eff_ref).await
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let eff = self.effective_config(config);
        let eff_ref = eff.as_ref().or(config);
        self.resolve(eff_ref).stream(input, eff_ref).await
    }
}

// ---------------------------------------------------------------------------
// ConfigurableAlternatives
// ---------------------------------------------------------------------------

/// A registry of named alternatives that can be switched at runtime via a
/// configuration key.
///
/// Unlike `RunnableConfigurableFields` which wraps actual `Runnable` instances,
/// `ConfigurableAlternatives` is a lightweight registry that simply maps string
/// keys to identify which alternative should be selected.
#[derive(Debug, Clone)]
pub struct ConfigurableAlternatives {
    default_key: String,
    alternatives: Vec<String>,
}

impl ConfigurableAlternatives {
    /// Create a new registry with the given default key.
    pub fn new(default_key: &str) -> Self {
        Self {
            default_key: default_key.to_string(),
            alternatives: vec![default_key.to_string()],
        }
    }

    /// Register a named alternative. Returns `&mut Self` for chaining.
    pub fn add_alternative(&mut self, key: impl Into<String>) -> &mut Self {
        let key = key.into();
        if !self.alternatives.contains(&key) {
            self.alternatives.push(key);
        }
        self
    }

    /// Check whether an alternative with the given key exists, returning the
    /// key if found.
    pub fn select(&self, key: &str) -> Option<&str> {
        self.alternatives
            .iter()
            .find(|k| k.as_str() == key)
            .map(|s| s.as_str())
    }

    /// Return all registered keys.
    pub fn keys(&self) -> &[String] {
        &self.alternatives
    }

    /// Return the default key.
    pub fn default_key(&self) -> &str {
        &self.default_key
    }

    /// Return the number of registered alternatives.
    pub fn len(&self) -> usize {
        self.alternatives.len()
    }

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // === ConfigurableFieldType validation tests ===

    #[test]
    fn test_field_type_string_validates_string() {
        assert!(ConfigurableFieldType::String.validate(&json!("hello")));
    }

    #[test]
    fn test_field_type_string_rejects_number() {
        assert!(!ConfigurableFieldType::String.validate(&json!(42)));
    }

    #[test]
    fn test_field_type_integer_validates_int() {
        assert!(ConfigurableFieldType::Integer.validate(&json!(42)));
    }

    #[test]
    fn test_field_type_integer_rejects_string() {
        assert!(!ConfigurableFieldType::Integer.validate(&json!("42")));
    }

    #[test]
    fn test_field_type_integer_rejects_float() {
        // 3.5 is not an integer
        assert!(!ConfigurableFieldType::Integer.validate(&json!(3.5)));
    }

    #[test]
    fn test_field_type_float_validates_float() {
        assert!(ConfigurableFieldType::Float.validate(&json!(3.14)));
    }

    #[test]
    fn test_field_type_float_validates_int_as_float() {
        // integers are valid floats
        assert!(ConfigurableFieldType::Float.validate(&json!(42)));
    }

    #[test]
    fn test_field_type_float_rejects_string() {
        assert!(!ConfigurableFieldType::Float.validate(&json!("3.14")));
    }

    #[test]
    fn test_field_type_boolean_validates_bool() {
        assert!(ConfigurableFieldType::Boolean.validate(&json!(true)));
        assert!(ConfigurableFieldType::Boolean.validate(&json!(false)));
    }

    #[test]
    fn test_field_type_boolean_rejects_int() {
        assert!(!ConfigurableFieldType::Boolean.validate(&json!(1)));
    }

    #[test]
    fn test_field_type_enum_validates_allowed() {
        let e = ConfigurableFieldType::Enum(vec!["a".into(), "b".into(), "c".into()]);
        assert!(e.validate(&json!("a")));
        assert!(e.validate(&json!("c")));
    }

    #[test]
    fn test_field_type_enum_rejects_disallowed() {
        let e = ConfigurableFieldType::Enum(vec!["a".into(), "b".into()]);
        assert!(!e.validate(&json!("x")));
    }

    #[test]
    fn test_field_type_enum_rejects_non_string() {
        let e = ConfigurableFieldType::Enum(vec!["1".into()]);
        assert!(!e.validate(&json!(1)));
    }

    #[test]
    fn test_field_type_json_accepts_anything() {
        assert!(ConfigurableFieldType::Json.validate(&json!(null)));
        assert!(ConfigurableFieldType::Json.validate(&json!(42)));
        assert!(ConfigurableFieldType::Json.validate(&json!("hi")));
        assert!(ConfigurableFieldType::Json.validate(&json!({"a": 1})));
        assert!(ConfigurableFieldType::Json.validate(&json!([1, 2, 3])));
    }

    // === ConfigurableFieldType default_value tests ===

    #[test]
    fn test_field_type_default_values() {
        assert_eq!(ConfigurableFieldType::String.default_value(), json!(""));
        assert_eq!(ConfigurableFieldType::Integer.default_value(), json!(0));
        assert_eq!(ConfigurableFieldType::Float.default_value(), json!(0.0));
        assert_eq!(ConfigurableFieldType::Boolean.default_value(), json!(false));
        assert_eq!(ConfigurableFieldType::Json.default_value(), json!(null));
    }

    #[test]
    fn test_field_type_enum_default_first_variant() {
        let e = ConfigurableFieldType::Enum(vec!["alpha".into(), "beta".into()]);
        assert_eq!(e.default_value(), json!("alpha"));
    }

    #[test]
    fn test_field_type_enum_default_empty() {
        let e = ConfigurableFieldType::Enum(vec![]);
        assert_eq!(e.default_value(), json!(null));
    }

    // === ConfigurableField builder tests ===

    #[test]
    fn test_configurable_field_new() {
        let f = ConfigurableField::new("model", "Model", ConfigurableFieldType::String);
        assert_eq!(f.id, "model");
        assert_eq!(f.name, "Model");
        assert!(f.description.is_none());
        assert!(f.default.is_none());
        assert!(!f.required);
    }

    #[test]
    fn test_configurable_field_builder() {
        let f = ConfigurableField::new("temp", "Temperature", ConfigurableFieldType::Float)
            .with_description("The sampling temperature")
            .with_default(json!(0.7))
            .required(true);
        assert_eq!(f.description.as_deref(), Some("The sampling temperature"));
        assert_eq!(f.default, Some(json!(0.7)));
        assert!(f.required);
    }

    #[test]
    fn test_configurable_field_validate_value_ok() {
        let f = ConfigurableField::new("count", "Count", ConfigurableFieldType::Integer);
        assert!(f.validate_value(&json!(10)).is_ok());
    }

    #[test]
    fn test_configurable_field_validate_value_err() {
        let f = ConfigurableField::new("count", "Count", ConfigurableFieldType::Integer);
        assert!(f.validate_value(&json!("not_a_number")).is_err());
    }

    // === ConfigurableSpec tests ===

    #[test]
    fn test_spec_empty() {
        let spec = ConfigurableSpec::new();
        assert!(spec.fields().is_empty());
        assert!(spec.required_fields().is_empty());
        // validating an empty object against an empty spec should succeed
        assert!(spec.validate(&json!({})).is_ok());
    }

    #[test]
    fn test_spec_add_and_get_field() {
        let spec = ConfigurableSpec::new().add_field(ConfigurableField::new(
            "model",
            "Model",
            ConfigurableFieldType::String,
        ));
        assert_eq!(spec.fields().len(), 1);
        assert!(spec.get_field("model").is_some());
        assert!(spec.get_field("missing").is_none());
    }

    #[test]
    fn test_spec_required_fields() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                    .required(true),
            )
            .add_field(ConfigurableField::new(
                "temp",
                "Temperature",
                ConfigurableFieldType::Float,
            ));
        let required = spec.required_fields();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].id, "model");
    }

    #[test]
    fn test_spec_validate_missing_required() {
        let spec = ConfigurableSpec::new().add_field(
            ConfigurableField::new("model", "Model", ConfigurableFieldType::String).required(true),
        );
        let result = spec.validate(&json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_spec_validate_wrong_type() {
        let spec = ConfigurableSpec::new().add_field(ConfigurableField::new(
            "count",
            "Count",
            ConfigurableFieldType::Integer,
        ));
        let result = spec.validate(&json!({"count": "not_int"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_spec_validate_success() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                    .required(true),
            )
            .add_field(ConfigurableField::new(
                "temp",
                "Temperature",
                ConfigurableFieldType::Float,
            ));
        let result = spec.validate(&json!({"model": "gpt-4", "temp": 0.5}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_spec_validate_required_present_optional_absent() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                    .required(true),
            )
            .add_field(ConfigurableField::new(
                "temp",
                "Temperature",
                ConfigurableFieldType::Float,
            ));
        let result = spec.validate(&json!({"model": "gpt-4"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_spec_validate_non_object() {
        let spec = ConfigurableSpec::new();
        let result = spec.validate(&json!("not an object"));
        assert!(result.is_err());
    }

    #[test]
    fn test_spec_apply_defaults() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                    .with_default(json!("gpt-3.5")),
            )
            .add_field(
                ConfigurableField::new("temp", "Temperature", ConfigurableFieldType::Float)
                    .with_default(json!(0.7)),
            );
        let mut config = json!({});
        spec.apply_defaults(&mut config);
        assert_eq!(config["model"], json!("gpt-3.5"));
        assert_eq!(config["temp"], json!(0.7));
    }

    #[test]
    fn test_spec_apply_defaults_does_not_overwrite() {
        let spec = ConfigurableSpec::new().add_field(
            ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                .with_default(json!("gpt-3.5")),
        );
        let mut config = json!({"model": "gpt-4"});
        spec.apply_defaults(&mut config);
        assert_eq!(config["model"], json!("gpt-4"));
    }

    #[test]
    fn test_spec_apply_defaults_no_default_leaves_absent() {
        let spec = ConfigurableSpec::new().add_field(ConfigurableField::new(
            "model",
            "Model",
            ConfigurableFieldType::String,
        ));
        let mut config = json!({});
        spec.apply_defaults(&mut config);
        assert!(config.get("model").is_none());
    }

    #[test]
    fn test_spec_all_defaults_applied() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("a", "A", ConfigurableFieldType::String)
                    .with_default(json!("x")),
            )
            .add_field(
                ConfigurableField::new("b", "B", ConfigurableFieldType::Integer)
                    .with_default(json!(42)),
            )
            .add_field(
                ConfigurableField::new("c", "C", ConfigurableFieldType::Boolean)
                    .with_default(json!(true)),
            );
        let mut config = json!({});
        spec.apply_defaults(&mut config);
        assert_eq!(config, json!({"a": "x", "b": 42, "c": true}));
    }

    // === RunnableConfigurableFields tests ===

    #[test]
    fn test_rcf_configure_validates_and_applies_defaults() {
        let spec = ConfigurableSpec::new()
            .add_field(
                ConfigurableField::new("model", "Model", ConfigurableFieldType::String)
                    .required(true),
            )
            .add_field(
                ConfigurableField::new("temp", "Temperature", ConfigurableFieldType::Float)
                    .with_default(json!(0.7)),
            );

        let dummy = Arc::new(crate::runnables::RunnablePassthrough::new()) as Arc<dyn Runnable>;
        let rcf = RunnableConfigurableFields::with_spec(dummy, spec);

        let result = rcf.configure(json!({"model": "gpt-4"})).unwrap();
        assert_eq!(result["model"], json!("gpt-4"));
        assert_eq!(result["temp"], json!(0.7));
    }

    #[test]
    fn test_rcf_configure_rejects_missing_required() {
        let spec = ConfigurableSpec::new().add_field(
            ConfigurableField::new("model", "Model", ConfigurableFieldType::String).required(true),
        );

        let dummy = Arc::new(crate::runnables::RunnablePassthrough::new()) as Arc<dyn Runnable>;
        let rcf = RunnableConfigurableFields::with_spec(dummy, spec);

        assert!(rcf.configure(json!({})).is_err());
    }

    #[test]
    fn test_rcf_get_spec() {
        let spec = ConfigurableSpec::new().add_field(ConfigurableField::new(
            "model",
            "Model",
            ConfigurableFieldType::String,
        ));
        let dummy = Arc::new(crate::runnables::RunnablePassthrough::new()) as Arc<dyn Runnable>;
        let rcf = RunnableConfigurableFields::with_spec(dummy, spec);
        assert!(rcf.get_spec().is_some());
        assert_eq!(rcf.get_spec().unwrap().fields().len(), 1);
    }

    #[test]
    fn test_rcf_no_spec_returns_none() {
        let dummy = Arc::new(crate::runnables::RunnablePassthrough::new()) as Arc<dyn Runnable>;
        let rcf = RunnableConfigurableFields::new(dummy, vec![]);
        assert!(rcf.get_spec().is_none());
    }

    // === ConfigurableAlternatives tests ===

    #[test]
    fn test_alternatives_new() {
        let alts = ConfigurableAlternatives::new("default");
        assert_eq!(alts.default_key(), "default");
        assert_eq!(alts.len(), 1);
        assert!(!alts.is_empty());
    }

    #[test]
    fn test_alternatives_add_and_select() {
        let mut alts = ConfigurableAlternatives::new("gpt-4");
        alts.add_alternative("claude");
        alts.add_alternative("gemini");

        assert_eq!(alts.select("gpt-4"), Some("gpt-4"));
        assert_eq!(alts.select("claude"), Some("claude"));
        assert_eq!(alts.select("gemini"), Some("gemini"));
        assert_eq!(alts.select("unknown"), None);
    }

    #[test]
    fn test_alternatives_keys() {
        let mut alts = ConfigurableAlternatives::new("a");
        alts.add_alternative("b");
        alts.add_alternative("c");
        assert_eq!(alts.keys(), &["a", "b", "c"]);
    }

    #[test]
    fn test_alternatives_len() {
        let mut alts = ConfigurableAlternatives::new("x");
        assert_eq!(alts.len(), 1);
        alts.add_alternative("y");
        assert_eq!(alts.len(), 2);
    }

    #[test]
    fn test_alternatives_no_duplicate() {
        let mut alts = ConfigurableAlternatives::new("x");
        alts.add_alternative("x"); // duplicate of default
        assert_eq!(alts.len(), 1);
    }

    #[test]
    fn test_alternatives_default_key() {
        let alts = ConfigurableAlternatives::new("primary");
        assert_eq!(alts.default_key(), "primary");
    }

    // === Edge case tests ===

    #[test]
    fn test_spec_no_config_object() {
        let spec = ConfigurableSpec::new();
        let mut config = json!(42);
        // apply_defaults on non-object is a no-op
        spec.apply_defaults(&mut config);
        assert_eq!(config, json!(42));
    }

    #[test]
    fn test_enum_validation_empty_variants() {
        let e = ConfigurableFieldType::Enum(vec![]);
        assert!(!e.validate(&json!("anything")));
    }
}
