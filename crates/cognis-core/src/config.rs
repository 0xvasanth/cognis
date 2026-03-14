//! Configuration management for the LLM framework.
//!
//! Provides type-safe configuration with validation, defaults, and environment
//! variable support. The central types are [`ConfigValue`] for representing
//! heterogeneous values, [`ConfigStore`] for hierarchical key-value storage,
//! and [`ConfigBuilder`] for ergonomic construction.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ConfigType
// ---------------------------------------------------------------------------

/// Describes the expected type of a configuration value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigType {
    String,
    Integer,
    Float,
    Bool,
    List,
    Map,
}

impl fmt::Display for ConfigType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigType::String => write!(f, "String"),
            ConfigType::Integer => write!(f, "Integer"),
            ConfigType::Float => write!(f, "Float"),
            ConfigType::Bool => write!(f, "Bool"),
            ConfigType::List => write!(f, "List"),
            ConfigType::Map => write!(f, "Map"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigError
// ---------------------------------------------------------------------------

/// Errors arising from configuration validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// A required key is missing.
    MissingRequired { key: String },
    /// A key has the wrong type.
    WrongType {
        key: String,
        expected: ConfigType,
        actual: ConfigType,
    },
    /// A custom validation error.
    Custom(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingRequired { key } => {
                write!(f, "missing required configuration key: {key}")
            }
            ConfigError::WrongType {
                key,
                expected,
                actual,
            } => write!(
                f,
                "wrong type for key '{key}': expected {expected}, got {actual}"
            ),
            ConfigError::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ConfigError {}

// ---------------------------------------------------------------------------
// ConfigValue
// ---------------------------------------------------------------------------

/// A dynamically-typed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ConfigValue>),
    Map(HashMap<String, ConfigValue>),
    Null,
}

impl ConfigValue {
    /// Try to extract a `&str` from a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to extract an `i64` from an `Integer` variant.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract an `f64` from a `Float` variant.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract a `bool` from a `Bool` variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns `true` if the value is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, ConfigValue::Null)
    }

    /// Convert the value to a `serde_json::Value`.
    pub fn to_json(&self) -> Value {
        match self {
            ConfigValue::String(s) => Value::String(s.clone()),
            ConfigValue::Integer(n) => Value::Number((*n).into()),
            ConfigValue::Float(f) => {
                Value::Number(serde_json::Number::from_f64(*f).unwrap_or_else(|| 0i64.into()))
            }
            ConfigValue::Bool(b) => Value::Bool(*b),
            ConfigValue::List(items) => Value::Array(items.iter().map(|v| v.to_json()).collect()),
            ConfigValue::Map(map) => {
                let obj: serde_json::Map<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.to_json())).collect();
                Value::Object(obj)
            }
            ConfigValue::Null => Value::Null,
        }
    }

    /// Return the [`ConfigType`] that describes this value.
    pub fn config_type(&self) -> ConfigType {
        match self {
            ConfigValue::String(_) => ConfigType::String,
            ConfigValue::Integer(_) => ConfigType::Integer,
            ConfigValue::Float(_) => ConfigType::Float,
            ConfigValue::Bool(_) => ConfigType::Bool,
            ConfigValue::List(_) => ConfigType::List,
            ConfigValue::Map(_) => ConfigType::Map,
            ConfigValue::Null => ConfigType::String, // treat null as string for type purposes
        }
    }
}

// --- From impls --------------------------------------------------------

impl From<String> for ConfigValue {
    fn from(s: String) -> Self {
        ConfigValue::String(s)
    }
}

impl From<&str> for ConfigValue {
    fn from(s: &str) -> Self {
        ConfigValue::String(s.to_owned())
    }
}

impl From<i64> for ConfigValue {
    fn from(n: i64) -> Self {
        ConfigValue::Integer(n)
    }
}

impl From<f64> for ConfigValue {
    fn from(f: f64) -> Self {
        ConfigValue::Float(f)
    }
}

impl From<bool> for ConfigValue {
    fn from(b: bool) -> Self {
        ConfigValue::Bool(b)
    }
}

// ---------------------------------------------------------------------------
// ConfigStore
// ---------------------------------------------------------------------------

/// A hierarchical key-value configuration store.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigStore {
    data: HashMap<String, ConfigValue>,
}

impl ConfigStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert or overwrite a value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<ConfigValue>) {
        self.data.insert(key.into(), value.into());
    }

    /// Look up a value by key.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.data.get(key)
    }

    /// Return the value for `key`, or `default` if not present.
    pub fn get_or(&self, key: &str, default: ConfigValue) -> ConfigValue {
        self.data.get(key).cloned().unwrap_or(default)
    }

    /// Convenience typed getter for `String`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(|v| v.as_str())
    }

    /// Convenience typed getter for `i64`.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.data.get(key).and_then(|v| v.as_i64())
    }

    /// Convenience typed getter for `f64`.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.data.get(key).and_then(|v| v.as_f64())
    }

    /// Convenience typed getter for `bool`.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.data.get(key).and_then(|v| v.as_bool())
    }

    /// Remove a key, returning the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        self.data.remove(key)
    }

    /// Return an iterator over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Merge `other` into `self`. Keys in `other` override keys in `self`.
    pub fn merge(&mut self, other: ConfigStore) {
        for (k, v) in other.data {
            self.data.insert(k, v);
        }
    }

    /// Create a scoped, read-only view of this store with the given prefix.
    ///
    /// For a store with keys `"db.host"`, `"db.port"`, calling
    /// `with_prefix("db")` yields a view where `get("host")` returns the
    /// value for `"db.host"`.
    pub fn with_prefix(&self, prefix: &str) -> ConfigView<'_> {
        ConfigView {
            store: self,
            prefix: format!("{}.", prefix),
        }
    }
}

impl Default for ConfigStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConfigView
// ---------------------------------------------------------------------------

/// A read-only, prefix-scoped view into a [`ConfigStore`].
#[derive(Debug)]
pub struct ConfigView<'a> {
    store: &'a ConfigStore,
    prefix: String,
}

impl<'a> ConfigView<'a> {
    /// Look up a value by key (relative to the prefix).
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        let full = format!("{}{}", self.prefix, key);
        self.store.get(&full)
    }

    /// Typed getter for `String`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    /// Typed getter for `i64`.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    /// Return all keys (with prefix stripped).
    pub fn keys(&self) -> Vec<String> {
        self.store
            .data
            .keys()
            .filter_map(|k| k.strip_prefix(&self.prefix).map(|s| s.to_owned()))
            .collect()
    }

    /// Materialise this view into an independent [`ConfigStore`].
    pub fn to_store(&self) -> ConfigStore {
        let mut store = ConfigStore::new();
        for (k, v) in &self.store.data {
            if let Some(stripped) = k.strip_prefix(&self.prefix) {
                store.data.insert(stripped.to_owned(), v.clone());
            }
        }
        store
    }
}

// ---------------------------------------------------------------------------
// ConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`ConfigStore`].
pub struct ConfigBuilder {
    data: HashMap<String, ConfigValue>,
}

impl ConfigBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Set a value, overwriting any previous value for the key.
    pub fn set(mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// Set a value only if the key is not already present.
    pub fn set_default(mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Self {
        let k = key.into();
        self.data.entry(k).or_insert_with(|| value.into());
        self
    }

    /// Read environment variables with the given prefix and populate the
    /// builder. For a prefix `"RUSTCHAIN"`, `RUSTCHAIN_API_KEY` becomes key
    /// `"api_key"` with a string value.
    pub fn from_env(mut self, prefix: &str) -> Self {
        let prefix_upper = format!("{}_", prefix.to_uppercase());
        for (key, value) in std::env::vars() {
            if let Some(suffix) = key.strip_prefix(&prefix_upper) {
                let config_key = suffix.to_lowercase();
                self.data.insert(config_key, ConfigValue::String(value));
            }
        }
        self
    }

    /// Load values from a JSON object. Non-object types are ignored.
    pub fn from_json(mut self, value: &Value) -> Self {
        if let Value::Object(map) = value {
            for (k, v) in map {
                self.data.insert(k.clone(), json_to_config_value(v));
            }
        }
        self
    }

    /// Consume the builder and produce a [`ConfigStore`].
    pub fn build(self) -> ConfigStore {
        ConfigStore { data: self.data }
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively convert a `serde_json::Value` to a `ConfigValue`.
fn json_to_config_value(v: &Value) -> ConfigValue {
    match v {
        Value::String(s) => ConfigValue::String(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                ConfigValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                ConfigValue::Float(f)
            } else {
                ConfigValue::Null
            }
        }
        Value::Bool(b) => ConfigValue::Bool(*b),
        Value::Array(arr) => ConfigValue::List(arr.iter().map(json_to_config_value).collect()),
        Value::Object(map) => {
            let m: HashMap<String, ConfigValue> = map
                .iter()
                .map(|(k, val)| (k.clone(), json_to_config_value(val)))
                .collect();
            ConfigValue::Map(m)
        }
        Value::Null => ConfigValue::Null,
    }
}

// ---------------------------------------------------------------------------
// ConfigValidator
// ---------------------------------------------------------------------------

/// Rule kinds used internally by [`ConfigValidator`].
enum ValidationRule {
    Required,
    Optional(ConfigValue),
    Typed(ConfigType),
}

/// Validates a [`ConfigStore`] against a declarative schema.
pub struct ConfigValidator {
    rules: Vec<(String, ValidationRule)>,
}

impl ConfigValidator {
    /// Create a new, empty validator.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Mark a key as required.
    pub fn required(mut self, key: impl Into<String>) -> Self {
        self.rules.push((key.into(), ValidationRule::Required));
        self
    }

    /// Mark a key as optional with a default value. If the key is absent in
    /// the store being validated, validation still passes.
    pub fn optional(mut self, key: impl Into<String>, default: impl Into<ConfigValue>) -> Self {
        self.rules
            .push((key.into(), ValidationRule::Optional(default.into())));
        self
    }

    /// Require that a key, if present, has the given type.
    pub fn typed(mut self, key: impl Into<String>, expected_type: ConfigType) -> Self {
        self.rules
            .push((key.into(), ValidationRule::Typed(expected_type)));
        self
    }

    /// Validate the store, returning all errors found.
    pub fn validate(&self, store: &ConfigStore) -> Result<(), Vec<ConfigError>> {
        let mut errors = Vec::new();
        for (key, rule) in &self.rules {
            match rule {
                ValidationRule::Required => {
                    if store.get(key).is_none() {
                        errors.push(ConfigError::MissingRequired { key: key.clone() });
                    }
                }
                ValidationRule::Optional(_default) => {
                    // Absence is fine; nothing to check.
                }
                ValidationRule::Typed(expected) => {
                    if let Some(val) = store.get(key) {
                        if !val.is_null() {
                            let actual = val.config_type();
                            if actual != *expected {
                                errors.push(ConfigError::WrongType {
                                    key: key.clone(),
                                    expected: *expected,
                                    actual,
                                });
                            }
                        }
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConfigSnapshot / ConfigChange
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a [`ConfigStore`] for diffing.
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    data: HashMap<String, ConfigValue>,
}

impl ConfigSnapshot {
    /// Capture the current state of a store.
    pub fn capture(store: &ConfigStore) -> Self {
        Self {
            data: store.data.clone(),
        }
    }

    /// Compute the differences between two snapshots.
    pub fn diff(&self, other: &ConfigSnapshot) -> Vec<ConfigChange> {
        let mut changes = Vec::new();

        // Keys in other but not in self → Added
        for (k, v) in &other.data {
            if !self.data.contains_key(k) {
                changes.push(ConfigChange::Added {
                    key: k.clone(),
                    value: v.clone(),
                });
            }
        }

        // Keys in self but not in other → Removed
        for (k, v) in &self.data {
            if !other.data.contains_key(k) {
                changes.push(ConfigChange::Removed {
                    key: k.clone(),
                    value: v.clone(),
                });
            }
        }

        // Keys in both but with different values → Modified
        for (k, old) in &self.data {
            if let Some(new) = other.data.get(k) {
                if old != new {
                    changes.push(ConfigChange::Modified {
                        key: k.clone(),
                        old: old.clone(),
                        new: new.clone(),
                    });
                }
            }
        }

        changes
    }
}

/// Describes a single difference between two [`ConfigSnapshot`]s.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigChange {
    Added {
        key: String,
        value: ConfigValue,
    },
    Removed {
        key: String,
        value: ConfigValue,
    },
    Modified {
        key: String,
        old: ConfigValue,
        new: ConfigValue,
    },
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- ConfigValue basic accessors ---

    #[test]
    fn config_value_as_str() {
        let v = ConfigValue::String("hello".into());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(ConfigValue::Integer(1).as_str(), None);
    }

    #[test]
    fn config_value_as_i64() {
        let v = ConfigValue::Integer(42);
        assert_eq!(v.as_i64(), Some(42));
        assert_eq!(ConfigValue::String("x".into()).as_i64(), None);
    }

    #[test]
    fn config_value_as_f64() {
        let v = ConfigValue::Float(3.14);
        assert_eq!(v.as_f64(), Some(3.14));
        assert_eq!(ConfigValue::Bool(true).as_f64(), None);
    }

    #[test]
    fn config_value_as_bool() {
        assert_eq!(ConfigValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConfigValue::Bool(false).as_bool(), Some(false));
        assert_eq!(ConfigValue::Null.as_bool(), None);
    }

    #[test]
    fn config_value_is_null() {
        assert!(ConfigValue::Null.is_null());
        assert!(!ConfigValue::Bool(false).is_null());
    }

    #[test]
    fn config_value_to_json_string() {
        let v = ConfigValue::String("abc".into());
        assert_eq!(v.to_json(), json!("abc"));
    }

    #[test]
    fn config_value_to_json_integer() {
        assert_eq!(ConfigValue::Integer(7).to_json(), json!(7));
    }

    #[test]
    fn config_value_to_json_float() {
        assert_eq!(ConfigValue::Float(1.5).to_json(), json!(1.5));
    }

    #[test]
    fn config_value_to_json_bool() {
        assert_eq!(ConfigValue::Bool(true).to_json(), json!(true));
    }

    #[test]
    fn config_value_to_json_null() {
        assert_eq!(ConfigValue::Null.to_json(), Value::Null);
    }

    #[test]
    fn config_value_to_json_list() {
        let v = ConfigValue::List(vec![ConfigValue::Integer(1), ConfigValue::Integer(2)]);
        assert_eq!(v.to_json(), json!([1, 2]));
    }

    #[test]
    fn config_value_to_json_map() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), ConfigValue::Integer(1));
        let v = ConfigValue::Map(m);
        assert_eq!(v.to_json(), json!({"a": 1}));
    }

    // --- From impls ---

    #[test]
    fn from_string() {
        let v: ConfigValue = String::from("hello").into();
        assert_eq!(v, ConfigValue::String("hello".into()));
    }

    #[test]
    fn from_str_ref() {
        let v: ConfigValue = "world".into();
        assert_eq!(v, ConfigValue::String("world".into()));
    }

    #[test]
    fn from_i64() {
        let v: ConfigValue = 100i64.into();
        assert_eq!(v, ConfigValue::Integer(100));
    }

    #[test]
    fn from_f64() {
        let v: ConfigValue = 2.719f64.into();
        assert_eq!(v, ConfigValue::Float(2.719));
    }

    #[test]
    fn from_bool() {
        let v: ConfigValue = true.into();
        assert_eq!(v, ConfigValue::Bool(true));
    }

    // --- ConfigValue::config_type ---

    #[test]
    fn config_type_string() {
        assert_eq!(
            ConfigValue::String("x".into()).config_type(),
            ConfigType::String
        );
    }

    #[test]
    fn config_type_integer() {
        assert_eq!(ConfigValue::Integer(0).config_type(), ConfigType::Integer);
    }

    #[test]
    fn config_type_float() {
        assert_eq!(ConfigValue::Float(0.0).config_type(), ConfigType::Float);
    }

    #[test]
    fn config_type_bool() {
        assert_eq!(ConfigValue::Bool(false).config_type(), ConfigType::Bool);
    }

    #[test]
    fn config_type_list() {
        assert_eq!(ConfigValue::List(vec![]).config_type(), ConfigType::List);
    }

    #[test]
    fn config_type_map() {
        assert_eq!(
            ConfigValue::Map(HashMap::new()).config_type(),
            ConfigType::Map
        );
    }

    // --- ConfigStore ---

    #[test]
    fn store_set_and_get() {
        let mut store = ConfigStore::new();
        store.set("key", "value");
        assert_eq!(store.get("key"), Some(&ConfigValue::String("value".into())));
    }

    #[test]
    fn store_get_missing() {
        let store = ConfigStore::new();
        assert_eq!(store.get("nope"), None);
    }

    #[test]
    fn store_get_or_present() {
        let mut store = ConfigStore::new();
        store.set("k", 10i64);
        assert_eq!(
            store.get_or("k", ConfigValue::Integer(99)),
            ConfigValue::Integer(10)
        );
    }

    #[test]
    fn store_get_or_missing() {
        let store = ConfigStore::new();
        assert_eq!(
            store.get_or("k", ConfigValue::Integer(99)),
            ConfigValue::Integer(99)
        );
    }

    #[test]
    fn store_typed_getters() {
        let mut store = ConfigStore::new();
        store.set("s", "hello");
        store.set("i", 42i64);
        store.set("f", 1.5f64);
        store.set("b", true);

        assert_eq!(store.get_str("s"), Some("hello"));
        assert_eq!(store.get_int("i"), Some(42));
        assert_eq!(store.get_float("f"), Some(1.5));
        assert_eq!(store.get_bool("b"), Some(true));
    }

    #[test]
    fn store_typed_getters_wrong_type() {
        let mut store = ConfigStore::new();
        store.set("s", "hello");
        assert_eq!(store.get_int("s"), None);
        assert_eq!(store.get_float("s"), None);
        assert_eq!(store.get_bool("s"), None);
    }

    #[test]
    fn store_remove() {
        let mut store = ConfigStore::new();
        store.set("k", "v");
        assert!(store.remove("k").is_some());
        assert!(store.get("k").is_none());
    }

    #[test]
    fn store_remove_missing() {
        let mut store = ConfigStore::new();
        assert!(store.remove("nope").is_none());
    }

    #[test]
    fn store_keys_len_empty() {
        let store = ConfigStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.keys().count(), 0);
    }

    #[test]
    fn store_keys_len() {
        let mut store = ConfigStore::new();
        store.set("a", "1");
        store.set("b", "2");
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn store_merge_overrides() {
        let mut base = ConfigStore::new();
        base.set("a", "original");
        base.set("b", "keep");

        let mut overlay = ConfigStore::new();
        overlay.set("a", "overridden");
        overlay.set("c", "new");

        base.merge(overlay);
        assert_eq!(base.get_str("a"), Some("overridden"));
        assert_eq!(base.get_str("b"), Some("keep"));
        assert_eq!(base.get_str("c"), Some("new"));
    }

    #[test]
    fn store_default_impl() {
        let store = ConfigStore::default();
        assert!(store.is_empty());
    }

    // --- ConfigView ---

    #[test]
    fn view_get() {
        let mut store = ConfigStore::new();
        store.set("db.host", "localhost");
        store.set("db.port", 5432i64);
        store.set("api.key", "secret");

        let view = store.with_prefix("db");
        assert_eq!(
            view.get("host"),
            Some(&ConfigValue::String("localhost".into()))
        );
        assert_eq!(view.get("port"), Some(&ConfigValue::Integer(5432)));
        assert_eq!(view.get("key"), None); // not under "db."
    }

    #[test]
    fn view_get_str() {
        let mut store = ConfigStore::new();
        store.set("app.name", "test");
        let view = store.with_prefix("app");
        assert_eq!(view.get_str("name"), Some("test"));
    }

    #[test]
    fn view_get_int() {
        let mut store = ConfigStore::new();
        store.set("app.port", 8080i64);
        let view = store.with_prefix("app");
        assert_eq!(view.get_int("port"), Some(8080));
    }

    #[test]
    fn view_keys() {
        let mut store = ConfigStore::new();
        store.set("db.host", "localhost");
        store.set("db.port", 5432i64);
        store.set("api.key", "secret");

        let view = store.with_prefix("db");
        let mut keys = view.keys();
        keys.sort();
        assert_eq!(keys, vec!["host", "port"]);
    }

    #[test]
    fn view_to_store() {
        let mut store = ConfigStore::new();
        store.set("db.host", "localhost");
        store.set("db.port", 5432i64);
        store.set("other", "ignored");

        let view = store.with_prefix("db");
        let sub = view.to_store();
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.get_str("host"), Some("localhost"));
        assert_eq!(sub.get_int("port"), Some(5432));
    }

    // --- ConfigBuilder ---

    #[test]
    fn builder_set() {
        let store = ConfigBuilder::new().set("key", "val").build();
        assert_eq!(store.get_str("key"), Some("val"));
    }

    #[test]
    fn builder_set_overwrites() {
        let store = ConfigBuilder::new()
            .set("k", "first")
            .set("k", "second")
            .build();
        assert_eq!(store.get_str("k"), Some("second"));
    }

    #[test]
    fn builder_set_default_no_overwrite() {
        let store = ConfigBuilder::new()
            .set("k", "existing")
            .set_default("k", "default")
            .build();
        assert_eq!(store.get_str("k"), Some("existing"));
    }

    #[test]
    fn builder_set_default_when_absent() {
        let store = ConfigBuilder::new().set_default("k", "default").build();
        assert_eq!(store.get_str("k"), Some("default"));
    }

    #[test]
    fn builder_from_json() {
        let j = json!({
            "host": "localhost",
            "port": 5432,
            "debug": true,
            "ratio": 0.75,
            "tags": ["a", "b"],
            "meta": {"x": 1}
        });
        let store = ConfigBuilder::new().from_json(&j).build();
        assert_eq!(store.get_str("host"), Some("localhost"));
        assert_eq!(store.get_int("port"), Some(5432));
        assert_eq!(store.get_bool("debug"), Some(true));
        assert_eq!(store.get_float("ratio"), Some(0.75));
        assert!(matches!(store.get("tags"), Some(ConfigValue::List(_))));
        assert!(matches!(store.get("meta"), Some(ConfigValue::Map(_))));
    }

    #[test]
    fn builder_from_json_non_object_ignored() {
        let store = ConfigBuilder::new()
            .from_json(&json!("not an object"))
            .build();
        assert!(store.is_empty());
    }

    #[test]
    fn builder_from_json_null_value() {
        let j = json!({"nothing": null});
        let store = ConfigBuilder::new().from_json(&j).build();
        assert!(store.get("nothing").unwrap().is_null());
    }

    #[test]
    fn builder_from_env() {
        // Set a temporary env var, build, then clean up.
        std::env::set_var("TESTCFG_MY_VAR", "hello");
        let store = ConfigBuilder::new().from_env("TESTCFG").build();
        std::env::remove_var("TESTCFG_MY_VAR");
        assert_eq!(store.get_str("my_var"), Some("hello"));
    }

    #[test]
    fn builder_from_env_ignores_other_prefixes() {
        std::env::set_var("OTHERCFG_X", "y");
        let store = ConfigBuilder::new().from_env("TESTCFG2").build();
        std::env::remove_var("OTHERCFG_X");
        assert!(store.get("x").is_none());
    }

    #[test]
    fn builder_default_impl() {
        let store = ConfigBuilder::default().build();
        assert!(store.is_empty());
    }

    // --- ConfigValidator ---

    #[test]
    fn validator_required_present() {
        let mut store = ConfigStore::new();
        store.set("api_key", "secret");
        let result = ConfigValidator::new().required("api_key").validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_required_missing() {
        let store = ConfigStore::new();
        let result = ConfigValidator::new().required("api_key").validate(&store);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(
            errs[0],
            ConfigError::MissingRequired {
                key: "api_key".into()
            }
        );
    }

    #[test]
    fn validator_multiple_required_missing() {
        let store = ConfigStore::new();
        let result = ConfigValidator::new()
            .required("a")
            .required("b")
            .validate(&store);
        let errs = result.unwrap_err();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn validator_typed_correct() {
        let mut store = ConfigStore::new();
        store.set("port", 8080i64);
        let result = ConfigValidator::new()
            .typed("port", ConfigType::Integer)
            .validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_typed_wrong() {
        let mut store = ConfigStore::new();
        store.set("port", "not a number");
        let result = ConfigValidator::new()
            .typed("port", ConfigType::Integer)
            .validate(&store);
        let errs = result.unwrap_err();
        assert_eq!(errs.len(), 1);
        match &errs[0] {
            ConfigError::WrongType {
                key,
                expected,
                actual,
            } => {
                assert_eq!(key, "port");
                assert_eq!(*expected, ConfigType::Integer);
                assert_eq!(*actual, ConfigType::String);
            }
            _ => panic!("expected WrongType error"),
        }
    }

    #[test]
    fn validator_typed_absent_is_ok() {
        let store = ConfigStore::new();
        let result = ConfigValidator::new()
            .typed("missing", ConfigType::Integer)
            .validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_typed_null_is_ok() {
        let mut store = ConfigStore::new();
        store.set("k", ConfigValue::Null);
        let result = ConfigValidator::new()
            .typed("k", ConfigType::Integer)
            .validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_optional_missing_is_ok() {
        let store = ConfigStore::new();
        let result = ConfigValidator::new()
            .optional("opt", "default_val")
            .validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_optional_present_is_ok() {
        let mut store = ConfigStore::new();
        store.set("opt", "custom");
        let result = ConfigValidator::new()
            .optional("opt", "default_val")
            .validate(&store);
        assert!(result.is_ok());
    }

    #[test]
    fn validator_default_impl() {
        let v = ConfigValidator::default();
        assert!(v.validate(&ConfigStore::new()).is_ok());
    }

    // --- ConfigError Display ---

    #[test]
    fn config_error_display_missing() {
        let e = ConfigError::MissingRequired {
            key: "api_key".into(),
        };
        assert_eq!(e.to_string(), "missing required configuration key: api_key");
    }

    #[test]
    fn config_error_display_wrong_type() {
        let e = ConfigError::WrongType {
            key: "port".into(),
            expected: ConfigType::Integer,
            actual: ConfigType::String,
        };
        assert_eq!(
            e.to_string(),
            "wrong type for key 'port': expected Integer, got String"
        );
    }

    #[test]
    fn config_error_display_custom() {
        let e = ConfigError::Custom("something went wrong".into());
        assert_eq!(e.to_string(), "something went wrong");
    }

    // --- ConfigSnapshot / ConfigChange ---

    #[test]
    fn snapshot_no_changes() {
        let mut store = ConfigStore::new();
        store.set("k", "v");
        let snap1 = ConfigSnapshot::capture(&store);
        let snap2 = ConfigSnapshot::capture(&store);
        assert!(snap1.diff(&snap2).is_empty());
    }

    #[test]
    fn snapshot_added() {
        let store1 = ConfigStore::new();
        let mut store2 = ConfigStore::new();
        store2.set("new_key", "new_val");

        let snap1 = ConfigSnapshot::capture(&store1);
        let snap2 = ConfigSnapshot::capture(&store2);
        let changes = snap1.diff(&snap2);
        assert_eq!(changes.len(), 1);
        assert!(matches!(&changes[0], ConfigChange::Added { key, .. } if key == "new_key"));
    }

    #[test]
    fn snapshot_removed() {
        let mut store1 = ConfigStore::new();
        store1.set("old_key", "old_val");
        let store2 = ConfigStore::new();

        let snap1 = ConfigSnapshot::capture(&store1);
        let snap2 = ConfigSnapshot::capture(&store2);
        let changes = snap1.diff(&snap2);
        assert_eq!(changes.len(), 1);
        assert!(matches!(&changes[0], ConfigChange::Removed { key, .. } if key == "old_key"));
    }

    #[test]
    fn snapshot_modified() {
        let mut store1 = ConfigStore::new();
        store1.set("k", "old");
        let mut store2 = ConfigStore::new();
        store2.set("k", "new");

        let snap1 = ConfigSnapshot::capture(&store1);
        let snap2 = ConfigSnapshot::capture(&store2);
        let changes = snap1.diff(&snap2);
        assert_eq!(changes.len(), 1);
        match &changes[0] {
            ConfigChange::Modified { key, old, new } => {
                assert_eq!(key, "k");
                assert_eq!(old, &ConfigValue::String("old".into()));
                assert_eq!(new, &ConfigValue::String("new".into()));
            }
            _ => panic!("expected Modified"),
        }
    }

    #[test]
    fn snapshot_mixed_changes() {
        let mut store1 = ConfigStore::new();
        store1.set("keep", "same");
        store1.set("change", "old");
        store1.set("remove_me", "bye");

        let mut store2 = ConfigStore::new();
        store2.set("keep", "same");
        store2.set("change", "new");
        store2.set("added", "hi");

        let snap1 = ConfigSnapshot::capture(&store1);
        let snap2 = ConfigSnapshot::capture(&store2);
        let changes = snap1.diff(&snap2);
        assert_eq!(changes.len(), 3); // added, removed, modified
    }

    // --- ConfigType Display ---

    #[test]
    fn config_type_display() {
        assert_eq!(ConfigType::String.to_string(), "String");
        assert_eq!(ConfigType::Integer.to_string(), "Integer");
        assert_eq!(ConfigType::Float.to_string(), "Float");
        assert_eq!(ConfigType::Bool.to_string(), "Bool");
        assert_eq!(ConfigType::List.to_string(), "List");
        assert_eq!(ConfigType::Map.to_string(), "Map");
    }

    // --- Edge cases and integration ---

    #[test]
    fn store_overwrite_different_type() {
        let mut store = ConfigStore::new();
        store.set("k", "string_val");
        store.set("k", 42i64);
        assert_eq!(store.get_int("k"), Some(42));
        assert_eq!(store.get_str("k"), None);
    }

    #[test]
    fn builder_chain_all() {
        let j = json!({"from_json": true});
        let store = ConfigBuilder::new()
            .set("explicit", "yes")
            .set_default("explicit", "no")
            .set_default("defaulted", "applied")
            .from_json(&j)
            .build();

        assert_eq!(store.get_str("explicit"), Some("yes"));
        assert_eq!(store.get_str("defaulted"), Some("applied"));
        assert_eq!(store.get_bool("from_json"), Some(true));
    }

    #[test]
    fn validator_combined_rules() {
        let mut store = ConfigStore::new();
        store.set("port", "not_int");

        let result = ConfigValidator::new()
            .required("api_key")
            .typed("port", ConfigType::Integer)
            .validate(&store);

        let errs = result.unwrap_err();
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn nested_json_to_config_value() {
        let j = json!({
            "nested": {
                "list": [1, "two", null]
            }
        });
        let store = ConfigBuilder::new().from_json(&j).build();
        if let Some(ConfigValue::Map(m)) = store.get("nested") {
            if let Some(ConfigValue::List(l)) = m.get("list") {
                assert_eq!(l.len(), 3);
                assert_eq!(l[0], ConfigValue::Integer(1));
                assert_eq!(l[1], ConfigValue::String("two".into()));
                assert!(l[2].is_null());
            } else {
                panic!("expected list in nested map");
            }
        } else {
            panic!("expected map for nested");
        }
    }

    #[test]
    fn view_empty_prefix() {
        let mut store = ConfigStore::new();
        store.set("x.y", "val");
        let view = store.with_prefix("x");
        assert_eq!(view.get_str("y"), Some("val"));
    }

    #[test]
    fn store_merge_empty() {
        let mut store = ConfigStore::new();
        store.set("a", "b");
        store.merge(ConfigStore::new());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn config_value_equality() {
        assert_eq!(ConfigValue::Integer(5), ConfigValue::Integer(5));
        assert_ne!(ConfigValue::Integer(5), ConfigValue::Integer(6));
        assert_ne!(ConfigValue::Integer(5), ConfigValue::Float(5.0));
    }

    #[test]
    fn config_value_clone() {
        let v = ConfigValue::List(vec![ConfigValue::String("a".into())]);
        let v2 = v.clone();
        assert_eq!(v, v2);
    }
}
