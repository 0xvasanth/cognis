//! Serializable trait and serialization helpers.
//!
//! Mirrors Python `langchain_core.load.serializable`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Base structure for all serialized forms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseSerialized {
    /// Schema version (always 1).
    pub lc: u32,
    /// Fully qualified path to the type: `[namespace..., class_name]`.
    pub id: Vec<String>,
    /// Optional human-readable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Serialized form for objects that can be reconstructed from kwargs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedConstructor {
    #[serde(flatten)]
    pub base: BaseSerialized,
    /// Always `"constructor"`.
    #[serde(rename = "type")]
    pub type_tag: String,
    /// Keyword arguments needed to reconstruct the object.
    pub kwargs: HashMap<String, Value>,
}

/// Serialized form for secret values (API keys, tokens, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSecret {
    #[serde(flatten)]
    pub base: BaseSerialized,
    /// Always `"secret"`.
    #[serde(rename = "type")]
    pub type_tag: String,
}

/// Serialized form for objects that don't support serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNotImplemented {
    #[serde(flatten)]
    pub base: BaseSerialized,
    /// Always `"not_implemented"`.
    #[serde(rename = "type")]
    pub type_tag: String,
    /// Debug representation of the object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repr: Option<String>,
}

/// Trait for objects that can be serialized to the LangChain constructor format.
///
/// Implementors declare their namespace and secrets, and `to_json()`
/// produces a `SerializedConstructor` that can be deserialized back
/// into the original object.
pub trait Serializable: Send + Sync {
    /// Whether this class supports LangChain serialization.
    fn is_lc_serializable(&self) -> bool {
        false
    }

    /// The namespace path for this class (e.g., `["langchain", "llms", "openai"]`).
    fn get_lc_namespace(&self) -> Vec<String>;

    /// Map of secret attribute names to their environment variable names.
    fn lc_secrets(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Additional attributes to include in serialization beyond model fields.
    fn lc_attributes(&self) -> HashMap<String, Value> {
        HashMap::new()
    }

    /// Unique identifier: `[namespace..., class_name]`.
    fn lc_id(&self) -> Vec<String>;

    /// Serialize this object to a constructor-based JSON representation.
    fn to_json(&self) -> Value;
}

/// Serialize a `Serializable` object to a `Value` (dict-like).
pub fn dumpd(obj: &dyn Serializable) -> Value {
    obj.to_json()
}

/// Serialize a `Serializable` object to a JSON string.
pub fn dumps(obj: &dyn Serializable, pretty: bool) -> String {
    let value = obj.to_json();
    if pretty {
        serde_json::to_string_pretty(&value).unwrap_or_default()
    } else {
        serde_json::to_string(&value).unwrap_or_default()
    }
}

/// Create a `SerializedNotImplemented` for any object.
pub fn to_json_not_implemented(type_name: &str, repr: &str, namespace: Vec<String>) -> Value {
    let mut id = namespace;
    id.push(type_name.to_string());

    serde_json::json!({
        "lc": 1,
        "type": "not_implemented",
        "id": id,
        "repr": repr,
    })
}

/// Create a `SerializedSecret` marker.
pub fn make_serialized_secret(secret_id: Vec<String>) -> Value {
    serde_json::json!({
        "lc": 1,
        "type": "secret",
        "id": secret_id,
    })
}
