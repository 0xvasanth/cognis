//! Serialization utilities for dumping objects.
//!
//! Mirrors Python `langchain_core.load.dump`.

use serde_json::Value;

/// Serialize any serde-compatible value to a JSON string.
pub fn dumps(obj: &Value, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(obj).unwrap_or_default()
    } else {
        serde_json::to_string(obj).unwrap_or_default()
    }
}

/// Serialize any serde-compatible value to a Value (identity for Value, conversion for others).
pub fn dumpd(obj: &Value) -> Value {
    obj.clone()
}

/// Serialize a `Serializable` object to a Value.
pub fn dumpd_serializable(obj: &dyn super::serializable::Serializable) -> Value {
    obj.to_json()
}

/// Serialize a `Serializable` object to a JSON string.
pub fn dumps_serializable(obj: &dyn super::serializable::Serializable, pretty: bool) -> String {
    let value = obj.to_json();
    if pretty {
        serde_json::to_string_pretty(&value).unwrap_or_default()
    } else {
        serde_json::to_string(&value).unwrap_or_default()
    }
}
