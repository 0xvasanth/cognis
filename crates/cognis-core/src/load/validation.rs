//! Validation utilities for LangChain serialization.
//!
//! Provides escape-based protection against injection attacks in serialized
//! objects. The approach uses an allowlist design: only dicts explicitly
//! produced by `Serializable::to_json()` are treated as LC objects during
//! deserialization.
//!
//! # How escaping works
//!
//! During serialization, plain dicts (user data) that contain an `"lc"` key
//! are wrapped:
//!
//! ```json
//! {"lc": 1, "...": "..."}
//! ```
//!
//! becomes:
//!
//! ```json
//! {"__lc_escaped__": {"lc": 1, "...": "..."}}
//! ```
//!
//! During deserialization, escaped dicts are unwrapped and returned as plain
//! dicts, **not** instantiated as LC objects.

use serde_json::{Map, Value};

/// Sentinel key used to mark escaped user dicts during serialization.
///
/// When a plain dict contains an `"lc"` key (which could be confused with LC
/// objects), we wrap it as `{"__lc_escaped__": {...original...}}`.
pub const LC_ESCAPED_KEY: &str = "__lc_escaped__";

/// Check if a JSON object needs escaping to prevent confusion with LC objects.
///
/// A dict needs escaping if:
/// 1. It has an `"lc"` key (could be confused with LC serialization format).
/// 2. It has only the escape key (would be mistaken for an already-escaped dict).
pub fn needs_escaping(obj: &Map<String, Value>) -> bool {
    obj.contains_key("lc") || (obj.len() == 1 && obj.contains_key(LC_ESCAPED_KEY))
}

/// Wrap a JSON object in the escape marker.
///
/// # Example
///
/// ```
/// use serde_json::{json, Value, Map};
/// use cognis_core::load::validation::escape_dict;
///
/// let mut m = Map::new();
/// m.insert("key".into(), json!("value"));
/// let escaped = escape_dict(m.clone());
/// assert_eq!(escaped.len(), 1);
/// assert!(escaped.contains_key("__lc_escaped__"));
/// ```
pub fn escape_dict(obj: Map<String, Value>) -> Map<String, Value> {
    let mut result = Map::new();
    result.insert(LC_ESCAPED_KEY.to_string(), Value::Object(obj));
    result
}

/// Check if a JSON object is an escaped user dict.
///
/// Returns `true` when the object has exactly one key and that key is
/// [`LC_ESCAPED_KEY`].
pub fn is_escaped_dict(obj: &Map<String, Value>) -> bool {
    obj.len() == 1 && obj.contains_key(LC_ESCAPED_KEY)
}

/// Check if a JSON value is a LangChain secret marker.
///
/// A secret marker is a JSON object with exactly three keys:
/// `"lc"` (value 1), `"type"` (value `"secret"`), and `"id"`.
pub fn is_lc_secret(obj: &Value) -> bool {
    match obj {
        Value::Object(map) => {
            map.len() == 3
                && map.get("lc") == Some(&Value::Number(1.into()))
                && map.get("type") == Some(&Value::String("secret".to_string()))
                && map.contains_key("id")
        }
        _ => false,
    }
}

/// Serialize a value, producing a `Value::Null` sentinel for non-serializable
/// values, and wrapping user dicts that contain `"lc"` keys with the escape
/// marker to prevent injection attacks.
///
/// This function is called recursively on kwarg values. In the Rust port we do
/// not have a `Serializable` trait object to detect, so all `Value::Object`
/// inputs are treated as plain user data. Objects that originate from
/// `Serializable::to_json()` should be handled *before* calling this function
/// (mirroring the Python `_serialize_lc_object` path).
pub fn serialize_value(obj: &Value) -> Value {
    match obj {
        Value::Object(map) => {
            // Check JSON-key validity: all keys are already strings in serde_json.
            if needs_escaping(map) {
                return Value::Object(escape_dict(map.clone()));
            }
            // Safe dict — recurse into values.
            let new_map: Map<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), serialize_value(v)))
                .collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(serialize_value).collect()),
        // Primitives are JSON-safe and returned as-is.
        Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => obj.clone(),
    }
}

/// Unescape a value, processing escape markers in dict values and lists.
///
/// When an escaped dict is encountered (`{"__lc_escaped__": ...}`), it is
/// unwrapped and the contents are returned **as-is** (no further processing).
/// The contents represent user data that should not be modified.
///
/// For regular dicts and lists, the function recurses to find any nested
/// escape markers.
pub fn unescape_value(obj: &Value) -> Value {
    match obj {
        Value::Object(map) => {
            if is_escaped_dict(map) {
                // Unwrap and return the user data as-is.
                return map.get(LC_ESCAPED_KEY).cloned().unwrap_or(Value::Null);
            }
            // Regular dict — recurse into values.
            let new_map: Map<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), unescape_value(v)))
                .collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(unescape_value).collect()),
        // Primitives pass through unchanged.
        _ => obj.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- needs_escaping ----

    #[test]
    fn test_needs_escaping_with_lc_key() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({"lc": 1, "type": "constructor"})).unwrap();
        assert!(needs_escaping(&map));
    }

    #[test]
    fn test_needs_escaping_with_escape_key_only() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({LC_ESCAPED_KEY: {"a": 1}})).unwrap();
        assert!(needs_escaping(&map));
    }

    #[test]
    fn test_no_escaping_needed_for_plain_dict() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({"key": "value", "another": 42})).unwrap();
        assert!(!needs_escaping(&map));
    }

    #[test]
    fn test_no_escaping_for_empty_dict() {
        let map = Map::new();
        assert!(!needs_escaping(&map));
    }

    #[test]
    fn test_escape_key_with_extra_keys_does_not_need_escaping() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({LC_ESCAPED_KEY: {}, "other": 1})).unwrap();
        // len != 1, so it should not need escaping (unless it also has "lc").
        assert!(!needs_escaping(&map));
    }

    // ---- escape_dict / is_escaped_dict ----

    #[test]
    fn test_escape_and_check() {
        let map: Map<String, Value> =
            serde_json::from_value(json!({"lc": 1, "id": ["foo"]})).unwrap();
        let escaped = escape_dict(map.clone());
        assert!(is_escaped_dict(&escaped));
        assert_eq!(escaped.len(), 1);
        assert_eq!(escaped.get(LC_ESCAPED_KEY).unwrap(), &Value::Object(map));
    }

    #[test]
    fn test_is_escaped_dict_false_for_regular() {
        let map: Map<String, Value> = serde_json::from_value(json!({"a": 1, "b": 2})).unwrap();
        assert!(!is_escaped_dict(&map));
    }

    // ---- is_lc_secret ----

    #[test]
    fn test_is_lc_secret_valid() {
        let secret = json!({"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]});
        assert!(is_lc_secret(&secret));
    }

    #[test]
    fn test_is_lc_secret_wrong_type() {
        let not_secret = json!({"lc": 1, "type": "constructor", "id": ["foo"]});
        assert!(!is_lc_secret(&not_secret));
    }

    #[test]
    fn test_is_lc_secret_missing_id() {
        let not_secret = json!({"lc": 1, "type": "secret"});
        assert!(!is_lc_secret(&not_secret));
    }

    #[test]
    fn test_is_lc_secret_extra_keys() {
        let not_secret = json!({"lc": 1, "type": "secret", "id": ["k"], "extra": true});
        assert!(!is_lc_secret(&not_secret));
    }

    #[test]
    fn test_is_lc_secret_non_object() {
        assert!(!is_lc_secret(&json!("hello")));
        assert!(!is_lc_secret(&json!(42)));
        assert!(!is_lc_secret(&Value::Null));
    }

    // ---- serialize_value ----

    #[test]
    fn test_serialize_primitives() {
        assert_eq!(serialize_value(&json!("hello")), json!("hello"));
        assert_eq!(serialize_value(&json!(42)), json!(42));
        assert_eq!(serialize_value(&json!(true)), json!(true));
        assert_eq!(serialize_value(&Value::Null), Value::Null);
    }

    #[test]
    fn test_serialize_safe_dict() {
        let input = json!({"name": "test", "count": 5});
        let result = serialize_value(&input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_serialize_dict_with_lc_key_is_escaped() {
        let input = json!({"lc": 1, "type": "constructor"});
        let result = serialize_value(&input);
        let expected = json!({LC_ESCAPED_KEY: {"lc": 1, "type": "constructor"}});
        assert_eq!(result, expected);
    }

    #[test]
    fn test_serialize_nested_dict_with_lc_key() {
        let input = json!({"metadata": {"lc": 1, "data": "foo"}});
        let result = serialize_value(&input);
        let expected = json!({"metadata": {LC_ESCAPED_KEY: {"lc": 1, "data": "foo"}}});
        assert_eq!(result, expected);
    }

    #[test]
    fn test_serialize_array_with_lc_dicts() {
        let input = json!([{"lc": 1}, "plain", 42]);
        let result = serialize_value(&input);
        let expected = json!([{LC_ESCAPED_KEY: {"lc": 1}}, "plain", 42]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_serialize_deeply_nested() {
        let input = json!({"a": {"b": {"c": {"lc": 2}}}});
        let result = serialize_value(&input);
        let expected = json!({"a": {"b": {"c": {LC_ESCAPED_KEY: {"lc": 2}}}}});
        assert_eq!(result, expected);
    }

    #[test]
    fn test_serialize_dict_that_looks_escaped() {
        // A dict with only __lc_escaped__ key also needs escaping (double-escape).
        let input = json!({LC_ESCAPED_KEY: {"fake": "data"}});
        let result = serialize_value(&input);
        let expected = json!({LC_ESCAPED_KEY: {LC_ESCAPED_KEY: {"fake": "data"}}});
        assert_eq!(result, expected);
    }

    // ---- unescape_value ----

    #[test]
    fn test_unescape_escaped_dict() {
        let input = json!({LC_ESCAPED_KEY: {"lc": 1, "type": "constructor"}});
        let result = unescape_value(&input);
        assert_eq!(result, json!({"lc": 1, "type": "constructor"}));
    }

    #[test]
    fn test_unescape_nested_escaped() {
        let input = json!({"metadata": {LC_ESCAPED_KEY: {"lc": 1}}});
        let result = unescape_value(&input);
        assert_eq!(result, json!({"metadata": {"lc": 1}}));
    }

    #[test]
    fn test_unescape_no_further_processing_of_contents() {
        // Contents of escaped dict are returned as-is, even if they contain
        // another escape key.
        let inner = json!({LC_ESCAPED_KEY: {"nested": true}});
        let input = json!({LC_ESCAPED_KEY: inner});
        let result = unescape_value(&input);
        // The inner value is returned as-is (no recursive unescaping).
        assert_eq!(result, inner);
    }

    #[test]
    fn test_unescape_array() {
        let input = json!([{LC_ESCAPED_KEY: {"lc": 1}}, "plain"]);
        let result = unescape_value(&input);
        assert_eq!(result, json!([{"lc": 1}, "plain"]));
    }

    #[test]
    fn test_unescape_primitives() {
        assert_eq!(unescape_value(&json!("hello")), json!("hello"));
        assert_eq!(unescape_value(&json!(42)), json!(42));
        assert_eq!(unescape_value(&Value::Null), Value::Null);
    }

    #[test]
    fn test_unescape_regular_dict_recurses() {
        let input = json!({"a": 1, "b": {LC_ESCAPED_KEY: {"lc": 2}}});
        let result = unescape_value(&input);
        assert_eq!(result, json!({"a": 1, "b": {"lc": 2}}));
    }

    // ---- round-trip ----

    #[test]
    fn test_serialize_then_unescape_roundtrip() {
        let original = json!({"lc": 1, "type": "constructor", "id": ["test"]});
        let serialized = serialize_value(&original);
        let deserialized = unescape_value(&serialized);
        assert_eq!(deserialized, original);
    }

    #[test]
    fn test_roundtrip_nested() {
        let original = json!({
            "safe_key": "value",
            "metadata": {"lc": 1, "data": [1, 2, 3]},
            "list": [{"lc": 2}, "text"]
        });
        let serialized = serialize_value(&original);
        let deserialized = unescape_value(&serialized);
        assert_eq!(deserialized, original);
    }

    #[test]
    fn test_roundtrip_safe_dict_unchanged() {
        let original = json!({"name": "test", "values": [1, 2, 3]});
        let serialized = serialize_value(&original);
        let deserialized = unescape_value(&serialized);
        assert_eq!(deserialized, original);
        // Safe dicts should not be modified at all.
        assert_eq!(serialized, original);
    }
}
