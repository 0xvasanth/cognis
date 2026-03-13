//! Deserialization utilities for loading objects.
//!
//! Mirrors Python `langchain_core.load.load`.

use std::collections::HashMap;

use serde_json::Value;

use crate::error::{Result, CognisError};

/// A reviver processes serialized dictionaries back into values.
///
/// In Python, this walks the JSON tree looking for `{"lc": 1, "type": "...", ...}`
/// markers and reconstructing objects. In Rust, since we can't dynamically
/// construct types at runtime without a registry, this instead validates
/// the structure and returns the Value tree with secrets resolved.
pub struct Reviver {
    /// Map of secret IDs to their actual values.
    secrets_map: HashMap<String, String>,
    /// Allowed namespaces for security (prevent arbitrary deserialization).
    valid_namespaces: Vec<String>,
}

impl Reviver {
    pub fn new() -> Self {
        Self {
            secrets_map: HashMap::new(),
            valid_namespaces: vec![
                "langchain_core".to_string(),
                "langchain".to_string(),
                "cognis_core".to_string(),
                "cognis".to_string(),
            ],
        }
    }

    /// Set secret values for resolving secret markers.
    pub fn with_secrets(mut self, secrets: HashMap<String, String>) -> Self {
        self.secrets_map = secrets;
        self
    }

    /// Add a valid namespace.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.valid_namespaces.push(namespace.into());
        self
    }

    /// Process a value, resolving any LangChain serialization markers.
    pub fn revive(&self, value: &Value) -> Result<Value> {
        match value {
            Value::Object(map) => {
                // Check if this is a LangChain serialized object
                if let Some(lc) = map.get("lc") {
                    if lc.as_u64() == Some(1) {
                        return self.revive_lc_object(map);
                    }
                }
                // Otherwise recurse into all values
                let mut result = serde_json::Map::new();
                for (k, v) in map {
                    result.insert(k.clone(), self.revive(v)?);
                }
                Ok(Value::Object(result))
            }
            Value::Array(arr) => {
                let result: Result<Vec<Value>> = arr.iter().map(|v| self.revive(v)).collect();
                Ok(Value::Array(result?))
            }
            other => Ok(other.clone()),
        }
    }

    fn revive_lc_object(&self, map: &serde_json::Map<String, Value>) -> Result<Value> {
        let type_tag = map.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match type_tag {
            "secret" => {
                // Resolve secret from secrets_map
                let id = map
                    .get("id")
                    .and_then(|id| id.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .unwrap_or_default();

                match self.secrets_map.get(&id) {
                    Some(value) => Ok(Value::String(value.clone())),
                    None => Err(CognisError::Other(format!(
                        "Secret '{}' not found in secrets map",
                        id
                    ))),
                }
            }
            "not_implemented" => {
                let repr = map
                    .get("repr")
                    .and_then(|r| r.as_str())
                    .unwrap_or("unknown");
                Err(CognisError::NotImplemented(format!(
                    "Cannot deserialize not_implemented object: {}",
                    repr
                )))
            }
            "constructor" => {
                // Validate namespace
                let id = map.get("id").and_then(|id| id.as_array());
                if let Some(id_parts) = id {
                    if let Some(first) = id_parts.first().and_then(|v| v.as_str()) {
                        if !self
                            .valid_namespaces
                            .iter()
                            .any(|ns| first.starts_with(ns.as_str()))
                        {
                            return Err(CognisError::Other(format!(
                                "Namespace '{}' is not in the allowed list",
                                first
                            )));
                        }
                    }
                }

                // Recurse into kwargs
                let mut result = serde_json::Map::new();
                result.insert("lc".to_string(), Value::Number(1.into()));
                result.insert("type".to_string(), Value::String("constructor".to_string()));
                if let Some(id) = map.get("id") {
                    result.insert("id".to_string(), id.clone());
                }
                if let Some(kwargs) = map.get("kwargs") {
                    result.insert("kwargs".to_string(), self.revive(kwargs)?);
                }
                Ok(Value::Object(result))
            }
            _ => {
                // Unknown type, pass through
                Ok(Value::Object(map.clone()))
            }
        }
    }
}

impl Default for Reviver {
    fn default() -> Self {
        Self::new()
    }
}

/// Deserialize a JSON string, resolving LangChain serialization markers.
pub fn loads(text: &str, secrets: Option<HashMap<String, String>>) -> Result<Value> {
    let parsed: Value = serde_json::from_str(text)
        .map_err(|e| CognisError::Other(format!("Failed to parse JSON: {}", e)))?;

    let mut reviver = Reviver::new();
    if let Some(s) = secrets {
        reviver = reviver.with_secrets(s);
    }
    reviver.revive(&parsed)
}

/// Deserialize a Value, resolving LangChain serialization markers.
pub fn load(obj: &Value, secrets: Option<HashMap<String, String>>) -> Result<Value> {
    let mut reviver = Reviver::new();
    if let Some(s) = secrets {
        reviver = reviver.with_secrets(s);
    }
    reviver.revive(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_loads_plain_json() {
        let result = loads(r#"{"key": "value"}"#, None).unwrap();
        assert_eq!(result, json!({"key": "value"}));
    }

    #[test]
    fn test_loads_invalid_json() {
        let result = loads("not json", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_revive_secret_resolved() {
        let mut secrets = HashMap::new();
        secrets.insert("OPENAI_API_KEY".to_string(), "sk-test123".to_string());

        let input = json!({
            "lc": 1,
            "type": "secret",
            "id": ["OPENAI_API_KEY"]
        });

        let result = load(&input, Some(secrets)).unwrap();
        assert_eq!(result, Value::String("sk-test123".to_string()));
    }

    #[test]
    fn test_revive_secret_missing() {
        let input = json!({
            "lc": 1,
            "type": "secret",
            "id": ["MISSING_KEY"]
        });

        let result = load(&input, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_revive_not_implemented() {
        let input = json!({
            "lc": 1,
            "type": "not_implemented",
            "id": ["test"],
            "repr": "SomeClass"
        });

        let result = load(&input, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_revive_constructor_valid_namespace() {
        let input = json!({
            "lc": 1,
            "type": "constructor",
            "id": ["langchain_core", "prompts", "PromptTemplate"],
            "kwargs": {"template": "Hello {name}"}
        });

        let result = load(&input, None).unwrap();
        assert_eq!(result["type"], "constructor");
        assert_eq!(result["kwargs"]["template"], "Hello {name}");
    }

    #[test]
    fn test_revive_constructor_invalid_namespace() {
        let input = json!({
            "lc": 1,
            "type": "constructor",
            "id": ["evil_namespace", "malware"],
            "kwargs": {}
        });

        let result = load(&input, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_revive_nested_secret_in_constructor() {
        let mut secrets = HashMap::new();
        secrets.insert("api_key".to_string(), "secret-value".to_string());

        let input = json!({
            "lc": 1,
            "type": "constructor",
            "id": ["cognis_core", "llms", "OpenAI"],
            "kwargs": {
                "model": "gpt-4",
                "api_key": {
                    "lc": 1,
                    "type": "secret",
                    "id": ["api_key"]
                }
            }
        });

        let result = load(&input, Some(secrets)).unwrap();
        assert_eq!(result["kwargs"]["api_key"], "secret-value");
        assert_eq!(result["kwargs"]["model"], "gpt-4");
    }

    #[test]
    fn test_reviver_with_custom_namespace() {
        let reviver = Reviver::new().with_namespace("custom_ns");

        let input = json!({
            "lc": 1,
            "type": "constructor",
            "id": ["custom_ns", "MyClass"],
            "kwargs": {}
        });

        let result = reviver.revive(&input).unwrap();
        assert_eq!(result["type"], "constructor");
    }

    #[test]
    fn test_load_array_with_markers() {
        let mut secrets = HashMap::new();
        secrets.insert("key".to_string(), "resolved".to_string());

        let input = json!([
            {"lc": 1, "type": "secret", "id": ["key"]},
            "plain_value"
        ]);

        let result = load(&input, Some(secrets)).unwrap();
        assert_eq!(result[0], "resolved");
        assert_eq!(result[1], "plain_value");
    }

    #[test]
    fn test_revive_unknown_type_passthrough() {
        let input = json!({
            "lc": 1,
            "type": "unknown_future_type",
            "id": ["something"],
            "data": "test"
        });

        let result = load(&input, None).unwrap();
        assert_eq!(result["type"], "unknown_future_type");
    }
}
