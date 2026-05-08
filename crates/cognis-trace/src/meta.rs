//! Well-known keys in `RunnableConfig.metadata`. The bridge handler reads
//! these on the trace root to populate trace-level fields. Users construct
//! them with the builder helpers below.

use serde_json::{Map, Value};

/// Well-known metadata keys.
pub mod keys {
    /// Langfuse-compatible `sessionId`.
    pub const SESSION_ID: &str = "session_id";
    /// Langfuse-compatible `userId`.
    pub const USER_ID: &str = "user_id";
    /// Release version (git sha or semver tag).
    pub const RELEASE: &str = "release";
    /// Trace version.
    pub const VERSION: &str = "version";
    /// Environment (production / staging / dev).
    pub const ENVIRONMENT: &str = "environment";
    /// Make trace public.
    pub const PUBLIC: &str = "public";
}

/// Builder of `RunnableConfig.metadata` entries with strongly-named helpers.
pub struct TraceMeta;

impl TraceMeta {
    /// Build a `(key, value)` pair for `session_id`.
    pub fn session(id: impl Into<String>) -> (&'static str, Value) {
        (keys::SESSION_ID, Value::String(id.into()))
    }

    /// Build a `(key, value)` pair for `user_id`.
    pub fn user(id: impl Into<String>) -> (&'static str, Value) {
        (keys::USER_ID, Value::String(id.into()))
    }

    /// Build a `(key, value)` pair for `release`.
    pub fn release(s: impl Into<String>) -> (&'static str, Value) {
        (keys::RELEASE, Value::String(s.into()))
    }

    /// Build a `(key, value)` pair for `environment`.
    pub fn environment(s: impl Into<String>) -> (&'static str, Value) {
        (keys::ENVIRONMENT, Value::String(s.into()))
    }

    /// Build a `(key, value)` pair for `version`.
    pub fn version(s: impl Into<String>) -> (&'static str, Value) {
        (keys::VERSION, Value::String(s.into()))
    }

    /// Build a `(key, value)` pair for `public`.
    pub fn public(b: bool) -> (&'static str, Value) {
        (keys::PUBLIC, Value::Bool(b))
    }
}

/// Insert a key/value into `metadata`, upgrading `Null` / non-object to an
/// object. Returns the updated value.
pub fn merge_into(metadata: Value, kv: (&str, Value)) -> Value {
    let mut map = match metadata {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    map.insert(kv.0.to_string(), kv.1);
    Value::Object(map)
}

/// Read a string-typed well-known key from `metadata` if present.
pub fn read_string(metadata: &Value, key: &str) -> Option<String> {
    metadata.as_object()?.get(key)?.as_str().map(str::to_owned)
}

/// Read a bool-typed well-known key from `metadata` if present.
pub fn read_bool(metadata: &Value, key: &str) -> Option<bool> {
    metadata.as_object()?.get(key)?.as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_into_null_creates_object() {
        let v = merge_into(Value::Null, TraceMeta::session("s1"));
        assert_eq!(v.as_object().unwrap()["session_id"], "s1");
    }

    #[test]
    fn merge_into_existing_object_inserts_key() {
        let v = merge_into(serde_json::json!({"a": 1}), TraceMeta::user("u1"));
        let obj = v.as_object().unwrap();
        assert_eq!(obj["a"], 1);
        assert_eq!(obj["user_id"], "u1");
    }

    #[test]
    fn merge_into_overwrites_same_key() {
        let v = merge_into(Value::Null, TraceMeta::session("first"));
        let v = merge_into(v, TraceMeta::session("second"));
        assert_eq!(v.as_object().unwrap()["session_id"], "second");
    }

    #[test]
    fn read_string_returns_none_when_missing() {
        assert_eq!(read_string(&Value::Null, keys::SESSION_ID), None);
        assert_eq!(read_string(&serde_json::json!({"a": 1}), keys::SESSION_ID), None);
    }

    #[test]
    fn read_string_finds_well_known_key() {
        let v = merge_into(Value::Null, TraceMeta::release("v1.2.3"));
        assert_eq!(read_string(&v, keys::RELEASE).as_deref(), Some("v1.2.3"));
    }

    #[test]
    fn read_bool_finds_public() {
        let v = merge_into(Value::Null, TraceMeta::public(true));
        assert_eq!(read_bool(&v, keys::PUBLIC), Some(true));
    }
}
