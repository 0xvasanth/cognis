//! Checkpoint serialization — converting graph state to/from storable formats.
//!
//! This module provides a pluggable serialization layer for checkpoint state
//! data (`serde_json::Value`), distinct from the [`super::serialization`]
//! module which serializes full [`Checkpoint`](crate::pregel::checkpoint::Checkpoint)
//! structs.
//!
//! # Serializers
//!
//! - [`JsonSerializer`] — standard JSON encoding
//! - [`CompactJsonSerializer`] — JSON with null fields and empty containers removed
//! - [`PrettyJsonSerializer`] — indented JSON for debugging
//! - [`VersionedSerializer`] — wraps another serializer with version metadata
//! - [`CompressedSerializer`] — mock compression via base64 encoding
//!
//! # Supporting types
//!
//! - [`CheckpointMetadata`] — metadata about a checkpoint (thread, step, etc.)
//! - [`CheckpointEntry`] — combines metadata with state for serialization
//! - [`CheckpointMigration`] — handles version-to-version state migrations

use std::collections::HashMap;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// SerializationFormat
// ---------------------------------------------------------------------------

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Standard JSON encoding.
    Json,
    /// MessagePack (mock — serializes as a JSON string with a format marker).
    MessagePack,
    /// Bincode (mock — serializes as a JSON string with a format marker).
    Bincode,
}

impl SerializationFormat {
    /// Human-readable name of this format.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::MessagePack => "msgpack",
            Self::Bincode => "bincode",
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointSerializer trait
// ---------------------------------------------------------------------------

/// Trait for serializing and deserializing checkpoint state values.
pub trait CheckpointSerializer: Send + Sync {
    /// Serialize a `Value` into bytes.
    fn serialize(&self, state: &Value) -> Result<Vec<u8>>;

    /// Deserialize bytes back into a `Value`.
    fn deserialize(&self, data: &[u8]) -> Result<Value>;
}

// ---------------------------------------------------------------------------
// JsonSerializer
// ---------------------------------------------------------------------------

/// Standard JSON serializer for checkpoint state.
#[derive(Debug, Clone, Default)]
pub struct JsonSerializer;

impl JsonSerializer {
    /// Create a new JSON serializer.
    pub fn new() -> Self {
        Self
    }
}

impl CheckpointSerializer for JsonSerializer {
    fn serialize(&self, state: &Value) -> Result<Vec<u8>> {
        serde_json::to_vec(state)
            .map_err(|e| LangGraphError::Other(format!("JSON serialization failed: {e}")))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Value> {
        serde_json::from_slice(data)
            .map_err(|e| LangGraphError::Other(format!("JSON deserialization failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// CompactJsonSerializer
// ---------------------------------------------------------------------------

/// JSON serializer that strips null fields and empty arrays/objects to save space.
#[derive(Debug, Clone, Default)]
pub struct CompactJsonSerializer;

impl CompactJsonSerializer {
    /// Create a new compact JSON serializer.
    pub fn new() -> Self {
        Self
    }

    /// Recursively remove null values, empty arrays, and empty objects.
    fn compact_value(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let filtered: serde_json::Map<String, Value> = map
                    .iter()
                    .filter_map(|(k, v)| {
                        if v.is_null() {
                            return None;
                        }
                        let compacted = Self::compact_value(v);
                        // Remove empty arrays and objects after compaction.
                        match &compacted {
                            Value::Array(arr) if arr.is_empty() => None,
                            Value::Object(obj) if obj.is_empty() => None,
                            _ => Some((k.clone(), compacted)),
                        }
                    })
                    .collect();
                Value::Object(filtered)
            }
            Value::Array(arr) => {
                let filtered: Vec<Value> = arr
                    .iter()
                    .filter(|v| !v.is_null())
                    .map(|v| Self::compact_value(v))
                    .collect();
                Value::Array(filtered)
            }
            other => other.clone(),
        }
    }
}

impl CheckpointSerializer for CompactJsonSerializer {
    fn serialize(&self, state: &Value) -> Result<Vec<u8>> {
        let compacted = Self::compact_value(state);
        serde_json::to_vec(&compacted)
            .map_err(|e| LangGraphError::Other(format!("Compact JSON serialization failed: {e}")))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Value> {
        serde_json::from_slice(data).map_err(|e| {
            LangGraphError::Other(format!("Compact JSON deserialization failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// PrettyJsonSerializer
// ---------------------------------------------------------------------------

/// Pretty-printed JSON serializer for debugging and inspection.
#[derive(Debug, Clone, Default)]
pub struct PrettyJsonSerializer;

impl PrettyJsonSerializer {
    /// Create a new pretty JSON serializer.
    pub fn new() -> Self {
        Self
    }
}

impl CheckpointSerializer for PrettyJsonSerializer {
    fn serialize(&self, state: &Value) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(state)
            .map_err(|e| LangGraphError::Other(format!("Pretty JSON serialization failed: {e}")))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Value> {
        serde_json::from_slice(data).map_err(|e| {
            LangGraphError::Other(format!("Pretty JSON deserialization failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// VersionedSerializer
// ---------------------------------------------------------------------------

/// A serializer wrapper that adds version metadata to serialized output.
///
/// The wire format is a JSON object:
/// ```json
/// {"version": N, "format": "...", "data": <inner-serialized-value>}
/// ```
///
/// This allows consumers to detect and handle version differences when
/// reading checkpoint data written by older versions.
pub struct VersionedSerializer {
    inner: Box<dyn CheckpointSerializer>,
    version: u64,
}

impl VersionedSerializer {
    /// Create a new versioned serializer wrapping `inner` at the given version.
    pub fn new(inner: Box<dyn CheckpointSerializer>, version: u64) -> Self {
        Self { inner, version }
    }

    /// Return the current version number.
    pub fn version(&self) -> u64 {
        self.version
    }
}

impl CheckpointSerializer for VersionedSerializer {
    fn serialize(&self, state: &Value) -> Result<Vec<u8>> {
        let inner_bytes = self.inner.serialize(state)?;
        // Parse the inner bytes back to Value so we can embed it in the envelope.
        let inner_value: Value = serde_json::from_slice(&inner_bytes).map_err(|e| {
            LangGraphError::Other(format!("VersionedSerializer inner parse failed: {e}"))
        })?;

        let envelope = serde_json::json!({
            "version": self.version,
            "format": "versioned",
            "data": inner_value,
        });

        serde_json::to_vec(&envelope).map_err(|e| {
            LangGraphError::Other(format!("VersionedSerializer envelope serialization: {e}"))
        })
    }

    fn deserialize(&self, data: &[u8]) -> Result<Value> {
        let envelope: Value = serde_json::from_slice(data).map_err(|e| {
            LangGraphError::Other(format!("VersionedSerializer envelope parse failed: {e}"))
        })?;

        // Extract the data field — accept any version (supports reading older versions).
        let _version = envelope.get("version").and_then(|v| v.as_u64()).ok_or_else(|| {
            LangGraphError::Other("VersionedSerializer: missing 'version' field".into())
        })?;

        let data_value = envelope.get("data").ok_or_else(|| {
            LangGraphError::Other("VersionedSerializer: missing 'data' field".into())
        })?;

        Ok(data_value.clone())
    }
}

// ---------------------------------------------------------------------------
// CompressedSerializer
// ---------------------------------------------------------------------------

/// Mock compression serializer that base64-encodes the inner serialized bytes.
///
/// This simulates compression without adding binary compression dependencies.
/// In a production system, this would use actual compression (e.g., deflate, zstd).
pub struct CompressedSerializer {
    inner: Box<dyn CheckpointSerializer>,
}

impl CompressedSerializer {
    /// Create a new compressed serializer wrapping `inner`.
    pub fn new(inner: Box<dyn CheckpointSerializer>) -> Self {
        Self { inner }
    }
}

impl CheckpointSerializer for CompressedSerializer {
    fn serialize(&self, state: &Value) -> Result<Vec<u8>> {
        let raw = self.inner.serialize(state)?;
        let encoded = BASE64.encode(&raw);
        Ok(encoded.into_bytes())
    }

    fn deserialize(&self, data: &[u8]) -> Result<Value> {
        let encoded = std::str::from_utf8(data)
            .map_err(|e| LangGraphError::Other(format!("CompressedSerializer: invalid UTF-8: {e}")))?;
        let decoded = BASE64.decode(encoded).map_err(|e| {
            LangGraphError::Other(format!("CompressedSerializer: base64 decode failed: {e}"))
        })?;
        self.inner.deserialize(&decoded)
    }
}

// ---------------------------------------------------------------------------
// CheckpointMetadata
// ---------------------------------------------------------------------------

/// Metadata describing a checkpoint's provenance and context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointMetadata {
    /// The thread this checkpoint belongs to.
    pub thread_id: String,
    /// Unique identifier for this checkpoint.
    pub checkpoint_id: String,
    /// The parent checkpoint ID, if any.
    pub parent_id: Option<String>,
    /// The step number in the execution.
    pub step: i64,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// Version counter for each channel.
    pub channel_versions: HashMap<String, u64>,
    /// The source of the checkpoint (e.g. "input", "loop", "update").
    pub source: String,
}

impl CheckpointMetadata {
    /// Create new metadata with required fields; other fields are set to defaults.
    pub fn new(thread_id: impl Into<String>, checkpoint_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            checkpoint_id: checkpoint_id.into(),
            parent_id: None,
            step: 0,
            created_at: String::new(),
            channel_versions: HashMap::new(),
            source: String::new(),
        }
    }

    /// Set the parent checkpoint ID.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the step number.
    pub fn with_step(mut self, step: i64) -> Self {
        self.step = step;
        self
    }

    /// Set the source label.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Add a channel version entry.
    pub fn with_channel_version(mut self, channel: impl Into<String>, version: u64) -> Self {
        self.channel_versions.insert(channel.into(), version);
        self
    }

    /// Serialize this metadata to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Deserialize metadata from a JSON `Value`.
    pub fn from_json(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone())
            .map_err(|e| LangGraphError::Other(format!("CheckpointMetadata parse failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// CheckpointEntry
// ---------------------------------------------------------------------------

/// Combines checkpoint metadata with the serialized state for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    /// Metadata describing this checkpoint.
    pub metadata: CheckpointMetadata,
    /// The checkpoint state as a JSON value.
    pub state: Value,
}

impl CheckpointEntry {
    /// Create a new entry from metadata and state.
    pub fn new(metadata: CheckpointMetadata, state: Value) -> Self {
        Self { metadata, state }
    }

    /// Serialize this entry using the given serializer.
    pub fn serialize_with(&self, serializer: &dyn CheckpointSerializer) -> Result<Vec<u8>> {
        let combined = serde_json::json!({
            "metadata": self.metadata.to_json(),
            "state": self.state,
        });
        serializer.serialize(&combined)
    }

    /// Deserialize an entry from bytes using the given serializer.
    pub fn deserialize_with(data: &[u8], serializer: &dyn CheckpointSerializer) -> Result<Self> {
        let combined = serializer.deserialize(data)?;

        let metadata_value = combined.get("metadata").ok_or_else(|| {
            LangGraphError::Other("CheckpointEntry: missing 'metadata' field".into())
        })?;
        let metadata = CheckpointMetadata::from_json(metadata_value)?;

        let state = combined
            .get("state")
            .cloned()
            .unwrap_or(Value::Null);

        Ok(Self { metadata, state })
    }
}

// ---------------------------------------------------------------------------
// CheckpointMigration
// ---------------------------------------------------------------------------

/// Handles version-to-version state migrations for checkpoint data.
///
/// Migrations are registered as `(from_version, to_version, transform_fn)`
/// triples and applied sequentially when upgrading from an old version to
/// a new one.
pub struct CheckpointMigration {
    migrations: Vec<(u64, u64, fn(Value) -> Value)>,
}

impl CheckpointMigration {
    /// Create a new migration registry with no registered migrations.
    pub fn new() -> Self {
        Self {
            migrations: Vec::new(),
        }
    }

    /// Register a migration from `from_version` to `to_version`.
    pub fn add_migration(&mut self, from_version: u64, to_version: u64, transform: fn(Value) -> Value) {
        self.migrations.push((from_version, to_version, transform));
    }

    /// Apply sequential migrations to move `data` from version `from` to version `to`.
    ///
    /// Migrations are applied in order: `from -> from+1 -> ... -> to`.
    /// Returns an error if any required migration step is missing.
    pub fn migrate(&self, data: Value, from: u64, to: u64) -> Result<Value> {
        if from == to {
            return Ok(data);
        }
        if from > to {
            return Err(LangGraphError::Other(format!(
                "Cannot migrate backwards from version {from} to {to}"
            )));
        }

        let mut current = data;
        let mut current_version = from;

        while current_version < to {
            let next_version = current_version + 1;
            let migration = self
                .migrations
                .iter()
                .find(|(f, t, _)| *f == current_version && *t == next_version);

            match migration {
                Some((_, _, transform)) => {
                    current = transform(current);
                    current_version = next_version;
                }
                None => {
                    return Err(LangGraphError::Other(format!(
                        "No migration registered from version {current_version} to {next_version}"
                    )));
                }
            }
        }

        Ok(current)
    }
}

impl Default for CheckpointMigration {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- JsonSerializer -------------------------------------------------------

    #[test]
    fn test_json_serializer_roundtrip() {
        let serializer = JsonSerializer::new();
        let state = json!({"key": "value", "number": 42});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_json_serializer_empty_object() {
        let serializer = JsonSerializer::new();
        let state = json!({});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_json_serializer_null() {
        let serializer = JsonSerializer::new();
        let state = Value::Null;
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_json_serializer_nested_objects() {
        let serializer = JsonSerializer::new();
        let state = json!({
            "level1": {
                "level2": {
                    "level3": [1, 2, {"deep": true}]
                }
            }
        });
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_json_serializer_array() {
        let serializer = JsonSerializer::new();
        let state = json!([1, "two", null, false, [3, 4]]);
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    // -- CompactJsonSerializer ------------------------------------------------

    #[test]
    fn test_compact_json_removes_nulls() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({"a": 1, "b": null, "c": "hello"});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert!(restored.get("b").is_none());
        assert_eq!(restored["a"], json!(1));
        assert_eq!(restored["c"], json!("hello"));
    }

    #[test]
    fn test_compact_json_removes_empty_arrays() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({"data": [1, 2], "empty": []});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert!(restored.get("empty").is_none());
        assert_eq!(restored["data"], json!([1, 2]));
    }

    #[test]
    fn test_compact_json_removes_empty_objects() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({"filled": {"x": 1}, "empty": {}});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert!(restored.get("empty").is_none());
        assert_eq!(restored["filled"], json!({"x": 1}));
    }

    #[test]
    fn test_compact_json_nested_nulls() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({
            "outer": {
                "keep": 1,
                "drop_null": null,
                "inner": {
                    "also_drop": null,
                    "also_keep": "yes"
                }
            }
        });
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert!(restored["outer"].get("drop_null").is_none());
        assert!(restored["outer"]["inner"].get("also_drop").is_none());
        assert_eq!(restored["outer"]["keep"], json!(1));
        assert_eq!(restored["outer"]["inner"]["also_keep"], json!("yes"));
    }

    #[test]
    fn test_compact_json_all_null_object_becomes_empty_then_removed() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({"parent": {"a": null, "b": null}});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        // The inner object has only nulls, so after removal it's empty, then it too is removed.
        assert!(restored.get("parent").is_none());
    }

    #[test]
    fn test_compact_json_preserves_false_and_zero() {
        let serializer = CompactJsonSerializer::new();
        let state = json!({"flag": false, "count": 0, "text": ""});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(restored["flag"], json!(false));
        assert_eq!(restored["count"], json!(0));
        assert_eq!(restored["text"], json!(""));
    }

    // -- PrettyJsonSerializer -------------------------------------------------

    #[test]
    fn test_pretty_json_formatting() {
        let serializer = PrettyJsonSerializer::new();
        let state = json!({"key": "value"});
        let bytes = serializer.serialize(&state).unwrap();
        let output = std::str::from_utf8(&bytes).unwrap();
        // Pretty output should contain newlines and indentation.
        assert!(output.contains('\n'));
        assert!(output.contains("  "));
    }

    #[test]
    fn test_pretty_json_roundtrip() {
        let serializer = PrettyJsonSerializer::new();
        let state = json!({"nested": {"arr": [1, 2, 3]}, "flag": true});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_pretty_json_larger_than_compact() {
        let state = json!({"a": 1, "b": [2, 3], "c": {"d": 4}});
        let compact_bytes = JsonSerializer::new().serialize(&state).unwrap();
        let pretty_bytes = PrettyJsonSerializer::new().serialize(&state).unwrap();
        assert!(pretty_bytes.len() > compact_bytes.len());
    }

    // -- VersionedSerializer --------------------------------------------------

    #[test]
    fn test_versioned_serializer_roundtrip() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = VersionedSerializer::new(inner, 1);
        let state = json!({"data": "test"});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_versioned_serializer_includes_version() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = VersionedSerializer::new(inner, 5);
        let state = json!({"x": 1});
        let bytes = serializer.serialize(&state).unwrap();
        let envelope: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope["version"], json!(5));
        assert_eq!(envelope["format"], json!("versioned"));
        assert_eq!(envelope["data"], state);
    }

    #[test]
    fn test_versioned_serializer_reads_older_version() {
        // Create data with version 1.
        let inner_v1 = Box::new(JsonSerializer::new());
        let serializer_v1 = VersionedSerializer::new(inner_v1, 1);
        let state = json!({"legacy": true});
        let bytes = serializer_v1.serialize(&state).unwrap();

        // Read it with a version 3 serializer — should still work.
        let inner_v3 = Box::new(JsonSerializer::new());
        let serializer_v3 = VersionedSerializer::new(inner_v3, 3);
        let restored = serializer_v3.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_versioned_serializer_rejects_missing_version() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = VersionedSerializer::new(inner, 1);
        let bad_data = serde_json::to_vec(&json!({"data": "no version"})).unwrap();
        let result = serializer.deserialize(&bad_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_versioned_serializer_rejects_missing_data() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = VersionedSerializer::new(inner, 1);
        let bad_data = serde_json::to_vec(&json!({"version": 1})).unwrap();
        let result = serializer.deserialize(&bad_data);
        assert!(result.is_err());
    }

    // -- CompressedSerializer -------------------------------------------------

    #[test]
    fn test_compressed_serializer_roundtrip() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = CompressedSerializer::new(inner);
        let state = json!({"message": "hello world", "count": 42});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_compressed_serializer_output_is_base64() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = CompressedSerializer::new(inner);
        let state = json!({"test": true});
        let bytes = serializer.serialize(&state).unwrap();
        let encoded_str = std::str::from_utf8(&bytes).unwrap();
        // Verify it's valid base64 by decoding.
        assert!(BASE64.decode(encoded_str).is_ok());
    }

    #[test]
    fn test_compressed_serializer_empty_state() {
        let inner = Box::new(JsonSerializer::new());
        let serializer = CompressedSerializer::new(inner);
        let state = json!({});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    // -- CheckpointMetadata ---------------------------------------------------

    #[test]
    fn test_metadata_new_defaults() {
        let meta = CheckpointMetadata::new("thread-1", "cp-1");
        assert_eq!(meta.thread_id, "thread-1");
        assert_eq!(meta.checkpoint_id, "cp-1");
        assert_eq!(meta.step, 0);
        assert!(meta.parent_id.is_none());
        assert!(meta.source.is_empty());
        assert!(meta.channel_versions.is_empty());
    }

    #[test]
    fn test_metadata_builder_pattern() {
        let meta = CheckpointMetadata::new("t1", "cp-1")
            .with_parent("cp-0")
            .with_step(5)
            .with_source("loop")
            .with_channel_version("messages", 3)
            .with_channel_version("counter", 1);

        assert_eq!(meta.parent_id, Some("cp-0".to_string()));
        assert_eq!(meta.step, 5);
        assert_eq!(meta.source, "loop");
        assert_eq!(meta.channel_versions.len(), 2);
        assert_eq!(meta.channel_versions["messages"], 3);
        assert_eq!(meta.channel_versions["counter"], 1);
    }

    #[test]
    fn test_metadata_to_json_and_back() {
        let meta = CheckpointMetadata::new("thread-1", "cp-42")
            .with_parent("cp-41")
            .with_step(10)
            .with_source("input")
            .with_channel_version("state", 7);

        let json_val = meta.to_json();
        let restored = CheckpointMetadata::from_json(&json_val).unwrap();

        assert_eq!(meta, restored);
    }

    #[test]
    fn test_metadata_from_json_invalid() {
        let bad_json = json!({"thread_id": 123}); // wrong types
        let result = CheckpointMetadata::from_json(&bad_json);
        assert!(result.is_err());
    }

    // -- CheckpointEntry ------------------------------------------------------

    #[test]
    fn test_checkpoint_entry_roundtrip_json() {
        let meta = CheckpointMetadata::new("t1", "cp-1")
            .with_step(3)
            .with_source("loop");
        let state = json!({"messages": ["hello"], "count": 1});
        let entry = CheckpointEntry::new(meta, state);

        let serializer = JsonSerializer::new();
        let bytes = entry.serialize_with(&serializer).unwrap();
        let restored = CheckpointEntry::deserialize_with(&bytes, &serializer).unwrap();

        assert_eq!(restored.metadata.thread_id, "t1");
        assert_eq!(restored.metadata.checkpoint_id, "cp-1");
        assert_eq!(restored.metadata.step, 3);
        assert_eq!(restored.state, json!({"messages": ["hello"], "count": 1}));
    }

    #[test]
    fn test_checkpoint_entry_roundtrip_versioned() {
        let meta = CheckpointMetadata::new("t2", "cp-99")
            .with_parent("cp-98")
            .with_source("update");
        let state = json!({"data": [1, 2, 3]});
        let entry = CheckpointEntry::new(meta, state);

        let serializer = VersionedSerializer::new(Box::new(JsonSerializer::new()), 2);
        let bytes = entry.serialize_with(&serializer).unwrap();
        let restored = CheckpointEntry::deserialize_with(&bytes, &serializer).unwrap();

        assert_eq!(restored.metadata.checkpoint_id, "cp-99");
        assert_eq!(restored.metadata.parent_id, Some("cp-98".to_string()));
        assert_eq!(restored.state, json!({"data": [1, 2, 3]}));
    }

    #[test]
    fn test_checkpoint_entry_roundtrip_compressed() {
        let meta = CheckpointMetadata::new("t3", "cp-5");
        let state = json!({"big": "value".repeat(100)});
        let entry = CheckpointEntry::new(meta, state);

        let serializer = CompressedSerializer::new(Box::new(JsonSerializer::new()));
        let bytes = entry.serialize_with(&serializer).unwrap();
        let restored = CheckpointEntry::deserialize_with(&bytes, &serializer).unwrap();

        assert_eq!(restored.metadata.thread_id, "t3");
        assert_eq!(restored.state["big"], json!("value".repeat(100)));
    }

    #[test]
    fn test_checkpoint_entry_with_null_state() {
        let meta = CheckpointMetadata::new("t1", "cp-null");
        let entry = CheckpointEntry::new(meta, Value::Null);

        let serializer = JsonSerializer::new();
        let bytes = entry.serialize_with(&serializer).unwrap();
        let restored = CheckpointEntry::deserialize_with(&bytes, &serializer).unwrap();

        assert_eq!(restored.state, Value::Null);
    }

    // -- CheckpointMigration --------------------------------------------------

    #[test]
    fn test_migration_single_step() {
        let mut migration = CheckpointMigration::new();
        migration.add_migration(1, 2, |mut v| {
            v.as_object_mut()
                .unwrap()
                .insert("version".to_string(), json!(2));
            v
        });

        let data = json!({"value": "original"});
        let result = migration.migrate(data, 1, 2).unwrap();
        assert_eq!(result["version"], json!(2));
        assert_eq!(result["value"], json!("original"));
    }

    #[test]
    fn test_migration_multi_step() {
        let mut migration = CheckpointMigration::new();
        migration.add_migration(1, 2, |mut v| {
            v.as_object_mut()
                .unwrap()
                .insert("added_in_v2".to_string(), json!(true));
            v
        });
        migration.add_migration(2, 3, |mut v| {
            v.as_object_mut()
                .unwrap()
                .insert("added_in_v3".to_string(), json!("hello"));
            v
        });
        migration.add_migration(3, 4, |mut v| {
            v.as_object_mut().unwrap().remove("old_field");
            v
        });

        let data = json!({"old_field": "remove_me", "keep": 42});
        let result = migration.migrate(data, 1, 4).unwrap();
        assert_eq!(result["added_in_v2"], json!(true));
        assert_eq!(result["added_in_v3"], json!("hello"));
        assert!(result.get("old_field").is_none());
        assert_eq!(result["keep"], json!(42));
    }

    #[test]
    fn test_migration_same_version_noop() {
        let migration = CheckpointMigration::new();
        let data = json!({"unchanged": true});
        let result = migration.migrate(data.clone(), 5, 5).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_migration_backwards_error() {
        let migration = CheckpointMigration::new();
        let result = migration.migrate(json!({}), 3, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_migration_missing_step_error() {
        let mut migration = CheckpointMigration::new();
        migration.add_migration(1, 2, |v| v);
        // Missing 2 -> 3
        migration.add_migration(3, 4, |v| v);

        let result = migration.migrate(json!({}), 1, 4);
        assert!(result.is_err());
    }

    // -- Edge cases -----------------------------------------------------------

    #[test]
    fn test_serializer_large_value() {
        let serializer = JsonSerializer::new();
        let large_array: Vec<i64> = (0..10_000).collect();
        let state = json!({"data": large_array});
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(restored["data"].as_array().unwrap().len(), 10_000);
    }

    #[test]
    fn test_serializer_deeply_nested() {
        let serializer = JsonSerializer::new();
        let mut state = json!({"leaf": "value"});
        for _ in 0..50 {
            state = json!({"nested": state});
        }
        let bytes = serializer.serialize(&state).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_serialization_format_as_str() {
        assert_eq!(SerializationFormat::Json.as_str(), "json");
        assert_eq!(SerializationFormat::MessagePack.as_str(), "msgpack");
        assert_eq!(SerializationFormat::Bincode.as_str(), "bincode");
    }

    #[test]
    fn test_serialization_format_serde_roundtrip() {
        for format in &[
            SerializationFormat::Json,
            SerializationFormat::MessagePack,
            SerializationFormat::Bincode,
        ] {
            let json_str = serde_json::to_string(format).unwrap();
            let restored: SerializationFormat = serde_json::from_str(&json_str).unwrap();
            assert_eq!(*format, restored);
        }
    }

    #[test]
    fn test_compact_json_roundtrip_no_nulls() {
        // If there are no nulls/empties, compact JSON should be identical to regular JSON.
        let state = json!({"a": 1, "b": "hello", "c": [1, 2, 3]});
        let compact = CompactJsonSerializer::new();
        let regular = JsonSerializer::new();

        let compact_bytes = compact.serialize(&state).unwrap();
        let regular_bytes = regular.serialize(&state).unwrap();

        let compact_restored = compact.deserialize(&compact_bytes).unwrap();
        let regular_restored = regular.deserialize(&regular_bytes).unwrap();

        assert_eq!(compact_restored, regular_restored);
    }

    #[test]
    fn test_metadata_channel_versions_multiple() {
        let meta = CheckpointMetadata::new("t1", "cp-1")
            .with_channel_version("ch1", 1)
            .with_channel_version("ch2", 5)
            .with_channel_version("ch3", 10);

        assert_eq!(meta.channel_versions.len(), 3);

        let json_val = meta.to_json();
        let restored = CheckpointMetadata::from_json(&json_val).unwrap();
        assert_eq!(restored.channel_versions, meta.channel_versions);
    }

    #[test]
    fn test_compressed_then_versioned_roundtrip() {
        let state = json!({"complex": {"data": [1, 2, 3], "flag": true}});

        let inner = Box::new(JsonSerializer::new());
        let versioned = Box::new(VersionedSerializer::new(inner, 1));
        let compressed = CompressedSerializer::new(versioned);

        let bytes = compressed.serialize(&state).unwrap();
        let restored = compressed.deserialize(&bytes).unwrap();
        assert_eq!(state, restored);
    }
}
