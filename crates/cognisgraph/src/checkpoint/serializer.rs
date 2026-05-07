//! Pluggable serialization format for checkpointers.
//!
//! V2's `SqliteCheckpointer` and `PostgresCheckpointer` historically
//! hardcoded `serde_json`. Production users sometimes want CBOR (smaller),
//! MessagePack (faster), or postcard (deterministic, embedded-friendly).
//! [`CheckpointSerializer`] is the integration point.
//!
//! The shipped impl is [`JsonSerializer`] (the default; matches existing
//! behaviour). Other formats live behind feature flags so the dep tree
//! stays minimal: `serializer-cbor`, `serializer-msgpack`,
//! `serializer-postcard`.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::Serialize;

use cognis_core::{CognisError, Result};

/// Pluggable serialize/deserialize for checkpoint payloads.
///
/// The format `name` is stored alongside the payload in the backend so
/// stale checkpoints can be deserialized with the matching format on
/// restart. Mismatches surface as a clear error rather than corrupted
/// state.
pub trait CheckpointSerializer: Send + Sync {
    /// Stable identifier (e.g. `"json"`, `"cbor"`). Stored next to the
    /// payload so deserialization can pick the matching format.
    fn name(&self) -> &str;

    /// Serialize an arbitrary `Serialize` value into bytes.
    fn serialize_bytes(&self, value: &serde_json::Value) -> Result<Vec<u8>>;

    /// Deserialize bytes back into a `serde_json::Value`. The two-step
    /// indirection (Value → final type) lets the trait stay non-generic
    /// while still supporting `S: Serialize + DeserializeOwned`.
    fn deserialize_bytes(&self, bytes: &[u8]) -> Result<serde_json::Value>;
}

/// JSON serializer (default). Matches the historical behaviour of the
/// SQLite / Postgres checkpointers.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonSerializer;

impl CheckpointSerializer for JsonSerializer {
    fn name(&self) -> &str {
        "json"
    }
    fn serialize_bytes(&self, value: &serde_json::Value) -> Result<Vec<u8>> {
        serde_json::to_vec(value)
            .map_err(|e| CognisError::Serialization(format!("json serialize: {e}")))
    }
    fn deserialize_bytes(&self, bytes: &[u8]) -> Result<serde_json::Value> {
        serde_json::from_slice(bytes)
            .map_err(|e| CognisError::Serialization(format!("json deserialize: {e}")))
    }
}

/// CBOR serializer. Compact binary; well-supported in the embedded /
/// IoT space. Behind feature `serializer-cbor`.
#[cfg(feature = "serializer-cbor")]
#[derive(Debug, Default, Clone, Copy)]
pub struct CborSerializer;

#[cfg(feature = "serializer-cbor")]
impl CheckpointSerializer for CborSerializer {
    fn name(&self) -> &str {
        "cbor"
    }
    fn serialize_bytes(&self, value: &serde_json::Value) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        ciborium::ser::into_writer(value, &mut out)
            .map_err(|e| CognisError::Serialization(format!("cbor serialize: {e}")))?;
        Ok(out)
    }
    fn deserialize_bytes(&self, bytes: &[u8]) -> Result<serde_json::Value> {
        ciborium::de::from_reader(bytes)
            .map_err(|e| CognisError::Serialization(format!("cbor deserialize: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Helpers used by the SQLite / Postgres checkpointer impls.
// ---------------------------------------------------------------------------

/// Serialize a typed checkpoint value via the given format.
pub fn encode<S: Serialize>(
    serializer: &Arc<dyn CheckpointSerializer>,
    value: &S,
) -> Result<Vec<u8>> {
    let v = serde_json::to_value(value)
        .map_err(|e| CognisError::Serialization(format!("checkpoint to_value: {e}")))?;
    serializer.serialize_bytes(&v)
}

/// Deserialize a typed checkpoint value via the given format.
pub fn decode<S: DeserializeOwned>(
    serializer: &Arc<dyn CheckpointSerializer>,
    bytes: &[u8],
) -> Result<S> {
    let v = serializer.deserialize_bytes(bytes)?;
    serde_json::from_value(v)
        .map_err(|e| CognisError::Serialization(format!("checkpoint from_value: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let s: Arc<dyn CheckpointSerializer> = Arc::new(JsonSerializer);
        let v = serde_json::json!({"a": 1, "b": "hi"});
        let bytes = s.serialize_bytes(&v).unwrap();
        let back = s.deserialize_bytes(&bytes).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn json_serializer_name() {
        assert_eq!(JsonSerializer.name(), "json");
    }

    #[test]
    fn typed_encode_decode_via_helpers() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Foo {
            n: u32,
            s: String,
        }
        let s: Arc<dyn CheckpointSerializer> = Arc::new(JsonSerializer);
        let foo = Foo {
            n: 7,
            s: "hi".into(),
        };
        let bytes = encode(&s, &foo).unwrap();
        let back: Foo = decode(&s, &bytes).unwrap();
        assert_eq!(back, foo);
    }
}
