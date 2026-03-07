//! Checkpoint serialization formats with a pluggable serializer interface.
//!
//! Provides multiple serialization backends for checkpoint data:
//! - [`JsonSerializer`] — human-readable JSON (default)
//! - [`BinarySerializer`] — compact binary with version header
//! - [`CompressedSerializer`] — wraps any serializer with deflate compression
//!
//! Also includes [`CheckpointBundle`] for bulk export/import and
//! [`CheckpointDiff`] for computing state differences between checkpoints.

use std::collections::HashMap;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::Checkpoint;

// ---------------------------------------------------------------------------
// CheckpointSerializer trait
// ---------------------------------------------------------------------------

/// Pluggable serializer for checkpoint data.
pub trait CheckpointSerializer: Send + Sync {
    /// Serialize a checkpoint into bytes.
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>>;

    /// Deserialize a checkpoint from bytes.
    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint>;

    /// Human-readable name of this format (e.g. `"json"`, `"binary"`).
    fn format_name(&self) -> &str;

    /// MIME content type (e.g. `"application/json"`).
    fn content_type(&self) -> &str;
}

// ---------------------------------------------------------------------------
// JsonSerializer
// ---------------------------------------------------------------------------

/// JSON serialization format for checkpoints.
#[derive(Debug, Clone)]
pub struct JsonSerializer {
    /// When `true`, output is pretty-printed with indentation.
    pub pretty: bool,
}

impl JsonSerializer {
    /// Create a compact JSON serializer.
    pub fn compact() -> Self {
        Self { pretty: false }
    }

    /// Create a pretty-printed JSON serializer.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

impl Default for JsonSerializer {
    fn default() -> Self {
        Self::compact()
    }
}

impl CheckpointSerializer for JsonSerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        let bytes = if self.pretty {
            serde_json::to_vec_pretty(checkpoint)
        } else {
            serde_json::to_vec(checkpoint)
        };
        bytes.map_err(|e| LangGraphError::Other(format!("JSON serialization failed: {e}")))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        serde_json::from_slice(data)
            .map_err(|e| LangGraphError::Other(format!("JSON deserialization failed: {e}")))
    }

    fn format_name(&self) -> &str {
        "json"
    }

    fn content_type(&self) -> &str {
        "application/json"
    }
}

// ---------------------------------------------------------------------------
// BinarySerializer
// ---------------------------------------------------------------------------

/// Magic bytes identifying the binary checkpoint format.
const BINARY_MAGIC: [u8; 4] = [0x4C, 0x47, 0x43, 0x50]; // "LGCP"

/// Current binary format version.
const BINARY_VERSION: u16 = 1;

/// Compact binary serialization format for checkpoints.
///
/// Wire format:
/// ```text
/// [4 bytes magic] [2 bytes version BE] [4 bytes payload length BE] [payload …]
/// ```
///
/// The payload is the serde_json-serialized checkpoint encoded as UTF-8 bytes.
/// A future version could swap in a more compact codec without breaking the
/// envelope format — consumers check the version field to decide how to decode.
#[derive(Debug, Clone, Default)]
pub struct BinarySerializer;

impl BinarySerializer {
    /// Header size in bytes: 4 (magic) + 2 (version) + 4 (length).
    const HEADER_SIZE: usize = 10;

    /// Create a new binary serializer.
    pub fn new() -> Self {
        Self
    }
}

impl CheckpointSerializer for BinarySerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        let payload = serde_json::to_vec(checkpoint)
            .map_err(|e| LangGraphError::Other(format!("Binary serialization failed: {e}")))?;

        let len = payload.len() as u32;
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE + payload.len());
        buf.extend_from_slice(&BINARY_MAGIC);
        buf.extend_from_slice(&BINARY_VERSION.to_be_bytes());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        if data.len() < Self::HEADER_SIZE {
            return Err(LangGraphError::Other(
                "Binary data too short for header".into(),
            ));
        }

        if data[..4] != BINARY_MAGIC {
            return Err(LangGraphError::Other("Invalid binary magic bytes".into()));
        }

        let version = u16::from_be_bytes([data[4], data[5]]);
        if version != BINARY_VERSION {
            return Err(LangGraphError::Other(format!(
                "Unsupported binary version: {version}"
            )));
        }

        let payload_len = u32::from_be_bytes([data[6], data[7], data[8], data[9]]) as usize;
        let payload_end = Self::HEADER_SIZE + payload_len;
        if data.len() < payload_end {
            return Err(LangGraphError::Other(
                "Binary data truncated: payload shorter than declared length".into(),
            ));
        }

        let payload = &data[Self::HEADER_SIZE..payload_end];
        serde_json::from_slice(payload)
            .map_err(|e| LangGraphError::Other(format!("Binary deserialization failed: {e}")))
    }

    fn format_name(&self) -> &str {
        "binary"
    }

    fn content_type(&self) -> &str {
        "application/octet-stream"
    }
}

// ---------------------------------------------------------------------------
// CompressedSerializer
// ---------------------------------------------------------------------------

/// Compression level for [`CompressedSerializer`].
#[derive(Debug, Clone, Copy)]
pub enum CompressionLevel {
    /// Fastest compression, larger output.
    Fast,
    /// Balanced speed/size.
    Default,
    /// Best compression ratio, slower.
    Best,
}

impl CompressionLevel {
    fn to_flate2(self) -> flate2::Compression {
        match self {
            Self::Fast => flate2::Compression::fast(),
            Self::Default => flate2::Compression::default(),
            Self::Best => flate2::Compression::best(),
        }
    }
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::Default
    }
}

/// A serializer wrapper that applies deflate compression on top of an inner
/// serializer.
pub struct CompressedSerializer {
    inner: Box<dyn CheckpointSerializer>,
    level: CompressionLevel,
}

impl CompressedSerializer {
    /// Wrap `inner` with default compression level.
    pub fn new(inner: Box<dyn CheckpointSerializer>) -> Self {
        Self {
            inner,
            level: CompressionLevel::Default,
        }
    }

    /// Wrap `inner` with an explicit compression level.
    pub fn with_level(inner: Box<dyn CheckpointSerializer>, level: CompressionLevel) -> Self {
        Self { inner, level }
    }
}

impl CheckpointSerializer for CompressedSerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        let raw = self.inner.serialize(checkpoint)?;

        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), self.level.to_flate2());
        encoder
            .write_all(&raw)
            .map_err(|e| LangGraphError::Other(format!("Compression failed: {e}")))?;
        encoder
            .finish()
            .map_err(|e| LangGraphError::Other(format!("Compression finish failed: {e}")))
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        let mut decoder = flate2::read::DeflateDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| LangGraphError::Other(format!("Decompression failed: {e}")))?;
        self.inner.deserialize(&decompressed)
    }

    fn format_name(&self) -> &str {
        "compressed"
    }

    fn content_type(&self) -> &str {
        "application/x-deflate"
    }
}

// ---------------------------------------------------------------------------
// CheckpointBundle
// ---------------------------------------------------------------------------

/// A bundle for bulk export/import of checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointBundle {
    /// Bundle format version.
    pub version: String,
    /// Serialization format used (e.g. `"json"`).
    pub format: String,
    /// ISO-8601 timestamp of when this bundle was created.
    pub created_at: String,
    /// The checkpoints in this bundle.
    pub checkpoints: Vec<Checkpoint>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl CheckpointBundle {
    /// Create a new empty bundle.
    pub fn new() -> Self {
        let ts = timestamp_now();
        Self {
            version: "1.0".to_string(),
            format: "json".to_string(),
            created_at: ts,
            checkpoints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Export this bundle to a file using the given serializer.
    pub fn export(
        &self,
        path: &std::path::Path,
        serializer: &dyn CheckpointSerializer,
    ) -> Result<()> {
        // Serialize each checkpoint individually, then wrap them in an envelope.
        let envelope = BundleEnvelope {
            version: self.version.clone(),
            format: serializer.format_name().to_string(),
            created_at: self.created_at.clone(),
            metadata: self.metadata.clone(),
            checkpoint_count: self.checkpoints.len(),
        };

        let envelope_bytes = serde_json::to_vec(&envelope)
            .map_err(|e| LangGraphError::Other(format!("Bundle envelope serialization: {e}")))?;

        let mut payload_parts: Vec<Vec<u8>> = Vec::with_capacity(self.checkpoints.len());
        for cp in &self.checkpoints {
            payload_parts.push(serializer.serialize(cp)?);
        }

        // File layout: [4-byte envelope len][envelope][for each checkpoint: 4-byte len, data]
        let mut file = std::fs::File::create(path)
            .map_err(|e| LangGraphError::Other(format!("Failed to create file: {e}")))?;

        let env_len = (envelope_bytes.len() as u32).to_be_bytes();
        file.write_all(&env_len)
            .and_then(|_| file.write_all(&envelope_bytes))
            .map_err(|e| LangGraphError::Other(format!("Write error: {e}")))?;

        for part in &payload_parts {
            let part_len = (part.len() as u32).to_be_bytes();
            file.write_all(&part_len)
                .and_then(|_| file.write_all(part))
                .map_err(|e| LangGraphError::Other(format!("Write error: {e}")))?;
        }

        Ok(())
    }

    /// Import a bundle from raw bytes using the given serializer.
    pub fn import(data: &[u8], serializer: &dyn CheckpointSerializer) -> Result<Self> {
        if data.len() < 4 {
            return Err(LangGraphError::Other("Bundle data too short".into()));
        }

        let env_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let env_end = 4 + env_len;
        if data.len() < env_end {
            return Err(LangGraphError::Other("Bundle envelope truncated".into()));
        }

        let envelope: BundleEnvelope = serde_json::from_slice(&data[4..env_end])
            .map_err(|e| LangGraphError::Other(format!("Bundle envelope parse error: {e}")))?;

        let mut pos = env_end;
        let mut checkpoints = Vec::with_capacity(envelope.checkpoint_count);
        for _ in 0..envelope.checkpoint_count {
            if pos + 4 > data.len() {
                return Err(LangGraphError::Other(
                    "Bundle data truncated (checkpoint header)".into(),
                ));
            }
            let cp_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            if pos + cp_len > data.len() {
                return Err(LangGraphError::Other(
                    "Bundle data truncated (checkpoint body)".into(),
                ));
            }
            checkpoints.push(serializer.deserialize(&data[pos..pos + cp_len])?);
            pos += cp_len;
        }

        Ok(Self {
            version: envelope.version,
            format: envelope.format,
            created_at: envelope.created_at,
            checkpoints,
            metadata: envelope.metadata,
        })
    }
}

impl Default for CheckpointBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal envelope stored at the head of a bundle file.
#[derive(Debug, Serialize, Deserialize)]
struct BundleEnvelope {
    version: String,
    format: String,
    created_at: String,
    metadata: HashMap<String, Value>,
    checkpoint_count: usize,
}

// ---------------------------------------------------------------------------
// CheckpointDiff / StateChange
// ---------------------------------------------------------------------------

/// A change to a single key in the checkpoint's channel values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateChange {
    /// A new key was added.
    Added { key: String, value: Value },
    /// An existing key was removed.
    Removed { key: String },
    /// An existing key changed value.
    Modified {
        key: String,
        old_value: Value,
        new_value: Value,
    },
}

/// A diff between two checkpoints' channel values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointDiff {
    /// ID of the source checkpoint.
    pub from_id: String,
    /// ID of the target checkpoint.
    pub to_id: String,
    /// Individual state changes.
    pub changes: Vec<StateChange>,
}

impl CheckpointDiff {
    /// Compute the diff from `old` to `new` based on their `channel_values`.
    pub fn compute_diff(old: &Checkpoint, new: &Checkpoint) -> Self {
        let mut changes = Vec::new();

        // Detect modified and removed keys.
        for (key, old_val) in &old.channel_values {
            match new.channel_values.get(key) {
                Some(new_val) if new_val != old_val => {
                    changes.push(StateChange::Modified {
                        key: key.clone(),
                        old_value: old_val.clone(),
                        new_value: new_val.clone(),
                    });
                }
                None => {
                    changes.push(StateChange::Removed { key: key.clone() });
                }
                _ => {} // unchanged
            }
        }

        // Detect added keys.
        for (key, new_val) in &new.channel_values {
            if !old.channel_values.contains_key(key) {
                changes.push(StateChange::Added {
                    key: key.clone(),
                    value: new_val.clone(),
                });
            }
        }

        // Sort for deterministic output.
        changes.sort_by(|a, b| {
            let key_a = match a {
                StateChange::Added { key, .. } => key,
                StateChange::Removed { key } => key,
                StateChange::Modified { key, .. } => key,
            };
            let key_b = match b {
                StateChange::Added { key, .. } => key,
                StateChange::Removed { key } => key,
                StateChange::Modified { key, .. } => key,
            };
            key_a.cmp(key_b)
        });

        Self {
            from_id: old.id.clone(),
            to_id: new.id.clone(),
            changes,
        }
    }

    /// Apply this diff to a checkpoint, producing a new checkpoint with the
    /// target state. The returned checkpoint keeps the same metadata as the
    /// input but with `channel_values` updated according to the diff.
    pub fn apply_diff(checkpoint: &Checkpoint, diff: &CheckpointDiff) -> Checkpoint {
        let mut new_values = checkpoint.channel_values.clone();

        for change in &diff.changes {
            match change {
                StateChange::Added { key, value } => {
                    new_values.insert(key.clone(), value.clone());
                }
                StateChange::Removed { key } => {
                    new_values.remove(key);
                }
                StateChange::Modified { key, new_value, .. } => {
                    new_values.insert(key.clone(), new_value.clone());
                }
            }
        }

        Checkpoint {
            v: checkpoint.v,
            id: diff.to_id.clone(),
            ts: checkpoint.ts.clone(),
            channel_values: new_values,
            channel_versions: checkpoint.channel_versions.clone(),
            versions_seen: checkpoint.versions_seen.clone(),
            updated_channels: checkpoint.updated_channels.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple UTC-ish timestamp (same style as the rest of the crate).
fn timestamp_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("1970-01-01T00:00:00Z+{}", secs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::checkpoint::empty_checkpoint;
    use serde_json::json;

    /// Helper: build a checkpoint with some channel values.
    fn sample_checkpoint() -> Checkpoint {
        let mut cp = empty_checkpoint();
        cp.channel_values
            .insert("messages".to_string(), json!(["hello", "world"]));
        cp.channel_values.insert("counter".to_string(), json!(42));
        cp.channel_versions.insert("messages".to_string(), 2);
        cp.channel_versions.insert("counter".to_string(), 1);
        cp
    }

    // -- JSON roundtrip ------------------------------------------------------

    #[test]
    fn test_json_serialize_deserialize_roundtrip() {
        let serializer = JsonSerializer::compact();
        let cp = sample_checkpoint();
        let bytes = serializer.serialize(&cp).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();

        assert_eq!(restored.id, cp.id);
        assert_eq!(restored.v, cp.v);
        assert_eq!(restored.channel_values, cp.channel_values);
        assert_eq!(restored.channel_versions, cp.channel_versions);
    }

    #[test]
    fn test_json_pretty_vs_compact() {
        let cp = sample_checkpoint();
        let compact = JsonSerializer::compact().serialize(&cp).unwrap();
        let pretty = JsonSerializer::pretty().serialize(&cp).unwrap();

        // Pretty-printed output must be larger because of whitespace.
        assert!(pretty.len() > compact.len());

        // Both must round-trip correctly.
        let from_compact = JsonSerializer::compact().deserialize(&compact).unwrap();
        let from_pretty = JsonSerializer::pretty().deserialize(&pretty).unwrap();
        assert_eq!(from_compact.id, from_pretty.id);
        assert_eq!(from_compact.channel_values, from_pretty.channel_values);
    }

    // -- Binary roundtrip ----------------------------------------------------

    #[test]
    fn test_binary_serialize_deserialize_roundtrip() {
        let serializer = BinarySerializer::new();
        let cp = sample_checkpoint();
        let bytes = serializer.serialize(&cp).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();

        assert_eq!(restored.id, cp.id);
        assert_eq!(restored.channel_values, cp.channel_values);
    }

    #[test]
    fn test_binary_version_header() {
        let serializer = BinarySerializer::new();
        let cp = sample_checkpoint();
        let bytes = serializer.serialize(&cp).unwrap();

        // Check magic bytes.
        assert_eq!(&bytes[..4], &BINARY_MAGIC);
        // Check version.
        let version = u16::from_be_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, BINARY_VERSION);
        // Check declared payload length matches actual.
        let payload_len = u32::from_be_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        assert_eq!(payload_len, bytes.len() - BinarySerializer::HEADER_SIZE);
    }

    // -- Compressed roundtrip ------------------------------------------------

    #[test]
    fn test_compressed_serialize_deserialize_roundtrip() {
        let inner = Box::new(JsonSerializer::compact());
        let serializer = CompressedSerializer::new(inner);
        let cp = sample_checkpoint();
        let bytes = serializer.serialize(&cp).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();

        assert_eq!(restored.id, cp.id);
        assert_eq!(restored.channel_values, cp.channel_values);
    }

    #[test]
    fn test_compression_reduces_size() {
        // Build a checkpoint with repetitive data so compression is effective.
        let mut cp = empty_checkpoint();
        let big_value: Vec<&str> = (0..200).map(|_| "repeated_string_value").collect();
        cp.channel_values
            .insert("big".to_string(), json!(big_value));

        let json_bytes = JsonSerializer::compact().serialize(&cp).unwrap();
        let compressed_bytes = CompressedSerializer::new(Box::new(JsonSerializer::compact()))
            .serialize(&cp)
            .unwrap();

        assert!(
            compressed_bytes.len() < json_bytes.len(),
            "compressed ({}) should be smaller than raw JSON ({})",
            compressed_bytes.len(),
            json_bytes.len()
        );
    }

    // -- CheckpointBundle ----------------------------------------------------

    #[test]
    fn test_checkpoint_bundle_export_import() {
        let mut bundle = CheckpointBundle::new();
        bundle.checkpoints.push(sample_checkpoint());
        bundle.checkpoints.push(sample_checkpoint());

        let serializer = JsonSerializer::compact();

        // Serialize to an in-memory buffer via export → bytes.
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("test_bundle_export.lgbundle");
        bundle.export(&tmp_path, &serializer).unwrap();

        let file_bytes = std::fs::read(&tmp_path).unwrap();
        let restored = CheckpointBundle::import(&file_bytes, &serializer).unwrap();

        assert_eq!(restored.checkpoints.len(), 2);
        assert_eq!(restored.version, "1.0");

        // Cleanup.
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_bundle_metadata() {
        let mut bundle = CheckpointBundle::new();
        bundle
            .metadata
            .insert("author".to_string(), json!("test-suite"));
        bundle
            .metadata
            .insert("tags".to_string(), json!(["checkpoint", "export"]));
        bundle.checkpoints.push(sample_checkpoint());

        let serializer = JsonSerializer::compact();
        let tmp_path = std::env::temp_dir().join("test_bundle_meta.lgbundle");
        bundle.export(&tmp_path, &serializer).unwrap();

        let file_bytes = std::fs::read(&tmp_path).unwrap();
        let restored = CheckpointBundle::import(&file_bytes, &serializer).unwrap();

        assert_eq!(restored.metadata["author"], json!("test-suite"));
        assert_eq!(restored.metadata["tags"], json!(["checkpoint", "export"]));

        let _ = std::fs::remove_file(&tmp_path);
    }

    // -- CheckpointDiff / StateChange ----------------------------------------

    #[test]
    fn test_checkpoint_diff_compute() {
        let mut old = sample_checkpoint();
        old.channel_values
            .insert("temp".to_string(), json!("will_be_removed"));

        let mut new = sample_checkpoint();
        new.id = "new-id".to_string();
        new.channel_values.insert("counter".to_string(), json!(99)); // modified
        new.channel_values
            .insert("added_key".to_string(), json!("new")); // added
                                                            // "temp" is absent → removed

        let diff = CheckpointDiff::compute_diff(&old, &new);

        assert_eq!(diff.from_id, old.id);
        assert_eq!(diff.to_id, "new-id");

        let added = diff
            .changes
            .iter()
            .filter(|c| matches!(c, StateChange::Added { .. }))
            .count();
        let removed = diff
            .changes
            .iter()
            .filter(|c| matches!(c, StateChange::Removed { .. }))
            .count();
        let modified = diff
            .changes
            .iter()
            .filter(|c| matches!(c, StateChange::Modified { .. }))
            .count();

        assert_eq!(added, 1);
        assert_eq!(removed, 1);
        assert_eq!(modified, 1);
    }

    #[test]
    fn test_diff_apply_produces_expected_state() {
        let old = sample_checkpoint();

        let mut new = sample_checkpoint();
        new.id = "target-id".to_string();
        new.channel_values.insert("counter".to_string(), json!(100));
        new.channel_values
            .insert("extra".to_string(), json!("bonus"));

        let diff = CheckpointDiff::compute_diff(&old, &new);
        let result = CheckpointDiff::apply_diff(&old, &diff);

        assert_eq!(result.id, "target-id");
        assert_eq!(result.channel_values["counter"], json!(100));
        assert_eq!(result.channel_values["extra"], json!("bonus"));
        assert_eq!(result.channel_values["messages"], json!(["hello", "world"]));
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_empty_checkpoint_serialization() {
        let cp = empty_checkpoint();

        for serializer in &[
            Box::new(JsonSerializer::compact()) as Box<dyn CheckpointSerializer>,
            Box::new(BinarySerializer::new()) as Box<dyn CheckpointSerializer>,
        ] {
            let bytes = serializer.serialize(&cp).unwrap();
            let restored = serializer.deserialize(&bytes).unwrap();
            assert_eq!(restored.id, cp.id);
            assert!(restored.channel_values.is_empty());
        }
    }

    #[test]
    fn test_large_state_serialization() {
        let mut cp = empty_checkpoint();
        // Insert 500 keys.
        for i in 0..500 {
            cp.channel_values.insert(
                format!("key_{i}"),
                json!({"index": i, "data": "x".repeat(100)}),
            );
        }

        let serializer = JsonSerializer::compact();
        let bytes = serializer.serialize(&cp).unwrap();
        let restored = serializer.deserialize(&bytes).unwrap();

        assert_eq!(restored.channel_values.len(), 500);
        assert_eq!(restored.channel_values["key_0"]["index"], json!(0));
    }

    #[test]
    fn test_format_name_and_content_type() {
        let json_ser = JsonSerializer::compact();
        assert_eq!(json_ser.format_name(), "json");
        assert_eq!(json_ser.content_type(), "application/json");

        let bin_ser = BinarySerializer::new();
        assert_eq!(bin_ser.format_name(), "binary");
        assert_eq!(bin_ser.content_type(), "application/octet-stream");

        let comp_ser = CompressedSerializer::new(Box::new(JsonSerializer::compact()));
        assert_eq!(comp_ser.format_name(), "compressed");
        assert_eq!(comp_ser.content_type(), "application/x-deflate");
    }

    #[test]
    fn test_state_change_variants() {
        let added = StateChange::Added {
            key: "k1".to_string(),
            value: json!(1),
        };
        let removed = StateChange::Removed {
            key: "k2".to_string(),
        };
        let modified = StateChange::Modified {
            key: "k3".to_string(),
            old_value: json!("old"),
            new_value: json!("new"),
        };

        // Verify serialization roundtrip for each variant.
        for change in &[added, removed, modified] {
            let json = serde_json::to_string(change).unwrap();
            let restored: StateChange = serde_json::from_str(&json).unwrap();
            assert_eq!(&restored, change);
        }
    }
}
