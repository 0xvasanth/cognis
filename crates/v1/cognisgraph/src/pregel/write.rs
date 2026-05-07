//! Channel writing for the Pregel execution engine.
//!
//! Provides [`PregelChannelWriter`] which collects writes from node executions
//! and applies them to channels, along with supporting types for write entries.

use std::collections::HashMap;

use serde_json::Value;

use crate::channels::base::BaseChannel;
use crate::constants::TASKS;
use crate::errors::{LangGraphError, Result};

/// Sentinel value indicating that the write value should be taken from the
/// node's output (passthrough).
pub const PASSTHROUGH: &str = "__passthrough__";

/// Sentinel value indicating that a write should be skipped entirely.
pub const SKIP_WRITE: &str = "__skip_write__";

/// A single channel write entry.
#[derive(Debug, Clone)]
pub struct ChannelWriteEntry {
    /// The channel name to write to.
    pub channel: String,
    /// The value to write. If `None`, indicates passthrough from node output.
    pub value: Option<Value>,
    /// Whether to skip writing if the resolved value is `null`.
    pub skip_none: bool,
}

impl ChannelWriteEntry {
    /// Create a new write entry for a specific channel and value.
    pub fn new(channel: impl Into<String>, value: Value) -> Self {
        Self {
            channel: channel.into(),
            value: Some(value),
            skip_none: false,
        }
    }

    /// Create a passthrough write entry that uses the node's output value.
    pub fn passthrough(channel: impl Into<String>) -> Self {
        Self {
            channel: channel.into(),
            value: None,
            skip_none: false,
        }
    }

    /// Set the `skip_none` flag.
    pub fn with_skip_none(mut self, skip_none: bool) -> Self {
        self.skip_none = skip_none;
        self
    }

    /// Resolve the actual value to write, substituting `node_output` for passthrough.
    pub fn resolve(&self, node_output: &Value) -> Option<Value> {
        let value = self.value.clone().unwrap_or_else(|| node_output.clone());

        if self.skip_none && value.is_null() {
            None
        } else {
            Some(value)
        }
    }
}

/// Collects writes from node execution and applies them to channels.
///
/// The writer validates entries, resolves passthrough values, and batches
/// updates to the channel map.
#[derive(Debug)]
pub struct PregelChannelWriter {
    /// Accumulated writes as (channel_name, value) pairs.
    writes: Vec<(String, Value)>,
}

impl PregelChannelWriter {
    /// Create a new empty channel writer.
    pub fn new() -> Self {
        Self { writes: Vec::new() }
    }

    /// Add a single write entry, resolving its value against the node output.
    ///
    /// # Arguments
    ///
    /// * `entry` - The write entry describing the target channel and value.
    /// * `node_output` - The output from the node action, used for passthrough.
    pub fn add_write(&mut self, entry: &ChannelWriteEntry, node_output: &Value) {
        if let Some(value) = entry.resolve(node_output) {
            self.writes.push((entry.channel.clone(), value));
        }
    }

    /// Add multiple write entries at once.
    pub fn add_writes(&mut self, entries: &[ChannelWriteEntry], node_output: &Value) {
        for entry in entries {
            self.add_write(entry, node_output);
        }
    }

    /// Add a raw (channel, value) write directly.
    pub fn add_raw_write(&mut self, channel: impl Into<String>, value: Value) {
        self.writes.push((channel.into(), value));
    }

    /// Get a reference to the accumulated writes.
    pub fn pending_writes(&self) -> &[(String, Value)] {
        &self.writes
    }

    /// Consume the writer and return all accumulated writes.
    pub fn into_writes(self) -> Vec<(String, Value)> {
        self.writes
    }

    /// Clear all accumulated writes (used when retrying a task).
    pub fn clear(&mut self) {
        self.writes.clear();
    }

    /// Apply all accumulated writes to the given channels.
    ///
    /// Each channel receives the list of values written to it. The channel's
    /// own `update` method determines how multiple values are merged.
    ///
    /// # Arguments
    ///
    /// * `channels` - Mutable reference to the channel map.
    ///
    /// # Returns
    ///
    /// A set of channel names that were actually updated (their state changed).
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::InvalidUpdateError`] if a write targets a
    /// reserved channel like `TASKS`.
    pub fn apply(
        &self,
        channels: &mut HashMap<String, Box<dyn BaseChannel>>,
    ) -> Result<Vec<String>> {
        // Validate writes.
        for (channel, _) in &self.writes {
            if channel == TASKS {
                return Err(LangGraphError::InvalidUpdateError(
                    "Cannot write to the reserved channel TASKS".into(),
                ));
            }
        }

        // Group writes by channel.
        let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
        for (channel, value) in &self.writes {
            grouped
                .entry(channel.clone())
                .or_default()
                .push(value.clone());
        }

        // Apply grouped writes to channels.
        let mut updated = Vec::new();
        for (channel_name, values) in grouped {
            if let Some(channel) = channels.get_mut(&channel_name) {
                if channel.update(values)? {
                    updated.push(channel_name);
                }
            } else {
                return Err(LangGraphError::InvalidUpdateError(format!(
                    "Channel '{}' not found",
                    channel_name
                )));
            }
        }

        Ok(updated)
    }
}

impl Default for PregelChannelWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Assemble a list of write entries into raw (channel, value) tuples.
///
/// This is a utility function that resolves passthrough values and filters
/// skipped writes.
pub fn assemble_writes(entries: &[ChannelWriteEntry], node_output: &Value) -> Vec<(String, Value)> {
    entries
        .iter()
        .filter_map(|entry| {
            entry
                .resolve(node_output)
                .map(|value| (entry.channel.clone(), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Simple test channel.
    #[derive(Debug, Clone)]
    struct TestChannel {
        key: String,
        value: Option<Value>,
    }

    impl crate::channels::base::BaseChannel for TestChannel {
        fn key(&self) -> &str {
            &self.key
        }

        fn box_clone(&self) -> Box<dyn crate::channels::base::BaseChannel> {
            Box::new(self.clone())
        }

        fn checkpoint(&self) -> Option<Value> {
            self.value.clone()
        }

        fn from_checkpoint(
            &self,
            checkpoint: Option<Value>,
        ) -> Box<dyn crate::channels::base::BaseChannel> {
            Box::new(TestChannel {
                key: self.key.clone(),
                value: checkpoint,
            })
        }

        fn get(&self) -> std::result::Result<Value, LangGraphError> {
            self.value.clone().ok_or(LangGraphError::EmptyChannelError)
        }

        fn is_available(&self) -> bool {
            self.value.is_some()
        }

        fn update(&mut self, values: Vec<Value>) -> std::result::Result<bool, LangGraphError> {
            if let Some(v) = values.into_iter().last() {
                self.value = Some(v);
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    fn make_channels(
        entries: Vec<(&str, Option<Value>)>,
    ) -> HashMap<String, Box<dyn crate::channels::base::BaseChannel>> {
        entries
            .into_iter()
            .map(|(name, value)| {
                let ch: Box<dyn crate::channels::base::BaseChannel> = Box::new(TestChannel {
                    key: name.to_string(),
                    value,
                });
                (name.to_string(), ch)
            })
            .collect()
    }

    #[test]
    fn test_channel_write_entry_new() {
        let entry = ChannelWriteEntry::new("output", json!(42));
        assert_eq!(entry.channel, "output");
        assert_eq!(entry.value, Some(json!(42)));
        assert!(!entry.skip_none);
    }

    #[test]
    fn test_channel_write_entry_passthrough() {
        let entry = ChannelWriteEntry::passthrough("output");
        assert_eq!(entry.channel, "output");
        assert!(entry.value.is_none());
    }

    #[test]
    fn test_channel_write_entry_resolve_with_value() {
        let entry = ChannelWriteEntry::new("out", json!(99));
        let resolved = entry.resolve(&json!({"ignored": true}));
        assert_eq!(resolved, Some(json!(99)));
    }

    #[test]
    fn test_channel_write_entry_resolve_passthrough() {
        let entry = ChannelWriteEntry::passthrough("out");
        let node_output = json!({"result": "hello"});
        let resolved = entry.resolve(&node_output);
        assert_eq!(resolved, Some(json!({"result": "hello"})));
    }

    #[test]
    fn test_channel_write_entry_resolve_skip_none() {
        let entry = ChannelWriteEntry::passthrough("out").with_skip_none(true);
        let resolved = entry.resolve(&Value::Null);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_channel_write_entry_resolve_skip_none_with_value() {
        let entry = ChannelWriteEntry::new("out", json!(42)).with_skip_none(true);
        let resolved = entry.resolve(&Value::Null);
        assert_eq!(resolved, Some(json!(42)));
    }

    #[test]
    fn test_writer_add_write() {
        let mut writer = PregelChannelWriter::new();
        let entry = ChannelWriteEntry::new("out", json!(1));
        writer.add_write(&entry, &Value::Null);

        assert_eq!(writer.pending_writes().len(), 1);
        assert_eq!(writer.pending_writes()[0].0, "out");
        assert_eq!(writer.pending_writes()[0].1, json!(1));
    }

    #[test]
    fn test_writer_add_writes() {
        let mut writer = PregelChannelWriter::new();
        let entries = vec![
            ChannelWriteEntry::new("a", json!(1)),
            ChannelWriteEntry::new("b", json!(2)),
        ];
        writer.add_writes(&entries, &Value::Null);

        assert_eq!(writer.pending_writes().len(), 2);
    }

    #[test]
    fn test_writer_add_raw_write() {
        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("chan", json!("raw"));

        assert_eq!(writer.pending_writes().len(), 1);
        assert_eq!(writer.pending_writes()[0].1, json!("raw"));
    }

    #[test]
    fn test_writer_clear() {
        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("x", json!(1));
        assert_eq!(writer.pending_writes().len(), 1);

        writer.clear();
        assert!(writer.pending_writes().is_empty());
    }

    #[test]
    fn test_writer_into_writes() {
        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("a", json!(1));
        writer.add_raw_write("b", json!(2));

        let writes = writer.into_writes();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].0, "a");
        assert_eq!(writes[1].0, "b");
    }

    #[test]
    fn test_writer_apply_success() {
        let mut channels = make_channels(vec![("a", None), ("b", None)]);

        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("a", json!(10));
        writer.add_raw_write("b", json!(20));

        let updated = writer.apply(&mut channels).unwrap();
        assert_eq!(updated.len(), 2);
        assert!(updated.contains(&"a".to_string()));
        assert!(updated.contains(&"b".to_string()));

        assert_eq!(channels["a"].get().unwrap(), json!(10));
        assert_eq!(channels["b"].get().unwrap(), json!(20));
    }

    #[test]
    fn test_writer_apply_reserved_channel() {
        let mut channels = make_channels(vec![]);

        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write(TASKS, json!(1));

        let result = writer.apply(&mut channels);
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::InvalidUpdateError(msg) => {
                assert!(msg.contains("reserved"));
            }
            other => panic!("Expected InvalidUpdateError, got: {other:?}"),
        }
    }

    #[test]
    fn test_writer_apply_unknown_channel() {
        let mut channels = make_channels(vec![]);

        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("nonexistent", json!(1));

        let result = writer.apply(&mut channels);
        assert!(result.is_err());
    }

    #[test]
    fn test_writer_apply_multiple_writes_same_channel() {
        let mut channels = make_channels(vec![("x", None)]);

        let mut writer = PregelChannelWriter::new();
        writer.add_raw_write("x", json!(1));
        writer.add_raw_write("x", json!(2));

        let updated = writer.apply(&mut channels).unwrap();
        assert_eq!(updated.len(), 1);
        // TestChannel uses last-value semantics.
        assert_eq!(channels["x"].get().unwrap(), json!(2));
    }

    #[test]
    fn test_assemble_writes() {
        let entries = vec![
            ChannelWriteEntry::new("a", json!(1)),
            ChannelWriteEntry::passthrough("b"),
            ChannelWriteEntry::passthrough("c").with_skip_none(true),
        ];
        let node_output = Value::Null;

        let writes = assemble_writes(&entries, &node_output);
        // "a" has an explicit value, "b" gets null (passthrough), "c" is skipped (skip_none + null).
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0], ("a".to_string(), json!(1)));
        assert_eq!(writes[1], ("b".to_string(), Value::Null));
    }

    #[test]
    fn test_assemble_writes_with_output() {
        let entries = vec![ChannelWriteEntry::passthrough("result")];
        let node_output = json!({"key": "value"});

        let writes = assemble_writes(&entries, &node_output);
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].1, json!({"key": "value"}));
    }

    #[test]
    fn test_default_writer() {
        let writer = PregelChannelWriter::default();
        assert!(writer.pending_writes().is_empty());
    }
}
