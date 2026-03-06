//! Checkpoint management for the Pregel execution engine.
//!
//! Provides the [`Checkpoint`] struct for serializing graph state at a point in
//! time, the [`CheckpointSaver`] trait for persistence backends, and helper
//! functions for creating and restoring checkpoints.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::channels::base::BaseChannel;
use crate::errors::{LangGraphError, Result};

/// Current checkpoint format version.
pub const LATEST_VERSION: u32 = 4;

/// A checkpoint representing the serialized state of a Pregel graph at a
/// specific point in execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint format version.
    pub v: u32,
    /// Unique identifier for this checkpoint (UUID v7 style, time-ordered).
    pub id: String,
    /// ISO-8601 timestamp of when the checkpoint was created.
    pub ts: String,
    /// Serialized channel values keyed by channel name.
    pub channel_values: HashMap<String, Value>,
    /// Version counter for each channel, incremented on each update.
    pub channel_versions: HashMap<String, u64>,
    /// For each node, the channel versions it has seen. Used to determine
    /// which channels have been updated since a node last ran.
    pub versions_seen: HashMap<String, HashMap<String, u64>>,
    /// Channels that were updated in this checkpoint (if tracked).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_channels: Option<Vec<String>>,
}

/// Metadata associated with a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// The source of the checkpoint (e.g. "loop", "input", "update").
    pub source: String,
    /// The step number in the execution.
    pub step: i64,
    /// Writes that produced this checkpoint.
    pub writes: Option<HashMap<String, Value>>,
    /// Additional key-value metadata.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// A checkpoint tuple containing the checkpoint along with its config and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTuple {
    /// The checkpoint data.
    pub checkpoint: Checkpoint,
    /// The config that identifies this checkpoint (thread_id, checkpoint_ns, etc.).
    pub config: HashMap<String, Value>,
    /// Metadata about the checkpoint.
    pub metadata: Option<CheckpointMetadata>,
    /// Config of the parent checkpoint, if any.
    pub parent_config: Option<HashMap<String, Value>>,
    /// Pending writes not yet committed.
    pub pending_writes: Option<Vec<(String, String, Value)>>,
}

/// A lightweight entry summarizing a checkpoint for time-travel listing.
///
/// Unlike [`CheckpointTuple`], this struct carries only the fields needed for
/// browsing the checkpoint history (e.g. picking a point to replay or fork from).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    /// Unique identifier for this checkpoint.
    pub checkpoint_id: String,
    /// The thread this checkpoint belongs to.
    pub thread_id: String,
    /// The node that produced this checkpoint (from metadata source/writes).
    pub node_name: String,
    /// Epoch milliseconds when the checkpoint was created.
    pub timestamp: u64,
    /// The full state snapshot at this point.
    pub state: Value,
}

/// Trait for checkpoint persistence backends.
///
/// Implementations store and retrieve checkpoints for durable graph execution.
/// Common backends include in-memory (for testing), SQLite, and Postgres.
#[async_trait]
pub trait CheckpointSaver: Send + Sync + std::fmt::Debug {
    /// Get the latest checkpoint matching the given config.
    ///
    /// The config typically contains `thread_id` and optionally `checkpoint_ns`
    /// and `checkpoint_id` to narrow the search.
    async fn get(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>>;

    /// Get a specific checkpoint by its config.
    async fn get_tuple(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>>;

    /// Save a checkpoint along with its metadata.
    ///
    /// Returns the updated config with the new checkpoint ID.
    async fn put(
        &self,
        config: &HashMap<String, Value>,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<HashMap<String, Value>>;

    /// Save pending writes for a checkpoint.
    ///
    /// These are writes that have been collected but not yet committed to
    /// a new checkpoint.
    async fn put_writes(
        &self,
        config: &HashMap<String, Value>,
        writes: Vec<(String, Value)>,
        task_id: &str,
    ) -> Result<()>;

    /// List checkpoints matching the given config, ordered newest first.
    async fn list(
        &self,
        config: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>>;

    /// List checkpoint entries for a given thread, ordered oldest first.
    ///
    /// Returns a lightweight [`CheckpointEntry`] for each checkpoint, suitable
    /// for browsing history and selecting a point for replay or forking.
    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<CheckpointEntry>>;
}

/// Create an empty checkpoint with default values.
///
/// This is the initial checkpoint used when starting a new graph execution.
pub fn empty_checkpoint() -> Checkpoint {
    let id = Uuid::now_v7().to_string();
    let ts = chrono_like_now();

    Checkpoint {
        v: LATEST_VERSION,
        id,
        ts,
        channel_values: HashMap::new(),
        channel_versions: HashMap::new(),
        versions_seen: HashMap::new(),
        updated_channels: None,
    }
}

/// Create a new checkpoint from the current channel state.
///
/// Snapshots each channel that has a version recorded in the previous checkpoint.
///
/// # Arguments
///
/// * `previous` - The previous checkpoint (used for versions).
/// * `channels` - Current channel instances. If `None`, reuses channel_values
///   from the previous checkpoint.
/// * `step` - The current execution step number (used for generating the ID).
/// * `id` - Optional explicit checkpoint ID. If `None`, a new UUID is generated.
/// * `updated_channels` - Optional set of channel names that were updated.
pub fn create_checkpoint(
    previous: &Checkpoint,
    channels: Option<&HashMap<String, Box<dyn BaseChannel>>>,
    _step: i64,
    id: Option<String>,
    updated_channels: Option<Vec<String>>,
) -> Checkpoint {
    let ts = chrono_like_now();
    let checkpoint_id = id.unwrap_or_else(|| Uuid::now_v7().to_string());

    let channel_values = match channels {
        Some(ch) => {
            let mut values = HashMap::new();
            for (key, channel) in ch {
                // Only include channels that have a recorded version.
                if previous.channel_versions.contains_key(key) {
                    if let Some(v) = channel.checkpoint() {
                        values.insert(key.clone(), v);
                    }
                }
            }
            values
        }
        None => previous.channel_values.clone(),
    };

    let sorted_updated = updated_channels.map(|mut v| {
        v.sort();
        v
    });

    Checkpoint {
        v: LATEST_VERSION,
        id: checkpoint_id,
        ts,
        channel_values,
        channel_versions: previous.channel_versions.clone(),
        versions_seen: previous.versions_seen.clone(),
        updated_channels: sorted_updated,
    }
}

/// Restore channels from a checkpoint.
///
/// For each channel spec, creates a new channel instance initialized from the
/// corresponding checkpoint value.
///
/// # Arguments
///
/// * `specs` - Map of channel name to a prototype channel used to create new instances.
/// * `checkpoint` - The checkpoint to restore from.
///
/// # Returns
///
/// A new map of channel name to restored channel instance.
pub fn channels_from_checkpoint(
    specs: &HashMap<String, Box<dyn BaseChannel>>,
    checkpoint: &Checkpoint,
) -> HashMap<String, Box<dyn BaseChannel>> {
    specs
        .iter()
        .map(|(key, spec)| {
            let value = checkpoint.channel_values.get(key).cloned();
            (key.clone(), spec.from_checkpoint(value))
        })
        .collect()
}

/// Create a deep copy of a checkpoint.
pub fn copy_checkpoint(checkpoint: &Checkpoint) -> Checkpoint {
    Checkpoint {
        v: checkpoint.v,
        id: checkpoint.id.clone(),
        ts: checkpoint.ts.clone(),
        channel_values: checkpoint.channel_values.clone(),
        channel_versions: checkpoint.channel_versions.clone(),
        versions_seen: checkpoint
            .versions_seen
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        updated_channels: checkpoint.updated_channels.clone(),
    }
}

/// Generate an ISO-8601 UTC timestamp string.
///
/// Uses a simple approach without pulling in the `chrono` crate.
fn chrono_like_now() -> String {
    // Use std::time to get a timestamp. For proper ISO formatting we approximate
    // since `chrono` is not in our dependencies.
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple ISO-like format: just use the unix timestamp.
    // A production implementation would use the `chrono` crate.
    format!("1970-01-01T00:00:00Z+{}", secs)
}

/// An in-memory checkpoint saver for testing purposes.
#[derive(Debug, Default)]
pub struct InMemoryCheckpointSaver {
    storage: tokio::sync::RwLock<Vec<(HashMap<String, Value>, CheckpointTuple)>>,
}

impl InMemoryCheckpointSaver {
    /// Create a new empty in-memory saver.
    pub fn new() -> Self {
        Self {
            storage: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl CheckpointSaver for InMemoryCheckpointSaver {
    async fn get(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        let storage = self.storage.read().await;
        let thread_id = config.get("thread_id");

        for (stored_config, tuple) in storage.iter().rev() {
            if thread_id.is_some() && stored_config.get("thread_id") == thread_id {
                return Ok(Some(tuple.clone()));
            }
        }
        Ok(None)
    }

    async fn get_tuple(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        let storage = self.storage.read().await;
        let thread_id = config.get("thread_id");
        let checkpoint_id = config.get("checkpoint_id");

        for (stored_config, tuple) in storage.iter().rev() {
            let thread_match = thread_id.is_none() || stored_config.get("thread_id") == thread_id;
            let id_match = checkpoint_id.is_none()
                || checkpoint_id.and_then(|v| v.as_str()) == Some(&tuple.checkpoint.id);

            if thread_match && id_match {
                return Ok(Some(tuple.clone()));
            }
        }
        Ok(None)
    }

    async fn put(
        &self,
        config: &HashMap<String, Value>,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<HashMap<String, Value>> {
        let mut new_config = config.clone();
        new_config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint.id.clone()),
        );

        let tuple = CheckpointTuple {
            checkpoint,
            config: new_config.clone(),
            metadata: Some(metadata),
            parent_config: Some(config.clone()),
            pending_writes: None,
        };

        let mut storage = self.storage.write().await;
        storage.push((new_config.clone(), tuple));

        Ok(new_config)
    }

    async fn put_writes(
        &self,
        config: &HashMap<String, Value>,
        writes: Vec<(String, Value)>,
        task_id: &str,
    ) -> Result<()> {
        let mut storage = self.storage.write().await;
        let thread_id = config.get("thread_id");
        let checkpoint_id = config
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        for (_, tuple) in storage.iter_mut().rev() {
            if thread_id.is_some()
                && tuple.config.get("thread_id") == thread_id
                && tuple.checkpoint.id == checkpoint_id
            {
                let pending = tuple.pending_writes.get_or_insert_with(Vec::new);
                for (channel, value) in &writes {
                    pending.push((task_id.to_string(), channel.clone(), value.clone()));
                }
                return Ok(());
            }
        }

        Err(LangGraphError::Other(
            "Checkpoint not found for put_writes".into(),
        ))
    }

    async fn list(
        &self,
        config: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>> {
        let storage = self.storage.read().await;
        let thread_id = config.get("thread_id");

        let mut results: Vec<CheckpointTuple> = storage
            .iter()
            .rev()
            .filter(|(stored_config, _)| {
                thread_id.is_none() || stored_config.get("thread_id") == thread_id
            })
            .map(|(_, tuple)| tuple.clone())
            .collect();

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<CheckpointEntry>> {
        let storage = self.storage.read().await;
        let tid = Value::String(thread_id.to_string());

        let entries: Vec<CheckpointEntry> = storage
            .iter()
            .filter(|(stored_config, _)| stored_config.get("thread_id") == Some(&tid))
            .map(|(_, tuple)| {
                let node_name = tuple
                    .metadata
                    .as_ref()
                    .and_then(|m| {
                        // Use the first key in writes as the node name, falling
                        // back to the source field.
                        m.writes
                            .as_ref()
                            .and_then(|w| w.keys().next().cloned())
                            .or_else(|| Some(m.source.clone()))
                    })
                    .unwrap_or_default();

                // Parse the timestamp from the checkpoint ts field.
                let timestamp = tuple
                    .checkpoint
                    .ts
                    .rsplit('+')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0)
                    * 1000;

                // Build the state from channel_values.
                let state = Value::Object(
                    tuple
                        .checkpoint
                        .channel_values
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );

                CheckpointEntry {
                    checkpoint_id: tuple.checkpoint.id.clone(),
                    thread_id: thread_id.to_string(),
                    node_name,
                    timestamp,
                    state,
                }
            })
            .collect();

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_empty_checkpoint() {
        let cp = empty_checkpoint();
        assert_eq!(cp.v, LATEST_VERSION);
        assert!(!cp.id.is_empty());
        assert!(!cp.ts.is_empty());
        assert!(cp.channel_values.is_empty());
        assert!(cp.channel_versions.is_empty());
        assert!(cp.versions_seen.is_empty());
        assert!(cp.updated_channels.is_none());
    }

    #[test]
    fn test_create_checkpoint_without_channels() {
        let previous = empty_checkpoint();
        let cp = create_checkpoint(&previous, None, 1, None, None);
        assert_eq!(cp.v, LATEST_VERSION);
        assert_ne!(cp.id, previous.id);
        assert!(cp.channel_values.is_empty());
    }

    #[test]
    fn test_create_checkpoint_with_channels() {
        let mut previous = empty_checkpoint();
        previous.channel_versions.insert("state".to_string(), 1);

        // Create a test channel.
        #[derive(Debug, Clone)]
        struct TestCh {
            key: String,
            value: Option<Value>,
        }
        impl BaseChannel for TestCh {
            fn key(&self) -> &str {
                &self.key
            }
            fn box_clone(&self) -> Box<dyn BaseChannel> {
                Box::new(self.clone())
            }
            fn checkpoint(&self) -> Option<Value> {
                self.value.clone()
            }
            fn from_checkpoint(&self, cp: Option<Value>) -> Box<dyn BaseChannel> {
                Box::new(TestCh {
                    key: self.key.clone(),
                    value: cp,
                })
            }
            fn get(&self) -> Result<Value> {
                self.value.clone().ok_or(LangGraphError::EmptyChannelError)
            }
            fn is_available(&self) -> bool {
                self.value.is_some()
            }
            fn update(&mut self, values: Vec<Value>) -> Result<bool> {
                if let Some(v) = values.into_iter().last() {
                    self.value = Some(v);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }

        let mut channels: HashMap<String, Box<dyn BaseChannel>> = HashMap::new();
        channels.insert(
            "state".to_string(),
            Box::new(TestCh {
                key: "state".to_string(),
                value: Some(json!({"count": 5})),
            }),
        );
        // Channel without a version should be excluded.
        channels.insert(
            "unversioned".to_string(),
            Box::new(TestCh {
                key: "unversioned".to_string(),
                value: Some(json!(42)),
            }),
        );

        let cp = create_checkpoint(&previous, Some(&channels), 2, None, None);
        assert_eq!(cp.channel_values.len(), 1);
        assert_eq!(cp.channel_values["state"], json!({"count": 5}));
        assert!(!cp.channel_values.contains_key("unversioned"));
    }

    #[test]
    fn test_create_checkpoint_with_explicit_id() {
        let previous = empty_checkpoint();
        let cp = create_checkpoint(&previous, None, 1, Some("custom-id-123".to_string()), None);
        assert_eq!(cp.id, "custom-id-123");
    }

    #[test]
    fn test_create_checkpoint_with_updated_channels() {
        let previous = empty_checkpoint();
        let cp = create_checkpoint(
            &previous,
            None,
            1,
            None,
            Some(vec!["b".to_string(), "a".to_string()]),
        );
        // Should be sorted.
        assert_eq!(
            cp.updated_channels,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_copy_checkpoint() {
        let mut original = empty_checkpoint();
        original.channel_values.insert("x".to_string(), json!(42));
        original.channel_versions.insert("x".to_string(), 3);
        original.versions_seen.insert("node_a".to_string(), {
            let mut m = HashMap::new();
            m.insert("x".to_string(), 2);
            m
        });

        let copy = copy_checkpoint(&original);
        assert_eq!(copy.v, original.v);
        assert_eq!(copy.id, original.id);
        assert_eq!(copy.ts, original.ts);
        assert_eq!(copy.channel_values, original.channel_values);
        assert_eq!(copy.channel_versions, original.channel_versions);
        assert_eq!(copy.versions_seen, original.versions_seen);
    }

    #[test]
    fn test_channels_from_checkpoint() {
        #[derive(Debug, Clone)]
        struct TestCh {
            key: String,
            value: Option<Value>,
        }
        impl BaseChannel for TestCh {
            fn key(&self) -> &str {
                &self.key
            }
            fn box_clone(&self) -> Box<dyn BaseChannel> {
                Box::new(self.clone())
            }
            fn checkpoint(&self) -> Option<Value> {
                self.value.clone()
            }
            fn from_checkpoint(&self, cp: Option<Value>) -> Box<dyn BaseChannel> {
                Box::new(TestCh {
                    key: self.key.clone(),
                    value: cp,
                })
            }
            fn get(&self) -> Result<Value> {
                self.value.clone().ok_or(LangGraphError::EmptyChannelError)
            }
            fn is_available(&self) -> bool {
                self.value.is_some()
            }
            fn update(&mut self, values: Vec<Value>) -> Result<bool> {
                Ok(values.into_iter().last().is_some())
            }
        }

        let mut specs: HashMap<String, Box<dyn BaseChannel>> = HashMap::new();
        specs.insert(
            "a".to_string(),
            Box::new(TestCh {
                key: "a".to_string(),
                value: None,
            }),
        );
        specs.insert(
            "b".to_string(),
            Box::new(TestCh {
                key: "b".to_string(),
                value: None,
            }),
        );

        let mut cp = empty_checkpoint();
        cp.channel_values
            .insert("a".to_string(), json!("restored_a"));

        let restored = channels_from_checkpoint(&specs, &cp);
        assert_eq!(restored.len(), 2);
        assert!(restored["a"].is_available());
        assert_eq!(restored["a"].get().unwrap(), json!("restored_a"));
        assert!(!restored["b"].is_available());
    }

    #[test]
    fn test_checkpoint_serialization_roundtrip() {
        let mut cp = empty_checkpoint();
        cp.channel_values
            .insert("state".to_string(), json!({"key": "value"}));
        cp.channel_versions.insert("state".to_string(), 5);

        let serialized = serde_json::to_string(&cp).unwrap();
        let deserialized: Checkpoint = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.v, cp.v);
        assert_eq!(deserialized.id, cp.id);
        assert_eq!(deserialized.channel_values, cp.channel_values);
        assert_eq!(deserialized.channel_versions, cp.channel_versions);
    }

    #[tokio::test]
    async fn test_in_memory_saver_put_and_get() {
        let saver = InMemoryCheckpointSaver::new();

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));

        let cp = empty_checkpoint();
        let metadata = CheckpointMetadata {
            source: "loop".to_string(),
            step: 1,
            writes: None,
            extra: HashMap::new(),
        };

        let new_config = saver.put(&config, cp.clone(), metadata).await.unwrap();
        assert!(new_config.contains_key("checkpoint_id"));

        let retrieved = saver.get(&config).await.unwrap();
        assert!(retrieved.is_some());
        let tuple = retrieved.unwrap();
        assert_eq!(tuple.checkpoint.id, cp.id);
    }

    #[tokio::test]
    async fn test_in_memory_saver_get_tuple() {
        let saver = InMemoryCheckpointSaver::new();

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        let metadata = CheckpointMetadata {
            source: "loop".to_string(),
            step: 1,
            writes: None,
            extra: HashMap::new(),
        };

        saver.put(&config, cp, metadata).await.unwrap();

        let mut lookup = config.clone();
        lookup.insert("checkpoint_id".to_string(), json!(cp_id));

        let retrieved = saver.get_tuple(&lookup).await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_in_memory_saver_list() {
        let saver = InMemoryCheckpointSaver::new();

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));

        for i in 0..5 {
            let cp = empty_checkpoint();
            let metadata = CheckpointMetadata {
                source: "loop".to_string(),
                step: i,
                writes: None,
                extra: HashMap::new(),
            };
            saver.put(&config, cp, metadata).await.unwrap();
        }

        let all = saver.list(&config, None).await.unwrap();
        assert_eq!(all.len(), 5);

        let limited = saver.list(&config, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_in_memory_saver_put_writes() {
        let saver = InMemoryCheckpointSaver::new();

        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));

        let cp = empty_checkpoint();
        let _cp_id = cp.id.clone();
        let metadata = CheckpointMetadata {
            source: "loop".to_string(),
            step: 0,
            writes: None,
            extra: HashMap::new(),
        };

        let new_config = saver.put(&config, cp, metadata).await.unwrap();

        let writes = vec![("state".to_string(), json!({"updated": true}))];
        saver
            .put_writes(&new_config, writes, "task-1")
            .await
            .unwrap();

        let retrieved = saver.get(&config).await.unwrap().unwrap();
        let pending = retrieved.pending_writes.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, "task-1");
        assert_eq!(pending[0].1, "state");
    }

    #[tokio::test]
    async fn test_in_memory_saver_get_empty() {
        let saver = InMemoryCheckpointSaver::new();
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("unknown"));

        let result = saver.get(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_in_memory_saver_put_writes_not_found() {
        let saver = InMemoryCheckpointSaver::new();
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));
        config.insert("checkpoint_id".to_string(), json!("nonexistent"));

        let result = saver.put_writes(&config, vec![], "task-1").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_metadata_serialization() {
        let metadata = CheckpointMetadata {
            source: "input".to_string(),
            step: 0,
            writes: Some({
                let mut m = HashMap::new();
                m.insert("channel".to_string(), json!("value"));
                m
            }),
            extra: HashMap::new(),
        };

        let serialized = serde_json::to_string(&metadata).unwrap();
        let deserialized: CheckpointMetadata = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.source, "input");
        assert_eq!(deserialized.step, 0);
        assert!(deserialized.writes.is_some());
    }
}
