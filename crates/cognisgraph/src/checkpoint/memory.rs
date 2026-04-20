//! In-memory checkpoint persistence for testing and development.
//!
//! Provides [`MemoryCheckpointSaver`], a thread-safe, ephemeral checkpoint
//! backend that stores all state in memory using `Arc<RwLock<HashMap>>`. This
//! is ideal for unit tests and rapid prototyping where durability is not
//! required.
//!
//! # Example
//!
//! ```
//! use cognisgraph::checkpoint::memory::MemoryCheckpointSaver;
//!
//! let saver = MemoryCheckpointSaver::new();
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::{
    Checkpoint, CheckpointEntry, CheckpointMetadata, CheckpointSaver, CheckpointTuple,
};

/// A stored checkpoint with its full context.
///
/// This struct extends the core [`Checkpoint`] with metadata, channel
/// information, and a creation timestamp for ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCheckpoint {
    /// Unique identifier for this checkpoint.
    pub checkpoint_id: String,
    /// The thread this checkpoint belongs to.
    pub thread_id: String,
    /// The serialized graph state.
    pub state: Value,
    /// Metadata describing how this checkpoint was created.
    pub metadata: CheckpointMeta,
    /// Serialized channel values keyed by channel name.
    pub channel_values: HashMap<String, Value>,
    /// Version counter for each channel.
    pub channel_versions: HashMap<String, u64>,
}

/// Metadata for an in-memory stored checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// The source of the checkpoint (e.g. "input", "loop", "update").
    pub source: String,
    /// The step number in the execution.
    pub step: i64,
    /// Writes that produced this checkpoint.
    pub writes: HashMap<String, Value>,
    /// When this checkpoint was created.
    #[serde(with = "system_time_serde")]
    pub created_at: SystemTime,
    /// The parent checkpoint ID, if any.
    pub parent_id: Option<String>,
}

mod system_time_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_millis(millis))
    }
}

/// Internal entry stored in the in-memory map.
#[derive(Debug, Clone)]
struct InternalEntry {
    /// The full checkpoint data (for trait compatibility).
    checkpoint: Checkpoint,
    /// The checkpoint metadata (core format).
    metadata: CheckpointMetadata,
    /// Config identifying the parent checkpoint, if any.
    parent_config: Option<HashMap<String, Value>>,
    /// Pending writes not yet committed to a new checkpoint.
    pending_writes: Option<Vec<(String, String, Value)>>,
    /// Monotonic insertion order for deterministic latest/list operations.
    seq: u64,
}

/// Thread-safe in-memory checkpoint store.
///
/// All checkpoints are keyed by `(thread_id, checkpoint_id)` and stored in a
/// `HashMap` behind an `Arc<RwLock<…>>`, making this type cheaply cloneable
/// and safe to share across async tasks.
///
/// Unlike the simpler [`InMemoryCheckpointSaver`](crate::pregel::checkpoint::InMemoryCheckpointSaver)
/// in the pregel module, this implementation provides additional utility
/// methods for deleting checkpoints, clearing threads, listing thread IDs,
/// and counting stored checkpoints.
#[derive(Debug, Clone)]
pub struct MemoryCheckpointSaver {
    /// The underlying storage: `(thread_id, checkpoint_id) -> InternalEntry`.
    storage: Arc<RwLock<HashMap<(String, String), InternalEntry>>>,
    /// Monotonically increasing sequence counter for insertion ordering.
    counter: Arc<RwLock<u64>>,
}

impl Default for MemoryCheckpointSaver {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCheckpointSaver {
    /// Create a new, empty in-memory checkpoint saver.
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Store a checkpoint.
    ///
    /// # Arguments
    ///
    /// * `thread_id` — The thread this checkpoint belongs to.
    /// * `checkpoint_id` — Unique identifier for this checkpoint.
    /// * `state` — The serialized graph state.
    /// * `metadata` — Metadata describing how this checkpoint was created.
    pub async fn put_stored(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
        state: Value,
        metadata: CheckpointMeta,
    ) -> Result<()> {
        let checkpoint = Checkpoint {
            v: crate::pregel::checkpoint::LATEST_VERSION,
            id: checkpoint_id.to_string(),
            ts: format!(
                "1970-01-01T00:00:00Z+{}",
                SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            ),
            channel_values: {
                let mut cv = HashMap::new();
                cv.insert("__state__".to_string(), state);
                cv
            },
            channel_versions: HashMap::new(),
            versions_seen: HashMap::new(),
            updated_channels: None,
        };

        let core_metadata = CheckpointMetadata {
            source: metadata.source.clone(),
            step: metadata.step,
            writes: if metadata.writes.is_empty() {
                None
            } else {
                Some(metadata.writes.clone())
            },
            extra: HashMap::new(),
        };

        let mut seq_counter = self.counter.write().await;
        *seq_counter += 1;
        let seq = *seq_counter;
        drop(seq_counter);

        let parent_config = metadata.parent_id.as_ref().map(|pid| {
            let mut pc = HashMap::new();
            pc.insert(
                "thread_id".to_string(),
                Value::String(thread_id.to_string()),
            );
            pc.insert("checkpoint_id".to_string(), Value::String(pid.clone()));
            pc
        });

        let entry = InternalEntry {
            checkpoint,
            metadata: core_metadata,
            parent_config,
            pending_writes: None,
            seq,
        };

        let mut storage = self.storage.write().await;
        storage.insert((thread_id.to_string(), checkpoint_id.to_string()), entry);

        Ok(())
    }

    /// Retrieve a specific checkpoint.
    pub async fn get_stored(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
    ) -> Result<Option<StoredCheckpoint>> {
        let storage = self.storage.read().await;
        let key = (thread_id.to_string(), checkpoint_id.to_string());
        Ok(storage
            .get(&key)
            .map(|entry| self.entry_to_stored(thread_id, entry)))
    }

    /// Retrieve the latest checkpoint for a thread (by insertion order).
    pub async fn get_latest(&self, thread_id: &str) -> Result<Option<StoredCheckpoint>> {
        let storage = self.storage.read().await;
        let latest = storage
            .iter()
            .filter(|((tid, _), _)| tid == thread_id)
            .max_by_key(|(_, entry)| entry.seq);

        Ok(latest.map(|((_, _), entry)| self.entry_to_stored(thread_id, entry)))
    }

    /// List all checkpoints for a thread, ordered newest first.
    pub async fn list_stored(&self, thread_id: &str) -> Result<Vec<StoredCheckpoint>> {
        let storage = self.storage.read().await;
        let mut entries: Vec<_> = storage
            .iter()
            .filter(|((tid, _), _)| tid == thread_id)
            .collect();

        entries.sort_by_key(|b| std::cmp::Reverse(b.1.seq));

        Ok(entries
            .into_iter()
            .map(|((_, _), entry)| self.entry_to_stored(thread_id, entry))
            .collect())
    }

    /// Delete a specific checkpoint. Returns `true` if it existed.
    pub async fn delete(&self, thread_id: &str, checkpoint_id: &str) -> Result<bool> {
        let mut storage = self.storage.write().await;
        let key = (thread_id.to_string(), checkpoint_id.to_string());
        Ok(storage.remove(&key).is_some())
    }

    /// Delete all checkpoints for a thread. Returns the number removed.
    pub async fn clear_thread(&self, thread_id: &str) -> Result<usize> {
        let mut storage = self.storage.write().await;
        let keys_to_remove: Vec<_> = storage
            .keys()
            .filter(|(tid, _)| tid == thread_id)
            .cloned()
            .collect();
        let count = keys_to_remove.len();
        for key in keys_to_remove {
            storage.remove(&key);
        }
        Ok(count)
    }

    /// List all thread IDs that have at least one checkpoint.
    pub async fn thread_ids(&self) -> Vec<String> {
        let storage = self.storage.read().await;
        let mut ids: Vec<String> = storage
            .keys()
            .map(|(tid, _)| tid.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort();
        ids
    }

    /// Return the total number of checkpoints across all threads.
    pub async fn total_checkpoints(&self) -> usize {
        let storage = self.storage.read().await;
        storage.len()
    }

    /// Convert an internal entry to a [`StoredCheckpoint`].
    fn entry_to_stored(&self, thread_id: &str, entry: &InternalEntry) -> StoredCheckpoint {
        let state = entry
            .checkpoint
            .channel_values
            .get("__state__")
            .cloned()
            .unwrap_or(Value::Object(
                entry
                    .checkpoint
                    .channel_values
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ));

        let parent_id = entry
            .parent_config
            .as_ref()
            .and_then(|pc| pc.get("checkpoint_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        StoredCheckpoint {
            checkpoint_id: entry.checkpoint.id.clone(),
            thread_id: thread_id.to_string(),
            state,
            metadata: CheckpointMeta {
                source: entry.metadata.source.clone(),
                step: entry.metadata.step,
                writes: entry.metadata.writes.clone().unwrap_or_default(),
                created_at: SystemTime::now(), // approximate; real ts is in checkpoint.ts
                parent_id,
            },
            channel_values: entry.checkpoint.channel_values.clone(),
            channel_versions: entry.checkpoint.channel_versions.clone(),
        }
    }

    /// Extract `(thread_id, checkpoint_ns, checkpoint_id)` from a config map.
    fn extract_config(config: &HashMap<String, Value>) -> (&str, &str, Option<&str>) {
        let thread_id = config
            .get("thread_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let checkpoint_ns = config
            .get("checkpoint_ns")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let checkpoint_id = config.get("checkpoint_id").and_then(|v| v.as_str());
        (thread_id, checkpoint_ns, checkpoint_id)
    }
}

#[async_trait]
impl CheckpointSaver for MemoryCheckpointSaver {
    async fn get(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        let (thread_id, _, checkpoint_id) = Self::extract_config(config);

        let storage = self.storage.read().await;

        if let Some(cid) = checkpoint_id {
            let key = (thread_id.to_string(), cid.to_string());
            return Ok(storage
                .get(&key)
                .map(|entry| self.entry_to_tuple(thread_id, entry)));
        }

        // No checkpoint_id: return the latest for this thread.
        let latest = storage
            .iter()
            .filter(|((tid, _), _)| tid == thread_id)
            .max_by_key(|(_, entry)| entry.seq);

        Ok(latest.map(|(_, entry)| self.entry_to_tuple(thread_id, entry)))
    }

    async fn get_tuple(&self, config: &HashMap<String, Value>) -> Result<Option<CheckpointTuple>> {
        self.get(config).await
    }

    async fn put(
        &self,
        config: &HashMap<String, Value>,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Result<HashMap<String, Value>> {
        let (thread_id, _, _) = Self::extract_config(config);
        let thread_id = thread_id.to_string();
        let checkpoint_id = checkpoint.id.clone();

        let parent_checkpoint_id = config
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let parent_config = parent_checkpoint_id.map(|pid| {
            let mut pc = HashMap::new();
            pc.insert("thread_id".to_string(), Value::String(thread_id.clone()));
            pc.insert("checkpoint_id".to_string(), Value::String(pid));
            pc
        });

        let mut seq_counter = self.counter.write().await;
        *seq_counter += 1;
        let seq = *seq_counter;
        drop(seq_counter);

        let entry = InternalEntry {
            checkpoint,
            metadata,
            parent_config,
            pending_writes: None,
            seq,
        };

        let mut storage = self.storage.write().await;
        storage.insert((thread_id.clone(), checkpoint_id.clone()), entry);

        let mut new_config = config.clone();
        new_config.insert("checkpoint_id".to_string(), Value::String(checkpoint_id));

        Ok(new_config)
    }

    async fn put_writes(
        &self,
        config: &HashMap<String, Value>,
        writes: Vec<(String, Value)>,
        task_id: &str,
    ) -> Result<()> {
        let (thread_id, _, checkpoint_id) = Self::extract_config(config);
        let checkpoint_id = checkpoint_id
            .ok_or_else(|| LangGraphError::Other("checkpoint_id required for put_writes".into()))?;

        let key = (thread_id.to_string(), checkpoint_id.to_string());
        let mut storage = self.storage.write().await;

        let entry = storage
            .get_mut(&key)
            .ok_or_else(|| LangGraphError::Other("Checkpoint not found for put_writes".into()))?;

        let pending = entry.pending_writes.get_or_insert_with(Vec::new);
        for (channel, value) in writes {
            pending.push((task_id.to_string(), channel, value));
        }

        Ok(())
    }

    async fn list(
        &self,
        config: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>> {
        let (thread_id, _, _) = Self::extract_config(config);

        let storage = self.storage.read().await;
        let mut entries: Vec<_> = storage
            .iter()
            .filter(|((tid, _), _)| tid == thread_id)
            .collect();

        // Sort newest first by sequence number.
        entries.sort_by_key(|b| std::cmp::Reverse(b.1.seq));

        let mut results: Vec<CheckpointTuple> = entries
            .into_iter()
            .map(|(_, entry)| self.entry_to_tuple(thread_id, entry))
            .collect();

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<CheckpointEntry>> {
        let storage = self.storage.read().await;
        let mut entries: Vec<_> = storage
            .iter()
            .filter(|((tid, _), _)| tid == thread_id)
            .collect();

        // Sort oldest first by sequence number.
        entries.sort_by_key(|(_, entry)| entry.seq);

        let results = entries
            .into_iter()
            .map(|(_, entry)| {
                let node_name = entry
                    .metadata
                    .writes
                    .as_ref()
                    .and_then(|w| w.keys().next().cloned())
                    .unwrap_or_else(|| entry.metadata.source.clone());

                let timestamp = entry
                    .checkpoint
                    .ts
                    .rsplit('+')
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0)
                    * 1000;

                let state = Value::Object(
                    entry
                        .checkpoint
                        .channel_values
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );

                CheckpointEntry {
                    checkpoint_id: entry.checkpoint.id.clone(),
                    thread_id: thread_id.to_string(),
                    node_name,
                    timestamp,
                    state,
                }
            })
            .collect();

        Ok(results)
    }
}

impl MemoryCheckpointSaver {
    /// Convert an internal entry to a [`CheckpointTuple`] for trait methods.
    fn entry_to_tuple(&self, thread_id: &str, entry: &InternalEntry) -> CheckpointTuple {
        let mut config = HashMap::new();
        config.insert(
            "thread_id".to_string(),
            Value::String(thread_id.to_string()),
        );
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(entry.checkpoint.id.clone()),
        );

        CheckpointTuple {
            checkpoint: entry.checkpoint.clone(),
            config,
            metadata: Some(entry.metadata.clone()),
            parent_config: entry.parent_config.clone(),
            pending_writes: entry.pending_writes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::checkpoint::empty_checkpoint;
    use serde_json::json;

    fn test_config(thread_id: &str) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!(thread_id));
        config
    }

    fn test_metadata(step: i64) -> CheckpointMetadata {
        CheckpointMetadata {
            source: "loop".to_string(),
            step,
            writes: None,
            extra: HashMap::new(),
        }
    }

    fn test_meta(step: i64) -> CheckpointMeta {
        CheckpointMeta {
            source: "loop".to_string(),
            step,
            writes: HashMap::new(),
            created_at: SystemTime::now(),
            parent_id: None,
        }
    }

    // ---------------------------------------------------------------
    // Tests for the CheckpointSaver trait implementation
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_put_and_get() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        let new_config = saver.put(&config, cp, test_metadata(1)).await.unwrap();

        assert!(new_config.contains_key("checkpoint_id"));

        let loaded = saver.get(&config).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().checkpoint.id, cp_id);
    }

    #[tokio::test]
    async fn test_get_by_checkpoint_id() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        let new_config = saver.put(&config, cp, test_metadata(1)).await.unwrap();

        let loaded = saver.get(&new_config).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().checkpoint.id, cp_id);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("nonexistent");

        let result = saver.get(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_tuple() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        saver.put(&config, cp, test_metadata(0)).await.unwrap();

        let mut lookup = config.clone();
        lookup.insert("checkpoint_id".to_string(), json!(cp_id));

        let result = saver.get_tuple(&lookup).await.unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_put_returns_config_with_checkpoint_id() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        let new_config = saver.put(&config, cp, test_metadata(0)).await.unwrap();

        assert_eq!(
            new_config.get("checkpoint_id").and_then(|v| v.as_str()),
            Some(cp_id.as_str())
        );
    }

    #[tokio::test]
    async fn test_latest_returns_most_recent() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let mut last_id = String::new();
        for i in 0..5 {
            let cp = empty_checkpoint();
            last_id = cp.id.clone();
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        let latest = saver.get(&config).await.unwrap().unwrap();
        assert_eq!(latest.checkpoint.id, last_id);
    }

    #[tokio::test]
    async fn test_list_newest_first() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let mut ids = Vec::new();
        for i in 0..4 {
            let cp = empty_checkpoint();
            ids.push(cp.id.clone());
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        let all = saver.list(&config, None).await.unwrap();
        assert_eq!(all.len(), 4);
        // First element should be the newest (last inserted).
        assert_eq!(all[0].checkpoint.id, ids[3]);
        assert_eq!(all[3].checkpoint.id, ids[0]);
    }

    #[tokio::test]
    async fn test_list_with_limit() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        for i in 0..5 {
            let cp = empty_checkpoint();
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        let limited = saver.list(&config, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_list_checkpoints_oldest_first() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let mut ids = Vec::new();
        for i in 0..3 {
            let cp = empty_checkpoint();
            ids.push(cp.id.clone());
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        let entries = saver.list_checkpoints("thread-1").await.unwrap();
        assert_eq!(entries.len(), 3);
        // Oldest first.
        assert_eq!(entries[0].checkpoint_id, ids[0]);
        assert_eq!(entries[2].checkpoint_id, ids[2]);
    }

    #[tokio::test]
    async fn test_put_writes() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let new_config = saver.put(&config, cp, test_metadata(0)).await.unwrap();

        let writes = vec![("state".to_string(), json!({"updated": true}))];
        saver
            .put_writes(&new_config, writes, "task-1")
            .await
            .unwrap();

        let loaded = saver.get(&new_config).await.unwrap().unwrap();
        let pending = loaded.pending_writes.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, "task-1");
        assert_eq!(pending[0].1, "state");
        assert_eq!(pending[0].2, json!({"updated": true}));
    }

    #[tokio::test]
    async fn test_put_writes_not_found() {
        let saver = MemoryCheckpointSaver::new();
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), json!("thread-1"));
        config.insert("checkpoint_id".to_string(), json!("nonexistent"));

        let result = saver.put_writes(&config, vec![], "task-1").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_put_writes_requires_checkpoint_id() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let result = saver.put_writes(&config, vec![], "task-1").await;
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // Tests for extended methods (delete, clear_thread, etc.)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_delete_existing() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let cp_id = cp.id.clone();
        saver.put(&config, cp, test_metadata(0)).await.unwrap();

        let removed = saver.delete("thread-1", &cp_id).await.unwrap();
        assert!(removed);

        let result = saver.get(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let saver = MemoryCheckpointSaver::new();
        let removed = saver.delete("thread-1", "no-such-id").await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_clear_thread() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        for i in 0..3 {
            let cp = empty_checkpoint();
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        let count = saver.clear_thread("thread-1").await.unwrap();
        assert_eq!(count, 3);

        let all = saver.list(&config, None).await.unwrap();
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn test_clear_thread_empty() {
        let saver = MemoryCheckpointSaver::new();
        let count = saver.clear_thread("nonexistent").await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_multi_thread_isolation() {
        let saver = MemoryCheckpointSaver::new();
        let config1 = test_config("thread-1");
        let config2 = test_config("thread-2");

        for i in 0..3 {
            let cp = empty_checkpoint();
            saver.put(&config1, cp, test_metadata(i)).await.unwrap();
        }
        for i in 0..2 {
            let cp = empty_checkpoint();
            saver.put(&config2, cp, test_metadata(i)).await.unwrap();
        }

        let list1 = saver.list(&config1, None).await.unwrap();
        assert_eq!(list1.len(), 3);

        let list2 = saver.list(&config2, None).await.unwrap();
        assert_eq!(list2.len(), 2);

        // Deleting thread-1 should not affect thread-2.
        saver.clear_thread("thread-1").await.unwrap();
        let list2_after = saver.list(&config2, None).await.unwrap();
        assert_eq!(list2_after.len(), 2);
    }

    #[tokio::test]
    async fn test_thread_ids() {
        let saver = MemoryCheckpointSaver::new();

        assert!(saver.thread_ids().await.is_empty());

        let config1 = test_config("alpha");
        let config2 = test_config("beta");

        saver
            .put(&config1, empty_checkpoint(), test_metadata(0))
            .await
            .unwrap();
        saver
            .put(&config2, empty_checkpoint(), test_metadata(0))
            .await
            .unwrap();
        saver
            .put(&config1, empty_checkpoint(), test_metadata(1))
            .await
            .unwrap();

        let ids = saver.thread_ids().await;
        assert_eq!(ids, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[tokio::test]
    async fn test_total_checkpoints() {
        let saver = MemoryCheckpointSaver::new();
        assert_eq!(saver.total_checkpoints().await, 0);

        let config = test_config("thread-1");

        for i in 0..4 {
            let cp = empty_checkpoint();
            saver.put(&config, cp, test_metadata(i)).await.unwrap();
        }

        assert_eq!(saver.total_checkpoints().await, 4);
    }

    // ---------------------------------------------------------------
    // Tests for put_stored / get_stored / get_latest / list_stored
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_put_stored_and_get_stored() {
        let saver = MemoryCheckpointSaver::new();

        saver
            .put_stored("thread-1", "cp-1", json!({"count": 1}), test_meta(0))
            .await
            .unwrap();

        let stored = saver.get_stored("thread-1", "cp-1").await.unwrap();
        assert!(stored.is_some());
        let s = stored.unwrap();
        assert_eq!(s.checkpoint_id, "cp-1");
        assert_eq!(s.thread_id, "thread-1");
        assert_eq!(s.state, json!({"count": 1}));
        assert_eq!(s.metadata.step, 0);
    }

    #[tokio::test]
    async fn test_get_latest_stored() {
        let saver = MemoryCheckpointSaver::new();

        saver
            .put_stored("thread-1", "cp-1", json!({"v": 1}), test_meta(0))
            .await
            .unwrap();
        saver
            .put_stored("thread-1", "cp-2", json!({"v": 2}), test_meta(1))
            .await
            .unwrap();

        let latest = saver.get_latest("thread-1").await.unwrap().unwrap();
        assert_eq!(latest.checkpoint_id, "cp-2");
    }

    #[tokio::test]
    async fn test_list_stored_ordering() {
        let saver = MemoryCheckpointSaver::new();

        saver
            .put_stored("thread-1", "cp-a", json!(1), test_meta(0))
            .await
            .unwrap();
        saver
            .put_stored("thread-1", "cp-b", json!(2), test_meta(1))
            .await
            .unwrap();
        saver
            .put_stored("thread-1", "cp-c", json!(3), test_meta(2))
            .await
            .unwrap();

        let list = saver.list_stored("thread-1").await.unwrap();
        assert_eq!(list.len(), 3);
        // Newest first.
        assert_eq!(list[0].checkpoint_id, "cp-c");
        assert_eq!(list[2].checkpoint_id, "cp-a");
    }

    #[tokio::test]
    async fn test_metadata_roundtrip() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let cp = empty_checkpoint();
        let mut metadata = test_metadata(3);
        metadata.source = "input".to_string();
        metadata.writes = Some({
            let mut m = HashMap::new();
            m.insert("ch".to_string(), json!("val"));
            m
        });

        saver.put(&config, cp, metadata).await.unwrap();

        let loaded = saver.get(&config).await.unwrap().unwrap();
        let meta = loaded.metadata.unwrap();
        assert_eq!(meta.source, "input");
        assert_eq!(meta.step, 3);
        assert!(meta.writes.is_some());
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        saver
            .put(&config, empty_checkpoint(), test_metadata(0))
            .await
            .unwrap();

        let saver2 = saver.clone();
        assert_eq!(saver2.total_checkpoints().await, 1);

        saver2
            .put(&config, empty_checkpoint(), test_metadata(1))
            .await
            .unwrap();
        assert_eq!(saver.total_checkpoints().await, 2);
    }

    #[tokio::test]
    async fn test_default_creates_empty() {
        let saver = MemoryCheckpointSaver::default();
        assert_eq!(saver.total_checkpoints().await, 0);
        assert!(saver.thread_ids().await.is_empty());
    }

    #[tokio::test]
    async fn test_channel_values_preserved() {
        let saver = MemoryCheckpointSaver::new();
        let config = test_config("thread-1");

        let mut cp = empty_checkpoint();
        cp.channel_values
            .insert("state".to_string(), json!({"count": 42}));

        saver.put(&config, cp, test_metadata(0)).await.unwrap();

        let loaded = saver.get(&config).await.unwrap().unwrap();
        assert_eq!(
            loaded.checkpoint.channel_values["state"],
            json!({"count": 42})
        );
    }
}
