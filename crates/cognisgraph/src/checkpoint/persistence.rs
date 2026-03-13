//! Persistence layer for LangGraph checkpoints.
//!
//! This module provides a high-level checkpoint persistence API with history
//! tracking, filtering, and rollback support. It mirrors Python LangGraph's
//! checkpoint saver pattern but uses synchronous interfaces for simplicity.
//!
//! # Types
//!
//! - [`CheckpointId`] — unique identifier for a checkpoint
//! - [`PersistentCheckpoint`] — a checkpoint with state, metadata, and lineage
//! - [`PersistentCheckpointSaver`] — trait for checkpoint CRUD operations
//! - [`InMemoryPersistentSaver`] — HashMap-backed ephemeral saver
//! - [`CheckpointHistory`] — tracks parent-child relationships between checkpoints
//! - [`CheckpointFilter`] — builder for querying checkpoints
//! - [`CheckpointManager`] — high-level manager combining saver + history + filtering
//! - [`CheckpointStats`] — summary statistics

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CheckpointId
// ---------------------------------------------------------------------------

/// A unique identifier for a checkpoint, wrapping a UUID-like string.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointId(String);

impl CheckpointId {
    /// Generate a new unique checkpoint ID using UUID v4.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a `CheckpointId` from an existing string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Return the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CheckpointId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint (persistence-layer)
// ---------------------------------------------------------------------------

/// A checkpoint representing serialized graph state at a point in execution.
///
/// This is the persistence-layer checkpoint, distinct from
/// [`crate::pregel::checkpoint::Checkpoint`] which is tied to the Pregel
/// execution engine. This struct is designed for general-purpose checkpoint
/// storage with metadata and lineage tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCheckpoint {
    /// Unique identifier for this checkpoint.
    pub id: CheckpointId,
    /// The thread (execution session) this checkpoint belongs to.
    pub thread_id: String,
    /// The serialized graph state.
    pub state: Value,
    /// The parent checkpoint ID, if this is not the root checkpoint.
    pub parent_id: Option<CheckpointId>,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// The step number in the execution.
    pub step: usize,
}

impl PersistentCheckpoint {
    /// Create a new checkpoint builder.
    pub fn builder(thread_id: impl Into<String>, state: Value) -> PersistentCheckpointBuilder {
        PersistentCheckpointBuilder {
            id: CheckpointId::new(),
            thread_id: thread_id.into(),
            state,
            parent_id: None,
            created_at: now_iso8601(),
            metadata: HashMap::new(),
            step: 0,
        }
    }

    /// Serialize this checkpoint to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id.as_str(),
            "thread_id": self.thread_id,
            "state": self.state,
            "parent_id": self.parent_id.as_ref().map(|id| id.as_str()),
            "created_at": self.created_at,
            "metadata": self.metadata,
            "step": self.step,
        })
    }

    /// Returns `true` if this checkpoint has no parent (i.e. it is a root).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

/// Builder for constructing a [`PersistentCheckpoint`].
pub struct PersistentCheckpointBuilder {
    id: CheckpointId,
    thread_id: String,
    state: Value,
    parent_id: Option<CheckpointId>,
    created_at: String,
    metadata: HashMap<String, Value>,
    step: usize,
}

impl PersistentCheckpointBuilder {
    /// Set the checkpoint ID.
    pub fn id(mut self, id: CheckpointId) -> Self {
        self.id = id;
        self
    }

    /// Set the parent checkpoint ID.
    pub fn parent_id(mut self, parent_id: CheckpointId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Set the creation timestamp.
    pub fn created_at(mut self, ts: impl Into<String>) -> Self {
        self.created_at = ts.into();
        self
    }

    /// Add a metadata key-value pair.
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the step number.
    pub fn step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Build the checkpoint.
    pub fn build(self) -> PersistentCheckpoint {
        PersistentCheckpoint {
            id: self.id,
            thread_id: self.thread_id,
            state: self.state,
            parent_id: self.parent_id,
            created_at: self.created_at,
            metadata: self.metadata,
            step: self.step,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointSaver trait (persistence-layer)
// ---------------------------------------------------------------------------

/// Trait for synchronous checkpoint persistence backends.
///
/// This is distinct from [`crate::pregel::checkpoint::CheckpointSaver`] which
/// is async and tied to the Pregel engine. This trait provides a simpler
/// synchronous API for general-purpose checkpoint storage.
pub trait PersistentCheckpointSaver {
    /// Save a checkpoint. Returns the checkpoint's ID.
    fn save(&mut self, checkpoint: PersistentCheckpoint) -> Result<CheckpointId, String>;

    /// Load a checkpoint by its ID.
    fn load(&self, id: &CheckpointId) -> Result<Option<PersistentCheckpoint>, String>;

    /// Load the latest checkpoint for a given thread.
    fn load_latest(&self, thread_id: &str) -> Result<Option<PersistentCheckpoint>, String>;

    /// List all checkpoints for a given thread, ordered by step.
    fn list(&self, thread_id: &str) -> Result<Vec<PersistentCheckpoint>, String>;

    /// Delete a checkpoint by its ID. Returns `true` if it existed.
    fn delete(&mut self, id: &CheckpointId) -> Result<bool, String>;
}

// ---------------------------------------------------------------------------
// InMemoryPersistentSaver
// ---------------------------------------------------------------------------

/// An in-memory implementation of [`PersistentCheckpointSaver`].
///
/// Stores checkpoints in a `HashMap`. Data is lost when the saver is dropped.
#[derive(Debug, Clone, Default)]
pub struct InMemoryPersistentSaver {
    checkpoints: HashMap<String, PersistentCheckpoint>,
}

impl InMemoryPersistentSaver {
    /// Create a new, empty in-memory saver.
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
        }
    }

    /// Return the total number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Return `true` if there are no stored checkpoints.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Return a sorted list of unique thread IDs.
    pub fn thread_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .checkpoints
            .values()
            .map(|cp| cp.thread_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort();
        ids
    }
}

impl PersistentCheckpointSaver for InMemoryPersistentSaver {
    fn save(&mut self, checkpoint: PersistentCheckpoint) -> Result<CheckpointId, String> {
        let id = checkpoint.id.clone();
        self.checkpoints.insert(id.as_str().to_string(), checkpoint);
        Ok(id)
    }

    fn load(&self, id: &CheckpointId) -> Result<Option<PersistentCheckpoint>, String> {
        Ok(self.checkpoints.get(id.as_str()).cloned())
    }

    fn load_latest(&self, thread_id: &str) -> Result<Option<PersistentCheckpoint>, String> {
        let mut thread_checkpoints: Vec<&PersistentCheckpoint> = self
            .checkpoints
            .values()
            .filter(|cp| cp.thread_id == thread_id)
            .collect();
        thread_checkpoints.sort_by_key(|cp| cp.step);
        Ok(thread_checkpoints.last().cloned().cloned())
    }

    fn list(&self, thread_id: &str) -> Result<Vec<PersistentCheckpoint>, String> {
        let mut results: Vec<PersistentCheckpoint> = self
            .checkpoints
            .values()
            .filter(|cp| cp.thread_id == thread_id)
            .cloned()
            .collect();
        results.sort_by_key(|cp| cp.step);
        Ok(results)
    }

    fn delete(&mut self, id: &CheckpointId) -> Result<bool, String> {
        Ok(self.checkpoints.remove(id.as_str()).is_some())
    }
}

// ---------------------------------------------------------------------------
// CheckpointHistory
// ---------------------------------------------------------------------------

/// Tracks parent-child relationships between checkpoints for lineage queries.
#[derive(Debug, Clone, Default)]
pub struct CheckpointHistory {
    /// Maps checkpoint ID -> parent checkpoint ID.
    parents: HashMap<String, Option<String>>,
    /// Maps parent checkpoint ID -> list of child checkpoint IDs.
    children: HashMap<String, Vec<String>>,
}

impl CheckpointHistory {
    /// Create a new, empty history tracker.
    pub fn new() -> Self {
        Self {
            parents: HashMap::new(),
            children: HashMap::new(),
        }
    }

    /// Register a checkpoint's lineage.
    pub fn add(&mut self, checkpoint: &PersistentCheckpoint) {
        let id = checkpoint.id.as_str().to_string();
        let parent_id = checkpoint
            .parent_id
            .as_ref()
            .map(|p| p.as_str().to_string());

        self.parents.insert(id.clone(), parent_id.clone());

        if let Some(ref pid) = parent_id {
            self.children.entry(pid.clone()).or_default().push(id);
        }
    }

    /// Get the full lineage from root to the given checkpoint ID.
    ///
    /// Returns an empty vector if the ID is not tracked.
    pub fn get_lineage(&self, id: &CheckpointId) -> Vec<CheckpointId> {
        let mut chain = Vec::new();
        let mut current = Some(id.as_str().to_string());

        while let Some(ref cid) = current {
            if !self.parents.contains_key(cid) {
                break;
            }
            chain.push(CheckpointId::from_str(cid));
            current = self.parents.get(cid).and_then(|p| p.clone());
        }

        chain.reverse();
        chain
    }

    /// Get the direct children of a checkpoint.
    pub fn get_children(&self, id: &CheckpointId) -> Vec<CheckpointId> {
        self.children
            .get(id.as_str())
            .map(|kids| kids.iter().map(|k| CheckpointId::from_str(k)).collect())
            .unwrap_or_default()
    }

    /// Get the depth of a checkpoint (number of ancestors including itself).
    ///
    /// Returns 0 if the ID is not tracked.
    pub fn depth(&self, id: &CheckpointId) -> usize {
        self.get_lineage(id).len()
    }
}

// ---------------------------------------------------------------------------
// CheckpointFilter
// ---------------------------------------------------------------------------

/// A filter for querying checkpoints. Supports builder-pattern construction.
#[derive(Debug, Clone, Default)]
pub struct CheckpointFilter {
    thread_id: Option<String>,
    step_min: Option<usize>,
    step_max: Option<usize>,
    metadata_key: Option<String>,
}

impl CheckpointFilter {
    /// Create a new empty filter that matches all checkpoints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by thread ID.
    pub fn with_thread(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Filter by step range (inclusive on both ends).
    pub fn with_step_range(mut self, min: usize, max: usize) -> Self {
        self.step_min = Some(min);
        self.step_max = Some(max);
        self
    }

    /// Filter to checkpoints that have a specific metadata key.
    pub fn with_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.metadata_key = Some(key.into());
        self
    }

    /// Check whether a checkpoint matches this filter.
    pub fn matches(&self, checkpoint: &PersistentCheckpoint) -> bool {
        if let Some(ref tid) = self.thread_id {
            if checkpoint.thread_id != *tid {
                return false;
            }
        }

        if let Some(min) = self.step_min {
            if checkpoint.step < min {
                return false;
            }
        }

        if let Some(max) = self.step_max {
            if checkpoint.step > max {
                return false;
            }
        }

        if let Some(ref key) = self.metadata_key {
            if !checkpoint.metadata.contains_key(key) {
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// CheckpointStats
// ---------------------------------------------------------------------------

/// Summary statistics about checkpoints managed by a [`CheckpointManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointStats {
    /// Total number of checkpoints.
    pub total_checkpoints: usize,
    /// Number of distinct threads.
    pub threads: usize,
    /// Average number of steps per thread.
    pub avg_steps_per_thread: f64,
}

impl CheckpointStats {
    /// Serialize these statistics to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "total_checkpoints": self.total_checkpoints,
            "threads": self.threads,
            "avg_steps_per_thread": self.avg_steps_per_thread,
        })
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// High-level checkpoint manager that combines a saver with history tracking
/// and filtering capabilities.
pub struct CheckpointManager {
    saver: Box<dyn PersistentCheckpointSaver>,
    history: CheckpointHistory,
    /// Track all thread IDs and their checkpoint IDs for stats/queries.
    thread_checkpoints: HashMap<String, Vec<CheckpointId>>,
}

impl CheckpointManager {
    /// Create a new manager backed by the given saver.
    pub fn new(saver: Box<dyn PersistentCheckpointSaver>) -> Self {
        Self {
            saver,
            history: CheckpointHistory::new(),
            thread_checkpoints: HashMap::new(),
        }
    }

    /// Create a new checkpoint and persist it.
    pub fn create_checkpoint(
        &mut self,
        thread_id: &str,
        state: Value,
        parent_id: Option<CheckpointId>,
        metadata: HashMap<String, Value>,
    ) -> Result<CheckpointId, String> {
        // Determine step: if parent exists, increment its step.
        let step = match &parent_id {
            Some(pid) => {
                let parent = self.saver.load(pid)?;
                parent.map(|p| p.step + 1).unwrap_or(0)
            }
            None => 0,
        };

        let mut builder = PersistentCheckpoint::builder(thread_id, state).step(step);

        if let Some(pid) = parent_id {
            builder = builder.parent_id(pid);
        }

        for (k, v) in metadata {
            builder = builder.metadata(k, v);
        }

        let checkpoint = builder.build();
        self.history.add(&checkpoint);
        self.thread_checkpoints
            .entry(thread_id.to_string())
            .or_default()
            .push(checkpoint.id.clone());

        self.saver.save(checkpoint)
    }

    /// Restore a checkpoint by its ID.
    pub fn restore(&self, id: &CheckpointId) -> Result<Option<PersistentCheckpoint>, String> {
        self.saver.load(id)
    }

    /// Restore the latest checkpoint for a given thread.
    pub fn restore_latest(&self, thread_id: &str) -> Result<Option<PersistentCheckpoint>, String> {
        self.saver.load_latest(thread_id)
    }

    /// Query checkpoints using a filter.
    pub fn query(&self, filter: &CheckpointFilter) -> Result<Vec<PersistentCheckpoint>, String> {
        // Collect from all known threads.
        let mut results = Vec::new();
        for thread_id in self.thread_checkpoints.keys() {
            let checkpoints = self.saver.list(thread_id)?;
            for cp in checkpoints {
                if filter.matches(&cp) {
                    results.push(cp);
                }
            }
        }
        results.sort_by_key(|cp| cp.step);
        Ok(results)
    }

    /// Rollback a thread to a specific step number.
    ///
    /// Returns the checkpoint at the target step, or `None` if no checkpoint
    /// exists at that step.
    pub fn rollback(
        &self,
        thread_id: &str,
        to_step: usize,
    ) -> Result<Option<PersistentCheckpoint>, String> {
        let checkpoints = self.saver.list(thread_id)?;
        Ok(checkpoints.into_iter().find(|cp| cp.step == to_step))
    }

    /// Compute summary statistics.
    pub fn stats(&self) -> CheckpointStats {
        let threads = self.thread_checkpoints.len();
        let total_checkpoints: usize = self.thread_checkpoints.values().map(|ids| ids.len()).sum();
        let avg_steps_per_thread = if threads > 0 {
            total_checkpoints as f64 / threads as f64
        } else {
            0.0
        };

        CheckpointStats {
            total_checkpoints,
            threads,
            avg_steps_per_thread,
        }
    }

    /// Get the history tracker (for lineage queries).
    pub fn history(&self) -> &CheckpointHistory {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a simple ISO-8601-ish timestamp string.
fn now_iso8601() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    format!("1970-01-01T00:00:00Z+{}", secs)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---------------------------------------------------------------
    // CheckpointId tests
    // ---------------------------------------------------------------

    #[test]
    fn test_checkpoint_id_new_is_unique() {
        let id1 = CheckpointId::new();
        let id2 = CheckpointId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_checkpoint_id_from_str() {
        let id = CheckpointId::from_str("abc-123");
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn test_checkpoint_id_display() {
        let id = CheckpointId::from_str("my-id");
        assert_eq!(format!("{}", id), "my-id");
    }

    #[test]
    fn test_checkpoint_id_equality() {
        let a = CheckpointId::from_str("same");
        let b = CheckpointId::from_str("same");
        assert_eq!(a, b);
    }

    #[test]
    fn test_checkpoint_id_inequality() {
        let a = CheckpointId::from_str("one");
        let b = CheckpointId::from_str("two");
        assert_ne!(a, b);
    }

    #[test]
    fn test_checkpoint_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CheckpointId::from_str("a"));
        set.insert(CheckpointId::from_str("a"));
        set.insert(CheckpointId::from_str("b"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_checkpoint_id_clone() {
        let id = CheckpointId::from_str("clone-me");
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_checkpoint_id_default() {
        let id = CheckpointId::default();
        assert!(!id.as_str().is_empty());
    }

    // ---------------------------------------------------------------
    // PersistentCheckpoint builder and serialization tests
    // ---------------------------------------------------------------

    #[test]
    fn test_checkpoint_builder_basic() {
        let cp = PersistentCheckpoint::builder("thread-1", json!({"count": 0}))
            .step(0)
            .build();

        assert_eq!(cp.thread_id, "thread-1");
        assert_eq!(cp.state, json!({"count": 0}));
        assert_eq!(cp.step, 0);
        assert!(cp.is_root());
        assert!(!cp.id.as_str().is_empty());
        assert!(!cp.created_at.is_empty());
    }

    #[test]
    fn test_checkpoint_builder_with_parent() {
        let parent_id = CheckpointId::from_str("parent-1");
        let cp = PersistentCheckpoint::builder("thread-1", json!({"count": 1}))
            .parent_id(parent_id.clone())
            .step(1)
            .build();

        assert!(!cp.is_root());
        assert_eq!(cp.parent_id, Some(parent_id));
    }

    #[test]
    fn test_checkpoint_builder_with_metadata() {
        let cp = PersistentCheckpoint::builder("thread-1", json!(null))
            .metadata("source", json!("loop"))
            .metadata("node", json!("agent"))
            .build();

        assert_eq!(cp.metadata.len(), 2);
        assert_eq!(cp.metadata["source"], json!("loop"));
        assert_eq!(cp.metadata["node"], json!("agent"));
    }

    #[test]
    fn test_checkpoint_builder_with_custom_id() {
        let custom_id = CheckpointId::from_str("custom-id-42");
        let cp = PersistentCheckpoint::builder("thread-1", json!(null))
            .id(custom_id.clone())
            .build();

        assert_eq!(cp.id, custom_id);
    }

    #[test]
    fn test_checkpoint_to_json() {
        let cp = PersistentCheckpoint::builder("thread-1", json!({"x": 1}))
            .step(3)
            .build();

        let j = cp.to_json();
        assert_eq!(j["thread_id"], json!("thread-1"));
        assert_eq!(j["state"], json!({"x": 1}));
        assert_eq!(j["step"], json!(3));
        assert!(j["id"].is_string());
        assert!(j["created_at"].is_string());
    }

    #[test]
    fn test_checkpoint_to_json_with_parent() {
        let parent_id = CheckpointId::from_str("p-1");
        let cp = PersistentCheckpoint::builder("t1", json!(null))
            .parent_id(parent_id)
            .build();

        let j = cp.to_json();
        assert_eq!(j["parent_id"], json!("p-1"));
    }

    #[test]
    fn test_checkpoint_to_json_without_parent() {
        let cp = PersistentCheckpoint::builder("t1", json!(null)).build();
        let j = cp.to_json();
        assert!(j["parent_id"].is_null());
    }

    #[test]
    fn test_checkpoint_is_root() {
        let root = PersistentCheckpoint::builder("t1", json!(null)).build();
        assert!(root.is_root());

        let child = PersistentCheckpoint::builder("t1", json!(null))
            .parent_id(CheckpointId::from_str("parent"))
            .build();
        assert!(!child.is_root());
    }

    // ---------------------------------------------------------------
    // InMemoryPersistentSaver tests
    // ---------------------------------------------------------------

    #[test]
    fn test_saver_new_is_empty() {
        let saver = InMemoryPersistentSaver::new();
        assert!(saver.is_empty());
        assert_eq!(saver.len(), 0);
        assert!(saver.thread_ids().is_empty());
    }

    #[test]
    fn test_saver_save_and_load() {
        let mut saver = InMemoryPersistentSaver::new();
        let cp = PersistentCheckpoint::builder("thread-1", json!({"step": 0}))
            .step(0)
            .build();
        let id = cp.id.clone();

        let saved_id = saver.save(cp).unwrap();
        assert_eq!(saved_id, id);
        assert_eq!(saver.len(), 1);

        let loaded = saver.load(&id).unwrap().unwrap();
        assert_eq!(loaded.thread_id, "thread-1");
        assert_eq!(loaded.state, json!({"step": 0}));
    }

    #[test]
    fn test_saver_load_nonexistent() {
        let saver = InMemoryPersistentSaver::new();
        let id = CheckpointId::from_str("nonexistent");
        assert!(saver.load(&id).unwrap().is_none());
    }

    #[test]
    fn test_saver_load_latest() {
        let mut saver = InMemoryPersistentSaver::new();

        for i in 0..5 {
            let cp = PersistentCheckpoint::builder("thread-1", json!({"step": i}))
                .step(i)
                .build();
            saver.save(cp).unwrap();
        }

        let latest = saver.load_latest("thread-1").unwrap().unwrap();
        assert_eq!(latest.step, 4);
        assert_eq!(latest.state, json!({"step": 4}));
    }

    #[test]
    fn test_saver_load_latest_nonexistent_thread() {
        let saver = InMemoryPersistentSaver::new();
        assert!(saver.load_latest("no-thread").unwrap().is_none());
    }

    #[test]
    fn test_saver_list() {
        let mut saver = InMemoryPersistentSaver::new();

        for i in 0..3 {
            let cp = PersistentCheckpoint::builder("thread-1", json!(i))
                .step(i)
                .build();
            saver.save(cp).unwrap();
        }

        // Different thread.
        let cp = PersistentCheckpoint::builder("thread-2", json!("other"))
            .step(0)
            .build();
        saver.save(cp).unwrap();

        let list = saver.list("thread-1").unwrap();
        assert_eq!(list.len(), 3);
        // Should be sorted by step.
        assert_eq!(list[0].step, 0);
        assert_eq!(list[1].step, 1);
        assert_eq!(list[2].step, 2);
    }

    #[test]
    fn test_saver_list_empty_thread() {
        let saver = InMemoryPersistentSaver::new();
        assert!(saver.list("empty").unwrap().is_empty());
    }

    #[test]
    fn test_saver_delete() {
        let mut saver = InMemoryPersistentSaver::new();
        let cp = PersistentCheckpoint::builder("t1", json!(null))
            .step(0)
            .build();
        let id = cp.id.clone();
        saver.save(cp).unwrap();

        assert!(saver.delete(&id).unwrap());
        assert!(saver.is_empty());
        assert!(saver.load(&id).unwrap().is_none());
    }

    #[test]
    fn test_saver_delete_nonexistent() {
        let mut saver = InMemoryPersistentSaver::new();
        let id = CheckpointId::from_str("nope");
        assert!(!saver.delete(&id).unwrap());
    }

    #[test]
    fn test_saver_thread_ids() {
        let mut saver = InMemoryPersistentSaver::new();
        saver
            .save(
                PersistentCheckpoint::builder("beta", json!(null))
                    .step(0)
                    .build(),
            )
            .unwrap();
        saver
            .save(
                PersistentCheckpoint::builder("alpha", json!(null))
                    .step(0)
                    .build(),
            )
            .unwrap();
        saver
            .save(
                PersistentCheckpoint::builder("alpha", json!(null))
                    .step(1)
                    .build(),
            )
            .unwrap();

        let ids = saver.thread_ids();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    // ---------------------------------------------------------------
    // CheckpointHistory tests
    // ---------------------------------------------------------------

    #[test]
    fn test_history_empty() {
        let history = CheckpointHistory::new();
        let id = CheckpointId::from_str("unknown");
        assert!(history.get_lineage(&id).is_empty());
        assert!(history.get_children(&id).is_empty());
        assert_eq!(history.depth(&id), 0);
    }

    #[test]
    fn test_history_single_root() {
        let mut history = CheckpointHistory::new();
        let cp = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("root"))
            .step(0)
            .build();
        history.add(&cp);

        let lineage = history.get_lineage(&CheckpointId::from_str("root"));
        assert_eq!(lineage.len(), 1);
        assert_eq!(lineage[0], CheckpointId::from_str("root"));
        assert_eq!(history.depth(&CheckpointId::from_str("root")), 1);
    }

    #[test]
    fn test_history_lineage_chain() {
        let mut history = CheckpointHistory::new();

        let root = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("cp-0"))
            .step(0)
            .build();
        history.add(&root);

        let child = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("cp-1"))
            .parent_id(CheckpointId::from_str("cp-0"))
            .step(1)
            .build();
        history.add(&child);

        let grandchild = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("cp-2"))
            .parent_id(CheckpointId::from_str("cp-1"))
            .step(2)
            .build();
        history.add(&grandchild);

        let lineage = history.get_lineage(&CheckpointId::from_str("cp-2"));
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0], CheckpointId::from_str("cp-0"));
        assert_eq!(lineage[1], CheckpointId::from_str("cp-1"));
        assert_eq!(lineage[2], CheckpointId::from_str("cp-2"));
    }

    #[test]
    fn test_history_children() {
        let mut history = CheckpointHistory::new();

        let parent = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("parent"))
            .step(0)
            .build();
        history.add(&parent);

        let child_a = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("child-a"))
            .parent_id(CheckpointId::from_str("parent"))
            .step(1)
            .build();
        history.add(&child_a);

        let child_b = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("child-b"))
            .parent_id(CheckpointId::from_str("parent"))
            .step(1)
            .build();
        history.add(&child_b);

        let children = history.get_children(&CheckpointId::from_str("parent"));
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_history_depth() {
        let mut history = CheckpointHistory::new();

        // Build a chain of 5 checkpoints.
        let ids: Vec<String> = (0..5).map(|i| format!("cp-{}", i)).collect();

        for i in 0..5 {
            let mut builder = PersistentCheckpoint::builder("t1", json!(null))
                .id(CheckpointId::from_str(&ids[i]))
                .step(i);
            if i > 0 {
                builder = builder.parent_id(CheckpointId::from_str(&ids[i - 1]));
            }
            history.add(&builder.build());
        }

        assert_eq!(history.depth(&CheckpointId::from_str("cp-0")), 1);
        assert_eq!(history.depth(&CheckpointId::from_str("cp-2")), 3);
        assert_eq!(history.depth(&CheckpointId::from_str("cp-4")), 5);
    }

    // ---------------------------------------------------------------
    // CheckpointFilter tests
    // ---------------------------------------------------------------

    #[test]
    fn test_filter_matches_all() {
        let filter = CheckpointFilter::new();
        let cp = PersistentCheckpoint::builder("t1", json!(null))
            .step(5)
            .build();
        assert!(filter.matches(&cp));
    }

    #[test]
    fn test_filter_by_thread() {
        let filter = CheckpointFilter::new().with_thread("t1");
        let cp1 = PersistentCheckpoint::builder("t1", json!(null)).build();
        let cp2 = PersistentCheckpoint::builder("t2", json!(null)).build();

        assert!(filter.matches(&cp1));
        assert!(!filter.matches(&cp2));
    }

    #[test]
    fn test_filter_by_step_range() {
        let filter = CheckpointFilter::new().with_step_range(2, 5);

        let cp1 = PersistentCheckpoint::builder("t1", json!(null))
            .step(1)
            .build();
        let cp2 = PersistentCheckpoint::builder("t1", json!(null))
            .step(3)
            .build();
        let cp3 = PersistentCheckpoint::builder("t1", json!(null))
            .step(6)
            .build();
        let cp4 = PersistentCheckpoint::builder("t1", json!(null))
            .step(2)
            .build();
        let cp5 = PersistentCheckpoint::builder("t1", json!(null))
            .step(5)
            .build();

        assert!(!filter.matches(&cp1));
        assert!(filter.matches(&cp2));
        assert!(!filter.matches(&cp3));
        assert!(filter.matches(&cp4));
        assert!(filter.matches(&cp5));
    }

    #[test]
    fn test_filter_by_metadata_key() {
        let filter = CheckpointFilter::new().with_metadata_key("source");

        let cp_with = PersistentCheckpoint::builder("t1", json!(null))
            .metadata("source", json!("loop"))
            .build();
        let cp_without = PersistentCheckpoint::builder("t1", json!(null)).build();

        assert!(filter.matches(&cp_with));
        assert!(!filter.matches(&cp_without));
    }

    #[test]
    fn test_filter_combined() {
        let filter = CheckpointFilter::new()
            .with_thread("t1")
            .with_step_range(1, 3)
            .with_metadata_key("source");

        let cp_match = PersistentCheckpoint::builder("t1", json!(null))
            .step(2)
            .metadata("source", json!("loop"))
            .build();
        let cp_wrong_thread = PersistentCheckpoint::builder("t2", json!(null))
            .step(2)
            .metadata("source", json!("loop"))
            .build();
        let cp_wrong_step = PersistentCheckpoint::builder("t1", json!(null))
            .step(5)
            .metadata("source", json!("loop"))
            .build();
        let cp_no_meta = PersistentCheckpoint::builder("t1", json!(null))
            .step(2)
            .build();

        assert!(filter.matches(&cp_match));
        assert!(!filter.matches(&cp_wrong_thread));
        assert!(!filter.matches(&cp_wrong_step));
        assert!(!filter.matches(&cp_no_meta));
    }

    // ---------------------------------------------------------------
    // CheckpointManager tests
    // ---------------------------------------------------------------

    fn new_manager() -> CheckpointManager {
        CheckpointManager::new(Box::new(InMemoryPersistentSaver::new()))
    }

    #[test]
    fn test_manager_create_checkpoint() {
        let mut mgr = new_manager();
        let id = mgr
            .create_checkpoint("t1", json!({"count": 0}), None, HashMap::new())
            .unwrap();

        let cp = mgr.restore(&id).unwrap().unwrap();
        assert_eq!(cp.thread_id, "t1");
        assert_eq!(cp.state, json!({"count": 0}));
        assert_eq!(cp.step, 0);
        assert!(cp.is_root());
    }

    #[test]
    fn test_manager_create_with_parent() {
        let mut mgr = new_manager();
        let id1 = mgr
            .create_checkpoint("t1", json!({"count": 0}), None, HashMap::new())
            .unwrap();
        let id2 = mgr
            .create_checkpoint("t1", json!({"count": 1}), Some(id1.clone()), HashMap::new())
            .unwrap();

        let cp2 = mgr.restore(&id2).unwrap().unwrap();
        assert_eq!(cp2.step, 1);
        assert!(!cp2.is_root());
    }

    #[test]
    fn test_manager_restore_latest() {
        let mut mgr = new_manager();
        let id1 = mgr
            .create_checkpoint("t1", json!(0), None, HashMap::new())
            .unwrap();
        let _id2 = mgr
            .create_checkpoint("t1", json!(1), Some(id1), HashMap::new())
            .unwrap();

        let latest = mgr.restore_latest("t1").unwrap().unwrap();
        assert_eq!(latest.step, 1);
        assert_eq!(latest.state, json!(1));
    }

    #[test]
    fn test_manager_restore_latest_nonexistent() {
        let mgr = new_manager();
        assert!(mgr.restore_latest("no-thread").unwrap().is_none());
    }

    #[test]
    fn test_manager_query_with_filter() {
        let mut mgr = new_manager();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), json!("loop"));

        let id1 = mgr
            .create_checkpoint("t1", json!(0), None, meta.clone())
            .unwrap();
        let _id2 = mgr
            .create_checkpoint("t1", json!(1), Some(id1), meta.clone())
            .unwrap();
        let _id3 = mgr
            .create_checkpoint("t2", json!(0), None, HashMap::new())
            .unwrap();

        let filter = CheckpointFilter::new()
            .with_thread("t1")
            .with_metadata_key("source");
        let results = mgr.query(&filter).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_manager_rollback() {
        let mut mgr = new_manager();

        let id0 = mgr
            .create_checkpoint("t1", json!({"count": 0}), None, HashMap::new())
            .unwrap();
        let id1 = mgr
            .create_checkpoint("t1", json!({"count": 1}), Some(id0), HashMap::new())
            .unwrap();
        let _id2 = mgr
            .create_checkpoint("t1", json!({"count": 2}), Some(id1), HashMap::new())
            .unwrap();

        let rolled_back = mgr.rollback("t1", 1).unwrap().unwrap();
        assert_eq!(rolled_back.step, 1);
        assert_eq!(rolled_back.state, json!({"count": 1}));
    }

    #[test]
    fn test_manager_rollback_to_step_0() {
        let mut mgr = new_manager();

        let id0 = mgr
            .create_checkpoint("t1", json!("initial"), None, HashMap::new())
            .unwrap();
        let _id1 = mgr
            .create_checkpoint("t1", json!("updated"), Some(id0), HashMap::new())
            .unwrap();

        let rolled_back = mgr.rollback("t1", 0).unwrap().unwrap();
        assert_eq!(rolled_back.step, 0);
        assert_eq!(rolled_back.state, json!("initial"));
    }

    #[test]
    fn test_manager_rollback_nonexistent_step() {
        let mut mgr = new_manager();
        mgr.create_checkpoint("t1", json!(null), None, HashMap::new())
            .unwrap();

        assert!(mgr.rollback("t1", 99).unwrap().is_none());
    }

    #[test]
    fn test_manager_rollback_nonexistent_thread() {
        let mgr = new_manager();
        assert!(mgr.rollback("nope", 0).unwrap().is_none());
    }

    #[test]
    fn test_manager_stats_empty() {
        let mgr = new_manager();
        let stats = mgr.stats();
        assert_eq!(stats.total_checkpoints, 0);
        assert_eq!(stats.threads, 0);
        assert_eq!(stats.avg_steps_per_thread, 0.0);
    }

    #[test]
    fn test_manager_stats_populated() {
        let mut mgr = new_manager();

        // Thread 1: 3 checkpoints.
        let id0 = mgr
            .create_checkpoint("t1", json!(0), None, HashMap::new())
            .unwrap();
        let id1 = mgr
            .create_checkpoint("t1", json!(1), Some(id0), HashMap::new())
            .unwrap();
        mgr.create_checkpoint("t1", json!(2), Some(id1), HashMap::new())
            .unwrap();

        // Thread 2: 1 checkpoint.
        mgr.create_checkpoint("t2", json!(0), None, HashMap::new())
            .unwrap();

        let stats = mgr.stats();
        assert_eq!(stats.total_checkpoints, 4);
        assert_eq!(stats.threads, 2);
        assert_eq!(stats.avg_steps_per_thread, 2.0);
    }

    #[test]
    fn test_manager_stats_to_json() {
        let stats = CheckpointStats {
            total_checkpoints: 10,
            threads: 3,
            avg_steps_per_thread: 3.33,
        };
        let j = stats.to_json();
        assert_eq!(j["total_checkpoints"], json!(10));
        assert_eq!(j["threads"], json!(3));
    }

    #[test]
    fn test_manager_history_lineage() {
        let mut mgr = new_manager();

        let id0 = mgr
            .create_checkpoint("t1", json!(0), None, HashMap::new())
            .unwrap();
        let id1 = mgr
            .create_checkpoint("t1", json!(1), Some(id0.clone()), HashMap::new())
            .unwrap();
        let id2 = mgr
            .create_checkpoint("t1", json!(2), Some(id1.clone()), HashMap::new())
            .unwrap();

        let lineage = mgr.history().get_lineage(&id2);
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0], id0);
        assert_eq!(lineage[1], id1);
        assert_eq!(lineage[2], id2);
    }

    // ---------------------------------------------------------------
    // Multi-thread tests
    // ---------------------------------------------------------------

    #[test]
    fn test_multi_thread_isolation() {
        let mut mgr = new_manager();

        mgr.create_checkpoint("t1", json!("t1-data"), None, HashMap::new())
            .unwrap();
        mgr.create_checkpoint("t2", json!("t2-data"), None, HashMap::new())
            .unwrap();

        let t1_latest = mgr.restore_latest("t1").unwrap().unwrap();
        let t2_latest = mgr.restore_latest("t2").unwrap().unwrap();

        assert_eq!(t1_latest.state, json!("t1-data"));
        assert_eq!(t2_latest.state, json!("t2-data"));
    }

    #[test]
    fn test_multi_thread_independent_steps() {
        let mut mgr = new_manager();

        // Thread 1: 5 steps.
        let mut prev: Option<CheckpointId> = None;
        for i in 0..5 {
            let id = mgr
                .create_checkpoint("t1", json!(i), prev.clone(), HashMap::new())
                .unwrap();
            prev = Some(id);
        }

        // Thread 2: 2 steps.
        let id_t2_0 = mgr
            .create_checkpoint("t2", json!(0), None, HashMap::new())
            .unwrap();
        mgr.create_checkpoint("t2", json!(1), Some(id_t2_0), HashMap::new())
            .unwrap();

        let t1_latest = mgr.restore_latest("t1").unwrap().unwrap();
        let t2_latest = mgr.restore_latest("t2").unwrap().unwrap();

        assert_eq!(t1_latest.step, 4);
        assert_eq!(t2_latest.step, 1);
    }

    #[test]
    fn test_query_across_threads() {
        let mut mgr = new_manager();

        mgr.create_checkpoint("t1", json!(0), None, HashMap::new())
            .unwrap();
        mgr.create_checkpoint("t2", json!(0), None, HashMap::new())
            .unwrap();

        let filter = CheckpointFilter::new().with_step_range(0, 0);
        let results = mgr.query(&filter).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ---------------------------------------------------------------
    // Edge cases
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_saver_operations() {
        let saver = InMemoryPersistentSaver::new();
        assert!(saver.load(&CheckpointId::from_str("x")).unwrap().is_none());
        assert!(saver.load_latest("t").unwrap().is_none());
        assert!(saver.list("t").unwrap().is_empty());
    }

    #[test]
    fn test_checkpoint_with_complex_state() {
        let mut saver = InMemoryPersistentSaver::new();
        let complex_state = json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"}
            ],
            "tools": [{"name": "search", "args": {"query": "test"}}],
            "nested": {"deep": {"value": [1, 2, 3]}}
        });

        let cp = PersistentCheckpoint::builder("t1", complex_state.clone())
            .step(0)
            .build();
        let id = saver.save(cp).unwrap();
        let loaded = saver.load(&id).unwrap().unwrap();
        assert_eq!(loaded.state, complex_state);
    }

    #[test]
    fn test_saver_overwrite_same_id() {
        let mut saver = InMemoryPersistentSaver::new();
        let id = CheckpointId::from_str("fixed-id");

        let cp1 = PersistentCheckpoint::builder("t1", json!("first"))
            .id(id.clone())
            .step(0)
            .build();
        saver.save(cp1).unwrap();

        let cp2 = PersistentCheckpoint::builder("t1", json!("second"))
            .id(id.clone())
            .step(1)
            .build();
        saver.save(cp2).unwrap();

        // Should have overwritten.
        assert_eq!(saver.len(), 1);
        let loaded = saver.load(&id).unwrap().unwrap();
        assert_eq!(loaded.state, json!("second"));
    }

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = CheckpointFilter::new();
        for step in 0..10 {
            let cp = PersistentCheckpoint::builder("any-thread", json!(null))
                .step(step)
                .build();
            assert!(filter.matches(&cp));
        }
    }

    #[test]
    fn test_manager_create_with_metadata() {
        let mut mgr = new_manager();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), json!("input"));
        meta.insert("node".to_string(), json!("agent"));

        let id = mgr
            .create_checkpoint("t1", json!(null), None, meta)
            .unwrap();
        let cp = mgr.restore(&id).unwrap().unwrap();
        assert_eq!(cp.metadata["source"], json!("input"));
        assert_eq!(cp.metadata["node"], json!("agent"));
    }

    #[test]
    fn test_history_no_children_for_leaf() {
        let mut history = CheckpointHistory::new();
        let cp = PersistentCheckpoint::builder("t1", json!(null))
            .id(CheckpointId::from_str("leaf"))
            .step(0)
            .build();
        history.add(&cp);

        assert!(history
            .get_children(&CheckpointId::from_str("leaf"))
            .is_empty());
    }
}
