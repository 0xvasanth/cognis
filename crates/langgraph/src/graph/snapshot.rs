//! Graph state snapshot and restore.
//!
//! This module provides [`GraphSnapshot`] for capturing the full execution
//! state of a graph at a point in time, [`SnapshotManager`] for
//! capture/restore/compare operations, and [`SnapshotStorage`] with concrete
//! implementations for persisting snapshots.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::checkpoint::serialization::StateChange;
use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// ExecutionStep
// ---------------------------------------------------------------------------

/// A single step in the graph execution path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionStep {
    /// Name of the node that executed.
    pub node_name: String,
    /// Input state supplied to the node.
    pub input: Value,
    /// Output state produced by the node.
    pub output: Value,
    /// Wall-clock duration of the step in milliseconds.
    pub duration_ms: u64,
    /// 0-based ordinal of this step within the execution.
    pub step_number: usize,
}

// ---------------------------------------------------------------------------
// GraphSnapshot
// ---------------------------------------------------------------------------

/// A full snapshot of graph execution state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSnapshot {
    /// Unique snapshot identifier.
    pub id: String,
    /// Name of the graph this snapshot belongs to.
    pub graph_name: String,
    /// Full state value at snapshot time.
    pub state: Value,
    /// The node being executed when the snapshot was taken (if any).
    pub current_node: Option<String>,
    /// Nodes waiting to execute.
    pub pending_nodes: Vec<String>,
    /// Nodes that have already executed.
    pub completed_nodes: Vec<String>,
    /// Ordered list of execution steps taken so far.
    pub execution_path: Vec<ExecutionStep>,
    /// ISO-8601 timestamp of snapshot creation.
    pub created_at: String,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// SnapshotSummary
// ---------------------------------------------------------------------------

/// Lightweight summary of a snapshot (no full state).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotSummary {
    /// Snapshot ID.
    pub id: String,
    /// Graph name.
    pub graph_name: String,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// Total number of nodes (pending + completed + current).
    pub node_count: usize,
    /// Number of completed execution steps.
    pub completed_steps: usize,
}

impl From<&GraphSnapshot> for SnapshotSummary {
    fn from(snap: &GraphSnapshot) -> Self {
        let mut node_count = snap.pending_nodes.len() + snap.completed_nodes.len();
        if snap.current_node.is_some() {
            node_count += 1;
        }
        Self {
            id: snap.id.clone(),
            graph_name: snap.graph_name.clone(),
            created_at: snap.created_at.clone(),
            node_count,
            completed_steps: snap.execution_path.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiff
// ---------------------------------------------------------------------------

/// Differences between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotDiff {
    /// State-level changes (reuses [`StateChange`] from the serialization
    /// module).
    pub state_changes: Vec<StateChange>,
    /// Execution path entries present in one snapshot but not the other.
    pub path_differences: Vec<String>,
    /// Nodes present in the second snapshot but not the first.
    pub added_nodes: Vec<String>,
    /// Nodes present in the first snapshot but not the second.
    pub removed_nodes: Vec<String>,
}

// ---------------------------------------------------------------------------
// SnapshotStorage trait
// ---------------------------------------------------------------------------

/// Pluggable storage backend for snapshots.
pub trait SnapshotStorage: Send + Sync {
    /// Persist raw snapshot bytes under the given ID.
    fn save(&self, id: &str, data: &[u8]) -> Result<()>;
    /// Load raw snapshot bytes by ID.
    fn load(&self, id: &str) -> Result<Vec<u8>>;
    /// List all stored snapshot IDs.
    fn list(&self) -> Result<Vec<String>>;
    /// Delete a snapshot by ID.
    fn delete(&self, id: &str) -> Result<()>;
    /// Check whether a snapshot with the given ID exists.
    fn exists(&self, id: &str) -> bool;
}

// ---------------------------------------------------------------------------
// InMemorySnapshotStorage
// ---------------------------------------------------------------------------

/// In-memory snapshot storage backed by a `HashMap`.
pub struct InMemorySnapshotStorage {
    data: Mutex<HashMap<String, Vec<u8>>>,
}

impl InMemorySnapshotStorage {
    /// Create a new empty in-memory storage.
    pub fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySnapshotStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotStorage for InMemorySnapshotStorage {
    fn save(&self, id: &str, data: &[u8]) -> Result<()> {
        self.data
            .lock()
            .map_err(|e| LangGraphError::Other(format!("lock poisoned: {e}")))?
            .insert(id.to_string(), data.to_vec());
        Ok(())
    }

    fn load(&self, id: &str) -> Result<Vec<u8>> {
        self.data
            .lock()
            .map_err(|e| LangGraphError::Other(format!("lock poisoned: {e}")))?
            .get(id)
            .cloned()
            .ok_or_else(|| LangGraphError::Other(format!("snapshot not found: {id}")))
    }

    fn list(&self) -> Result<Vec<String>> {
        Ok(self
            .data
            .lock()
            .map_err(|e| LangGraphError::Other(format!("lock poisoned: {e}")))?
            .keys()
            .cloned()
            .collect())
    }

    fn delete(&self, id: &str) -> Result<()> {
        let removed = self
            .data
            .lock()
            .map_err(|e| LangGraphError::Other(format!("lock poisoned: {e}")))?
            .remove(id);
        if removed.is_some() {
            Ok(())
        } else {
            Err(LangGraphError::Other(format!("snapshot not found: {id}")))
        }
    }

    fn exists(&self, id: &str) -> bool {
        self.data
            .lock()
            .map(|d| d.contains_key(id))
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// FileSnapshotStorage
// ---------------------------------------------------------------------------

/// File-system snapshot storage. Each snapshot is stored as a JSON file
/// named `{id}.json` inside the configured directory.
pub struct FileSnapshotStorage {
    dir: PathBuf,
}

impl FileSnapshotStorage {
    /// Create a new file storage rooted at `dir`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| LangGraphError::Other(format!("failed to create dir: {e}")))?;
        Ok(Self { dir })
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}

impl SnapshotStorage for FileSnapshotStorage {
    fn save(&self, id: &str, data: &[u8]) -> Result<()> {
        std::fs::write(self.path_for(id), data)
            .map_err(|e| LangGraphError::Other(format!("write failed: {e}")))
    }

    fn load(&self, id: &str) -> Result<Vec<u8>> {
        std::fs::read(self.path_for(id))
            .map_err(|e| LangGraphError::Other(format!("read failed: {e}")))
    }

    fn list(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(&self.dir)
            .map_err(|e| LangGraphError::Other(format!("read_dir failed: {e}")))?;
        for entry in entries {
            let entry =
                entry.map_err(|e| LangGraphError::Other(format!("dir entry error: {e}")))?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(id) = name.strip_suffix(".json") {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }

    fn delete(&self, id: &str) -> Result<()> {
        let path = self.path_for(id);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| LangGraphError::Other(format!("delete failed: {e}")))
        } else {
            Err(LangGraphError::Other(format!("snapshot not found: {id}")))
        }
    }

    fn exists(&self, id: &str) -> bool {
        self.path_for(id).exists()
    }
}

// ---------------------------------------------------------------------------
// SnapshotManager
// ---------------------------------------------------------------------------

/// High-level manager for capturing, restoring, comparing, and persisting
/// graph snapshots.
pub struct SnapshotManager {
    graph_name: String,
}

impl SnapshotManager {
    /// Create a new manager for the given graph name.
    pub fn new(graph_name: impl Into<String>) -> Self {
        Self {
            graph_name: graph_name.into(),
        }
    }

    /// Capture a snapshot from the current graph state.
    ///
    /// The caller supplies the runtime state value and optional metadata;
    /// node tracking fields default to empty and can be populated by the
    /// execution engine as needed.
    pub fn capture(&self, state: Value, metadata: HashMap<String, Value>) -> GraphSnapshot {
        GraphSnapshot {
            id: Uuid::new_v4().to_string(),
            graph_name: self.graph_name.clone(),
            state,
            current_node: None,
            pending_nodes: Vec::new(),
            completed_nodes: Vec::new(),
            execution_path: Vec::new(),
            created_at: chrono_now(),
            metadata,
        }
    }

    /// Capture a snapshot with full execution context.
    #[allow(clippy::too_many_arguments)]
    pub fn capture_full(
        &self,
        state: Value,
        current_node: Option<String>,
        pending_nodes: Vec<String>,
        completed_nodes: Vec<String>,
        execution_path: Vec<ExecutionStep>,
        metadata: HashMap<String, Value>,
    ) -> GraphSnapshot {
        GraphSnapshot {
            id: Uuid::new_v4().to_string(),
            graph_name: self.graph_name.clone(),
            state,
            current_node,
            pending_nodes,
            completed_nodes,
            execution_path,
            created_at: chrono_now(),
            metadata,
        }
    }

    /// Restore a [`Value`] state from a snapshot.
    ///
    /// Currently this simply clones the state out of the snapshot. Future
    /// versions may perform validation against the graph schema.
    pub fn restore(&self, snapshot: &GraphSnapshot) -> Result<Value> {
        if snapshot.graph_name != self.graph_name {
            return Err(LangGraphError::Other(format!(
                "snapshot graph name mismatch: expected '{}', got '{}'",
                self.graph_name, snapshot.graph_name
            )));
        }
        Ok(snapshot.state.clone())
    }

    /// Persist a snapshot using the given storage backend.
    pub fn save(&self, snapshot: &GraphSnapshot, storage: &dyn SnapshotStorage) -> Result<()> {
        let bytes = serde_json::to_vec(snapshot)
            .map_err(|e| LangGraphError::Other(format!("serialization failed: {e}")))?;
        storage.save(&snapshot.id, &bytes)
    }

    /// Load a snapshot by ID from the given storage backend.
    pub fn load(&self, id: &str, storage: &dyn SnapshotStorage) -> Result<GraphSnapshot> {
        let bytes = storage.load(id)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| LangGraphError::Other(format!("deserialization failed: {e}")))
    }

    /// List summaries of all snapshots in the given storage.
    pub fn list(&self, storage: &dyn SnapshotStorage) -> Result<Vec<SnapshotSummary>> {
        let ids = storage.list()?;
        let mut summaries = Vec::new();
        for id in ids {
            if let Ok(snap) = self.load(&id, storage) {
                summaries.push(SnapshotSummary::from(&snap));
            }
        }
        Ok(summaries)
    }

    /// Delete a snapshot by ID.
    pub fn delete(&self, id: &str, storage: &dyn SnapshotStorage) -> Result<()> {
        storage.delete(id)
    }

    /// Compare two snapshots and return their differences.
    pub fn compare(&self, snap1: &GraphSnapshot, snap2: &GraphSnapshot) -> SnapshotDiff {
        // State changes --------------------------------------------------
        let state_changes = compute_state_changes(&snap1.state, &snap2.state);

        // Path differences -----------------------------------------------
        let path1: Vec<String> = snap1
            .execution_path
            .iter()
            .map(|s| format!("{}@{}", s.node_name, s.step_number))
            .collect();
        let path2: Vec<String> = snap2
            .execution_path
            .iter()
            .map(|s| format!("{}@{}", s.node_name, s.step_number))
            .collect();
        let mut path_differences = Vec::new();
        for p in &path1 {
            if !path2.contains(p) {
                path_differences.push(format!("-{p}"));
            }
        }
        for p in &path2 {
            if !path1.contains(p) {
                path_differences.push(format!("+{p}"));
            }
        }

        // Node differences -----------------------------------------------
        let all_nodes_1 = all_node_names(snap1);
        let all_nodes_2 = all_node_names(snap2);
        let added_nodes = all_nodes_2
            .iter()
            .filter(|n| !all_nodes_1.contains(n))
            .cloned()
            .collect();
        let removed_nodes = all_nodes_1
            .iter()
            .filter(|n| !all_nodes_2.contains(n))
            .cloned()
            .collect();

        SnapshotDiff {
            state_changes,
            path_differences,
            added_nodes,
            removed_nodes,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect all unique node names referenced in a snapshot.
fn all_node_names(snap: &GraphSnapshot) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    if let Some(cn) = &snap.current_node {
        if !names.contains(cn) {
            names.push(cn.clone());
        }
    }
    for n in &snap.pending_nodes {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    for n in &snap.completed_nodes {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    names.sort();
    names
}

/// Compute [`StateChange`]s between two [`Value`]s treated as objects.
fn compute_state_changes(old: &Value, new: &Value) -> Vec<StateChange> {
    let empty = serde_json::Map::new();
    let old_map = old.as_object().unwrap_or(&empty);
    let new_map = new.as_object().unwrap_or(&empty);

    let mut changes = Vec::new();

    for (key, old_val) in old_map {
        match new_map.get(key) {
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
            _ => {}
        }
    }
    for (key, new_val) in new_map {
        if !old_map.contains_key(key) {
            changes.push(StateChange::Added {
                key: key.clone(),
                value: new_val.clone(),
            });
        }
    }

    changes.sort_by(|a, b| {
        let ka = match a {
            StateChange::Added { key, .. } => key,
            StateChange::Removed { key } => key,
            StateChange::Modified { key, .. } => key,
        };
        let kb = match b {
            StateChange::Added { key, .. } => key,
            StateChange::Removed { key } => key,
            StateChange::Modified { key, .. } => key,
        };
        ka.cmp(kb)
    });

    changes
}

/// Return an ISO-8601 timestamp string (UTC) without pulling in the `chrono`
/// crate.  Uses [`std::time::SystemTime`].
fn chrono_now() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple UTC formatting: seconds since epoch -> date-time string.
    // We do minimal formatting here; production code could use chrono.
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;

    // Days since 1970-01-01.
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's `civil_from_days`.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Helper: create a simple snapshot with some defaults.
    fn make_snapshot(
        manager: &SnapshotManager,
        state: Value,
        completed: Vec<&str>,
        pending: Vec<&str>,
        current: Option<&str>,
    ) -> GraphSnapshot {
        let mut snap = manager.capture(state, HashMap::new());
        snap.completed_nodes = completed.into_iter().map(|s| s.to_string()).collect();
        snap.pending_nodes = pending.into_iter().map(|s| s.to_string()).collect();
        snap.current_node = current.map(|s| s.to_string());
        snap
    }

    // 1. Capture snapshot from state
    #[test]
    fn test_capture_snapshot() {
        let mgr = SnapshotManager::new("test_graph");
        let state = json!({"counter": 1});
        let snap = mgr.capture(state.clone(), HashMap::new());

        assert_eq!(snap.graph_name, "test_graph");
        assert_eq!(snap.state, state);
        assert!(!snap.id.is_empty());
        assert!(!snap.created_at.is_empty());
        assert!(snap.current_node.is_none());
        assert!(snap.pending_nodes.is_empty());
        assert!(snap.completed_nodes.is_empty());
        assert!(snap.execution_path.is_empty());
    }

    // 2. Restore snapshot to state
    #[test]
    fn test_restore_snapshot() {
        let mgr = SnapshotManager::new("g1");
        let state = json!({"x": 42});
        let snap = mgr.capture(state.clone(), HashMap::new());
        let restored = mgr.restore(&snap).unwrap();
        assert_eq!(restored, state);
    }

    // 3. Restore fails on graph name mismatch
    #[test]
    fn test_restore_graph_name_mismatch() {
        let mgr1 = SnapshotManager::new("g1");
        let mgr2 = SnapshotManager::new("g2");
        let snap = mgr1.capture(json!({}), HashMap::new());
        assert!(mgr2.restore(&snap).is_err());
    }

    // 4. InMemorySnapshotStorage save & load
    #[test]
    fn test_in_memory_save_load() {
        let storage = InMemorySnapshotStorage::new();
        let mgr = SnapshotManager::new("g");
        let snap = mgr.capture(json!({"a": 1}), HashMap::new());
        mgr.save(&snap, &storage).unwrap();

        let loaded = mgr.load(&snap.id, &storage).unwrap();
        assert_eq!(loaded.id, snap.id);
        assert_eq!(loaded.state, snap.state);
    }

    // 5. InMemorySnapshotStorage exists
    #[test]
    fn test_in_memory_exists() {
        let storage = InMemorySnapshotStorage::new();
        storage.save("abc", b"data").unwrap();
        assert!(storage.exists("abc"));
        assert!(!storage.exists("missing"));
    }

    // 6. InMemorySnapshotStorage delete
    #[test]
    fn test_in_memory_delete() {
        let storage = InMemorySnapshotStorage::new();
        storage.save("abc", b"data").unwrap();
        storage.delete("abc").unwrap();
        assert!(!storage.exists("abc"));
        assert!(storage.delete("abc").is_err());
    }

    // 7. InMemorySnapshotStorage list
    #[test]
    fn test_in_memory_list() {
        let storage = InMemorySnapshotStorage::new();
        storage.save("a", b"1").unwrap();
        storage.save("b", b"2").unwrap();
        let mut ids = storage.list().unwrap();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    // 8. FileSnapshotStorage CRUD
    #[test]
    fn test_file_storage_crud() {
        let dir = tempfile::tempdir().unwrap();
        let storage = FileSnapshotStorage::new(dir.path()).unwrap();

        // save
        storage.save("snap1", b"hello").unwrap();
        assert!(storage.exists("snap1"));

        // load
        let data = storage.load("snap1").unwrap();
        assert_eq!(data, b"hello");

        // list
        let ids = storage.list().unwrap();
        assert_eq!(ids, vec!["snap1".to_string()]);

        // delete
        storage.delete("snap1").unwrap();
        assert!(!storage.exists("snap1"));
        assert!(storage.delete("snap1").is_err());
    }

    // 9. List snapshots through manager
    #[test]
    fn test_list_snapshots() {
        let storage = InMemorySnapshotStorage::new();
        let mgr = SnapshotManager::new("g");

        let s1 = mgr.capture(json!({"a": 1}), HashMap::new());
        let s2 = mgr.capture(json!({"b": 2}), HashMap::new());
        mgr.save(&s1, &storage).unwrap();
        mgr.save(&s2, &storage).unwrap();

        let summaries = mgr.list(&storage).unwrap();
        assert_eq!(summaries.len(), 2);
    }

    // 10. Snapshot summary generation
    #[test]
    fn test_snapshot_summary() {
        let mgr = SnapshotManager::new("mygraph");
        let mut snap = mgr.capture(json!({}), HashMap::new());
        snap.completed_nodes = vec!["a".into(), "b".into()];
        snap.pending_nodes = vec!["c".into()];
        snap.current_node = Some("d".into());
        snap.execution_path = vec![
            ExecutionStep {
                node_name: "a".into(),
                input: json!({}),
                output: json!({}),
                duration_ms: 10,
                step_number: 0,
            },
            ExecutionStep {
                node_name: "b".into(),
                input: json!({}),
                output: json!({}),
                duration_ms: 20,
                step_number: 1,
            },
        ];

        let summary = SnapshotSummary::from(&snap);
        assert_eq!(summary.id, snap.id);
        assert_eq!(summary.graph_name, "mygraph");
        assert_eq!(summary.node_count, 4); // a, b, c, d
        assert_eq!(summary.completed_steps, 2);
    }

    // 11. Compare two snapshots
    #[test]
    fn test_compare_snapshots() {
        let mgr = SnapshotManager::new("g");
        let mut s1 = make_snapshot(&mgr, json!({"x": 1, "y": 2}), vec!["a"], vec!["b"], None);
        s1.execution_path.push(ExecutionStep {
            node_name: "a".into(),
            input: json!({}),
            output: json!({}),
            duration_ms: 5,
            step_number: 0,
        });

        let mut s2 = make_snapshot(
            &mgr,
            json!({"x": 1, "z": 3}),
            vec!["a", "b"],
            vec![],
            Some("c"),
        );
        s2.execution_path.push(ExecutionStep {
            node_name: "a".into(),
            input: json!({}),
            output: json!({}),
            duration_ms: 5,
            step_number: 0,
        });
        s2.execution_path.push(ExecutionStep {
            node_name: "b".into(),
            input: json!({}),
            output: json!({}),
            duration_ms: 7,
            step_number: 1,
        });

        let diff = mgr.compare(&s1, &s2);

        // State: y removed, z added
        assert!(!diff.state_changes.is_empty());
        // Node c added in s2
        assert!(diff.added_nodes.contains(&"c".to_string()));
        // path difference: b@1 added in s2
        assert!(!diff.path_differences.is_empty());
    }

    // 12. ExecutionStep tracking
    #[test]
    fn test_execution_step_tracking() {
        let mgr = SnapshotManager::new("g");
        let steps = vec![
            ExecutionStep {
                node_name: "init".into(),
                input: json!({"in": 0}),
                output: json!({"out": 1}),
                duration_ms: 100,
                step_number: 0,
            },
            ExecutionStep {
                node_name: "process".into(),
                input: json!({"out": 1}),
                output: json!({"out": 2}),
                duration_ms: 200,
                step_number: 1,
            },
        ];
        let snap = mgr.capture_full(
            json!({"out": 2}),
            Some("finish".into()),
            vec![],
            vec!["init".into(), "process".into()],
            steps.clone(),
            HashMap::new(),
        );
        assert_eq!(snap.execution_path.len(), 2);
        assert_eq!(snap.execution_path[0].node_name, "init");
        assert_eq!(snap.execution_path[1].duration_ms, 200);
        assert_eq!(snap.current_node, Some("finish".to_string()));
    }

    // 13. Snapshot metadata
    #[test]
    fn test_snapshot_metadata() {
        let mgr = SnapshotManager::new("g");
        let mut meta = HashMap::new();
        meta.insert("user".to_string(), json!("alice"));
        meta.insert("reason".to_string(), json!("debugging"));
        let snap = mgr.capture(json!({}), meta.clone());
        assert_eq!(snap.metadata.get("user"), Some(&json!("alice")));
        assert_eq!(snap.metadata.get("reason"), Some(&json!("debugging")));
    }

    // 14. Empty state snapshot
    #[test]
    fn test_empty_state_snapshot() {
        let mgr = SnapshotManager::new("g");
        let snap = mgr.capture(json!({}), HashMap::new());
        assert_eq!(snap.state, json!({}));
        let restored = mgr.restore(&snap).unwrap();
        assert_eq!(restored, json!({}));
    }

    // 15. Multiple snapshots for same graph
    #[test]
    fn test_multiple_snapshots_same_graph() {
        let storage = InMemorySnapshotStorage::new();
        let mgr = SnapshotManager::new("g");

        let s1 = mgr.capture(json!({"step": 1}), HashMap::new());
        let s2 = mgr.capture(json!({"step": 2}), HashMap::new());
        let s3 = mgr.capture(json!({"step": 3}), HashMap::new());

        mgr.save(&s1, &storage).unwrap();
        mgr.save(&s2, &storage).unwrap();
        mgr.save(&s3, &storage).unwrap();

        let summaries = mgr.list(&storage).unwrap();
        assert_eq!(summaries.len(), 3);

        // Each has a unique id
        let mut ids: Vec<_> = summaries.iter().map(|s| s.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 3);
    }

    // 16. Snapshot serialization roundtrip
    #[test]
    fn test_snapshot_serialization_roundtrip() {
        let mgr = SnapshotManager::new("roundtrip_graph");
        let mut snap = mgr.capture(json!({"key": "value", "num": 42}), HashMap::new());
        snap.current_node = Some("node_a".into());
        snap.pending_nodes = vec!["node_b".into()];
        snap.completed_nodes = vec!["node_c".into()];
        snap.execution_path.push(ExecutionStep {
            node_name: "node_c".into(),
            input: json!({}),
            output: json!({"key": "value"}),
            duration_ms: 15,
            step_number: 0,
        });

        let json_bytes = serde_json::to_vec(&snap).unwrap();
        let deserialized: GraphSnapshot = serde_json::from_slice(&json_bytes).unwrap();
        assert_eq!(snap, deserialized);
    }

    // 17. Pending/completed nodes tracking
    #[test]
    fn test_pending_completed_nodes_tracking() {
        let mgr = SnapshotManager::new("g");
        let snap = make_snapshot(
            &mgr,
            json!({}),
            vec!["done1", "done2"],
            vec!["wait1", "wait2", "wait3"],
            Some("running"),
        );
        assert_eq!(snap.completed_nodes, vec!["done1", "done2"]);
        assert_eq!(snap.pending_nodes, vec!["wait1", "wait2", "wait3"]);
        assert_eq!(snap.current_node, Some("running".to_string()));

        let summary = SnapshotSummary::from(&snap);
        // 2 completed + 3 pending + 1 current = 6
        assert_eq!(summary.node_count, 6);
    }

    // 18. Delete snapshot via manager
    #[test]
    fn test_delete_snapshot_via_manager() {
        let storage = InMemorySnapshotStorage::new();
        let mgr = SnapshotManager::new("g");
        let snap = mgr.capture(json!({}), HashMap::new());
        mgr.save(&snap, &storage).unwrap();
        assert!(storage.exists(&snap.id));
        mgr.delete(&snap.id, &storage).unwrap();
        assert!(!storage.exists(&snap.id));
    }
}
