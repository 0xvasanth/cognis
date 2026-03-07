//! State snapshot and time-travel debugging.
//!
//! This module provides types for capturing graph state at any execution point
//! and replaying/inspecting execution history. The main components are:
//!
//! - [`SnapshotId`] — unique identifier for a snapshot
//! - [`StateSnapshot`] — captured state at a point in execution
//! - [`SnapshotStore`] — indexed, capacity-bounded snapshot storage
//! - [`TimeTravelDebugger`] — navigation over snapshot history
//! - [`SnapshotComparator`] — structural comparison of two snapshots
//! - [`SnapshotDiff`] — the result of comparing two snapshots

use std::collections::HashMap;
use std::time::Instant;

use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// SnapshotId
// ---------------------------------------------------------------------------

/// Unique identifier for a state snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SnapshotId(String);

impl SnapshotId {
    /// Generate a new unique snapshot ID (UUID v4).
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a `SnapshotId` from a string slice.
    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Return the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SnapshotId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// StateSnapshot
// ---------------------------------------------------------------------------

/// A captured snapshot of graph state at a specific execution point.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Unique identifier for this snapshot.
    pub id: SnapshotId,
    /// The full graph state at snapshot time.
    pub state: Value,
    /// Name of the node being executed when the snapshot was taken.
    pub node_name: String,
    /// The step number (0-based ordinal) in the execution.
    pub step: usize,
    /// The instant at which the snapshot was created.
    pub timestamp: Instant,
    /// Optional parent snapshot ID (for lineage tracking).
    pub parent_id: Option<SnapshotId>,
    /// Arbitrary metadata attached to this snapshot.
    pub metadata: HashMap<String, Value>,
}

impl StateSnapshot {
    /// Create a new snapshot with the given state, node name, and step number.
    pub fn new(state: Value, node_name: impl Into<String>, step: usize) -> Self {
        Self {
            id: SnapshotId::new(),
            state,
            node_name: node_name.into(),
            step,
            timestamp: Instant::now(),
            parent_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the parent snapshot ID (builder pattern).
    pub fn with_parent(mut self, parent_id: SnapshotId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Add a metadata key-value pair (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize the snapshot to a JSON [`Value`].
    ///
    /// Note: `timestamp` is represented as elapsed nanoseconds since creation
    /// (relative to `Instant` epoch) and `Instant` cannot be serialized
    /// absolutely, so it is omitted from the JSON representation.
    pub fn to_json(&self) -> Value {
        let parent = match &self.parent_id {
            Some(pid) => Value::String(pid.as_str().to_string()),
            None => Value::Null,
        };
        json!({
            "id": self.id.as_str(),
            "state": self.state,
            "node_name": self.node_name,
            "step": self.step,
            "parent_id": parent,
            "metadata": self.metadata,
        })
    }

    /// Compute a simple diff between this snapshot's state and another's.
    ///
    /// Returns a JSON object with keys `added`, `removed`, and `changed`
    /// describing the differences at the top level of the state objects.
    pub fn state_diff(&self, other: &StateSnapshot) -> Value {
        let diff = SnapshotComparator::diff(self, other);
        diff.to_json()
    }
}

// ---------------------------------------------------------------------------
// SnapshotStore
// ---------------------------------------------------------------------------

/// Indexed, capacity-bounded store for [`StateSnapshot`]s.
///
/// Snapshots are stored in insertion order. When the store reaches its
/// configured maximum capacity, the oldest snapshot is evicted on each new
/// insert.
pub struct SnapshotStore {
    /// Ordered list of snapshots (oldest first).
    snapshots: Vec<StateSnapshot>,
    /// Maximum number of snapshots to retain. `None` means unlimited.
    max_snapshots: Option<usize>,
}

impl SnapshotStore {
    /// Create a new store with no capacity limit.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots: None,
        }
    }

    /// Create a new store that retains at most `n` snapshots.
    pub fn with_max_snapshots(n: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots: Some(n),
        }
    }

    /// Save a snapshot into the store.
    ///
    /// If the store is at capacity, the oldest snapshot is evicted first.
    pub fn save(&mut self, snapshot: StateSnapshot) {
        if let Some(max) = self.max_snapshots {
            if max > 0 && self.snapshots.len() >= max {
                self.snapshots.remove(0);
            }
        }
        self.snapshots.push(snapshot);
    }

    /// Retrieve a snapshot by its ID.
    pub fn get(&self, id: &SnapshotId) -> Option<&StateSnapshot> {
        self.snapshots.iter().find(|s| s.id == *id)
    }

    /// Retrieve all snapshots recorded at a specific step number.
    pub fn get_by_step(&self, step: usize) -> Vec<&StateSnapshot> {
        self.snapshots.iter().filter(|s| s.step == step).collect()
    }

    /// Retrieve all snapshots recorded for a specific node.
    pub fn get_by_node(&self, node: &str) -> Vec<&StateSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.node_name == node)
            .collect()
    }

    /// Return the most recently saved snapshot, if any.
    pub fn latest(&self) -> Option<&StateSnapshot> {
        self.snapshots.last()
    }

    /// Return all snapshots in insertion order (oldest first).
    pub fn history(&self) -> Vec<&StateSnapshot> {
        self.snapshots.iter().collect()
    }

    /// Return the number of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Remove all snapshots from the store.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

impl Default for SnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TimeTravelDebugger
// ---------------------------------------------------------------------------

/// Enables navigating and replaying graph execution from stored snapshots.
///
/// The debugger borrows a [`SnapshotStore`] and provides high-level navigation
/// methods such as `goto_step`, `rewind`, and `diff_steps`.
pub struct TimeTravelDebugger<'a> {
    store: &'a SnapshotStore,
}

impl<'a> TimeTravelDebugger<'a> {
    /// Create a new debugger over the given snapshot store.
    pub fn new(store: &'a SnapshotStore) -> Self {
        Self { store }
    }

    /// Find the first snapshot recorded at the given step number.
    pub fn goto_step(&self, step: usize) -> Option<&'a StateSnapshot> {
        self.store
            .snapshots
            .iter()
            .find(|s| s.step == step)
    }

    /// Find a snapshot by its ID.
    pub fn goto_snapshot(&self, id: &SnapshotId) -> Option<&'a StateSnapshot> {
        self.store.get(id)
    }

    /// Go back `steps` positions from the latest snapshot.
    ///
    /// `rewind(0)` returns the latest snapshot. `rewind(1)` returns the
    /// second-to-last, and so on.
    pub fn rewind(&self, steps: usize) -> Option<&'a StateSnapshot> {
        let len = self.store.snapshots.len();
        if len == 0 || steps >= len {
            return None;
        }
        Some(&self.store.snapshots[len - 1 - steps])
    }

    /// Compute the state diff between the first snapshots found at `step1`
    /// and `step2`.
    ///
    /// Returns `None` if either step has no snapshot.
    pub fn diff_steps(&self, step1: usize, step2: usize) -> Option<Value> {
        let s1 = self.goto_step(step1)?;
        let s2 = self.goto_step(step2)?;
        Some(s1.state_diff(s2))
    }

    /// Return the execution path as a list of `(step, node_name)` pairs in
    /// insertion order.
    pub fn execution_path(&self) -> Vec<(usize, String)> {
        self.store
            .snapshots
            .iter()
            .map(|s| (s.step, s.node_name.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SnapshotComparator
// ---------------------------------------------------------------------------

/// Compares two [`StateSnapshot`]s structurally.
pub struct SnapshotComparator;

impl SnapshotComparator {
    /// Compute the structural diff between two snapshots' states.
    pub fn diff(a: &StateSnapshot, b: &StateSnapshot) -> SnapshotDiff {
        let empty = serde_json::Map::new();
        let map_a = a.state.as_object().unwrap_or(&empty);
        let map_b = b.state.as_object().unwrap_or(&empty);

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for (key, val_a) in map_a {
            match map_b.get(key) {
                Some(val_b) if val_a != val_b => {
                    changed.push((key.clone(), val_a.clone(), val_b.clone()));
                }
                None => {
                    removed.push(key.clone());
                }
                _ => {}
            }
        }

        for key in map_b.keys() {
            if !map_a.contains_key(key) {
                added.push(key.clone());
            }
        }

        // Sort for deterministic output.
        added.sort();
        removed.sort();
        changed.sort_by(|a, b| a.0.cmp(&b.0));

        SnapshotDiff {
            added,
            removed,
            changed,
        }
    }

    /// Check whether two snapshots have equivalent state (ignoring ID,
    /// timestamp, metadata, etc.).
    pub fn is_equivalent(a: &StateSnapshot, b: &StateSnapshot) -> bool {
        a.state == b.state
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiff
// ---------------------------------------------------------------------------

/// The result of comparing two snapshot states.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotDiff {
    /// Keys present in the second snapshot but not the first.
    pub added: Vec<String>,
    /// Keys present in the first snapshot but not the second.
    pub removed: Vec<String>,
    /// Keys present in both but with different values: `(key, old, new)`.
    pub changed: Vec<(String, Value, Value)>,
}

impl SnapshotDiff {
    /// Returns `true` if no differences were found.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    /// Serialize the diff to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let changed_json: Vec<Value> = self
            .changed
            .iter()
            .map(|(k, old, new)| {
                json!({
                    "key": k,
                    "old": old,
                    "new": new,
                })
            })
            .collect();

        json!({
            "added": self.added,
            "removed": self.removed,
            "changed": changed_json,
        })
    }

    /// Return a human-readable summary of the diff.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if !self.added.is_empty() {
            parts.push(format!("added: [{}]", self.added.join(", ")));
        }
        if !self.removed.is_empty() {
            parts.push(format!("removed: [{}]", self.removed.join(", ")));
        }
        if !self.changed.is_empty() {
            let keys: Vec<&str> = self.changed.iter().map(|(k, _, _)| k.as_str()).collect();
            parts.push(format!("changed: [{}]", keys.join(", ")));
        }

        if parts.is_empty() {
            "no differences".to_string()
        } else {
            parts.join("; ")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ===================================================================
    // SnapshotId tests
    // ===================================================================

    #[test]
    fn test_snapshot_id_new_generates_unique_ids() {
        let id1 = SnapshotId::new();
        let id2 = SnapshotId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_snapshot_id_from_str_roundtrip() {
        let id = SnapshotId::from_str("my-custom-id");
        assert_eq!(id.as_str(), "my-custom-id");
    }

    #[test]
    fn test_snapshot_id_equality() {
        let a = SnapshotId::from_str("abc");
        let b = SnapshotId::from_str("abc");
        let c = SnapshotId::from_str("xyz");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_snapshot_id_display() {
        let id = SnapshotId::from_str("display-test");
        assert_eq!(format!("{}", id), "display-test");
    }

    #[test]
    fn test_snapshot_id_default_generates_id() {
        let id = SnapshotId::default();
        assert!(!id.as_str().is_empty());
    }

    // ===================================================================
    // StateSnapshot tests
    // ===================================================================

    #[test]
    fn test_state_snapshot_new() {
        let snap = StateSnapshot::new(json!({"count": 1}), "node_a", 0);
        assert_eq!(snap.state, json!({"count": 1}));
        assert_eq!(snap.node_name, "node_a");
        assert_eq!(snap.step, 0);
        assert!(snap.parent_id.is_none());
        assert!(snap.metadata.is_empty());
    }

    #[test]
    fn test_state_snapshot_with_parent() {
        let parent = SnapshotId::from_str("parent-1");
        let snap = StateSnapshot::new(json!({}), "n", 1).with_parent(parent.clone());
        assert_eq!(snap.parent_id, Some(parent));
    }

    #[test]
    fn test_state_snapshot_with_metadata() {
        let snap = StateSnapshot::new(json!({}), "n", 0)
            .with_metadata("user", json!("alice"))
            .with_metadata("reason", json!("debug"));
        assert_eq!(snap.metadata.get("user"), Some(&json!("alice")));
        assert_eq!(snap.metadata.get("reason"), Some(&json!("debug")));
    }

    #[test]
    fn test_state_snapshot_to_json() {
        let snap = StateSnapshot::new(json!({"x": 42}), "compute", 3)
            .with_metadata("tag", json!("test"));
        let j = snap.to_json();
        assert_eq!(j["node_name"], "compute");
        assert_eq!(j["step"], 3);
        assert_eq!(j["state"]["x"], 42);
        assert_eq!(j["metadata"]["tag"], "test");
        assert!(j["parent_id"].is_null());
    }

    #[test]
    fn test_state_snapshot_to_json_with_parent() {
        let parent = SnapshotId::from_str("p1");
        let snap = StateSnapshot::new(json!({}), "n", 0).with_parent(parent);
        let j = snap.to_json();
        assert_eq!(j["parent_id"], "p1");
    }

    #[test]
    fn test_state_snapshot_state_diff() {
        let a = StateSnapshot::new(json!({"x": 1, "y": 2}), "n", 0);
        let b = StateSnapshot::new(json!({"x": 1, "z": 3}), "n", 1);
        let diff = a.state_diff(&b);
        // y removed, z added
        assert!(diff["removed"].as_array().unwrap().contains(&json!("y")));
        assert!(diff["added"].as_array().unwrap().contains(&json!("z")));
        assert!(diff["changed"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_state_snapshot_state_diff_changed() {
        let a = StateSnapshot::new(json!({"x": 1}), "n", 0);
        let b = StateSnapshot::new(json!({"x": 99}), "n", 1);
        let diff = a.state_diff(&b);
        let changed = diff["changed"].as_array().unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0]["key"], "x");
        assert_eq!(changed[0]["old"], 1);
        assert_eq!(changed[0]["new"], 99);
    }

    // ===================================================================
    // SnapshotStore tests
    // ===================================================================

    #[test]
    fn test_snapshot_store_new_is_empty() {
        let store = SnapshotStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.latest().is_none());
    }

    #[test]
    fn test_snapshot_store_save_and_get() {
        let mut store = SnapshotStore::new();
        let snap = StateSnapshot::new(json!({"a": 1}), "node1", 0);
        let id = snap.id.clone();
        store.save(snap);
        assert_eq!(store.len(), 1);
        assert!(store.get(&id).is_some());
        assert_eq!(store.get(&id).unwrap().node_name, "node1");
    }

    #[test]
    fn test_snapshot_store_get_missing() {
        let store = SnapshotStore::new();
        assert!(store.get(&SnapshotId::from_str("nonexistent")).is_none());
    }

    #[test]
    fn test_snapshot_store_capacity_eviction() {
        let mut store = SnapshotStore::with_max_snapshots(3);
        let ids: Vec<SnapshotId> = (0..5)
            .map(|i| {
                let snap = StateSnapshot::new(json!({"i": i}), "n", i);
                let id = snap.id.clone();
                store.save(snap);
                id
            })
            .collect();

        assert_eq!(store.len(), 3);
        // First two should have been evicted.
        assert!(store.get(&ids[0]).is_none());
        assert!(store.get(&ids[1]).is_none());
        // Last three should remain.
        assert!(store.get(&ids[2]).is_some());
        assert!(store.get(&ids[3]).is_some());
        assert!(store.get(&ids[4]).is_some());
    }

    #[test]
    fn test_snapshot_store_get_by_step() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({}), "a", 0));
        store.save(StateSnapshot::new(json!({}), "b", 1));
        store.save(StateSnapshot::new(json!({}), "c", 1));
        store.save(StateSnapshot::new(json!({}), "d", 2));

        assert_eq!(store.get_by_step(0).len(), 1);
        assert_eq!(store.get_by_step(1).len(), 2);
        assert_eq!(store.get_by_step(2).len(), 1);
        assert_eq!(store.get_by_step(99).len(), 0);
    }

    #[test]
    fn test_snapshot_store_get_by_node() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({}), "alpha", 0));
        store.save(StateSnapshot::new(json!({}), "beta", 1));
        store.save(StateSnapshot::new(json!({}), "alpha", 2));

        assert_eq!(store.get_by_node("alpha").len(), 2);
        assert_eq!(store.get_by_node("beta").len(), 1);
        assert_eq!(store.get_by_node("gamma").len(), 0);
    }

    #[test]
    fn test_snapshot_store_latest() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({"v": 1}), "a", 0));
        store.save(StateSnapshot::new(json!({"v": 2}), "b", 1));
        store.save(StateSnapshot::new(json!({"v": 3}), "c", 2));

        let latest = store.latest().unwrap();
        assert_eq!(latest.state, json!({"v": 3}));
        assert_eq!(latest.node_name, "c");
    }

    #[test]
    fn test_snapshot_store_history_order() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({}), "first", 0));
        store.save(StateSnapshot::new(json!({}), "second", 1));
        store.save(StateSnapshot::new(json!({}), "third", 2));

        let history = store.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].node_name, "first");
        assert_eq!(history[1].node_name, "second");
        assert_eq!(history[2].node_name, "third");
    }

    #[test]
    fn test_snapshot_store_clear() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({}), "n", 0));
        store.save(StateSnapshot::new(json!({}), "n", 1));
        assert_eq!(store.len(), 2);
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    // ===================================================================
    // TimeTravelDebugger tests
    // ===================================================================

    fn populated_store() -> SnapshotStore {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({"count": 0}), "init", 0));
        store.save(StateSnapshot::new(json!({"count": 1}), "process", 1));
        store.save(StateSnapshot::new(json!({"count": 2}), "validate", 2));
        store.save(StateSnapshot::new(json!({"count": 3}), "output", 3));
        store
    }

    #[test]
    fn test_debugger_goto_step() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);

        let snap = dbg.goto_step(2).unwrap();
        assert_eq!(snap.node_name, "validate");
        assert_eq!(snap.state, json!({"count": 2}));
    }

    #[test]
    fn test_debugger_goto_step_missing() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);
        assert!(dbg.goto_step(99).is_none());
    }

    #[test]
    fn test_debugger_goto_snapshot() {
        let mut store = SnapshotStore::new();
        let snap = StateSnapshot::new(json!({"x": 1}), "n", 0);
        let id = snap.id.clone();
        store.save(snap);

        let dbg = TimeTravelDebugger::new(&store);
        assert!(dbg.goto_snapshot(&id).is_some());
        assert!(dbg.goto_snapshot(&SnapshotId::from_str("nope")).is_none());
    }

    #[test]
    fn test_debugger_rewind() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);

        // rewind(0) = latest
        let latest = dbg.rewind(0).unwrap();
        assert_eq!(latest.node_name, "output");

        // rewind(2) = second from start
        let two_back = dbg.rewind(2).unwrap();
        assert_eq!(two_back.node_name, "process");

        // rewind(3) = first
        let first = dbg.rewind(3).unwrap();
        assert_eq!(first.node_name, "init");

        // rewind beyond history
        assert!(dbg.rewind(4).is_none());
        assert!(dbg.rewind(100).is_none());
    }

    #[test]
    fn test_debugger_rewind_empty_store() {
        let store = SnapshotStore::new();
        let dbg = TimeTravelDebugger::new(&store);
        assert!(dbg.rewind(0).is_none());
    }

    #[test]
    fn test_debugger_diff_steps() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);

        let diff = dbg.diff_steps(0, 3).unwrap();
        let changed = diff["changed"].as_array().unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0]["key"], "count");
        assert_eq!(changed[0]["old"], 0);
        assert_eq!(changed[0]["new"], 3);
    }

    #[test]
    fn test_debugger_diff_steps_missing() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);
        assert!(dbg.diff_steps(0, 99).is_none());
        assert!(dbg.diff_steps(99, 0).is_none());
    }

    #[test]
    fn test_debugger_execution_path() {
        let store = populated_store();
        let dbg = TimeTravelDebugger::new(&store);

        let path = dbg.execution_path();
        assert_eq!(
            path,
            vec![
                (0, "init".to_string()),
                (1, "process".to_string()),
                (2, "validate".to_string()),
                (3, "output".to_string()),
            ]
        );
    }

    #[test]
    fn test_debugger_execution_path_empty() {
        let store = SnapshotStore::new();
        let dbg = TimeTravelDebugger::new(&store);
        assert!(dbg.execution_path().is_empty());
    }

    // ===================================================================
    // SnapshotComparator tests
    // ===================================================================

    #[test]
    fn test_comparator_diff_added_removed_changed() {
        let a = StateSnapshot::new(json!({"x": 1, "y": 2, "z": 3}), "n", 0);
        let b = StateSnapshot::new(json!({"x": 1, "y": 99, "w": 4}), "n", 1);
        let diff = SnapshotComparator::diff(&a, &b);

        assert_eq!(diff.added, vec!["w".to_string()]);
        assert_eq!(diff.removed, vec!["z".to_string()]);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].0, "y");
        assert_eq!(diff.changed[0].1, json!(2));
        assert_eq!(diff.changed[0].2, json!(99));
    }

    #[test]
    fn test_comparator_diff_identical() {
        let a = StateSnapshot::new(json!({"x": 1}), "n", 0);
        let b = StateSnapshot::new(json!({"x": 1}), "n", 1);
        let diff = SnapshotComparator::diff(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_comparator_diff_empty_states() {
        let a = StateSnapshot::new(json!({}), "n", 0);
        let b = StateSnapshot::new(json!({}), "n", 1);
        let diff = SnapshotComparator::diff(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_comparator_diff_non_object_states() {
        let a = StateSnapshot::new(json!(42), "n", 0);
        let b = StateSnapshot::new(json!("hello"), "n", 1);
        let diff = SnapshotComparator::diff(&a, &b);
        // Both treated as empty objects -> no diff.
        assert!(diff.is_empty());
    }

    #[test]
    fn test_comparator_is_equivalent_true() {
        let a = StateSnapshot::new(json!({"x": 1}), "node_a", 0);
        let b = StateSnapshot::new(json!({"x": 1}), "node_b", 5);
        assert!(SnapshotComparator::is_equivalent(&a, &b));
    }

    #[test]
    fn test_comparator_is_equivalent_false() {
        let a = StateSnapshot::new(json!({"x": 1}), "n", 0);
        let b = StateSnapshot::new(json!({"x": 2}), "n", 0);
        assert!(!SnapshotComparator::is_equivalent(&a, &b));
    }

    // ===================================================================
    // SnapshotDiff tests
    // ===================================================================

    #[test]
    fn test_snapshot_diff_is_empty() {
        let diff = SnapshotDiff {
            added: vec![],
            removed: vec![],
            changed: vec![],
        };
        assert!(diff.is_empty());
    }

    #[test]
    fn test_snapshot_diff_is_not_empty_with_added() {
        let diff = SnapshotDiff {
            added: vec!["key".into()],
            removed: vec![],
            changed: vec![],
        };
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_snapshot_diff_to_json() {
        let diff = SnapshotDiff {
            added: vec!["new_key".into()],
            removed: vec!["old_key".into()],
            changed: vec![("mod_key".into(), json!(1), json!(2))],
        };
        let j = diff.to_json();
        assert_eq!(j["added"], json!(["new_key"]));
        assert_eq!(j["removed"], json!(["old_key"]));
        let changed = j["changed"].as_array().unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0]["key"], "mod_key");
    }

    #[test]
    fn test_snapshot_diff_summary_no_differences() {
        let diff = SnapshotDiff {
            added: vec![],
            removed: vec![],
            changed: vec![],
        };
        assert_eq!(diff.summary(), "no differences");
    }

    #[test]
    fn test_snapshot_diff_summary_with_all_kinds() {
        let diff = SnapshotDiff {
            added: vec!["a".into(), "b".into()],
            removed: vec!["c".into()],
            changed: vec![("d".into(), json!(1), json!(2))],
        };
        let s = diff.summary();
        assert!(s.contains("added: [a, b]"));
        assert!(s.contains("removed: [c]"));
        assert!(s.contains("changed: [d]"));
    }

    #[test]
    fn test_snapshot_diff_summary_added_only() {
        let diff = SnapshotDiff {
            added: vec!["x".into()],
            removed: vec![],
            changed: vec![],
        };
        assert_eq!(diff.summary(), "added: [x]");
    }

    // ===================================================================
    // Integration / edge-case tests
    // ===================================================================

    #[test]
    fn test_store_with_max_one_keeps_only_latest() {
        let mut store = SnapshotStore::with_max_snapshots(1);
        store.save(StateSnapshot::new(json!({"v": 1}), "a", 0));
        store.save(StateSnapshot::new(json!({"v": 2}), "b", 1));
        assert_eq!(store.len(), 1);
        assert_eq!(store.latest().unwrap().state, json!({"v": 2}));
    }

    #[test]
    fn test_full_workflow_save_navigate_diff() {
        let mut store = SnapshotStore::new();
        store.save(StateSnapshot::new(json!({"x": 0, "y": "hello"}), "start", 0));
        store.save(StateSnapshot::new(json!({"x": 1, "y": "hello"}), "step1", 1));
        store.save(StateSnapshot::new(json!({"x": 2, "y": "world", "z": true}), "step2", 2));

        let dbg = TimeTravelDebugger::new(&store);

        // Navigate
        assert_eq!(dbg.goto_step(0).unwrap().node_name, "start");
        assert_eq!(dbg.rewind(0).unwrap().node_name, "step2");

        // Diff
        let diff = dbg.diff_steps(0, 2).unwrap();
        assert!(diff["added"].as_array().unwrap().contains(&json!("z")));
        assert!(diff["changed"].as_array().unwrap().len() >= 1);

        // Execution path
        let path = dbg.execution_path();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_same_state_different_snapshots_equivalent() {
        let a = StateSnapshot::new(json!({"key": "value"}), "node_x", 10);
        let b = StateSnapshot::new(json!({"key": "value"}), "node_y", 20);
        assert!(SnapshotComparator::is_equivalent(&a, &b));
        let diff = SnapshotComparator::diff(&a, &b);
        assert!(diff.is_empty());
    }
}
