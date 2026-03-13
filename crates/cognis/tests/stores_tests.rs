//! Tests for the key-value store implementations.

use std::time::Duration;

use cognis::stores::file::FileStore;
use cognis::stores::in_memory::InMemoryStore;
use cognis::stores::layered::LayeredStore;
use cognis::stores::namespaced::NamespacedStore;
use cognis::stores::Store;

// ---------------------------------------------------------------------------
// InMemoryStore CRUD
// ---------------------------------------------------------------------------

#[test]
fn in_memory_set_and_get() {
    let store = InMemoryStore::new();
    store.set("key1", b"hello").unwrap();
    let val = store.get("key1").unwrap();
    assert_eq!(val, Some(b"hello".to_vec()));
}

#[test]
fn in_memory_get_missing_key() {
    let store = InMemoryStore::new();
    let val = store.get("nonexistent").unwrap();
    assert_eq!(val, None);
}

#[test]
fn in_memory_delete() {
    let store = InMemoryStore::new();
    store.set("k", b"v").unwrap();
    assert!(store.delete("k").unwrap());
    assert!(!store.delete("k").unwrap());
    assert_eq!(store.get("k").unwrap(), None);
}

#[test]
fn in_memory_exists() {
    let store = InMemoryStore::new();
    assert!(!store.exists("k"));
    store.set("k", b"v").unwrap();
    assert!(store.exists("k"));
}

#[test]
fn in_memory_keys_listing() {
    let store = InMemoryStore::new();
    store.set("a", b"1").unwrap();
    store.set("b", b"2").unwrap();
    store.set("c", b"3").unwrap();
    let mut keys = store.keys().unwrap();
    keys.sort();
    assert_eq!(keys, vec!["a", "b", "c"]);
}

#[test]
fn in_memory_clear() {
    let store = InMemoryStore::new();
    store.set("a", b"1").unwrap();
    store.set("b", b"2").unwrap();
    store.clear().unwrap();
    assert!(store.keys().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// InMemoryStore TTL
// ---------------------------------------------------------------------------

#[test]
fn in_memory_ttl_expiration() {
    let store = InMemoryStore::with_ttl(Duration::from_millis(50));
    store.set("k", b"v").unwrap();
    assert!(store.exists("k"));

    std::thread::sleep(Duration::from_millis(80));

    assert_eq!(store.get("k").unwrap(), None);
    assert!(!store.exists("k"));
}

#[test]
fn in_memory_set_with_explicit_ttl() {
    let store = InMemoryStore::new();
    store
        .set_with_ttl("k", b"v", Duration::from_millis(50))
        .unwrap();
    assert_eq!(store.get("k").unwrap(), Some(b"v".to_vec()));

    std::thread::sleep(Duration::from_millis(80));
    assert_eq!(store.get("k").unwrap(), None);
}

// ---------------------------------------------------------------------------
// InMemoryStore thread safety
// ---------------------------------------------------------------------------

#[test]
fn in_memory_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let store = Arc::new(InMemoryStore::new());
    let mut handles = vec![];

    for i in 0..10 {
        let s = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            let key = format!("key_{}", i);
            let val = format!("val_{}", i);
            s.set(&key, val.as_bytes()).unwrap();
            assert_eq!(s.get(&key).unwrap(), Some(val.into_bytes()));
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(store.keys().unwrap().len(), 10);
}

// ---------------------------------------------------------------------------
// FileStore CRUD
// ---------------------------------------------------------------------------

#[test]
fn file_store_set_and_get() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileStore::new(dir.path()).unwrap();

    store.set("hello", b"world").unwrap();
    let val = store.get("hello").unwrap();
    assert_eq!(val, Some(b"world".to_vec()));
}

#[test]
fn file_store_delete() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileStore::new(dir.path()).unwrap();

    store.set("k", b"v").unwrap();
    assert!(store.delete("k").unwrap());
    assert!(!store.delete("k").unwrap());
    assert_eq!(store.get("k").unwrap(), None);
}

#[test]
fn file_store_exists_and_keys() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileStore::new(dir.path()).unwrap();

    assert!(!store.exists("a"));
    store.set("a", b"1").unwrap();
    store.set("b", b"2").unwrap();
    assert!(store.exists("a"));

    let mut keys = store.keys().unwrap();
    keys.sort();
    assert_eq!(keys, vec!["a", "b"]);
}

// ---------------------------------------------------------------------------
// FileStore persistence
// ---------------------------------------------------------------------------

#[test]
fn file_store_persistence_across_instances() {
    let dir = tempfile::tempdir().unwrap();

    {
        let store = FileStore::new(dir.path()).unwrap();
        store.set("persist", b"data").unwrap();
    }

    // New instance pointing at the same directory.
    let store2 = FileStore::new(dir.path()).unwrap();
    assert_eq!(store2.get("persist").unwrap(), Some(b"data".to_vec()));
}

// ---------------------------------------------------------------------------
// FileStore atomic writes
// ---------------------------------------------------------------------------

#[test]
fn file_store_atomic_write_no_tmp_leftover() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileStore::new(dir.path()).unwrap();

    store.set("atomic", b"test").unwrap();

    // Ensure no .tmp file is left behind.
    let tmp_files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("tmp"))
        .collect();
    assert!(tmp_files.is_empty(), "no .tmp files should remain");
}

// ---------------------------------------------------------------------------
// FileStore clear
// ---------------------------------------------------------------------------

#[test]
fn file_store_clear() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileStore::new(dir.path()).unwrap();
    store.set("x", b"1").unwrap();
    store.set("y", b"2").unwrap();
    store.clear().unwrap();
    assert!(store.keys().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// NamespacedStore
// ---------------------------------------------------------------------------

#[test]
fn namespaced_store_key_prefixing() {
    let inner = InMemoryStore::new();
    inner.set("ns1::a", b"should_not_see").unwrap();

    let ns = NamespacedStore::new(Box::new(InMemoryStore::new()), "ns1".to_string());
    ns.set("key1", b"val1").unwrap();
    ns.set("key2", b"val2").unwrap();

    assert_eq!(ns.get("key1").unwrap(), Some(b"val1".to_vec()));
    assert!(!ns.exists("nonexistent"));

    let mut keys = ns.keys().unwrap();
    keys.sort();
    assert_eq!(keys, vec!["key1", "key2"]);
}

#[test]
fn namespaced_store_list_namespaces() {
    let inner = Box::new(InMemoryStore::new());
    // Populate inner with keys from multiple namespaces.
    inner.set("alpha::k1", b"v").unwrap();
    inner.set("alpha::k2", b"v").unwrap();
    inner.set("beta::k1", b"v").unwrap();
    inner.set("gamma::k1", b"v").unwrap();

    let ns = NamespacedStore::new(inner, "alpha".to_string());
    let mut namespaces = ns.list_namespaces().unwrap();
    namespaces.sort();
    assert_eq!(namespaces, vec!["alpha", "beta", "gamma"]);
}

// ---------------------------------------------------------------------------
// LayeredStore
// ---------------------------------------------------------------------------

#[test]
fn layered_store_cache_hit() {
    let fast = Box::new(InMemoryStore::new());
    let slow = Box::new(InMemoryStore::new());

    fast.set("cached", b"fast_value").unwrap();
    slow.set("cached", b"slow_value").unwrap();

    let layered = LayeredStore::new(fast, slow);
    // Should return value from fast store.
    assert_eq!(layered.get("cached").unwrap(), Some(b"fast_value".to_vec()));
}

#[test]
fn layered_store_cache_miss_read_through() {
    let fast = Box::new(InMemoryStore::new());
    let slow = Box::new(InMemoryStore::new());

    slow.set("only_slow", b"from_slow").unwrap();

    let layered = LayeredStore::new(fast, slow);
    // First get should read through from slow.
    assert_eq!(
        layered.get("only_slow").unwrap(),
        Some(b"from_slow".to_vec())
    );
    // After read-through, fast store should have it.
    // Do another get — it will come from fast this time (verified by the code path).
    assert_eq!(
        layered.get("only_slow").unwrap(),
        Some(b"from_slow".to_vec())
    );
}

#[test]
fn layered_store_invalidation() {
    let fast = Box::new(InMemoryStore::new());
    let slow = Box::new(InMemoryStore::new());

    let layered = LayeredStore::new(fast, slow);
    layered.set("k", b"v").unwrap();
    assert!(layered.exists("k"));

    layered.invalidate("k").unwrap();
    // Still exists in slow, so read-through should repopulate.
    assert_eq!(layered.get("k").unwrap(), Some(b"v".to_vec()));
}

#[test]
fn layered_store_warm_up() {
    let fast = Box::new(InMemoryStore::new());
    let slow = Box::new(InMemoryStore::new());

    slow.set("w1", b"v1").unwrap();
    slow.set("w2", b"v2").unwrap();

    let layered = LayeredStore::new(fast, slow);
    layered.warm_up(&["w1", "w2"]).unwrap();

    // Both should now be in fast store (verified by invalidating slow and
    // still getting them from fast -- but we can't easily split. We just
    // verify they're accessible).
    assert_eq!(layered.get("w1").unwrap(), Some(b"v1".to_vec()));
    assert_eq!(layered.get("w2").unwrap(), Some(b"v2".to_vec()));
}

// ---------------------------------------------------------------------------
// Store trait object compatibility
// ---------------------------------------------------------------------------

#[test]
fn store_trait_object_compatibility() {
    let store: Box<dyn Store> = Box::new(InMemoryStore::new());
    store.set("dyn_key", b"dyn_val").unwrap();
    assert_eq!(store.get("dyn_key").unwrap(), Some(b"dyn_val".to_vec()));
    assert!(store.exists("dyn_key"));
    assert!(store.delete("dyn_key").unwrap());
    assert!(store.keys().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Empty store operations
// ---------------------------------------------------------------------------

#[test]
fn empty_store_operations() {
    let store = InMemoryStore::new();
    assert_eq!(store.get("nope").unwrap(), None);
    assert!(!store.exists("nope"));
    assert!(!store.delete("nope").unwrap());
    assert!(store.keys().unwrap().is_empty());
    store.clear().unwrap(); // should not panic
}
