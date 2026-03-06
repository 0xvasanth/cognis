use rustchain_core::utils::{generate_id, merge_dicts};
use serde_json::json;

#[test]
fn merge_dicts_objects() {
    let left = json!({"a": 1, "b": 2});
    let right = json!({"b": 3, "c": 4});
    let merged = merge_dicts(&left, &[&right]).unwrap();
    // Integers are summed: 2 + 3 = 5
    assert_eq!(merged, json!({"a": 1, "b": 5, "c": 4}));
}

#[test]
fn merge_dicts_nested_objects() {
    let left = json!({"outer": {"a": 1, "b": 2}});
    let right = json!({"outer": {"b": 3, "c": 4}});
    let merged = merge_dicts(&left, &[&right]).unwrap();
    // Nested integers are also summed: 2 + 3 = 5
    assert_eq!(merged, json!({"outer": {"a": 1, "b": 5, "c": 4}}));
}

#[test]
fn merge_dicts_arrays() {
    // When left is not an object, merge_dicts returns left as-is.
    // Use merge_lists for array merging, or wrap in an object.
    let left = json!({"arr": [1, 2]});
    let right = json!({"arr": [3, 4]});
    let merged = merge_dicts(&left, &[&right]).unwrap();
    assert_eq!(merged, json!({"arr": [1, 2, 3, 4]}));
}

#[test]
fn merge_dicts_non_object_returns_left() {
    let left = json!(1);
    let right = json!(2);
    let merged = merge_dicts(&left, &[&right]).unwrap();
    // Non-object left is returned as-is
    assert_eq!(merged, json!(1));
}

#[test]
fn merge_dicts_non_object_right_skipped() {
    let left = json!({"a": 1});
    let right = json!("string");
    let merged = merge_dicts(&left, &[&right]).unwrap();
    // Non-object right is skipped
    assert_eq!(merged, json!({"a": 1}));
}

#[test]
fn generate_id_returns_uuid_format() {
    let id = generate_id();
    // UUID v4 format: 8-4-4-4-12 hex chars
    assert_eq!(id.len(), 36);
    assert_eq!(id.chars().filter(|c| *c == '-').count(), 4);
}

#[test]
fn generate_id_unique() {
    let id1 = generate_id();
    let id2 = generate_id();
    assert_ne!(id1, id2);
}
