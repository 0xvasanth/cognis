//! Deep merge utilities for JSON values.
//!
//! Mirrors Python `langchain_core.utils._merge`.
//!
//! These functions are critical for merging configuration dictionaries,
//! tool call chunks, and streaming message deltas throughout the framework.

use serde_json::{Map, Value};

/// Error type for merge operations.
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// The same key exists in both dicts but with incompatible types.
    #[error("additional_kwargs[\"{key}\"] already exists in this message, but with a different type")]
    TypeMismatch { key: String },

    /// A value has an unsupported type for merging.
    #[error("Additional kwargs key {key} already exists in left dict and value has unsupported type")]
    UnsupportedType { key: String },

    /// Two values cannot be merged.
    #[error("Unable to merge left and right. Both must be of type string, object, or array, or else be two equal values")]
    IncompatibleValues,
}

/// Keys whose integer values use last-wins instead of summing.
const INT_LAST_WINS_KEYS: &[&str] = &["index", "created", "timestamp"];

/// Keys whose string values are kept as-is when equal (not concatenated).
const STR_IDENTITY_KEYS: &[&str] = &["id", "output_version", "model_provider"];

/// Deeply merge two or more JSON objects.
///
/// Handles the specific scenarios required by the LangChain message delta
/// merging protocol:
///
/// - If a key exists in `left` with a `Null` value, the value from `right` is used.
/// - If the value in `right` is `Null`, the key is skipped.
/// - Strings are concatenated (with special handling for `"index"` keys starting
///   with `"lc_"` and identity keys like `"id"`).
/// - Objects are merged recursively.
/// - Arrays are merged via [`merge_lists`].
/// - Equal scalars are kept as-is.
/// - Integers are summed (except for `"index"`, `"created"`, `"timestamp"` which
///   use last-wins).
///
/// # Errors
///
/// Returns [`MergeError::TypeMismatch`] if the same key has different JSON types
/// in `left` and `right`, or [`MergeError::UnsupportedType`] if a value cannot
/// be merged.
pub fn merge_dicts(left: &Value, others: &[&Value]) -> Result<Value, MergeError> {
    let initial = match left {
        Value::Object(m) => m.clone(),
        _ => return Ok(left.clone()),
    };

    let mut merged = initial;

    for right in others {
        let right_obj = match right {
            Value::Object(m) => m,
            _ => continue,
        };

        for (right_k, right_v) in right_obj {
            if !merged.contains_key(right_k)
                || (!right_v.is_null() && merged.get(right_k).is_some_and(Value::is_null))
            {
                merged.insert(right_k.clone(), right_v.clone());
            } else if right_v.is_null() {
                continue;
            } else {
                let left_v = merged.get(right_k).unwrap();

                // Type check: both values must have the same JSON "kind"
                if std::mem::discriminant(left_v) != std::mem::discriminant(right_v) {
                    return Err(MergeError::TypeMismatch {
                        key: right_k.clone(),
                    });
                }

                match (left_v, right_v) {
                    (Value::String(ls), Value::String(rs)) => {
                        // Special handling for "index" keys with "lc_" prefix
                        if right_k == "index" && ls.starts_with("lc_") {
                            continue;
                        }
                        // Identity keys: skip if values are equal
                        if STR_IDENTITY_KEYS.contains(&right_k.as_str()) && ls == rs {
                            continue;
                        }
                        let concatenated = format!("{}{}", ls, rs);
                        merged.insert(right_k.clone(), Value::String(concatenated));
                    }
                    (Value::Object(_), Value::Object(_)) => {
                        let sub = merge_dicts(left_v, &[right_v])?;
                        merged.insert(right_k.clone(), sub);
                    }
                    (Value::Array(_), Value::Array(_)) => {
                        let sub = merge_lists(Some(left_v), &[Some(right_v)])?;
                        if let Some(result) = sub {
                            merged.insert(right_k.clone(), result);
                        }
                    }
                    _ if left_v == right_v => {
                        // Equal values: keep as-is
                        continue;
                    }
                    (Value::Number(ln), Value::Number(rn)) => {
                        if INT_LAST_WINS_KEYS.contains(&right_k.as_str()) {
                            // Last-wins for identification/temporal fields
                            merged.insert(right_k.clone(), right_v.clone());
                        } else {
                            // Sum integers/floats
                            let sum = if let (Some(li), Some(ri)) = (ln.as_i64(), rn.as_i64()) {
                                Value::from(li + ri)
                            } else if let (Some(lf), Some(rf)) = (ln.as_f64(), rn.as_f64()) {
                                Value::from(lf + rf)
                            } else {
                                right_v.clone()
                            };
                            merged.insert(right_k.clone(), sum);
                        }
                    }
                    _ => {
                        return Err(MergeError::UnsupportedType {
                            key: right_k.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(Value::Object(merged))
}

/// Merge two or more JSON arrays, handling `Null` values.
///
/// Elements that are objects with an `"index"` field are merged by matching
/// index values (deduplication). Other elements are simply appended.
///
/// The `"index"` matching also checks that `"id"` fields are not inconsistent
/// (both present and different).
///
/// # Errors
///
/// Propagates any [`MergeError`] from recursive [`merge_dicts`] calls.
pub fn merge_lists(
    left: Option<&Value>,
    others: &[Option<&Value>],
) -> Result<Option<Value>, MergeError> {
    let mut merged: Option<Vec<Value>> = match left {
        Some(Value::Array(arr)) => Some(arr.clone()),
        Some(Value::Null) | None => None,
        Some(other) => Some(vec![other.clone()]),
    };

    for other in others {
        let other_arr = match other {
            Some(Value::Array(arr)) => arr,
            None | Some(Value::Null) => continue,
            Some(other_val) => {
                // Treat a non-array as a single-element list
                if let Some(ref mut v) = merged {
                    v.push((*other_val).clone());
                } else {
                    merged = Some(vec![(*other_val).clone()]);
                }
                continue;
            }
        };

        if merged.is_none() {
            merged = Some(other_arr.clone());
            continue;
        }

        let m = merged.as_mut().unwrap();

        for e in other_arr {
            if let Value::Object(e_obj) = e {
                if has_valid_index(e_obj) {
                    let to_merge = find_merge_target(m, e_obj);

                    if let Some(idx) = to_merge {
                        let new_e = prepare_merge_element(m, idx, e_obj);
                        let existing = &m[idx];
                        let result = merge_dicts(existing, &[&new_e])?;
                        m[idx] = result;
                    } else {
                        m.push(e.clone());
                    }
                    continue;
                }
            }
            m.push(e.clone());
        }
    }

    Ok(merged.map(Value::Array))
}

/// Merge two JSON values of any type.
///
/// - `Null` values are skipped (the non-null side wins).
/// - Strings are concatenated.
/// - Objects are deep-merged via [`merge_dicts`].
/// - Arrays are merged via [`merge_lists`].
/// - Equal values are kept as-is.
///
/// # Errors
///
/// Returns [`MergeError`] if the values have incompatible types or cannot be
/// merged.
pub fn merge_obj(left: &Value, right: &Value) -> Result<Value, MergeError> {
    if left.is_null() {
        return Ok(right.clone());
    }
    if right.is_null() {
        return Ok(left.clone());
    }
    if std::mem::discriminant(left) != std::mem::discriminant(right) {
        return Err(MergeError::IncompatibleValues);
    }
    match (left, right) {
        (Value::String(ls), Value::String(rs)) => Ok(Value::String(format!("{}{}", ls, rs))),
        (Value::Object(_), Value::Object(_)) => merge_dicts(left, &[right]),
        (Value::Array(_), Value::Array(_)) => {
            merge_lists(Some(left), &[Some(right)]).map(|v| v.unwrap_or(Value::Null))
        }
        _ if left == right => Ok(left.clone()),
        _ => Err(MergeError::IncompatibleValues),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check if an object has a valid `"index"` field (integer or `"lc_"` prefixed string).
fn has_valid_index(obj: &Map<String, Value>) -> bool {
    match obj.get("index") {
        Some(Value::Number(n)) => n.as_i64().is_some() || n.as_u64().is_some(),
        Some(Value::String(s)) => s.starts_with("lc_"),
        _ => false,
    }
}

/// Find the index in `merged` of an element whose `"index"` matches `e_obj`'s
/// and whose `"id"` is not inconsistent.
fn find_merge_target(merged: &[Value], e_obj: &Map<String, Value>) -> Option<usize> {
    let e_index = e_obj.get("index")?;

    for (i, existing) in merged.iter().enumerate() {
        if let Value::Object(existing_obj) = existing {
            if let Some(existing_index) = existing_obj.get("index") {
                if existing_index == e_index && ids_compatible(existing_obj, e_obj) {
                    return Some(i);
                }
            }
        }
    }
    None
}

/// Check that the `"id"` fields of two objects are not inconsistent.
/// IDs are compatible if either is absent, null, empty string, or they are equal.
fn ids_compatible(left: &Map<String, Value>, right: &Map<String, Value>) -> bool {
    let left_id = left.get("id");
    let right_id = right.get("id");

    let left_empty = match left_id {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) if s.is_empty() => true,
        _ => false,
    };
    let right_empty = match right_id {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) if s.is_empty() => true,
        _ => false,
    };

    if left_empty || right_empty {
        return true;
    }
    left_id == right_id
}

/// Prepare the element to merge into an existing target, handling the special
/// `"type": "non_standard"` protocol for tool call chunks.
fn prepare_merge_element(
    merged: &[Value],
    target_idx: usize,
    e_obj: &Map<String, Value>,
) -> Value {
    let left_type = merged[target_idx]
        .as_object()
        .and_then(|o| o.get("type"))
        .and_then(Value::as_str);

    let e_type = e_obj.get("type").and_then(Value::as_str);

    // Special handling for non_standard type merging
    if let (Some(_left_t), Some("non_standard")) = (left_type, e_type) {
        if let Some(Value::Object(value_obj)) = e_obj.get("value") {
            let left_is_non_standard = left_type == Some("non_standard");

            let mut new_e = Map::new();
            let filtered: Map<String, Value> = value_obj
                .iter()
                .filter(|(k, _)| k.as_str() != "type")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            if left_is_non_standard {
                // non_standard + non_standard
                new_e.insert("value".to_string(), Value::Object(filtered));
                if let Some(idx) = e_obj.get("index") {
                    new_e.insert("index".to_string(), idx.clone());
                }
            } else {
                // standard + non_standard
                new_e.insert("extras".to_string(), Value::Object(filtered));
            }

            return Value::Object(new_e);
        }
    }

    // Default: strip the "type" key if present
    if e_obj.contains_key("type") {
        let filtered: Map<String, Value> = e_obj
            .iter()
            .filter(|(k, _)| k.as_str() != "type")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Value::Object(filtered)
    } else {
        Value::Object(e_obj.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // merge_dicts tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_dicts_basic() {
        let left = json!({"a": 1, "b": 2});
        let right = json!({"b": 3, "c": 4});
        let result = merge_dicts(&left, &[&right]).unwrap();
        // b should be summed: 2 + 3 = 5
        assert_eq!(result, json!({"a": 1, "b": 5, "c": 4}));
    }

    #[test]
    fn test_merge_dicts_new_key() {
        let left = json!({"a": 1});
        let right = json!({"b": 2});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_merge_dicts_null_left_value() {
        let left = json!({"a": null});
        let right = json!({"a": "hello"});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": "hello"}));
    }

    #[test]
    fn test_merge_dicts_null_right_value_skipped() {
        let left = json!({"a": "hello"});
        let right = json!({"a": null});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": "hello"}));
    }

    #[test]
    fn test_merge_dicts_string_concatenation() {
        let left = json!({"content": "hello "});
        let right = json!({"content": "world"});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"content": "hello world"}));
    }

    #[test]
    fn test_merge_dicts_nested_objects() {
        let left = json!({"function_call": {"arguments": "{\n"}});
        let right = json!({"function_call": {"arguments": "  \"name\": \"test\""}});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(
            result,
            json!({"function_call": {"arguments": "{\n  \"name\": \"test\""}})
        );
    }

    #[test]
    fn test_merge_dicts_null_nested_overwritten() {
        let left = json!({"function_call": {"arguments": null}});
        let right = json!({"function_call": {"arguments": "{\n"}});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"function_call": {"arguments": "{\n"}}));
    }

    #[test]
    fn test_merge_dicts_type_mismatch_error() {
        let left = json!({"a": "string"});
        let right = json!({"a": 42});
        let result = merge_dicts(&left, &[&right]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MergeError::TypeMismatch { .. }));
    }

    #[test]
    fn test_merge_dicts_list_merge() {
        let left = json!({"items": [1, 2]});
        let right = json!({"items": [3, 4]});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"items": [1, 2, 3, 4]}));
    }

    #[test]
    fn test_merge_dicts_equal_scalars_no_change() {
        let left = json!({"a": true});
        let right = json!({"a": true});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": true}));
    }

    #[test]
    fn test_merge_dicts_integer_sum() {
        let left = json!({"count": 5});
        let right = json!({"count": 3});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"count": 8}));
    }

    #[test]
    fn test_merge_dicts_index_last_wins() {
        let left = json!({"index": 0});
        let right = json!({"index": 1});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"index": 1}));
    }

    #[test]
    fn test_merge_dicts_created_last_wins() {
        let left = json!({"created": 1000});
        let right = json!({"created": 2000});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"created": 2000}));
    }

    #[test]
    fn test_merge_dicts_timestamp_last_wins() {
        let left = json!({"timestamp": 100});
        let right = json!({"timestamp": 200});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"timestamp": 200}));
    }

    #[test]
    fn test_merge_dicts_str_index_lc_prefix_skipped() {
        let left = json!({"index": "lc_abc"});
        let right = json!({"index": "lc_def"});
        let result = merge_dicts(&left, &[&right]).unwrap();
        // lc_ prefixed index strings are not concatenated
        assert_eq!(result, json!({"index": "lc_abc"}));
    }

    #[test]
    fn test_merge_dicts_id_identity() {
        let left = json!({"id": "run-123"});
        let right = json!({"id": "run-123"});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"id": "run-123"}));
    }

    #[test]
    fn test_merge_dicts_multiple_others() {
        let left = json!({"a": 1});
        let r1 = json!({"a": 2, "b": 10});
        let r2 = json!({"a": 3, "c": 20});
        let result = merge_dicts(&left, &[&r1, &r2]).unwrap();
        // a: 1+2=3, then 3==3 → equal → skip, so a stays 3
        assert_eq!(result, json!({"a": 3, "b": 10, "c": 20}));
    }

    #[test]
    fn test_merge_dicts_empty_right() {
        let left = json!({"a": 1});
        let right = json!({});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    #[test]
    fn test_merge_dicts_empty_left() {
        let left = json!({});
        let right = json!({"a": 1});
        let result = merge_dicts(&left, &[&right]).unwrap();
        assert_eq!(result, json!({"a": 1}));
    }

    // -----------------------------------------------------------------------
    // merge_lists tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_lists_basic_append() {
        let left = json!([1, 2, 3]);
        let right = json!([4, 5]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap();
        assert_eq!(result, Some(json!([1, 2, 3, 4, 5])));
    }

    #[test]
    fn test_merge_lists_none_left() {
        let right = json!([1, 2]);
        let result = merge_lists(None, &[Some(&right)]).unwrap();
        assert_eq!(result, Some(json!([1, 2])));
    }

    #[test]
    fn test_merge_lists_none_right() {
        let left = json!([1, 2]);
        let result = merge_lists(Some(&left), &[None]).unwrap();
        assert_eq!(result, Some(json!([1, 2])));
    }

    #[test]
    fn test_merge_lists_both_none() {
        let result = merge_lists(None, &[None]).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_merge_lists_index_dedup() {
        let left = json!([
            {"index": 0, "content": "hello "}
        ]);
        let right = json!([
            {"index": 0, "content": "world"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"], "hello world");
    }

    #[test]
    fn test_merge_lists_index_different_appended() {
        let left = json!([
            {"index": 0, "content": "hello"}
        ]);
        let right = json!([
            {"index": 1, "content": "world"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_merge_lists_index_with_matching_ids() {
        let left = json!([
            {"index": 0, "id": "call_1", "content": "a"}
        ]);
        let right = json!([
            {"index": 0, "id": "call_1", "content": "b"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"], "ab");
    }

    #[test]
    fn test_merge_lists_index_with_conflicting_ids() {
        let left = json!([
            {"index": 0, "id": "call_1", "content": "a"}
        ]);
        let right = json!([
            {"index": 0, "id": "call_2", "content": "b"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        // Different IDs: not merged, appended
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_merge_lists_index_with_null_id() {
        let left = json!([
            {"index": 0, "id": null, "content": "a"}
        ]);
        let right = json!([
            {"index": 0, "id": "call_1", "content": "b"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        // Null ID is compatible
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_merge_lists_index_with_empty_id() {
        let left = json!([
            {"index": 0, "id": "", "content": "a"}
        ]);
        let right = json!([
            {"index": 0, "id": "call_1", "content": "b"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_merge_lists_lc_index() {
        let left = json!([
            {"index": "lc_abc", "content": "hello "}
        ]);
        let right = json!([
            {"index": "lc_abc", "content": "world"}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["content"], "hello world");
    }

    #[test]
    fn test_merge_lists_without_index_appended() {
        let left = json!([{"name": "a"}]);
        let right = json!([{"name": "b"}]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_merge_lists_non_standard_type_handling() {
        let left = json!([
            {"index": 0, "type": "function", "function": {"name": "test"}}
        ]);
        let right = json!([
            {"index": 0, "type": "non_standard", "value": {"type": "x", "extra_field": "data"}}
        ]);
        let result = merge_lists(Some(&left), &[Some(&right)]).unwrap().unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        // Standard + non_standard: extras key should be present
        assert!(arr[0].get("extras").is_some());
        assert_eq!(arr[0]["extras"]["extra_field"], "data");
        // "type" key should be filtered from extras
        assert!(arr[0]["extras"].get("type").is_none());
    }

    #[test]
    fn test_merge_lists_multiple_others() {
        let left = json!([1]);
        let r1 = json!([2]);
        let r2 = json!([3]);
        let result = merge_lists(Some(&left), &[Some(&r1), Some(&r2)])
            .unwrap()
            .unwrap();
        assert_eq!(result, json!([1, 2, 3]));
    }

    // -----------------------------------------------------------------------
    // merge_obj tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_obj_null_left() {
        let result = merge_obj(&Value::Null, &json!("hello")).unwrap();
        assert_eq!(result, json!("hello"));
    }

    #[test]
    fn test_merge_obj_null_right() {
        let result = merge_obj(&json!("hello"), &Value::Null).unwrap();
        assert_eq!(result, json!("hello"));
    }

    #[test]
    fn test_merge_obj_strings() {
        let result = merge_obj(&json!("hello "), &json!("world")).unwrap();
        assert_eq!(result, json!("hello world"));
    }

    #[test]
    fn test_merge_obj_dicts() {
        let result = merge_obj(&json!({"a": 1}), &json!({"b": 2})).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_merge_obj_lists() {
        let result = merge_obj(&json!([1, 2]), &json!([3, 4])).unwrap();
        assert_eq!(result, json!([1, 2, 3, 4]));
    }

    #[test]
    fn test_merge_obj_equal_values() {
        let result = merge_obj(&json!(42), &json!(42)).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_merge_obj_type_mismatch() {
        let result = merge_obj(&json!("string"), &json!(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_obj_unequal_scalars() {
        let result = merge_obj(&json!(true), &json!(false));
        assert!(result.is_err());
    }
}
