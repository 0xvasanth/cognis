//! Deep-merge two `serde_json::Value`s.
//!
//! Lifted out of `cognis2-graph::state` (where it's used by the
//! `#[reducer(merge)]` codegen) so user code can deep-merge JSON without
//! pulling in the graph crate.

use serde_json::Value;

/// Deep-merge `source` into `target`.
///
/// Rules:
/// - Both objects: keys union; for shared keys, recurse.
/// - Both arrays: target is replaced by source. (Use append-style
///   reducers if you want concatenation.)
/// - Anything else: source replaces target.
pub fn deep_merge_json(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(t_map), Value::Object(s_map)) => {
            for (k, v) in s_map {
                deep_merge_json(t_map.entry(k).or_insert(Value::Null), v);
            }
        }
        (slot, source) => {
            *slot = source;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_union_at_top_level() {
        let mut t = json!({"a": 1, "b": 2});
        deep_merge_json(&mut t, json!({"b": 20, "c": 3}));
        assert_eq!(t, json!({"a": 1, "b": 20, "c": 3}));
    }

    #[test]
    fn nested_objects_merge_recursively() {
        let mut t = json!({"x": {"a": 1, "b": 2}});
        deep_merge_json(&mut t, json!({"x": {"b": 20, "c": 3}}));
        assert_eq!(t, json!({"x": {"a": 1, "b": 20, "c": 3}}));
    }

    #[test]
    fn arrays_replace_target() {
        let mut t = json!([1, 2, 3]);
        deep_merge_json(&mut t, json!([9]));
        assert_eq!(t, json!([9]));
    }

    #[test]
    fn scalar_replaces_target() {
        let mut t = json!(1);
        deep_merge_json(&mut t, json!("hello"));
        assert_eq!(t, json!("hello"));
    }
}
