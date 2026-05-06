//! Channel reducers — strategies for merging per-step updates into state.
//!
//! `#[derive(GraphState)]` desugars `#[reducer(append|add|last|merge|...)]`
//! into inline merge logic. This trait + impls are exposed for:
//! - Programmatic reducer composition (advanced graph builders).
//! - Documentation of the available strategies.

use std::ops::AddAssign;

/// A strategy for merging an `update: T` into a `current: &mut T`.
pub trait Reducer<T>: Send + Sync {
    /// Merge `update` into `current` in place.
    fn reduce(&self, current: &mut T, update: T);
}

/// Append: `Vec<T>::extend(update)`. Matches `#[reducer(append)]`.
pub struct Append;
impl<T: Send + Sync> Reducer<Vec<T>> for Append {
    fn reduce(&self, current: &mut Vec<T>, update: Vec<T>) {
        current.extend(update);
    }
}

/// Add: `current += update` (numeric `AddAssign`). Matches `#[reducer(add)]`.
pub struct Add;
impl<T> Reducer<T> for Add
where
    T: AddAssign + Send + Sync,
{
    fn reduce(&self, current: &mut T, update: T) {
        *current += update;
    }
}

/// LastValue: `*current = update` if update is not the type's "empty"
/// — for `Option<U>` only assigns when `Some`. The `#[reducer(last)]`
/// macro variant wraps the field type in `Option<T>` in the Update
/// struct, so the user-facing semantics are "assign only if Some".
pub struct LastValue;
impl<T: Send + Sync> Reducer<Option<T>> for LastValue {
    fn reduce(&self, current: &mut Option<T>, update: Option<T>) {
        if let Some(v) = update {
            *current = Some(v);
        }
    }
}

/// Merge: deep-merge for `serde_json::Value`. Matches `#[reducer(merge)]`.
pub struct Merge;
impl Reducer<serde_json::Value> for Merge {
    fn reduce(&self, current: &mut serde_json::Value, update: serde_json::Value) {
        crate::state::__merge_json(current, update);
    }
}

/// Custom user-supplied closure. Matches `#[reducer(custom = "path::fn")]`
/// in spirit (the macro inlines the call rather than going through this
/// type, but `Custom` is provided for programmatic composition).
pub struct Custom<F>(pub F);
impl<T, F> Reducer<T> for Custom<F>
where
    T: Send + Sync,
    F: Fn(&mut T, T) + Send + Sync,
{
    fn reduce(&self, current: &mut T, update: T) {
        (self.0)(current, update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_extends_vec() {
        let mut v = vec![1, 2];
        Append.reduce(&mut v, vec![3, 4]);
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn add_increments() {
        let mut n = 5u32;
        Add.reduce(&mut n, 3);
        assert_eq!(n, 8);
    }

    #[test]
    fn last_value_some_overwrites() {
        let mut o: Option<&str> = Some("old");
        LastValue.reduce(&mut o, Some("new"));
        assert_eq!(o, Some("new"));
    }

    #[test]
    fn last_value_none_keeps() {
        let mut o: Option<&str> = Some("keep");
        LastValue.reduce(&mut o, None);
        assert_eq!(o, Some("keep"));
    }

    #[test]
    fn merge_deep_merges_json() {
        let mut t = serde_json::json!({"a": 1});
        Merge.reduce(&mut t, serde_json::json!({"b": 2}));
        assert_eq!(t, serde_json::json!({"a": 1, "b": 2}));
    }

    #[test]
    fn custom_closure_applies() {
        let mut s = String::from("hello");
        Custom(|cur: &mut String, upd: String| cur.push_str(&upd))
            .reduce(&mut s, " world".into());
        assert_eq!(s, "hello world");
    }
}
