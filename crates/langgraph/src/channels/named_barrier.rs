//! NamedBarrierValue channel — waits until all named participants have reported.
//!
//! This channel acts as a synchronization barrier. It becomes available only when
//! all registered names have sent a value. Once all names are seen, the channel
//! can be consumed and the barrier resets.

use std::collections::HashSet;

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A barrier channel that waits for all named participants.
///
/// The channel becomes available (i.e., [`get`](BaseChannel::get) succeeds and
/// [`is_available`](BaseChannel::is_available) returns `true`) only when all
/// registered names have been seen via [`update`](BaseChannel::update). After
/// consumption, the seen set resets for the next barrier cycle.
#[derive(Debug, Clone)]
pub struct NamedBarrierValue {
    /// The key name of this channel.
    key: String,
    /// The set of names that must be seen before the barrier is complete.
    names: HashSet<String>,
    /// The set of names that have been seen so far.
    seen: HashSet<String>,
}

impl NamedBarrierValue {
    /// Create a new `NamedBarrierValue` channel with the given key and expected names.
    pub fn new(key: impl Into<String>, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            key: key.into(),
            names: names.into_iter().map(|n| n.into()).collect(),
            seen: HashSet::new(),
        }
    }

    /// Whether the barrier is complete (all names have been seen).
    pub fn is_complete(&self) -> bool {
        !self.names.is_empty() && self.seen == self.names
    }
}

impl BaseChannel for NamedBarrierValue {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        if self.seen.is_empty() {
            None
        } else {
            let seen_vec: Vec<Value> = self.seen.iter().map(|s| Value::from(s.as_str())).collect();
            Some(Value::Array(seen_vec))
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        let seen = match checkpoint {
            Some(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => HashSet::new(),
        };
        Box::new(NamedBarrierValue {
            key: self.key.clone(),
            names: self.names.clone(),
            seen,
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        if self.is_complete() {
            Ok(Value::Bool(true))
        } else {
            Err(LangGraphError::EmptyChannelError)
        }
    }

    fn is_available(&self) -> bool {
        self.is_complete()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        let mut changed = false;
        for v in values {
            if let Value::String(name) = v {
                if self.names.contains(&name) && self.seen.insert(name) {
                    changed = true;
                }
            } else if let Some(name) = v.as_str() {
                let name = name.to_string();
                if self.names.contains(&name) && self.seen.insert(name) {
                    changed = true;
                }
            }
        }
        Ok(changed)
    }

    fn consume(&mut self) -> bool {
        if self.is_complete() {
            self.seen.clear();
            true
        } else {
            false
        }
    }
}

/// A barrier channel that waits for both all named participants AND a finish signal.
///
/// Like [`NamedBarrierValue`], but also requires `finish()` to be called before
/// the channel becomes available. After consumption, both the seen set and
/// the finished flag are reset.
#[derive(Debug, Clone)]
pub struct NamedBarrierValueAfterFinish {
    key: String,
    names: HashSet<String>,
    seen: HashSet<String>,
    finished: bool,
}

impl NamedBarrierValueAfterFinish {
    pub fn new(key: impl Into<String>, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            key: key.into(),
            names: names.into_iter().map(|n| n.into()).collect(),
            seen: HashSet::new(),
            finished: false,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.finished && !self.names.is_empty() && self.seen == self.names
    }
}

impl BaseChannel for NamedBarrierValueAfterFinish {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        let seen_vec: Vec<Value> = self.seen.iter().map(|s| Value::from(s.as_str())).collect();
        Some(serde_json::json!([seen_vec, self.finished]))
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        match checkpoint {
            Some(Value::Array(arr)) if arr.len() == 2 => {
                let seen = match &arr[0] {
                    Value::Array(names) => names
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect(),
                    _ => HashSet::new(),
                };
                let finished = arr[1].as_bool().unwrap_or(false);
                Box::new(NamedBarrierValueAfterFinish {
                    key: self.key.clone(),
                    names: self.names.clone(),
                    seen,
                    finished,
                })
            }
            _ => Box::new(NamedBarrierValueAfterFinish::new(
                self.key.clone(),
                self.names.iter().cloned(),
            )),
        }
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        if self.is_complete() {
            Ok(Value::Bool(true))
        } else {
            Err(LangGraphError::EmptyChannelError)
        }
    }

    fn is_available(&self) -> bool {
        self.is_complete()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        let mut changed = false;
        for v in values {
            if let Some(name) = v.as_str() {
                let name = name.to_string();
                if self.names.contains(&name) && self.seen.insert(name) {
                    changed = true;
                }
            }
        }
        Ok(changed)
    }

    fn consume(&mut self) -> bool {
        if self.is_complete() {
            self.seen.clear();
            self.finished = false;
            true
        } else {
            false
        }
    }

    fn finish(&mut self) -> bool {
        if !self.finished {
            self.finished = true;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_barrier_not_available() {
        let ch = NamedBarrierValue::new("barrier", vec!["a", "b", "c"]);
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_partial_update_not_available() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        assert!(!ch.is_available());
        assert!(!ch.is_complete());
    }

    #[test]
    fn test_complete_barrier() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        ch.update(vec![Value::from("b")]).unwrap();
        assert!(ch.is_available());
        assert!(ch.is_complete());
        assert_eq!(ch.get().unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_all_at_once() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["x", "y"]);
        let changed = ch
            .update(vec![Value::from("x"), Value::from("y")])
            .unwrap();
        assert!(changed);
        assert!(ch.is_complete());
    }

    #[test]
    fn test_unknown_name_ignored() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a"]);
        let changed = ch.update(vec![Value::from("unknown")]).unwrap();
        assert!(!changed);
        assert!(!ch.is_complete());
    }

    #[test]
    fn test_duplicate_name_no_change() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        let changed = ch.update(vec![Value::from("a")]).unwrap();
        assert!(!changed); // Already seen
    }

    #[test]
    fn test_consume_resets_barrier() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();
        assert!(ch.is_complete());

        let changed = ch.consume();
        assert!(changed);
        assert!(!ch.is_complete());
        assert!(!ch.is_available());
    }

    #[test]
    fn test_consume_on_incomplete_noop() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        let changed = ch.consume();
        assert!(!changed);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = NamedBarrierValue::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();

        let ckpt = ch.checkpoint();
        assert!(ckpt.is_some());

        let restored = ch.from_checkpoint(ckpt);
        // Restored channel has "a" seen, needs "b".
        assert!(!restored.is_available());
    }

    #[test]
    fn test_checkpoint_empty() {
        let ch = NamedBarrierValue::new("barrier", vec!["a"]);
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_from_checkpoint_none() {
        let ch = NamedBarrierValue::new("barrier", vec!["a"]);
        let restored = ch.from_checkpoint(None);
        assert!(!restored.is_available());
    }

    #[test]
    fn test_empty_names_never_complete() {
        let ch: NamedBarrierValue = NamedBarrierValue::new("barrier", Vec::<String>::new());
        assert!(!ch.is_complete());
    }

    #[test]
    fn test_after_finish_not_available_without_finish() {
        let mut ch = NamedBarrierValueAfterFinish::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();
        assert!(!ch.is_available()); // All seen but not finished
    }

    #[test]
    fn test_after_finish_not_available_without_all_names() {
        let mut ch = NamedBarrierValueAfterFinish::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        ch.finish();
        assert!(!ch.is_available()); // Finished but not all seen
    }

    #[test]
    fn test_after_finish_available_when_both() {
        let mut ch = NamedBarrierValueAfterFinish::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();
        ch.finish();
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_after_finish_consume_resets_both() {
        let mut ch = NamedBarrierValueAfterFinish::new("barrier", vec!["a"]);
        ch.update(vec![Value::from("a")]).unwrap();
        ch.finish();
        assert!(ch.consume());
        assert!(!ch.is_available());
        assert!(!ch.finished);
        assert!(ch.seen.is_empty());
    }

    #[test]
    fn test_after_finish_checkpoint_roundtrip() {
        let mut ch = NamedBarrierValueAfterFinish::new("barrier", vec!["a", "b"]);
        ch.update(vec![Value::from("a")]).unwrap();
        ch.finish();

        let ckpt = ch.checkpoint();
        let restored = ch.from_checkpoint(ckpt);
        // a is seen and finished, but b is not seen
        assert!(!restored.is_available());
    }
}
