//! UntrackedValue channel — like LastValue but never checkpointed.
//!
//! This channel stores the most recent value but excludes it from checkpoints.
//! Useful for transient state that should not be persisted or restored.

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A channel that stores the last value but is never included in checkpoints.
///
/// This is identical to [`LastValue`](super::last_value::LastValue) in behavior,
/// except that [`checkpoint`](BaseChannel::checkpoint) always returns `None` and
/// [`from_checkpoint`](BaseChannel::from_checkpoint) always creates an empty channel.
#[derive(Debug, Clone)]
pub struct UntrackedValue {
    /// The key name of this channel.
    key: String,
    /// The current stored value.
    value: Option<Value>,
    /// Whether to guard against multiple values per step.
    guard: bool,
}

impl UntrackedValue {
    /// Create a new empty `UntrackedValue` channel with the given key.
    ///
    /// By default, guard mode is enabled (rejects multiple values per step).
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            guard: true,
        }
    }

    /// Create a new `UntrackedValue` channel with an initial value.
    pub fn with_value(key: impl Into<String>, value: Value) -> Self {
        Self {
            key: key.into(),
            value: Some(value),
            guard: true,
        }
    }

    /// Create a new unguarded `UntrackedValue` channel.
    ///
    /// An unguarded channel accepts multiple values per step and takes the last one.
    pub fn unguarded(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            guard: false,
        }
    }

    /// Set the guard mode.
    pub fn with_guard(mut self, guard: bool) -> Self {
        self.guard = guard;
        self
    }
}

impl BaseChannel for UntrackedValue {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        // Never checkpointed.
        None
    }

    fn from_checkpoint(&self, _checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        // Always create an empty channel — untracked values are not restored.
        Box::new(UntrackedValue {
            key: self.key.clone(),
            value: None,
            guard: self.guard,
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        self.value
            .clone()
            .ok_or(LangGraphError::EmptyChannelError)
    }

    fn is_available(&self) -> bool {
        self.value.is_some()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }
        if self.guard && values.len() > 1 {
            return Err(LangGraphError::InvalidUpdateError(format!(
                "UntrackedValue channel '{}' received {} values, expected at most 1",
                self.key,
                values.len()
            )));
        }
        self.value = values.into_iter().last();
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_channel_is_empty() {
        let ch = UntrackedValue::new("test");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_with_value() {
        let ch = UntrackedValue::with_value("test", Value::from(42));
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::from(42));
    }

    #[test]
    fn test_update_single_value() {
        let mut ch = UntrackedValue::new("test");
        let changed = ch.update(vec![Value::from("data")]).unwrap();
        assert!(changed);
        assert_eq!(ch.get().unwrap(), Value::from("data"));
    }

    #[test]
    fn test_update_multiple_values_errors() {
        let mut ch = UntrackedValue::new("test");
        let result = ch.update(vec![Value::from(1), Value::from(2)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_empty() {
        let mut ch = UntrackedValue::new("test");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_never_checkpointed() {
        let mut ch = UntrackedValue::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_from_checkpoint_always_empty() {
        let ch = UntrackedValue::new("test");
        let restored = ch.from_checkpoint(Some(Value::from("data")));
        assert!(!restored.is_available());
    }

    #[test]
    fn test_update_replaces_value() {
        let mut ch = UntrackedValue::with_value("test", Value::from("old"));
        ch.update(vec![Value::from("new")]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from("new"));
    }

    #[test]
    fn test_unguarded_multiple_values_takes_last() {
        let mut ch = UntrackedValue::unguarded("test");
        ch.update(vec![Value::from(1), Value::from(2), Value::from(3)])
            .unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(3));
    }

    #[test]
    fn test_with_guard_builder() {
        let ch = UntrackedValue::new("test").with_guard(false);
        assert!(!ch.guard);
    }
}
