//! EphemeralValue channel — stores a value from the preceding step only.
//!
//! This channel stores a value that is automatically cleared after being consumed.
//! It is useful for passing data between consecutive steps without persisting it
//! across the entire graph execution.

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A channel that stores a value for one step only.
///
/// The value is cleared when [`consume`](BaseChannel::consume) is called (i.e.,
/// after a subscribed task runs). If `guard` is `true`, an error is raised when
/// multiple values are written in a single step.
#[derive(Debug, Clone)]
pub struct EphemeralValue {
    /// The key name of this channel.
    key: String,
    /// The current stored value.
    value: Option<Value>,
    /// Whether to guard against multiple values per step.
    guard: bool,
}

impl EphemeralValue {
    /// Create a new `EphemeralValue` channel.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            guard: false,
        }
    }

    /// Create a new guarded `EphemeralValue` channel.
    ///
    /// A guarded channel will return an error if multiple values are written
    /// in a single step.
    pub fn guarded(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            guard: true,
        }
    }

    /// Set the guard mode.
    pub fn with_guard(mut self, guard: bool) -> Self {
        self.guard = guard;
        self
    }
}

impl BaseChannel for EphemeralValue {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        // Ephemeral values are not checkpointed.
        None
    }

    fn from_checkpoint(&self, _checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        // Always create an empty channel — ephemeral values are not restored.
        Box::new(EphemeralValue {
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
            // When unguarded, clear value on empty update (Python behavior)
            if !self.guard && self.value.is_some() {
                self.value = None;
                return Ok(true);
            }
            return Ok(false);
        }
        if self.guard && values.len() > 1 {
            return Err(LangGraphError::InvalidUpdateError(format!(
                "EphemeralValue channel '{}' received {} values, expected at most 1 (guard mode)",
                self.key,
                values.len()
            )));
        }
        self.value = values.into_iter().last();
        Ok(true)
    }

    fn consume(&mut self) -> bool {
        if self.value.is_some() {
            self.value = None;
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
    fn test_new_channel_is_empty() {
        let ch = EphemeralValue::new("test");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_update_and_get() {
        let mut ch = EphemeralValue::new("test");
        let changed = ch.update(vec![Value::from("data")]).unwrap();
        assert!(changed);
        assert_eq!(ch.get().unwrap(), Value::from("data"));
    }

    #[test]
    fn test_consume_clears_value() {
        let mut ch = EphemeralValue::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(ch.is_available());

        let changed = ch.consume();
        assert!(changed);
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_consume_on_empty() {
        let mut ch = EphemeralValue::new("test");
        let changed = ch.consume();
        assert!(!changed);
    }

    #[test]
    fn test_not_checkpointed() {
        let mut ch = EphemeralValue::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_from_checkpoint_always_empty() {
        let ch = EphemeralValue::new("test");
        let restored = ch.from_checkpoint(Some(Value::from("data")));
        assert!(!restored.is_available());
    }

    #[test]
    fn test_guarded_single_value_ok() {
        let mut ch = EphemeralValue::guarded("test");
        let result = ch.update(vec![Value::from("ok")]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_guarded_multiple_values_error() {
        let mut ch = EphemeralValue::guarded("test");
        let result = ch.update(vec![Value::from(1), Value::from(2)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::InvalidUpdateError(msg) => {
                assert!(msg.contains("guard mode"));
            }
            other => panic!("Expected InvalidUpdateError, got: {:?}", other),
        }
    }

    #[test]
    fn test_non_guarded_multiple_values_takes_last() {
        let mut ch = EphemeralValue::new("test");
        ch.update(vec![Value::from(1), Value::from(2), Value::from(3)])
            .unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(3));
    }

    #[test]
    fn test_empty_update() {
        let mut ch = EphemeralValue::new("test");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_with_guard_builder() {
        let ch = EphemeralValue::new("test").with_guard(true);
        assert!(ch.guard);
    }

    #[test]
    fn test_unguarded_empty_update_clears_value() {
        let mut ch = EphemeralValue::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(ch.is_available());

        // Empty update on unguarded should clear
        let changed = ch.update(vec![]).unwrap();
        assert!(changed);
        assert!(!ch.is_available());
    }

    #[test]
    fn test_guarded_empty_update_does_not_clear() {
        let mut ch = EphemeralValue::guarded("test");
        ch.update(vec![Value::from("data")]).unwrap();

        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
        assert!(ch.is_available());
    }
}
