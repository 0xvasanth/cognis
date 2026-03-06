//! AnyValue channel — stores the last value received, assuming all values are equal.
//!
//! Similar to [`LastValue`](super::last_value::LastValue), but accepts multiple
//! values per step (assuming they are all equivalent). When updated with an empty
//! list, the channel clears its value.

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A channel that stores the last value received from any writer.
///
/// Unlike [`LastValue`](super::last_value::LastValue), this channel accepts
/// multiple values per step — it assumes they are all semantically equal and
/// simply keeps the last one. When updated with an empty list, the value is
/// cleared.
#[derive(Debug, Clone)]
pub struct AnyValue {
    /// The key name of this channel.
    key: String,
    /// The current stored value.
    value: Option<Value>,
}

impl AnyValue {
    /// Create a new empty `AnyValue` channel with the given key.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
        }
    }

    /// Create a new `AnyValue` channel with an initial value.
    pub fn with_value(key: impl Into<String>, value: Value) -> Self {
        Self {
            key: key.into(),
            value: Some(value),
        }
    }
}

impl BaseChannel for AnyValue {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        self.value.clone()
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        Box::new(AnyValue {
            key: self.key.clone(),
            value: checkpoint,
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
            // Clear the value on empty update.
            if self.value.is_some() {
                self.value = None;
                return Ok(true);
            }
            return Ok(false);
        }
        // Take the last value (all are assumed equal).
        self.value = values.into_iter().last();
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_channel_is_empty() {
        let ch = AnyValue::new("test");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_with_value() {
        let ch = AnyValue::with_value("test", Value::from(42));
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::from(42));
    }

    #[test]
    fn test_update_single_value() {
        let mut ch = AnyValue::new("test");
        let changed = ch.update(vec![Value::from("hello")]).unwrap();
        assert!(changed);
        assert_eq!(ch.get().unwrap(), Value::from("hello"));
    }

    #[test]
    fn test_update_multiple_values_takes_last() {
        let mut ch = AnyValue::new("test");
        let changed = ch.update(vec![Value::from(1), Value::from(2), Value::from(3)]).unwrap();
        assert!(changed);
        assert_eq!(ch.get().unwrap(), Value::from(3));
    }

    #[test]
    fn test_empty_update_clears_value() {
        let mut ch = AnyValue::with_value("test", Value::from(42));
        assert!(ch.is_available());

        let changed = ch.update(vec![]).unwrap();
        assert!(changed);
        assert!(!ch.is_available());
    }

    #[test]
    fn test_empty_update_on_empty_channel() {
        let mut ch = AnyValue::new("test");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = AnyValue::new("test");
        ch.update(vec![Value::from("state")]).unwrap();

        let ckpt = ch.checkpoint();
        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(restored.get().unwrap(), Value::from("state"));
    }

    #[test]
    fn test_from_checkpoint_none() {
        let ch = AnyValue::new("test");
        let restored = ch.from_checkpoint(None);
        assert!(!restored.is_available());
    }
}
