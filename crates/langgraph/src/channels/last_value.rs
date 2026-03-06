//! LastValue channel — stores the most recent value received.
//!
//! This channel accepts at most one value per step. If multiple values are
//! provided in a single update, it returns an error. This enforces that
//! exactly one node writes to this channel per step.

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// A channel that stores the last value received.
///
/// Only one value may be written per step. If multiple nodes attempt to write
/// to this channel in the same step, an [`LangGraphError::InvalidUpdateError`]
/// is returned.
#[derive(Debug, Clone)]
pub struct LastValue {
    /// The key name of this channel.
    key: String,
    /// The current stored value.
    value: Option<Value>,
}

impl LastValue {
    /// Create a new empty `LastValue` channel with the given key.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
        }
    }

    /// Create a new `LastValue` channel with an initial value.
    pub fn with_value(key: impl Into<String>, value: Value) -> Self {
        Self {
            key: key.into(),
            value: Some(value),
        }
    }
}

impl BaseChannel for LastValue {
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
        Box::new(LastValue {
            key: self.key.clone(),
            value: checkpoint,
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        self.value.clone().ok_or(LangGraphError::EmptyChannelError)
    }

    fn is_available(&self) -> bool {
        self.value.is_some()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }
        if values.len() > 1 {
            return Err(LangGraphError::InvalidUpdateError(format!(
                "LastValue channel '{}' received {} values, expected at most 1",
                self.key,
                values.len()
            )));
        }
        self.value = Some(values.into_iter().next().unwrap());
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_channel_is_empty() {
        let ch = LastValue::new("test");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_with_value() {
        let ch = LastValue::with_value("test", Value::from(42));
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::from(42));
    }

    #[test]
    fn test_update_single_value() {
        let mut ch = LastValue::new("test");
        let changed = ch.update(vec![Value::from("hello")]).unwrap();
        assert!(changed);
        assert_eq!(ch.get().unwrap(), Value::from("hello"));
    }

    #[test]
    fn test_update_empty_values() {
        let mut ch = LastValue::new("test");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
        assert!(!ch.is_available());
    }

    #[test]
    fn test_update_multiple_values_errors() {
        let mut ch = LastValue::new("test");
        let result = ch.update(vec![Value::from(1), Value::from(2)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::InvalidUpdateError(msg) => {
                assert!(msg.contains("2 values"));
            }
            other => panic!("Expected InvalidUpdateError, got: {:?}", other),
        }
    }

    #[test]
    fn test_update_replaces_value() {
        let mut ch = LastValue::with_value("test", Value::from(1));
        ch.update(vec![Value::from(2)]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::from(2));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = LastValue::new("test");
        ch.update(vec![Value::from("state")]).unwrap();

        let ckpt = ch.checkpoint();
        assert!(ckpt.is_some());

        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(restored.get().unwrap(), Value::from("state"));
    }

    #[test]
    fn test_from_checkpoint_none() {
        let ch = LastValue::new("test");
        let restored = ch.from_checkpoint(None);
        assert!(!restored.is_available());
    }

    #[test]
    fn test_key() {
        let ch = LastValue::new("my_channel");
        assert_eq!(ch.key(), "my_channel");
    }
}

/// A channel that stores a value but only becomes available after `finish()` is called.
///
/// Used for delayed-availability patterns where a value should only be readable
/// once a certain phase of execution is complete.
#[derive(Debug, Clone)]
pub struct LastValueAfterFinish {
    key: String,
    value: Option<Value>,
    finished: bool,
}

impl LastValueAfterFinish {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: None,
            finished: false,
        }
    }

    pub fn with_value(key: impl Into<String>, value: Value) -> Self {
        Self {
            key: key.into(),
            value: Some(value),
            finished: false,
        }
    }
}

impl BaseChannel for LastValueAfterFinish {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        // Checkpoint as [value, finished]
        let val = self.value.clone().unwrap_or(Value::Null);
        Some(serde_json::json!([val, self.finished]))
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        match checkpoint {
            Some(Value::Array(arr)) if arr.len() == 2 => {
                let value = if arr[0].is_null() {
                    None
                } else {
                    Some(arr[0].clone())
                };
                let finished = arr[1].as_bool().unwrap_or(false);
                Box::new(LastValueAfterFinish {
                    key: self.key.clone(),
                    value,
                    finished,
                })
            }
            _ => Box::new(LastValueAfterFinish::new(self.key.clone())),
        }
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        if !self.finished {
            return Err(LangGraphError::EmptyChannelError);
        }
        self.value.clone().ok_or(LangGraphError::EmptyChannelError)
    }

    fn is_available(&self) -> bool {
        self.finished && self.value.is_some()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }
        if values.len() > 1 {
            return Err(LangGraphError::InvalidUpdateError(format!(
                "LastValueAfterFinish channel '{}' received {} values, expected at most 1",
                self.key,
                values.len()
            )));
        }
        self.value = Some(values.into_iter().next().unwrap());
        Ok(true)
    }

    fn consume(&mut self) -> bool {
        if self.finished && self.value.is_some() {
            self.value = None;
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
mod tests_after_finish {
    use super::*;

    #[test]
    fn test_after_finish_not_available_before_finish() {
        let mut ch = LastValueAfterFinish::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_after_finish_available_after_finish() {
        let mut ch = LastValueAfterFinish::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        ch.finish();
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::from("data"));
    }

    #[test]
    fn test_after_finish_consume_resets() {
        let mut ch = LastValueAfterFinish::new("test");
        ch.update(vec![Value::from("data")]).unwrap();
        ch.finish();
        assert!(ch.consume());
        assert!(!ch.is_available());
        assert!(!ch.finished);
    }

    #[test]
    fn test_after_finish_checkpoint_roundtrip() {
        let mut ch = LastValueAfterFinish::new("test");
        ch.update(vec![Value::from(42)]).unwrap();
        ch.finish();

        let ckpt = ch.checkpoint();
        let restored = ch.from_checkpoint(ckpt);
        assert!(restored.is_available());
        assert_eq!(restored.get().unwrap(), Value::from(42));
    }
}
