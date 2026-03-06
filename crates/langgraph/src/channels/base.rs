//! Base channel trait for LangGraph state management.
//!
//! Channels are the core abstraction for managing state between graph steps in the
//! Pregel execution model. Each channel stores a value and defines how updates from
//! multiple nodes are combined.

use serde_json::Value;

use crate::errors::LangGraphError;

/// Base trait for all channels. Channels store and manage state between graph steps.
///
/// In the Pregel execution model, channels act as the communication medium between
/// nodes. Each node reads from and writes to channels. The channel implementation
/// determines how concurrent writes are resolved.
pub trait BaseChannel: Send + Sync + std::fmt::Debug {
    /// The key name of this channel.
    fn key(&self) -> &str;

    /// Return a clone of this channel (boxed).
    fn box_clone(&self) -> Box<dyn BaseChannel>;

    /// Return a serializable checkpoint of the channel's current state.
    /// Returns `None` if the channel is empty.
    fn checkpoint(&self) -> Option<Value>;

    /// Create a new channel from a checkpoint value.
    /// If checkpoint is `None`, creates an empty channel.
    #[allow(clippy::wrong_self_convention)]
    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel>;

    /// Get the current value.
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::EmptyChannelError`] if the channel has not been
    /// updated yet.
    fn get(&self) -> Result<Value, LangGraphError>;

    /// Whether the channel has a value available.
    fn is_available(&self) -> bool;

    /// Update the channel with a sequence of values.
    /// Returns `true` if the channel's value changed.
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::InvalidUpdateError`] if the values are not
    /// compatible with this channel type.
    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError>;

    /// Notify the channel that a subscribed task ran. Returns `true` if state changed.
    fn consume(&mut self) -> bool {
        false
    }

    /// Notify the channel that the Pregel run is finishing. Returns `true` if state changed.
    fn finish(&mut self) -> bool {
        false
    }
}

impl Clone for Box<dyn BaseChannel> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal test channel to verify the trait compiles and works.
    #[derive(Debug, Clone)]
    struct TestChannel {
        key: String,
        value: Option<Value>,
    }

    impl BaseChannel for TestChannel {
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
            Box::new(TestChannel {
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
            if let Some(v) = values.into_iter().last() {
                self.value = Some(v);
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    #[test]
    fn test_base_channel_trait() {
        let mut ch = TestChannel {
            key: "test".to_string(),
            value: None,
        };
        assert_eq!(ch.key(), "test");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());

        let changed = ch.update(vec![Value::from(42)]).unwrap();
        assert!(changed);
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), Value::from(42));
    }

    #[test]
    fn test_box_clone() {
        let ch = TestChannel {
            key: "clone_test".to_string(),
            value: Some(Value::from("hello")),
        };
        let boxed: Box<dyn BaseChannel> = Box::new(ch);
        let cloned = boxed.clone();
        assert_eq!(cloned.key(), "clone_test");
        assert_eq!(cloned.get().unwrap(), Value::from("hello"));
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let ch = TestChannel {
            key: "ckpt".to_string(),
            value: Some(Value::from(99)),
        };
        let ckpt = ch.checkpoint();
        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(restored.get().unwrap(), Value::from(99));
    }

    #[test]
    fn test_default_consume_and_finish() {
        let mut ch = TestChannel {
            key: "defaults".to_string(),
            value: None,
        };
        assert!(!ch.consume());
        assert!(!ch.finish());
    }
}
