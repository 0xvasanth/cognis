use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;

/// Dispatches to a named runnable based on a `key` field in the input.
///
/// Expects input of the form `{"key": "<name>", "input": <value>}`.
pub struct RouterRunnable {
    runnables: HashMap<String, Arc<dyn Runnable>>,
}

impl RouterRunnable {
    pub fn new(runnables: HashMap<String, Arc<dyn Runnable>>) -> Self {
        Self { runnables }
    }
}

#[async_trait]
impl Runnable for RouterRunnable {
    fn name(&self) -> &str {
        "RouterRunnable"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let key = input
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                RustChainError::InvalidKey("Input must have a string 'key' field".into())
            })?
            .to_string();

        let inner_input = input.get("input").cloned().unwrap_or(Value::Null);

        let runnable = self.runnables.get(&key).ok_or_else(|| {
            RustChainError::InvalidKey(format!(
                "No runnable found for key '{}'. Available: {:?}",
                key,
                self.runnables.keys().collect::<Vec<_>>()
            ))
        })?;

        runnable.invoke(inner_input, config).await
    }
}
