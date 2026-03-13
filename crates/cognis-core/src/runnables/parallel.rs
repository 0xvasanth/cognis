use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};
use tokio::task::JoinSet;

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::{ensure_config, RunnableConfig};

/// Runs multiple runnables in parallel, merging outputs into a JSON object keyed by step name.
pub struct RunnableParallel {
    name: Option<String>,
    steps: HashMap<String, Arc<dyn Runnable>>,
}

impl RunnableParallel {
    pub fn new(steps: HashMap<String, Arc<dyn Runnable>>) -> Self {
        Self { name: None, steps }
    }

    /// Set a custom name for this parallel.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

#[async_trait]
impl Runnable for RunnableParallel {
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("RunnableParallel")
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let cfg = ensure_config(config);
        let mut join_set = JoinSet::new();

        for (key, runnable) in &self.steps {
            let key = key.clone();
            let runnable = Arc::clone(runnable);
            let input = input.clone();
            let cfg = cfg.clone();

            join_set.spawn(async move {
                let result = runnable.invoke(input, Some(&cfg)).await;
                (key, result)
            });
        }

        let mut map = Map::new();
        while let Some(result) = join_set.join_next().await {
            let (key, value) = result.map_err(|e| CognisError::Other(e.to_string()))?;
            map.insert(key, value?);
        }

        Ok(Value::Object(map))
    }
}
