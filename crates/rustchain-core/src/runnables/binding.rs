use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::{merge_configs, ensure_config, RunnableConfig};

/// Binds additional kwargs and/or config to an inner runnable.
///
/// If the input is an object, `kwargs` are merged into it before invocation.
/// If `config_patch` is set, it is merged with the provided config.
pub struct RunnableBinding {
    bound: Arc<dyn Runnable>,
    kwargs: HashMap<String, Value>,
    config_patch: Option<RunnableConfig>,
}

impl RunnableBinding {
    pub fn new(
        bound: Arc<dyn Runnable>,
        kwargs: HashMap<String, Value>,
        config_patch: Option<RunnableConfig>,
    ) -> Self {
        Self {
            bound,
            kwargs,
            config_patch,
        }
    }
}

#[async_trait]
impl Runnable for RunnableBinding {
    fn name(&self) -> &str {
        "RunnableBinding"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let merged_input = if let Value::Object(mut map) = input {
            for (k, v) in &self.kwargs {
                map.insert(k.clone(), v.clone());
            }
            Value::Object(map)
        } else {
            input
        };

        let effective_config = match &self.config_patch {
            Some(patch) => {
                let base = ensure_config(config);
                merge_configs(&base, patch)
            }
            None => ensure_config(config),
        };

        self.bound.invoke(merged_input, Some(&effective_config)).await
    }
}
