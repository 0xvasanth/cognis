use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// Describes a configurable field that can be used to select alternative runnables.
#[derive(Debug, Clone)]
pub struct ConfigurableField {
    /// Unique identifier for the field, used as the key in `RunnableConfig::configurable`.
    pub id: String,
    /// Human-readable name for the field.
    pub name: Option<String>,
    /// Description of what this field controls.
    pub description: Option<String>,
}

/// A runnable that selects between alternative implementations based on configuration.
///
/// When invoked, it checks `RunnableConfig::configurable` for field values that match
/// registered alternatives. If no match is found, it falls back to the default runnable.
pub struct RunnableConfigurableFields {
    default: Arc<dyn Runnable>,
    fields: Vec<ConfigurableField>,
    alternatives: HashMap<String, HashMap<String, Arc<dyn Runnable>>>,
}

impl RunnableConfigurableFields {
    /// Create a new `RunnableConfigurableFields` with a default runnable and configurable fields.
    pub fn new(default: Arc<dyn Runnable>, fields: Vec<ConfigurableField>) -> Self {
        Self {
            default,
            fields,
            alternatives: HashMap::new(),
        }
    }

    /// Register alternative runnables for a given field.
    pub fn with_alternatives(
        mut self,
        field_id: &str,
        alternatives: HashMap<String, Arc<dyn Runnable>>,
    ) -> Self {
        self.alternatives.insert(field_id.into(), alternatives);
        self
    }

    fn resolve(&self, config: Option<&RunnableConfig>) -> Arc<dyn Runnable> {
        if let Some(config) = config {
            for field in &self.fields {
                if let Some(val) = config.configurable.get(&field.id) {
                    if let Some(key) = val.as_str() {
                        if let Some(alts) = self.alternatives.get(&field.id) {
                            if let Some(alt) = alts.get(key) {
                                return alt.clone();
                            }
                        }
                    }
                }
            }
        }
        self.default.clone()
    }
}

#[async_trait]
impl Runnable for RunnableConfigurableFields {
    fn name(&self) -> &str {
        "RunnableConfigurableFields"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        self.resolve(config).invoke(input, config).await
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        self.resolve(config).stream(input, config).await
    }
}
