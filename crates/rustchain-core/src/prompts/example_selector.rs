use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

/// Trait for dynamically selecting examples for few-shot prompts.
///
/// Implementations can select examples based on similarity, recency,
/// or any other criterion.
#[async_trait]
pub trait BaseExampleSelector: Send + Sync {
    /// Select examples based on the input variables.
    async fn select_examples(
        &self,
        input: &HashMap<String, Value>,
    ) -> Result<Vec<HashMap<String, Value>>>;

    /// Add a new example to the selector's store.
    async fn add_example(&self, example: HashMap<String, Value>) -> Result<()>;
}
