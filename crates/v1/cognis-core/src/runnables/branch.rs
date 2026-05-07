use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;

/// Conditional routing: evaluates condition runnables in order,
/// executing the first branch whose condition returns a truthy value.
/// Falls back to the default if no condition matches.
pub struct RunnableBranch {
    branches: Vec<(Arc<dyn Runnable>, Arc<dyn Runnable>)>,
    default: Arc<dyn Runnable>,
}

impl RunnableBranch {
    /// Create a new branch with condition/action pairs and a default.
    ///
    /// Each condition runnable should return a `Value::Bool`.
    pub fn new(
        branches: Vec<(Arc<dyn Runnable>, Arc<dyn Runnable>)>,
        default: Arc<dyn Runnable>,
    ) -> Self {
        Self { branches, default }
    }
}

fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[async_trait]
impl Runnable for RunnableBranch {
    fn name(&self) -> &str {
        "RunnableBranch"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        for (condition, action) in &self.branches {
            let result = condition.invoke(input.clone(), config).await?;
            if is_truthy(&result) {
                return action.invoke(input, config).await;
            }
        }
        self.default.invoke(input, config).await
    }
}
