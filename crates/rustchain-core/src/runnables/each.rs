use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;

/// Maps a runnable over each element of an input array.
///
/// Input must be `Value::Array`. Returns `Value::Array` of results.
pub struct RunnableEach {
    bound: Arc<dyn Runnable>,
}

impl RunnableEach {
    pub fn new(bound: Arc<dyn Runnable>) -> Self {
        Self { bound }
    }
}

#[async_trait]
impl Runnable for RunnableEach {
    fn name(&self) -> &str {
        "RunnableEach"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let arr = input
            .as_array()
            .ok_or_else(|| RustChainError::TypeMismatch {
                expected: "Array".into(),
                got: match &input {
                    Value::Null => "Null",
                    Value::Bool(_) => "Bool",
                    Value::Number(_) => "Number",
                    Value::String(_) => "String",
                    Value::Object(_) => "Object",
                    Value::Array(_) => unreachable!(),
                }
                .into(),
            })?;

        let mut results = Vec::with_capacity(arr.len());
        for item in arr {
            results.push(self.bound.invoke(item.clone(), config).await?);
        }
        Ok(Value::Array(results))
    }
}
