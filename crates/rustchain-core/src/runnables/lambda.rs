use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;

type AsyncFn = Box<
    dyn Fn(Value, Option<RunnableConfig>) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>>
        + Send
        + Sync,
>;

/// Wraps an async closure as a Runnable.
pub struct RunnableLambda {
    name: String,
    func: AsyncFn,
}

impl RunnableLambda {
    /// Create a lambda that only takes input (ignores config).
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let f = move |input: Value,
                      _config: Option<RunnableConfig>|
              -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> {
            Box::pin(f(input))
        };
        Self {
            name: name.into(),
            func: Box::new(f),
        }
    }

    /// Create a lambda that takes both input and config.
    pub fn with_config<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value, Option<RunnableConfig>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let f = move |input: Value,
                      config: Option<RunnableConfig>|
              -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> {
            Box::pin(f(input, config))
        };
        Self {
            name: name.into(),
            func: Box::new(f),
        }
    }
}

#[async_trait]
impl Runnable for RunnableLambda {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        (self.func)(input, config.cloned()).await
    }
}
