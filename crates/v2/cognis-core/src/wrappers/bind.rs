//! `Bind` — pre-apply a `RunnableConfig` mutation to a runnable.
//!
//! Useful when a chain expects a fixed config flavor (extra tags, a
//! deadline, an observer) without forcing the caller to build it every time.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

type ConfigFn = dyn Fn(&mut RunnableConfig) + Send + Sync;

/// Wraps a runnable with a config-mutator that runs before each invocation.
pub struct Bind<R, I, O> {
    inner: R,
    apply: Arc<ConfigFn>,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Bind<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build a wrapper that applies `apply` to a clone of the caller's
    /// config before delegating.
    pub fn new<F>(inner: R, apply: F) -> Self
    where
        F: Fn(&mut RunnableConfig) + Send + Sync + 'static,
    {
        Self {
            inner,
            apply: Arc::new(apply),
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for Bind<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, mut config: RunnableConfig) -> Result<O> {
        (self.apply)(&mut config);
        self.inner.invoke(input, config).await
    }
    fn name(&self) -> &str {
        "Bind"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CapturesTag {
        captured: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Runnable<u32, u32> for CapturesTag {
        async fn invoke(&self, input: u32, config: RunnableConfig) -> Result<u32> {
            self.captured.lock().unwrap().extend(config.tags.clone());
            Ok(input)
        }
    }

    #[tokio::test]
    async fn bind_applies_mutation() {
        let inner = CapturesTag {
            captured: std::sync::Mutex::new(Vec::new()),
        };
        let bound = Bind::new(inner, |c| c.tags.push("bound".into()));
        let _ = bound.invoke(1, RunnableConfig::default()).await.unwrap();
        let cap = bound.inner.captured.lock().unwrap().clone();
        assert!(cap.contains(&"bound".to_string()));
    }
}
