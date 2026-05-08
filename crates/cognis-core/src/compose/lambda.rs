//! Wrap a closure as a `Runnable<I, O>`.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

type LambdaFn<I, O> =
    dyn Fn(I, RunnableConfig) -> Pin<Box<dyn Future<Output = Result<O>> + Send>> + Send + Sync;

/// Wraps an async or sync closure so it can act as a `Runnable<I, O>`.
pub struct Lambda<I, O> {
    func: Arc<LambdaFn<I, O>>,
    name: &'static str,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<I, O> Clone for Lambda<I, O> {
    fn clone(&self) -> Self {
        Self {
            func: self.func.clone(),
            name: self.name,
            _phantom: PhantomData,
        }
    }
}

impl<I, O> Lambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build from an async closure that ignores the config.
    pub fn from_async<F, Fut>(f: F) -> Self
    where
        F: Fn(I) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
    {
        Self {
            func: Arc::new(move |i, _cfg| {
                Box::pin(f(i)) as Pin<Box<dyn Future<Output = Result<O>> + Send>>
            }),
            name: "Lambda",
            _phantom: PhantomData,
        }
    }

    /// Build from an async closure that uses the config.
    pub fn from_async_with_config<F, Fut>(f: F) -> Self
    where
        F: Fn(I, RunnableConfig) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
    {
        Self {
            func: Arc::new(move |i, c| {
                Box::pin(f(i, c)) as Pin<Box<dyn Future<Output = Result<O>> + Send>>
            }),
            name: "Lambda",
            _phantom: PhantomData,
        }
    }

    /// Build from a sync closure.
    pub fn from_sync<F>(f: F) -> Self
    where
        F: Fn(I) -> Result<O> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(move |i, _cfg| {
                let result = f(i);
                Box::pin(async move { result }) as Pin<Box<dyn Future<Output = Result<O>> + Send>>
            }),
            name: "Lambda",
            _phantom: PhantomData,
        }
    }

    /// Override the name reported via `Runnable::name`.
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

#[async_trait]
impl<I, O> Runnable<I, O> for Lambda<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        (self.func)(input, config).await
    }

    fn name(&self) -> &str {
        self.name
    }
}

/// Convenience constructor — equivalent to [`Lambda::from_async`].
pub fn lambda<F, Fut, I, O>(f: F) -> Lambda<I, O>
where
    F: Fn(I) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    Lambda::from_async(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn from_async_runs() {
        let l = lambda(|i: u32| async move { Ok(i + 1) });
        assert_eq!(l.invoke(2, RunnableConfig::default()).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn from_sync_runs() {
        let l: Lambda<u32, u32> = Lambda::from_sync(|i| Ok(i * 2));
        assert_eq!(l.invoke(5, RunnableConfig::default()).await.unwrap(), 10);
    }

    #[tokio::test]
    async fn with_name_overrides() {
        let l = lambda(|i: u32| async move { Ok(i) }).with_name("my_lambda");
        assert_eq!(l.name(), "my_lambda");
    }
}
