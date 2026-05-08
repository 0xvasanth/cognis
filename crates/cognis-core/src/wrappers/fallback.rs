//! Fallback wrapper — try `primary`, switch to `fallback` on error.

use std::marker::PhantomData;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Try the primary runnable; on any error, fall through to the fallback.
///
/// Chain multiple fallbacks by repeated wrapping.
pub struct Fallback<P, F, I, O> {
    primary: P,
    fallback: F,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<P, F, I, O> Fallback<P, F, I, O>
where
    P: Runnable<I, O>,
    F: Runnable<I, O>,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    /// Build a fallback wrapper.
    pub fn new(primary: P, fallback: F) -> Self {
        Self {
            primary,
            fallback,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<P, F, I, O> Runnable<I, O> for Fallback<P, F, I, O>
where
    P: Runnable<I, O>,
    F: Runnable<I, O>,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        match self.primary.invoke(input.clone(), config.clone()).await {
            Ok(v) => Ok(v),
            Err(_) => self.fallback.invoke(input, config).await,
        }
    }
    fn name(&self) -> &str {
        "Fallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CognisError;

    struct Err1;
    struct Ok99;

    #[async_trait]
    impl Runnable<u32, u32> for Err1 {
        async fn invoke(&self, _: u32, _: RunnableConfig) -> Result<u32> {
            Err(CognisError::Internal("nope".into()))
        }
    }

    #[async_trait]
    impl Runnable<u32, u32> for Ok99 {
        async fn invoke(&self, _: u32, _: RunnableConfig) -> Result<u32> {
            Ok(99)
        }
    }

    #[tokio::test]
    async fn falls_through_on_error() {
        let f = Fallback::new(Err1, Ok99);
        assert_eq!(f.invoke(0, RunnableConfig::default()).await.unwrap(), 99);
    }

    #[tokio::test]
    async fn primary_wins_on_success() {
        let f = Fallback::new(Ok99, Err1);
        assert_eq!(f.invoke(0, RunnableConfig::default()).await.unwrap(), 99);
    }
}
