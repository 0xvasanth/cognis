//! Sequential composition: feed `A`'s output as `B`'s input.

use std::marker::PhantomData;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Sequential composition of two runnables.
pub struct Pipe<A, B, I, M, O> {
    a: A,
    b: B,
    _types: PhantomData<fn(I) -> O>,
    _mid: PhantomData<fn() -> M>,
}

impl<A, B, I, M, O> Pipe<A, B, I, M, O>
where
    A: Runnable<I, M>,
    B: Runnable<M, O>,
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    /// Build a pipe from two runnables.
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _types: PhantomData,
            _mid: PhantomData,
        }
    }
}

#[async_trait]
impl<A, B, I, M, O> Runnable<I, O> for Pipe<A, B, I, M, O>
where
    A: Runnable<I, M>,
    B: Runnable<M, O>,
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        let mid = self.a.invoke(input, config.clone()).await?;
        self.b.invoke(mid, config).await
    }

    fn name(&self) -> &str {
        "Pipe"
    }

    fn input_schema(&self) -> Option<serde_json::Value> {
        self.a.input_schema()
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        self.b.output_schema()
    }
}

/// Convenience: `pipe(a, b)` is `Pipe::new(a, b)`. Prefer
/// `a.pipe(b)` from [`crate::RunnableExt`] for chains.
pub fn pipe<A, B, I, M, O>(a: A, b: B) -> Pipe<A, B, I, M, O>
where
    A: Runnable<I, M>,
    B: Runnable<M, O>,
    I: Send + 'static,
    M: Send + 'static,
    O: Send + 'static,
{
    Pipe::new(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Add(u32);

    #[async_trait]
    impl Runnable<u32, u32> for Add {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input + self.0)
        }
    }

    struct Mul(u32);

    #[async_trait]
    impl Runnable<u32, u32> for Mul {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input * self.0)
        }
    }

    #[tokio::test]
    async fn pipe_method_chains() {
        let chain = Pipe::new(Add(2), Mul(3));
        let out = chain.invoke(4, RunnableConfig::default()).await.unwrap();
        assert_eq!(out, (4 + 2) * 3);
    }

    #[tokio::test]
    async fn pipe_fn_works() {
        let chain = pipe(Add(1), Mul(10));
        let out = chain.invoke(2, RunnableConfig::default()).await.unwrap();
        assert_eq!(out, (2 + 1) * 10);
    }
}
