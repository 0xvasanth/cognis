//! Apply a runnable to every element of a `Vec<I>`, preserving order.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Wraps a `Runnable<I, O>` to act on a `Vec<I>` element-wise.
///
/// Concurrency is bounded by `RunnableConfig::max_concurrency`. Output
/// order matches input order.
pub struct Each<R, I, O> {
    inner: R,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Each<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap a runnable so it operates element-wise on a `Vec<I>`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<R, I, O> Runnable<Vec<I>, Vec<O>> for Each<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, inputs: Vec<I>, config: RunnableConfig) -> Result<Vec<O>> {
        let concurrency = config.max_concurrency.max(1);
        let cfg = Arc::new(config);
        stream::iter(inputs)
            .map(|i| {
                let cfg = cfg.clone();
                async move {
                    self.inner
                        .invoke(i, RunnableConfig::clone_for_subcall(&cfg))
                        .await
                }
            })
            .buffered(concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
    }
    fn name(&self) -> &str {
        "Each"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Inc;

    #[async_trait]
    impl Runnable<u32, u32> for Inc {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input + 1)
        }
    }

    #[tokio::test]
    async fn maps_each_element() {
        let e = Each::new(Inc);
        let out = e
            .invoke(vec![1, 2, 3, 4], RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, vec![2, 3, 4, 5]);
    }
}
