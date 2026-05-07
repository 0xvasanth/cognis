//! Predicate-based dispatch.

use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

type Predicate<I> = dyn Fn(&I) -> bool + Send + Sync;

/// One arm of a [`Branch`]: a sync predicate paired with a runnable.
pub struct BranchCase<I, O> {
    /// Predicate inspected without consuming the input.
    pub predicate: Arc<Predicate<I>>,
    /// Runnable invoked when the predicate returns `true`.
    pub runnable: Arc<dyn Runnable<I, O>>,
}

/// Conditional dispatch: try each case's predicate in order; the first
/// match runs. If none match, the default runs.
pub struct Branch<I, O> {
    cases: Vec<BranchCase<I, O>>,
    default: Arc<dyn Runnable<I, O>>,
}

impl<I, O> Branch<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build with a default runnable used when no case matches.
    pub fn new(default: Arc<dyn Runnable<I, O>>) -> Self {
        Self {
            cases: Vec::new(),
            default,
        }
    }

    /// Add a case.
    pub fn case<P>(mut self, predicate: P, runnable: Arc<dyn Runnable<I, O>>) -> Self
    where
        P: Fn(&I) -> bool + Send + Sync + 'static,
    {
        self.cases.push(BranchCase {
            predicate: Arc::new(predicate),
            runnable,
        });
        self
    }
}

#[async_trait]
impl<I, O> Runnable<I, O> for Branch<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        for c in &self.cases {
            if (c.predicate)(&input) {
                return c.runnable.invoke(input, config).await;
            }
        }
        self.default.invoke(input, config).await
    }
    fn name(&self) -> &str {
        "Branch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Const(u32);

    #[async_trait]
    impl Runnable<u32, u32> for Const {
        async fn invoke(&self, _: u32, _: RunnableConfig) -> Result<u32> {
            Ok(self.0)
        }
    }

    #[tokio::test]
    async fn dispatches_to_first_match() {
        let b: Branch<u32, u32> = Branch::new(Arc::new(Const(0)))
            .case(|i| *i < 10, Arc::new(Const(1)))
            .case(|i| *i < 100, Arc::new(Const(2)));
        assert_eq!(b.invoke(5, RunnableConfig::default()).await.unwrap(), 1);
        assert_eq!(b.invoke(50, RunnableConfig::default()).await.unwrap(), 2);
        assert_eq!(b.invoke(500, RunnableConfig::default()).await.unwrap(), 0);
    }
}
