//! Parallel fan-out: invoke multiple runnables on the same input.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Runs multiple runnables concurrently on a shared input and returns
/// their outputs keyed by name.
///
/// Each branch must share the same input type `I` (cloned per branch) and
/// the same output type `O`. For heterogeneous outputs, wrap each branch's
/// output in an enum or use `serde_json::Value`.
pub struct Parallel<I, O> {
    branches: Vec<(String, Arc<dyn Runnable<I, O>>)>,
}

impl<I, O> Default for Parallel<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    fn default() -> Self {
        Self {
            branches: Vec::new(),
        }
    }
}

impl<I, O> Parallel<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named branch.
    pub fn branch(mut self, name: impl Into<String>, runnable: Arc<dyn Runnable<I, O>>) -> Self {
        self.branches.push((name.into(), runnable));
        self
    }
}

#[async_trait]
impl<I, O> Runnable<I, HashMap<String, O>> for Parallel<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<HashMap<String, O>> {
        let futs = self.branches.iter().map(|(name, r)| {
            let r = r.clone();
            let cfg = config.clone();
            let i = input.clone();
            let n = name.clone();
            async move {
                let out = r.invoke(i, cfg).await?;
                Ok::<(String, O), crate::CognisError>((n, out))
            }
        });
        let mut out = HashMap::with_capacity(self.branches.len());
        for r in join_all(futs).await {
            let (k, v) = r?;
            out.insert(k, v);
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "Parallel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddN(u32);

    #[async_trait]
    impl Runnable<u32, u32> for AddN {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input + self.0)
        }
    }

    #[tokio::test]
    async fn fans_out_and_collects() {
        let p: Parallel<u32, u32> = Parallel::new()
            .branch("plus_one", Arc::new(AddN(1)))
            .branch("plus_ten", Arc::new(AddN(10)));
        let out = p.invoke(5, RunnableConfig::default()).await.unwrap();
        assert_eq!(out["plus_one"], 6);
        assert_eq!(out["plus_ten"], 15);
        assert_eq!(out.len(), 2);
    }
}
