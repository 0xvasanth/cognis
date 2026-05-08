//! `Assign` — fan out an input to N runnables in parallel and merge their
//! outputs into a single map alongside the input itself.
//!
//! This is the LCEL `RunnablePassthrough.assign(...)` pattern: keep the
//! original input, but enrich it with values computed by branches.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use serde::Serialize;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Output type of [`Assign::invoke`].
#[derive(Debug, Clone, Serialize)]
pub struct AssignOutput<I, O> {
    /// The original input, passed through unchanged.
    pub input: I,
    /// Per-branch outputs keyed by branch name.
    pub assigned: HashMap<String, O>,
}

/// Runs each branch on a clone of the input and bundles the results.
pub struct Assign<I, O> {
    branches: Vec<(String, Arc<dyn Runnable<I, O>>)>,
}

impl<I, O> Default for Assign<I, O>
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

impl<I, O> Assign<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    /// Empty assign builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named branch.
    ///
    /// **Panics** on duplicate names. Two branches under the same key
    /// would race into the result map and the loser would be silently
    /// dropped — almost always a programming error.
    pub fn branch(mut self, name: impl Into<String>, runnable: Arc<dyn Runnable<I, O>>) -> Self {
        let name = name.into();
        if self.branches.iter().any(|(n, _)| n == &name) {
            panic!("Assign::branch: duplicate branch name `{name}`");
        }
        self.branches.push((name, runnable));
        self
    }
}

#[async_trait]
impl<I, O> Runnable<I, AssignOutput<I, O>> for Assign<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<AssignOutput<I, O>> {
        let futs = self.branches.iter().map(|(name, r)| {
            let r = r.clone();
            let cfg = config.clone();
            let n = name.clone();
            let i = input.clone();
            async move {
                let out = r.invoke(i, cfg).await?;
                Ok::<(String, O), crate::CognisError>((n, out))
            }
        });
        let mut assigned = HashMap::with_capacity(self.branches.len());
        for r in join_all(futs).await {
            let (k, v) = r?;
            assigned.insert(k, v);
        }
        Ok(AssignOutput { input, assigned })
    }
    fn name(&self) -> &str {
        "Assign"
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
    async fn assign_passes_input_through_and_collects_branches() {
        let a: Assign<u32, u32> = Assign::new()
            .branch("plus_one", Arc::new(AddN(1)))
            .branch("plus_ten", Arc::new(AddN(10)));
        let out = a.invoke(5, RunnableConfig::default()).await.unwrap();
        assert_eq!(out.input, 5);
        assert_eq!(out.assigned["plus_one"], 6);
        assert_eq!(out.assigned["plus_ten"], 15);
    }

    #[test]
    #[should_panic(expected = "duplicate branch name")]
    fn duplicate_branch_name_panics() {
        let _: Assign<u32, u32> = Assign::new()
            .branch("dup", Arc::new(AddN(1)))
            .branch("dup", Arc::new(AddN(2)));
    }
}
