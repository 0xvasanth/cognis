//! In-process checkpointer. Useful for tests and short-lived processes.
//! NOT durable across restarts.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use uuid::Uuid;

use cognis2_core::{CognisError, Result};

use crate::state::GraphState;

use super::Checkpointer;

/// Stores `(run_id, namespace, step) -> S` in a Mutex-protected HashMap.
/// State must be `Clone` because saving stores a clone and loading returns one.
pub struct InMemoryCheckpointer<S: GraphState + Clone> {
    runs: Mutex<HashMap<(Uuid, String), HashMap<u64, S>>>,
    active: Mutex<HashMap<(Uuid, String, u64), Vec<super::ActiveSnapshot>>>,
    namespace: String,
}

impl<S: GraphState + Clone> Default for InMemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: GraphState + Clone> InMemoryCheckpointer<S> {
    /// Empty checkpointer.
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
            active: Mutex::new(HashMap::new()),
            namespace: String::new(),
        }
    }

    /// Set the namespace for subgraph isolation. Operations on this instance
    /// scope to `(run_id, namespace, step)`.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }
}

#[async_trait]
impl<S: GraphState + Clone> Checkpointer<S> for InMemoryCheckpointer<S> {
    async fn save(&self, run_id: Uuid, step: u64, state: &S) -> Result<()> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|e| CognisError::Internal(format!("checkpointer mutex poisoned: {e}")))?;
        runs.entry((run_id, self.namespace.clone()))
            .or_default()
            .insert(step, state.clone());
        Ok(())
    }

    async fn load(&self, run_id: Uuid, step: Option<u64>) -> Result<Option<S>> {
        let runs = self
            .runs
            .lock()
            .map_err(|e| CognisError::Internal(format!("checkpointer mutex poisoned: {e}")))?;
        let Some(steps) = runs.get(&(run_id, self.namespace.clone())) else {
            return Ok(None);
        };
        match step {
            Some(s) => Ok(steps.get(&s).cloned()),
            None => {
                let max = steps.keys().copied().max();
                Ok(max.and_then(|s| steps.get(&s).cloned()))
            }
        }
    }

    async fn list(&self, run_id: Uuid) -> Result<Vec<u64>> {
        let runs = self
            .runs
            .lock()
            .map_err(|e| CognisError::Internal(format!("checkpointer mutex poisoned: {e}")))?;
        let mut steps: Vec<u64> = runs
            .get(&(run_id, self.namespace.clone()))
            .map(|s| s.keys().copied().collect())
            .unwrap_or_default();
        steps.sort();
        Ok(steps)
    }

    async fn save_active(
        &self,
        run_id: Uuid,
        step: u64,
        active: &[super::ActiveSnapshot],
    ) -> Result<()> {
        let mut a = self
            .active
            .lock()
            .map_err(|e| CognisError::Internal(format!("active mutex poisoned: {e}")))?;
        a.insert((run_id, self.namespace.clone(), step), active.to_vec());
        Ok(())
    }

    async fn load_active(&self, run_id: Uuid, step: u64) -> Result<Vec<super::ActiveSnapshot>> {
        let a = self
            .active
            .lock()
            .map_err(|e| CognisError::Internal(format!("active mutex poisoned: {e}")))?;
        Ok(a.get(&(run_id, self.namespace.clone(), step))
            .cloned()
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default, Clone, Debug, PartialEq)]
    struct S {
        n: u32,
    }
    #[derive(Default)]
    struct SU {
        n: u32,
    }
    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, u: Self::Update) {
            self.n += u.n;
        }
    }

    #[tokio::test]
    async fn save_then_load_explicit_step() {
        let cp = InMemoryCheckpointer::<S>::new();
        let id = Uuid::new_v4();
        cp.save(id, 0, &S { n: 1 }).await.unwrap();
        cp.save(id, 1, &S { n: 2 }).await.unwrap();
        cp.save(id, 2, &S { n: 3 }).await.unwrap();

        assert_eq!(cp.load(id, Some(0)).await.unwrap(), Some(S { n: 1 }));
        assert_eq!(cp.load(id, Some(1)).await.unwrap(), Some(S { n: 2 }));
        assert_eq!(cp.load(id, Some(99)).await.unwrap(), None);
    }

    #[tokio::test]
    async fn load_latest_when_step_is_none() {
        let cp = InMemoryCheckpointer::<S>::new();
        let id = Uuid::new_v4();
        cp.save(id, 0, &S { n: 1 }).await.unwrap();
        cp.save(id, 5, &S { n: 9 }).await.unwrap();
        cp.save(id, 2, &S { n: 4 }).await.unwrap();
        assert_eq!(cp.load(id, None).await.unwrap(), Some(S { n: 9 }));
    }

    #[tokio::test]
    async fn list_returns_sorted_steps() {
        let cp = InMemoryCheckpointer::<S>::new();
        let id = Uuid::new_v4();
        for s in [3u64, 1, 4, 1, 5, 9, 2, 6] {
            cp.save(id, s, &S { n: s as u32 }).await.unwrap();
        }
        assert_eq!(cp.list(id).await.unwrap(), vec![1, 2, 3, 4, 5, 6, 9]);
    }

    #[tokio::test]
    async fn unknown_run_returns_empty() {
        let cp = InMemoryCheckpointer::<S>::new();
        let unknown = Uuid::new_v4();
        assert_eq!(cp.load(unknown, None).await.unwrap(), None);
        assert!(cp.list(unknown).await.unwrap().is_empty());
    }
}
