//! `Configurable<R>` — runtime swap between named alternative runnables.
//!
//! V1 had `configurable_fields()` and `configurable_alternatives()` listed
//! as one of the P1 affordances. The Rust-native implementation:
//! - One enum-like wrapper, `Configurable<I, O>`.
//! - Caller registers alternatives by name.
//! - Per-call selection via `RunnableConfig::extras::<ConfigKey>()`.
//!
//! ```ignore
//! let model = Configurable::new(default_model)
//!     .alternative("smaller", smaller_model)
//!     .alternative("bigger", bigger_model);
//!
//! // Default behaviour:
//! model.invoke(input, RunnableConfig::default()).await?;
//!
//! // Switch at runtime:
//! let mut cfg = RunnableConfig::default();
//! cfg.extras.insert(ConfigKey::new("bigger"));
//! model.invoke(input, cfg).await?;
//! ```

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Key inserted into `RunnableConfig::extras` to choose an alternative.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfigKey {
    /// Name of the alternative to use.
    pub name: String,
}

impl ConfigKey {
    /// Construct.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Runtime-swappable runnable. Holds a default plus N named alternatives.
pub struct Configurable<I, O> {
    default: Arc<dyn Runnable<I, O>>,
    alternatives: HashMap<String, Arc<dyn Runnable<I, O>>>,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<I, O> Configurable<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Build with a default runnable.
    pub fn new(default: Arc<dyn Runnable<I, O>>) -> Self {
        Self {
            default,
            alternatives: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Add an alternative under `name`. Selection is via
    /// `cfg.extras.insert(ConfigKey::new("name"))` at invoke time.
    pub fn alternative(mut self, name: impl Into<String>, r: Arc<dyn Runnable<I, O>>) -> Self {
        self.alternatives.insert(name.into(), r);
        self
    }
}

#[async_trait]
impl<I, O> Runnable<I, O> for Configurable<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        let chosen = config
            .extras
            .get::<ConfigKey>()
            .and_then(|k| self.alternatives.get(&k.name))
            .cloned()
            .unwrap_or_else(|| self.default.clone());
        chosen.invoke(input, config).await
    }
    fn name(&self) -> &str {
        "Configurable"
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
    async fn picks_default_when_no_key() {
        let c: Configurable<u32, u32> =
            Configurable::new(Arc::new(Const(1))).alternative("alt", Arc::new(Const(2)));
        assert_eq!(c.invoke(0, RunnableConfig::default()).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn picks_alternative_when_keyed() {
        let c: Configurable<u32, u32> =
            Configurable::new(Arc::new(Const(1))).alternative("alt", Arc::new(Const(2)));
        let mut cfg = RunnableConfig::default();
        cfg.extras.insert(ConfigKey::new("alt"));
        assert_eq!(c.invoke(0, cfg).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn unknown_key_falls_back_to_default() {
        let c: Configurable<u32, u32> = Configurable::new(Arc::new(Const(1)));
        let mut cfg = RunnableConfig::default();
        cfg.extras.insert(ConfigKey::new("missing"));
        assert_eq!(c.invoke(0, cfg).await.unwrap(), 1);
    }
}
