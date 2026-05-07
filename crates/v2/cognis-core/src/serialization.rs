//! Minimal Runnable serialization layer.
//!
//! V1 had a heavy "serializable runnable" hierarchy modeled after
//! Python's serialize-everything pattern. V2 takes a smaller,
//! Rust-native shape:
//!
//! - [`RunnableDefinition`] is a tagged enum describing a runnable's
//!   shape (kind + config).
//! - The [`Serializable`] trait (`Runnable + Serializable`) lets a
//!   runnable emit its definition.
//! - [`Loader`] reconstructs runnables from definitions via a registry
//!   of named factories. Lambda closures are by name only — caller must
//!   register the lambda factory under the same name used during dump.
//!
//! This is enough to ship/restore composed pipelines (Pipe/Each/Pipe of
//! Pipe) without a full serialize-everything story.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::runnable::Runnable;
use crate::{CognisError, Result};

/// Serializable description of a runnable's shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunnableDefinition {
    /// Sequential `a.pipe(b)`.
    Pipe {
        /// Left-side definition.
        a: Box<RunnableDefinition>,
        /// Right-side definition.
        b: Box<RunnableDefinition>,
    },
    /// Element-wise wrapper.
    Each {
        /// Inner definition.
        inner: Box<RunnableDefinition>,
    },
    /// Identity runnable.
    Passthrough,
    /// Named lambda — the caller's `Loader` must register a factory
    /// under `name`.
    Lambda {
        /// Registered factory name.
        name: String,
    },
    /// Free-form opaque definition. Runnables that can't fully describe
    /// themselves emit this; the caller must know how to rebuild from
    /// `name` + `params`.
    Opaque {
        /// Type identifier for the loader.
        name: String,
        /// Arbitrary config payload.
        params: serde_json::Value,
    },
}

/// Trait runnables implement to participate in dump/load. Default is
/// `Opaque { name = type name }` — override to capture useful structure.
pub trait Serializable {
    /// Emit a definition describing this runnable's shape.
    fn to_definition(&self) -> RunnableDefinition;
}

/// One factory in a `Loader`: takes opaque params, produces a runnable.
type Factory<I, O> = Box<dyn Fn(&serde_json::Value) -> Arc<dyn Runnable<I, O>> + Send + Sync>;

/// Reconstructs runnables from definitions. Caller registers factories
/// by name for each `Lambda` / `Opaque` kind they expect to load.
pub struct Loader<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    factories: HashMap<String, Factory<I, O>>,
}

impl<I, O> Default for Loader<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    fn default() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }
}

impl<I, O> Loader<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Empty loader.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory for a named lambda or opaque kind. The factory
    /// receives the params payload (`serde_json::Value::Null` for
    /// lambdas) and must produce an `Arc<dyn Runnable<I, O>>`.
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn(&serde_json::Value) -> Arc<dyn Runnable<I, O>> + Send + Sync + 'static,
    {
        self.factories.insert(name.into(), Box::new(factory));
    }

    /// Reconstruct a `Runnable<I, O>` from its definition.
    pub fn load(&self, def: &RunnableDefinition) -> Result<Arc<dyn Runnable<I, O>>> {
        match def {
            RunnableDefinition::Lambda { name } | RunnableDefinition::Opaque { name, .. } => {
                let params = match def {
                    RunnableDefinition::Opaque { params, .. } => params.clone(),
                    _ => serde_json::Value::Null,
                };
                self.factories.get(name).map(|f| f(&params)).ok_or_else(|| {
                    CognisError::Configuration(format!(
                        "Loader: no factory registered for `{name}`"
                    ))
                })
            }
            // Composite kinds aren't directly buildable here — they need
            // type-aligned inner definitions, which a generic loader
            // can't enforce. Callers reconstruct composites via their
            // own builder using the inner definitions.
            other => Err(CognisError::Configuration(format!(
                "Loader::load: composite definitions ({other:?}) must be reconstructed by caller — \
                 use the inner definitions to build the composition explicitly"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnable::RunnableConfig;
    use async_trait::async_trait;

    struct Const(u32);
    #[async_trait]
    impl Runnable<u32, u32> for Const {
        async fn invoke(&self, _: u32, _: RunnableConfig) -> Result<u32> {
            Ok(self.0)
        }
    }

    #[test]
    fn roundtrip_definition_serde() {
        let d = RunnableDefinition::Pipe {
            a: Box::new(RunnableDefinition::Lambda {
                name: "step_a".into(),
            }),
            b: Box::new(RunnableDefinition::Passthrough),
        };
        let s = serde_json::to_string(&d).unwrap();
        let back: RunnableDefinition = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, RunnableDefinition::Pipe { .. }));
    }

    #[tokio::test]
    async fn loader_resolves_named_factory() {
        let mut loader = Loader::<u32, u32>::new();
        loader.register("k", |_| Arc::new(Const(7)));
        let r = loader
            .load(&RunnableDefinition::Lambda { name: "k".into() })
            .unwrap();
        assert_eq!(r.invoke(0, RunnableConfig::default()).await.unwrap(), 7);
    }

    #[tokio::test]
    async fn loader_unknown_factory_errors() {
        let loader = Loader::<u32, u32>::new();
        assert!(loader
            .load(&RunnableDefinition::Lambda {
                name: "nope".into(),
            })
            .is_err());
    }
}
