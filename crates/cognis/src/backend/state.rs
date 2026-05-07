//! Typed key/value state backend.
//!
//! Distinct from the file-shaped [`Backend`](super::Backend): this is for
//! agent scratch state — typed values keyed by string. Useful when the
//! agent needs to remember structured data across turns without putting
//! it in chat history.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};

use cognis_core::{CognisError, Result};

/// Typed key/value state.
///
/// Values are deserialized lazily — the backend stores opaque bytes (JSON)
/// and decodes on `get`.
#[async_trait]
pub trait StateBackend: Send + Sync {
    /// Read a typed value by key.
    async fn get<T: DeserializeOwned + Send + 'static>(&self, key: &str) -> Result<Option<T>>;
    /// Write a typed value by key.
    async fn set<T: Serialize + Send + Sync + 'static>(&self, key: &str, value: &T) -> Result<()>;
    /// Delete a key.
    async fn delete(&self, key: &str) -> Result<()>;
    /// All keys with the given prefix.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

/// In-process state backend.
#[derive(Default)]
pub struct InMemoryStateBackend {
    inner: Mutex<HashMap<String, String>>,
    _t: PhantomData<()>,
}

impl InMemoryStateBackend {
    /// Empty backend.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StateBackend for InMemoryStateBackend {
    async fn get<T: DeserializeOwned + Send + 'static>(&self, key: &str) -> Result<Option<T>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("state mutex: {e}")))?;
        match inner.get(key) {
            None => Ok(None),
            Some(json) => serde_json::from_str(json)
                .map(Some)
                .map_err(|e| CognisError::Serialization(format!("state get `{key}`: {e}"))),
        }
    }
    async fn set<T: Serialize + Send + Sync + 'static>(&self, key: &str, value: &T) -> Result<()> {
        let json = serde_json::to_string(value)
            .map_err(|e| CognisError::Serialization(format!("state set `{key}`: {e}")))?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("state mutex: {e}")))?;
        inner.insert(key.to_string(), json);
        Ok(())
    }
    async fn delete(&self, key: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("state mutex: {e}")))?;
        inner.remove(key);
        Ok(())
    }
    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("state mutex: {e}")))?;
        let mut out: Vec<String> = inner
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Foo {
        n: u32,
    }

    #[tokio::test]
    async fn roundtrip() {
        let b = InMemoryStateBackend::new();
        b.set("a/x", &Foo { n: 1 }).await.unwrap();
        b.set("a/y", &Foo { n: 2 }).await.unwrap();
        assert_eq!(b.get::<Foo>("a/x").await.unwrap(), Some(Foo { n: 1 }));
        assert_eq!(b.get::<Foo>("missing").await.unwrap(), None);
        assert_eq!(b.list("a/").await.unwrap(), vec!["a/x", "a/y"]);
        b.delete("a/x").await.unwrap();
        assert_eq!(b.get::<Foo>("a/x").await.unwrap(), None);
    }
}
