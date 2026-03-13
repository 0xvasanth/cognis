//! Namespaced store wrapper.
//!
//! Transparently prefixes every key with a namespace string, allowing
//! multiple logical stores to share a single physical backend.

use cognis_core::error::Result;

use super::Store;

/// A store wrapper that prefixes all keys with a namespace.
///
/// The namespace and the original key are separated by `::`.
pub struct NamespacedStore {
    inner: Box<dyn Store>,
    namespace: String,
    separator: String,
}

impl NamespacedStore {
    /// Wrap `inner` so that all keys are prefixed with `namespace::`.
    pub fn new(inner: Box<dyn Store>, namespace: String) -> Self {
        Self {
            inner,
            namespace,
            separator: "::".to_string(),
        }
    }

    /// Wrap `inner` with a custom separator between namespace and key.
    pub fn with_separator(inner: Box<dyn Store>, namespace: String, separator: String) -> Self {
        Self {
            inner,
            namespace,
            separator,
        }
    }

    /// Build the full prefixed key.
    fn prefixed(&self, key: &str) -> String {
        format!("{}{}{}", self.namespace, self.separator, key)
    }

    /// Strip the namespace prefix from a key, returning `None` if it does not
    /// belong to this namespace.
    fn strip_prefix<'a>(&self, full_key: &'a str) -> Option<&'a str> {
        let prefix = format!("{}{}", self.namespace, self.separator);
        full_key.strip_prefix(&prefix)
    }

    /// List all distinct namespace prefixes found in the inner store.
    pub fn list_namespaces(&self) -> Result<Vec<String>> {
        let all_keys = self.inner.keys()?;
        let mut namespaces: Vec<String> = all_keys
            .iter()
            .filter_map(|k| k.find(&self.separator).map(|idx| k[..idx].to_string()))
            .collect();
        namespaces.sort();
        namespaces.dedup();
        Ok(namespaces)
    }
}

impl Store for NamespacedStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.inner.get(&self.prefixed(key))
    }

    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.inner.set(&self.prefixed(key), value)
    }

    fn delete(&self, key: &str) -> Result<bool> {
        self.inner.delete(&self.prefixed(key))
    }

    fn exists(&self, key: &str) -> bool {
        self.inner.exists(&self.prefixed(key))
    }

    fn keys(&self) -> Result<Vec<String>> {
        let all_keys = self.inner.keys()?;
        Ok(all_keys
            .iter()
            .filter_map(|k| self.strip_prefix(k).map(|s| s.to_string()))
            .collect())
    }

    fn clear(&self) -> Result<()> {
        let my_keys = self.keys()?;
        for key in my_keys {
            self.inner.delete(&self.prefixed(&key))?;
        }
        Ok(())
    }
}
