use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;

/// Configuration for a runnable invocation.
#[derive(Clone)]
pub struct RunnableConfig {
    pub tags: Vec<String>,
    pub metadata: HashMap<String, Value>,
    pub run_name: Option<String>,
    pub max_concurrency: Option<usize>,
    pub recursion_limit: usize,
    pub configurable: HashMap<String, Value>,
    pub run_id: Option<Uuid>,
    pub callbacks: Vec<Arc<dyn CallbackHandler>>,
}

impl Default for RunnableConfig {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            metadata: HashMap::new(),
            run_name: None,
            max_concurrency: None,
            recursion_limit: 25,
            configurable: HashMap::new(),
            run_id: None,
            callbacks: Vec::new(),
        }
    }
}

impl std::fmt::Debug for RunnableConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableConfig")
            .field("tags", &self.tags)
            .field("metadata", &self.metadata)
            .field("run_name", &self.run_name)
            .field("max_concurrency", &self.max_concurrency)
            .field("recursion_limit", &self.recursion_limit)
            .field("configurable", &self.configurable)
            .field("run_id", &self.run_id)
            .field("callbacks", &format!("[{} handlers]", self.callbacks.len()))
            .finish()
    }
}

/// Returns a clone of the config if provided, otherwise returns the default.
pub fn ensure_config(config: Option<&RunnableConfig>) -> RunnableConfig {
    match config {
        Some(c) => c.clone(),
        None => RunnableConfig::default(),
    }
}

/// A builder for selectively patching a `RunnableConfig`.
///
/// Unlike `merge_configs`, patching replaces fields (not appending) when set.
pub struct ConfigPatch {
    tags: Option<Vec<String>>,
    metadata: Option<HashMap<String, Value>>,
    callbacks: Option<Vec<Arc<dyn CallbackHandler>>>,
    run_name: Option<String>,
    max_concurrency: Option<usize>,
    recursion_limit: Option<usize>,
    configurable: Option<HashMap<String, Value>>,
    run_id: Option<Uuid>,
}

impl ConfigPatch {
    pub fn new() -> Self {
        Self {
            tags: None,
            metadata: None,
            callbacks: None,
            run_name: None,
            max_concurrency: None,
            recursion_limit: None,
            configurable: None,
            run_id: None,
        }
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self { self.tags = Some(tags); self }
    pub fn metadata(mut self, metadata: HashMap<String, Value>) -> Self { self.metadata = Some(metadata); self }
    pub fn callbacks(mut self, callbacks: Vec<Arc<dyn CallbackHandler>>) -> Self { self.callbacks = Some(callbacks); self }
    pub fn run_name(mut self, name: impl Into<String>) -> Self { self.run_name = Some(name.into()); self }
    pub fn max_concurrency(mut self, n: usize) -> Self { self.max_concurrency = Some(n); self }
    pub fn recursion_limit(mut self, n: usize) -> Self { self.recursion_limit = Some(n); self }
    pub fn configurable(mut self, configurable: HashMap<String, Value>) -> Self { self.configurable = Some(configurable); self }
    pub fn run_id(mut self, id: Uuid) -> Self { self.run_id = Some(id); self }

    /// Apply this patch to a config, returning a new config.
    pub fn apply(&self, config: &RunnableConfig) -> RunnableConfig {
        RunnableConfig {
            tags: self.tags.clone().unwrap_or_else(|| config.tags.clone()),
            metadata: self.metadata.clone().unwrap_or_else(|| config.metadata.clone()),
            callbacks: self.callbacks.clone().unwrap_or_else(|| config.callbacks.clone()),
            run_name: self.run_name.clone().or_else(|| config.run_name.clone()),
            max_concurrency: self.max_concurrency.or(config.max_concurrency),
            recursion_limit: self.recursion_limit.unwrap_or(config.recursion_limit),
            configurable: self.configurable.clone().unwrap_or_else(|| config.configurable.clone()),
            run_id: self.run_id.or(config.run_id),
        }
    }
}

impl Default for ConfigPatch {
    fn default() -> Self { Self::new() }
}

/// Patch a config with selective overrides using the builder pattern.
///
/// Unlike `merge_configs`, this only overwrites fields that are explicitly provided.
/// Tags and metadata are replaced (not appended) if present.
pub fn patch_config(config: &RunnableConfig, patch: &ConfigPatch) -> RunnableConfig {
    patch.apply(config)
}

/// Create a list of configs of the given length from a single config or list.
///
/// If given a single config, clones it `length` times.
/// This is used to fan out configs for batch operations.
pub fn get_config_list(config: &RunnableConfig, length: usize) -> Vec<RunnableConfig> {
    (0..length).map(|_| config.clone()).collect()
}

/// Merges two configs. The overlay's scalar values win if set.
/// Tags and callbacks are appended. Metadata and configurable are overlaid.
pub fn merge_configs(base: &RunnableConfig, overlay: &RunnableConfig) -> RunnableConfig {
    let mut tags = base.tags.clone();
    tags.extend(overlay.tags.iter().cloned());

    let mut metadata = base.metadata.clone();
    metadata.extend(overlay.metadata.iter().map(|(k, v)| (k.clone(), v.clone())));

    let mut configurable = base.configurable.clone();
    configurable.extend(
        overlay
            .configurable
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );

    let mut callbacks = base.callbacks.clone();
    callbacks.extend(overlay.callbacks.iter().cloned());

    RunnableConfig {
        tags,
        metadata,
        run_name: overlay.run_name.clone().or_else(|| base.run_name.clone()),
        max_concurrency: overlay.max_concurrency.or(base.max_concurrency),
        recursion_limit: overlay.recursion_limit,
        configurable,
        run_id: overlay.run_id.or(base.run_id),
        callbacks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_default_config() {
        let config = RunnableConfig::default();
        assert!(config.tags.is_empty());
        assert!(config.metadata.is_empty());
        assert_eq!(config.recursion_limit, 25);
        assert!(config.run_name.is_none());
        assert!(config.max_concurrency.is_none());
        assert!(config.run_id.is_none());
    }

    #[test]
    fn test_ensure_config_none() {
        let config = ensure_config(None);
        assert_eq!(config.recursion_limit, 25);
    }

    #[test]
    fn test_ensure_config_some() {
        let mut c = RunnableConfig::default();
        c.recursion_limit = 50;
        let result = ensure_config(Some(&c));
        assert_eq!(result.recursion_limit, 50);
    }

    #[test]
    fn test_merge_configs_tags_appended() {
        let mut base = RunnableConfig::default();
        base.tags = vec!["a".into()];
        let mut overlay = RunnableConfig::default();
        overlay.tags = vec!["b".into()];
        let result = merge_configs(&base, &overlay);
        assert_eq!(result.tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_merge_configs_metadata_overlaid() {
        let mut base = RunnableConfig::default();
        base.metadata.insert("k1".into(), json!("v1"));
        base.metadata.insert("k2".into(), json!("v2"));
        let mut overlay = RunnableConfig::default();
        overlay.metadata.insert("k2".into(), json!("v2_new"));
        let result = merge_configs(&base, &overlay);
        assert_eq!(result.metadata.get("k1"), Some(&json!("v1")));
        assert_eq!(result.metadata.get("k2"), Some(&json!("v2_new")));
    }

    #[test]
    fn test_merge_configs_run_name_overlay_wins() {
        let mut base = RunnableConfig::default();
        base.run_name = Some("base_name".into());
        let mut overlay = RunnableConfig::default();
        overlay.run_name = Some("overlay_name".into());
        let result = merge_configs(&base, &overlay);
        assert_eq!(result.run_name, Some("overlay_name".into()));
    }

    #[test]
    fn test_patch_config_selective() {
        let mut config = RunnableConfig::default();
        config.tags = vec!["original".into()];
        config.recursion_limit = 10;

        let patch = ConfigPatch::new()
            .tags(vec!["new_tag".into()])
            .run_name("my_run");
        let patched = patch_config(&config, &patch);
        assert_eq!(patched.tags, vec!["new_tag".to_string()]);
        assert_eq!(patched.recursion_limit, 10); // unchanged
        assert_eq!(patched.run_name, Some("my_run".into()));
    }

    #[test]
    fn test_get_config_list() {
        let config = RunnableConfig::default();
        let list = get_config_list(&config, 3);
        assert_eq!(list.len(), 3);
        for c in &list {
            assert_eq!(c.recursion_limit, 25);
        }
    }
}
