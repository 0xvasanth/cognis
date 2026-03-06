//! Configuration utilities for LangGraph.
//!
//! Provides helpers for creating, merging, and patching runnable configurations.
//! This is the Rust equivalent of Python's `langgraph.utils.config` module.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Configuration keys used internally by LangGraph.
pub const CONFIG_KEY_RUNTIME: &str = "__pregel_runtime";

/// A runnable configuration that controls execution behavior.
///
/// This is the Rust equivalent of LangChain's `RunnableConfig`. It holds
/// metadata, tags, callbacks, and configurable values that flow through
/// the execution pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnableConfig {
    /// Tags for tracing and filtering.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Metadata key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,

    /// Configurable parameters.
    #[serde(default)]
    pub configurable: HashMap<String, Value>,

    /// Maximum recursion depth.
    #[serde(default)]
    pub recursion_limit: Option<u32>,
}

impl RunnableConfig {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration with the specified recursion limit.
    pub fn with_recursion_limit(mut self, limit: u32) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    /// Add a tag to this configuration.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add metadata to this configuration.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set a configurable value.
    pub fn with_configurable(mut self, key: impl Into<String>, value: Value) -> Self {
        self.configurable.insert(key.into(), value);
        self
    }
}

/// Ensure that a config is present, returning a default if `None`.
///
/// This is the Rust equivalent of Python's `ensure_config`.
pub fn ensure_config(config: Option<RunnableConfig>) -> RunnableConfig {
    config.unwrap_or_default()
}

/// Merge two configurations, with values from `override_config` taking precedence.
///
/// Tags and metadata are merged (combined), while configurable values from
/// the override replace those in the base.
pub fn merge_configs(base: &RunnableConfig, override_config: &RunnableConfig) -> RunnableConfig {
    let mut result = base.clone();

    // Merge tags (combine, dedup)
    for tag in &override_config.tags {
        if !result.tags.contains(tag) {
            result.tags.push(tag.clone());
        }
    }

    // Merge metadata (override wins)
    for (key, value) in &override_config.metadata {
        result.metadata.insert(key.clone(), value.clone());
    }

    // Merge configurable (override wins)
    for (key, value) in &override_config.configurable {
        result.configurable.insert(key.clone(), value.clone());
    }

    // Override recursion limit if set
    if override_config.recursion_limit.is_some() {
        result.recursion_limit = override_config.recursion_limit;
    }

    result
}

/// Patch the configurable section of a config with additional values.
///
/// This is the Rust equivalent of Python's `patch_configurable`.
pub fn patch_configurable(
    config: &RunnableConfig,
    updates: HashMap<String, Value>,
) -> RunnableConfig {
    let mut result = config.clone();
    for (key, value) in updates {
        result.configurable.insert(key, value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_runnable_config_new() {
        let config = RunnableConfig::new();
        assert!(config.tags.is_empty());
        assert!(config.metadata.is_empty());
        assert!(config.configurable.is_empty());
        assert_eq!(config.recursion_limit, None);
    }

    #[test]
    fn test_runnable_config_builder() {
        let config = RunnableConfig::new()
            .with_recursion_limit(25)
            .with_tag("test")
            .with_metadata("key", json!("value"))
            .with_configurable("thread_id", json!("abc123"));

        assert_eq!(config.recursion_limit, Some(25));
        assert_eq!(config.tags, vec!["test".to_string()]);
        assert_eq!(config.metadata.get("key"), Some(&json!("value")));
        assert_eq!(
            config.configurable.get("thread_id"),
            Some(&json!("abc123"))
        );
    }

    #[test]
    fn test_ensure_config_some() {
        let config = RunnableConfig::new().with_recursion_limit(10);
        let result = ensure_config(Some(config));
        assert_eq!(result.recursion_limit, Some(10));
    }

    #[test]
    fn test_ensure_config_none() {
        let result = ensure_config(None);
        assert_eq!(result.recursion_limit, None);
        assert!(result.tags.is_empty());
    }

    #[test]
    fn test_merge_configs_tags() {
        let base = RunnableConfig::new().with_tag("a").with_tag("b");
        let over = RunnableConfig::new().with_tag("b").with_tag("c");
        let result = merge_configs(&base, &over);
        assert_eq!(result.tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_merge_configs_metadata() {
        let base = RunnableConfig::new()
            .with_metadata("a", json!(1))
            .with_metadata("b", json!(2));
        let over = RunnableConfig::new()
            .with_metadata("b", json!(20))
            .with_metadata("c", json!(30));
        let result = merge_configs(&base, &over);
        assert_eq!(result.metadata.get("a"), Some(&json!(1)));
        assert_eq!(result.metadata.get("b"), Some(&json!(20)));
        assert_eq!(result.metadata.get("c"), Some(&json!(30)));
    }

    #[test]
    fn test_merge_configs_configurable() {
        let base = RunnableConfig::new().with_configurable("thread_id", json!("old"));
        let over = RunnableConfig::new().with_configurable("thread_id", json!("new"));
        let result = merge_configs(&base, &over);
        assert_eq!(
            result.configurable.get("thread_id"),
            Some(&json!("new"))
        );
    }

    #[test]
    fn test_merge_configs_recursion_limit() {
        let base = RunnableConfig::new().with_recursion_limit(10);
        let over = RunnableConfig::new();
        let result = merge_configs(&base, &over);
        assert_eq!(result.recursion_limit, Some(10));

        let over2 = RunnableConfig::new().with_recursion_limit(20);
        let result2 = merge_configs(&base, &over2);
        assert_eq!(result2.recursion_limit, Some(20));
    }

    #[test]
    fn test_patch_configurable() {
        let config = RunnableConfig::new().with_configurable("a", json!(1));
        let mut updates = HashMap::new();
        updates.insert("b".to_string(), json!(2));
        updates.insert("a".to_string(), json!(10));
        let result = patch_configurable(&config, updates);
        assert_eq!(result.configurable.get("a"), Some(&json!(10)));
        assert_eq!(result.configurable.get("b"), Some(&json!(2)));
    }

    #[test]
    fn test_runnable_config_serialize_deserialize() {
        let config = RunnableConfig::new()
            .with_recursion_limit(25)
            .with_tag("test")
            .with_metadata("key", json!("value"));

        let json_str = serde_json::to_string(&config).unwrap();
        let deserialized: RunnableConfig = serde_json::from_str(&json_str).unwrap();

        assert_eq!(deserialized.recursion_limit, Some(25));
        assert_eq!(deserialized.tags, vec!["test".to_string()]);
        assert_eq!(deserialized.metadata.get("key"), Some(&json!("value")));
    }
}
