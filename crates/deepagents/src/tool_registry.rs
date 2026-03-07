//! Tool registry for managing tool availability, permissions, and discovery.
//!
//! The [`ToolRegistry`] provides a centralized store for tools available to an agent,
//! with support for permission checks, call counting, filtering, and enable/disable control.

use rustchain_core::tools::ToolSchema;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// Permission level for a registered tool.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolPermission {
    /// Tool may be called freely.
    Allowed,
    /// Tool is denied and must not be called.
    Denied,
    /// Tool requires human approval before each call.
    RequiresApproval,
    /// Tool is rate-limited to a maximum number of calls within a time window.
    RateLimited { max_calls: usize, window: Duration },
}

/// A registered tool entry containing metadata, permission, and runtime counters.
pub struct ToolEntry {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The tool's input schema.
    pub schema: ToolSchema,
    /// Permission level.
    pub permission: ToolPermission,
    /// Category grouping (e.g. "filesystem", "web", "code").
    pub category: String,
    /// Arbitrary tags for filtering.
    pub tags: Vec<String>,
    /// Number of times this tool has been called.
    pub call_count: AtomicUsize,
    /// Whether the tool is currently enabled.
    pub enabled: AtomicBool,
}

impl std::fmt::Debug for ToolEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolEntry")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("schema", &self.schema)
            .field("permission", &self.permission)
            .field("category", &self.category)
            .field("tags", &self.tags)
            .field("call_count", &self.call_count.load(Ordering::Relaxed))
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .finish()
    }
}

impl ToolEntry {
    /// Create a new tool entry with the given name, description, and schema.
    /// Defaults to `Allowed` permission, empty category/tags, zero calls, and enabled.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: ToolSchema,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            permission: ToolPermission::Allowed,
            category: String::new(),
            tags: Vec::new(),
            call_count: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Set the permission level.
    pub fn with_permission(mut self, permission: ToolPermission) -> Self {
        self.permission = permission;
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Set the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Filter criteria for searching tools in the registry.
#[derive(Debug, Clone, Default)]
pub struct ToolFilter {
    /// Only tools in this category.
    pub category: Option<String>,
    /// Only tools that have this tag.
    pub tag: Option<String>,
    /// Only return enabled tools (default `true`).
    pub enabled_only: bool,
    /// Substring match on tool name.
    pub name_pattern: Option<String>,
}

impl ToolFilter {
    /// Create a new filter with default settings (enabled_only = true).
    pub fn new() -> Self {
        Self {
            category: None,
            tag: None,
            enabled_only: true,
            name_pattern: None,
        }
    }

    /// Filter by category.
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Filter by tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Set whether to return only enabled tools.
    pub fn enabled_only(mut self, enabled_only: bool) -> Self {
        self.enabled_only = enabled_only;
        self
    }

    /// Filter by name substring.
    pub fn name_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.name_pattern = Some(pattern.into());
        self
    }

    /// Check whether a tool entry matches this filter.
    fn matches(&self, entry: &ToolEntry) -> bool {
        if self.enabled_only && !entry.enabled.load(Ordering::Relaxed) {
            return false;
        }
        if let Some(ref cat) = self.category {
            if entry.category != *cat {
                return false;
            }
        }
        if let Some(ref tag) = self.tag {
            if !entry.tags.iter().any(|t| t == tag) {
                return false;
            }
        }
        if let Some(ref pattern) = self.name_pattern {
            if !entry.name.contains(pattern.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Thread-safe registry for managing tools available to an agent.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, Arc<ToolEntry>>>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tools = self.tools.read().unwrap();
        f.debug_struct("ToolRegistry")
            .field("tool_count", &tools.len())
            .finish()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return a builder for constructing a registry with initial tools.
    pub fn builder() -> ToolRegistryBuilder {
        ToolRegistryBuilder::new()
    }

    /// Register a tool entry. If a tool with the same name already exists it is replaced.
    pub fn register(&self, entry: ToolEntry) {
        let name = entry.name.clone();
        let mut tools = self.tools.write().unwrap();
        tools.insert(name, Arc::new(entry));
    }

    /// Remove a tool by name, returning the entry if it existed.
    pub fn unregister(&self, name: &str) -> Option<Arc<ToolEntry>> {
        let mut tools = self.tools.write().unwrap();
        tools.remove(name)
    }

    /// Get a tool entry by name.
    pub fn get(&self, name: &str) -> Option<Arc<ToolEntry>> {
        let tools = self.tools.read().unwrap();
        tools.get(name).cloned()
    }

    /// Check the permission for a tool. Returns `Denied` if the tool is not found.
    pub fn check_permission(&self, name: &str) -> ToolPermission {
        let tools = self.tools.read().unwrap();
        match tools.get(name) {
            Some(entry) => entry.permission.clone(),
            None => ToolPermission::Denied,
        }
    }

    /// Record a call to the named tool by incrementing its call counter.
    pub fn record_call(&self, name: &str) {
        let tools = self.tools.read().unwrap();
        if let Some(entry) = tools.get(name) {
            entry.call_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Search for tools matching the given filter.
    pub fn search(&self, filter: &ToolFilter) -> Vec<Arc<ToolEntry>> {
        let tools = self.tools.read().unwrap();
        tools
            .values()
            .filter(|entry| filter.matches(entry))
            .cloned()
            .collect()
    }

    /// List all distinct categories present in the registry.
    pub fn list_categories(&self) -> Vec<String> {
        let tools = self.tools.read().unwrap();
        let mut cats: Vec<String> = tools
            .values()
            .map(|e| e.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cats.sort();
        cats
    }

    /// Enable a tool by name.
    pub fn enable(&self, name: &str) {
        let tools = self.tools.read().unwrap();
        if let Some(entry) = tools.get(name) {
            entry.enabled.store(true, Ordering::Relaxed);
        }
    }

    /// Disable a tool by name.
    pub fn disable(&self, name: &str) {
        let tools = self.tools.read().unwrap();
        if let Some(entry) = tools.get(name) {
            entry.enabled.store(false, Ordering::Relaxed);
        }
    }

    /// Get schemas for all currently enabled tools (suitable for sending to an LLM).
    pub fn get_schemas(&self) -> Vec<ToolSchema> {
        let tools = self.tools.read().unwrap();
        tools
            .values()
            .filter(|e| e.enabled.load(Ordering::Relaxed))
            .map(|e| e.schema.clone())
            .collect()
    }

    /// Return the number of registered tools.
    pub fn len(&self) -> usize {
        let tools = self.tools.read().unwrap();
        tools.len()
    }

    /// Check whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Builder for constructing a [`ToolRegistry`] with initial tools.
#[derive(Default)]
pub struct ToolRegistryBuilder {
    entries: Vec<ToolEntry>,
}

impl ToolRegistryBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a tool entry to the builder.
    pub fn tool(mut self, entry: ToolEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Build the registry.
    pub fn build(self) -> ToolRegistry {
        let registry = ToolRegistry::new();
        for entry in self.entries {
            registry.register(entry);
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::tools::ToolSchema;

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            name: name.to_string(),
            description: format!("{} tool", name),
            parameters: None,
            extras: None,
        }
    }

    fn make_entry(name: &str, category: &str) -> ToolEntry {
        ToolEntry::new(name, format!("Description for {}", name), make_schema(name))
            .with_category(category)
    }

    #[test]
    fn test_register_and_get() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("read_file", "filesystem"));
        let entry = registry.get("read_file").unwrap();
        assert_eq!(entry.name, "read_file");
        assert_eq!(entry.category, "filesystem");
    }

    #[test]
    fn test_get_missing_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_unregister() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool_a", "cat"));
        assert_eq!(registry.len(), 1);
        let removed = registry.unregister("tool_a");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "tool_a");
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_unregister_missing() {
        let registry = ToolRegistry::new();
        assert!(registry.unregister("nope").is_none());
    }

    #[test]
    fn test_len_and_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        registry.register(make_entry("t1", "c"));
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_check_permission_allowed() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool", "c").with_permission(ToolPermission::Allowed));
        assert_eq!(registry.check_permission("tool"), ToolPermission::Allowed);
    }

    #[test]
    fn test_check_permission_denied_for_missing() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.check_permission("missing"), ToolPermission::Denied);
    }

    #[test]
    fn test_check_permission_requires_approval() {
        let registry = ToolRegistry::new();
        registry
            .register(make_entry("tool", "c").with_permission(ToolPermission::RequiresApproval));
        assert_eq!(
            registry.check_permission("tool"),
            ToolPermission::RequiresApproval
        );
    }

    #[test]
    fn test_check_permission_rate_limited() {
        let registry = ToolRegistry::new();
        let perm = ToolPermission::RateLimited {
            max_calls: 10,
            window: Duration::from_secs(60),
        };
        registry.register(make_entry("tool", "c").with_permission(perm.clone()));
        assert_eq!(registry.check_permission("tool"), perm);
    }

    #[test]
    fn test_record_call() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool", "c"));
        registry.record_call("tool");
        registry.record_call("tool");
        registry.record_call("tool");
        let entry = registry.get("tool").unwrap();
        assert_eq!(entry.call_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_record_call_missing_tool_does_not_panic() {
        let registry = ToolRegistry::new();
        registry.record_call("nonexistent"); // should not panic
    }

    #[test]
    fn test_enable_disable() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool", "c"));
        assert!(registry
            .get("tool")
            .unwrap()
            .enabled
            .load(Ordering::Relaxed));

        registry.disable("tool");
        assert!(!registry
            .get("tool")
            .unwrap()
            .enabled
            .load(Ordering::Relaxed));

        registry.enable("tool");
        assert!(registry
            .get("tool")
            .unwrap()
            .enabled
            .load(Ordering::Relaxed));
    }

    #[test]
    fn test_search_by_category() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("read_file", "filesystem"));
        registry.register(make_entry("write_file", "filesystem"));
        registry.register(make_entry("http_get", "web"));

        let results = registry.search(&ToolFilter::new().category("filesystem"));
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.category == "filesystem"));
    }

    #[test]
    fn test_search_by_tag() {
        let registry = ToolRegistry::new();
        registry.register(
            make_entry("grep", "filesystem").with_tags(vec!["search".into(), "text".into()]),
        );
        registry.register(make_entry("web_search", "web").with_tags(vec!["search".into()]));
        registry.register(make_entry("write_file", "filesystem"));

        let results = registry.search(&ToolFilter::new().tag("search"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_name_pattern() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("read_file", "filesystem"));
        registry.register(make_entry("write_file", "filesystem"));
        registry.register(make_entry("http_get", "web"));

        let results = registry.search(&ToolFilter::new().name_pattern("file"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_enabled_only() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool_a", "c"));
        registry.register(make_entry("tool_b", "c"));
        registry.disable("tool_b");

        let enabled = registry.search(&ToolFilter::new());
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].name, "tool_a");

        let all = registry.search(&ToolFilter::new().enabled_only(false));
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_search_combined_filters() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("read_file", "filesystem").with_tags(vec!["io".into()]));
        registry.register(make_entry("write_file", "filesystem").with_tags(vec!["io".into()]));
        registry.register(make_entry("read_url", "web").with_tags(vec!["io".into()]));

        let results = registry.search(
            &ToolFilter::new()
                .category("filesystem")
                .tag("io")
                .name_pattern("read"),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "read_file");
    }

    #[test]
    fn test_list_categories() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("a", "web"));
        registry.register(make_entry("b", "filesystem"));
        registry.register(make_entry("c", "web"));
        registry.register(make_entry("d", "code"));

        let cats = registry.list_categories();
        assert_eq!(cats, vec!["code", "filesystem", "web"]);
    }

    #[test]
    fn test_get_schemas_only_enabled() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool_a", "c"));
        registry.register(make_entry("tool_b", "c"));
        registry.disable("tool_b");

        let schemas = registry.get_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "tool_a");
    }

    #[test]
    fn test_builder() {
        let registry = ToolRegistry::builder()
            .tool(make_entry("t1", "c1"))
            .tool(make_entry("t2", "c2"))
            .build();

        assert_eq!(registry.len(), 2);
        assert!(registry.get("t1").is_some());
        assert!(registry.get("t2").is_some());
    }

    #[test]
    fn test_register_replaces_existing() {
        let registry = ToolRegistry::new();
        registry.register(make_entry("tool", "old_cat"));
        registry.register(make_entry("tool", "new_cat"));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("tool").unwrap().category, "new_cat");
    }

    #[test]
    fn test_default_creates_empty_registry() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_clone_shares_state() {
        let registry = ToolRegistry::new();
        let clone = registry.clone();
        registry.register(make_entry("tool", "c"));
        assert_eq!(clone.len(), 1);
    }
}
