//! Plugin system for extending agent capabilities at runtime.
//!
//! Plugins provide a structured way to bundle and manage extensions that add
//! tools, middleware, state transformers, or event handlers to a running agent.
//!
//! The [`PluginRegistry`] manages plugin lifecycle (load, activate, deactivate,
//! unload) and supports dependency resolution via topological sorting.

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};

use crate::agent::DeepAgentError;

/// Result type for plugin operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// PluginCapability
// ---------------------------------------------------------------------------

/// Describes what kind of extension a plugin provides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PluginCapability {
    /// The plugin provides one or more tools.
    ToolProvider,
    /// The plugin provides middleware.
    MiddlewareProvider,
    /// The plugin transforms agent state.
    StateTransformer,
    /// The plugin handles events.
    EventHandler,
    /// A custom capability not covered by the built-in variants.
    Custom(String),
}

impl fmt::Display for PluginCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolProvider => write!(f, "ToolProvider"),
            Self::MiddlewareProvider => write!(f, "MiddlewareProvider"),
            Self::StateTransformer => write!(f, "StateTransformer"),
            Self::EventHandler => write!(f, "EventHandler"),
            Self::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

// ---------------------------------------------------------------------------
// PluginMetadata
// ---------------------------------------------------------------------------

/// Metadata describing a plugin.
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Unique plugin name.
    pub name: String,
    /// Semver version string.
    pub version: String,
    /// Optional author.
    pub author: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Capabilities this plugin provides.
    pub capabilities: Vec<PluginCapability>,
    /// Names of other plugins this plugin depends on.
    pub dependencies: Vec<String>,
}

impl PluginMetadata {
    /// Create metadata with the given name and version.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            author: None,
            description: None,
            capabilities: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Set the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a capability.
    pub fn with_capability(mut self, cap: PluginCapability) -> Self {
        self.capabilities.push(cap);
        self
    }

    /// Add a dependency on another plugin by name.
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Serialize metadata to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "capabilities": self.capabilities.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
            "dependencies": self.dependencies,
        })
    }

    /// Check whether this plugin advertises the given capability.
    pub fn has_capability(&self, cap: &PluginCapability) -> bool {
        self.capabilities.contains(cap)
    }
}

// ---------------------------------------------------------------------------
// PluginStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a plugin instance.
#[derive(Debug, Clone, PartialEq)]
pub enum PluginStatus {
    /// The plugin has been registered but not yet loaded.
    Registered,
    /// The plugin has been loaded (initialised) but is not active.
    Loaded,
    /// The plugin is active and contributing to the agent.
    Active,
    /// The plugin has been explicitly disabled.
    Disabled,
    /// The plugin encountered an error.
    Error(String),
}

impl PluginStatus {
    /// Returns `true` if the plugin is currently active.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns `true` if the plugin is in an error state.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

impl fmt::Display for PluginStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registered => write!(f, "Registered"),
            Self::Loaded => write!(f, "Loaded"),
            Self::Active => write!(f, "Active"),
            Self::Disabled => write!(f, "Disabled"),
            Self::Error(msg) => write!(f, "Error({})", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// Trait that all plugins must implement.
///
/// The lifecycle methods are called by [`PluginInstance`] during state
/// transitions. Default implementations are no-ops so plugins only need to
/// override hooks they care about.
pub trait Plugin: Send {
    /// Returns a reference to the plugin's metadata.
    fn metadata(&self) -> &PluginMetadata;

    /// Called when the plugin is loaded. Perform one-time initialisation here.
    fn on_load(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when the plugin is being unloaded. Clean up resources here.
    fn on_unload(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when the plugin transitions to the active state.
    fn on_activate(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called when the plugin transitions from active to deactivated.
    fn on_deactivate(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PluginInstance
// ---------------------------------------------------------------------------

/// Wraps a [`Plugin`] with lifecycle status tracking.
pub struct PluginInstance {
    plugin: Box<dyn Plugin>,
    status: PluginStatus,
}

impl PluginInstance {
    /// Create a new instance wrapping the given plugin.
    /// The initial status is [`PluginStatus::Registered`].
    pub fn new(plugin: Box<dyn Plugin>) -> Self {
        Self {
            plugin,
            status: PluginStatus::Registered,
        }
    }

    /// Load the plugin. Valid from `Registered` status only.
    pub fn load(&mut self) -> Result<()> {
        match &self.status {
            PluginStatus::Registered => {}
            other => {
                let msg = format!(
                    "cannot load plugin '{}': current status is {}",
                    self.name(),
                    other
                );
                self.status = PluginStatus::Error(msg.clone());
                return Err(DeepAgentError::Other(msg));
            }
        }
        match self.plugin.on_load() {
            Ok(()) => {
                self.status = PluginStatus::Loaded;
                Ok(())
            }
            Err(e) => {
                self.status = PluginStatus::Error(e.to_string());
                Err(e)
            }
        }
    }

    /// Activate the plugin. Valid from `Loaded` or `Disabled` status.
    pub fn activate(&mut self) -> Result<()> {
        match &self.status {
            PluginStatus::Loaded | PluginStatus::Disabled => {}
            other => {
                let msg = format!(
                    "cannot activate plugin '{}': current status is {}",
                    self.name(),
                    other
                );
                self.status = PluginStatus::Error(msg.clone());
                return Err(DeepAgentError::Other(msg));
            }
        }
        match self.plugin.on_activate() {
            Ok(()) => {
                self.status = PluginStatus::Active;
                Ok(())
            }
            Err(e) => {
                self.status = PluginStatus::Error(e.to_string());
                Err(e)
            }
        }
    }

    /// Deactivate the plugin. Valid from `Active` status only.
    pub fn deactivate(&mut self) -> Result<()> {
        match &self.status {
            PluginStatus::Active => {}
            other => {
                let msg = format!(
                    "cannot deactivate plugin '{}': current status is {}",
                    self.name(),
                    other
                );
                self.status = PluginStatus::Error(msg.clone());
                return Err(DeepAgentError::Other(msg));
            }
        }
        match self.plugin.on_deactivate() {
            Ok(()) => {
                self.status = PluginStatus::Disabled;
                Ok(())
            }
            Err(e) => {
                self.status = PluginStatus::Error(e.to_string());
                Err(e)
            }
        }
    }

    /// Unload the plugin. Valid from `Loaded` or `Disabled` status.
    pub fn unload(&mut self) -> Result<()> {
        match &self.status {
            PluginStatus::Loaded | PluginStatus::Disabled => {}
            other => {
                let msg = format!(
                    "cannot unload plugin '{}': current status is {}",
                    self.name(),
                    other
                );
                self.status = PluginStatus::Error(msg.clone());
                return Err(DeepAgentError::Other(msg));
            }
        }
        match self.plugin.on_unload() {
            Ok(()) => {
                self.status = PluginStatus::Registered;
                Ok(())
            }
            Err(e) => {
                self.status = PluginStatus::Error(e.to_string());
                Err(e)
            }
        }
    }

    /// Returns the current status.
    pub fn status(&self) -> &PluginStatus {
        &self.status
    }

    /// Returns a reference to the plugin's metadata.
    pub fn metadata(&self) -> &PluginMetadata {
        self.plugin.metadata()
    }

    /// Convenience accessor for the plugin name.
    pub fn name(&self) -> &str {
        &self.plugin.metadata().name
    }
}

// ---------------------------------------------------------------------------
// PluginRegistry
// ---------------------------------------------------------------------------

/// Central registry for managing plugin instances and their lifecycles.
pub struct PluginRegistry {
    plugins: HashMap<String, PluginInstance>,
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register and immediately load a plugin.
    ///
    /// Returns an error if a plugin with the same name is already registered.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.plugins.contains_key(&name) {
            return Err(DeepAgentError::Other(format!(
                "plugin '{}' is already registered",
                name
            )));
        }
        let mut instance = PluginInstance::new(plugin);
        instance.load()?;
        self.plugins.insert(name, instance);
        Ok(())
    }

    /// Activate a registered plugin by name.
    pub fn activate(&mut self, name: &str) -> Result<()> {
        let instance = self.plugins.get_mut(name).ok_or_else(|| {
            DeepAgentError::Other(format!("plugin '{}' not found", name))
        })?;
        instance.activate()
    }

    /// Deactivate a registered plugin by name.
    pub fn deactivate(&mut self, name: &str) -> Result<()> {
        let instance = self.plugins.get_mut(name).ok_or_else(|| {
            DeepAgentError::Other(format!("plugin '{}' not found", name))
        })?;
        instance.deactivate()
    }

    /// Unregister a plugin by name. The plugin is unloaded first if necessary.
    ///
    /// Active plugins must be deactivated before they can be unregistered.
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        let instance = self.plugins.get(name).ok_or_else(|| {
            DeepAgentError::Other(format!("plugin '{}' not found", name))
        })?;
        if instance.status().is_active() {
            return Err(DeepAgentError::Other(format!(
                "cannot unregister active plugin '{}'; deactivate it first",
                name
            )));
        }
        // Unload then remove.
        let instance = self.plugins.get_mut(name).unwrap();
        // If loaded or disabled, unload. If already registered (shouldn't happen after
        // register() which auto-loads), just remove.
        match instance.status() {
            PluginStatus::Loaded | PluginStatus::Disabled => {
                instance.unload()?;
            }
            _ => {}
        }
        self.plugins.remove(name);
        Ok(())
    }

    /// Get a reference to a plugin instance by name.
    pub fn get(&self, name: &str) -> Option<&PluginInstance> {
        self.plugins.get(name)
    }

    /// Return all currently active plugin instances.
    pub fn active_plugins(&self) -> Vec<&PluginInstance> {
        self.plugins
            .values()
            .filter(|inst| inst.status().is_active())
            .collect()
    }

    /// Return all plugins that advertise the given capability.
    pub fn plugins_with_capability(&self, cap: &PluginCapability) -> Vec<&PluginInstance> {
        self.plugins
            .values()
            .filter(|inst| inst.metadata().has_capability(cap))
            .collect()
    }

    /// Resolve the dependency chain for a plugin, returning names in
    /// topological order (dependencies before dependents).
    ///
    /// Returns an error if any dependency is missing from the registry.
    pub fn resolve_dependencies(&self, name: &str) -> Result<Vec<String>> {
        let mut visited: HashMap<String, bool> = HashMap::new(); // false = in-progress, true = done
        let mut order: Vec<String> = Vec::new();
        self.visit_deps(name, &mut visited, &mut order)?;
        Ok(order)
    }

    /// Depth-first topological sort helper.
    fn visit_deps(
        &self,
        name: &str,
        visited: &mut HashMap<String, bool>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        if let Some(&done) = visited.get(name) {
            if done {
                return Ok(());
            } else {
                return Err(DeepAgentError::Other(format!(
                    "circular dependency detected involving '{}'",
                    name
                )));
            }
        }

        let instance = self.plugins.get(name).ok_or_else(|| {
            DeepAgentError::Other(format!(
                "dependency '{}' is not registered",
                name
            ))
        })?;

        visited.insert(name.to_string(), false);

        for dep in &instance.metadata().dependencies {
            self.visit_deps(dep, visited, order)?;
        }

        visited.insert(name.to_string(), true);
        order.push(name.to_string());
        Ok(())
    }

    /// Return the number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// List all plugins as JSON values.
    pub fn list(&self) -> Vec<Value> {
        self.plugins
            .values()
            .map(|inst| {
                let mut v = inst.metadata().to_json();
                v.as_object_mut()
                    .unwrap()
                    .insert("status".to_string(), json!(inst.status().to_string()));
                v
            })
            .collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SimplePlugin
// ---------------------------------------------------------------------------

/// A basic [`Plugin`] implementation useful for testing.
///
/// Tracks how many times each lifecycle hook has been called.
pub struct SimplePlugin {
    metadata: PluginMetadata,
    /// Number of times `on_load` has been called.
    pub load_count: usize,
    /// Number of times `on_unload` has been called.
    pub unload_count: usize,
    /// Number of times `on_activate` has been called.
    pub activate_count: usize,
    /// Number of times `on_deactivate` has been called.
    pub deactivate_count: usize,
}

impl SimplePlugin {
    /// Create a new simple plugin with the given name and version.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            metadata: PluginMetadata::new(name, version),
            load_count: 0,
            unload_count: 0,
            activate_count: 0,
            deactivate_count: 0,
        }
    }

    /// Add a capability.
    pub fn with_capability(mut self, cap: PluginCapability) -> Self {
        self.metadata.capabilities.push(cap);
        self
    }

    /// Add a dependency.
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.metadata.dependencies.push(dep.into());
        self
    }

    /// Set the author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.metadata.author = Some(author.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.metadata.description = Some(desc.into());
        self
    }
}

impl Plugin for SimplePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn on_load(&mut self) -> Result<()> {
        self.load_count += 1;
        Ok(())
    }

    fn on_unload(&mut self) -> Result<()> {
        self.unload_count += 1;
        Ok(())
    }

    fn on_activate(&mut self) -> Result<()> {
        self.activate_count += 1;
        Ok(())
    }

    fn on_deactivate(&mut self) -> Result<()> {
        self.deactivate_count += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple(name: &str) -> Box<SimplePlugin> {
        Box::new(SimplePlugin::new(name, "1.0.0"))
    }

    fn simple_with_caps(name: &str, caps: Vec<PluginCapability>) -> Box<SimplePlugin> {
        let mut p = SimplePlugin::new(name, "1.0.0");
        p.metadata.capabilities = caps;
        Box::new(p)
    }

    fn simple_with_deps(name: &str, deps: Vec<&str>) -> Box<SimplePlugin> {
        let mut p = SimplePlugin::new(name, "1.0.0");
        p.metadata.dependencies = deps.into_iter().map(String::from).collect();
        Box::new(p)
    }

    // -- PluginCapability display and equality --

    #[test]
    fn test_capability_display_tool_provider() {
        assert_eq!(PluginCapability::ToolProvider.to_string(), "ToolProvider");
    }

    #[test]
    fn test_capability_display_middleware_provider() {
        assert_eq!(
            PluginCapability::MiddlewareProvider.to_string(),
            "MiddlewareProvider"
        );
    }

    #[test]
    fn test_capability_display_state_transformer() {
        assert_eq!(
            PluginCapability::StateTransformer.to_string(),
            "StateTransformer"
        );
    }

    #[test]
    fn test_capability_display_event_handler() {
        assert_eq!(PluginCapability::EventHandler.to_string(), "EventHandler");
    }

    #[test]
    fn test_capability_display_custom() {
        let cap = PluginCapability::Custom("metrics".into());
        assert_eq!(cap.to_string(), "Custom(metrics)");
    }

    #[test]
    fn test_capability_equality() {
        assert_eq!(PluginCapability::ToolProvider, PluginCapability::ToolProvider);
        assert_ne!(PluginCapability::ToolProvider, PluginCapability::EventHandler);
        assert_eq!(
            PluginCapability::Custom("a".into()),
            PluginCapability::Custom("a".into())
        );
        assert_ne!(
            PluginCapability::Custom("a".into()),
            PluginCapability::Custom("b".into())
        );
    }

    // -- PluginMetadata builder and to_json --

    #[test]
    fn test_metadata_new_defaults() {
        let m = PluginMetadata::new("test", "0.1.0");
        assert_eq!(m.name, "test");
        assert_eq!(m.version, "0.1.0");
        assert!(m.author.is_none());
        assert!(m.description.is_none());
        assert!(m.capabilities.is_empty());
        assert!(m.dependencies.is_empty());
    }

    #[test]
    fn test_metadata_builder_chain() {
        let m = PluginMetadata::new("p", "2.0.0")
            .with_author("Alice")
            .with_description("A plugin")
            .with_capability(PluginCapability::ToolProvider)
            .with_capability(PluginCapability::EventHandler)
            .with_dependency("base");
        assert_eq!(m.author.as_deref(), Some("Alice"));
        assert_eq!(m.description.as_deref(), Some("A plugin"));
        assert_eq!(m.capabilities.len(), 2);
        assert_eq!(m.dependencies, vec!["base"]);
    }

    #[test]
    fn test_metadata_has_capability() {
        let m = PluginMetadata::new("p", "1.0.0")
            .with_capability(PluginCapability::ToolProvider);
        assert!(m.has_capability(&PluginCapability::ToolProvider));
        assert!(!m.has_capability(&PluginCapability::EventHandler));
    }

    #[test]
    fn test_metadata_to_json() {
        let m = PluginMetadata::new("myplugin", "1.0.0")
            .with_author("Bob")
            .with_capability(PluginCapability::ToolProvider);
        let j = m.to_json();
        assert_eq!(j["name"], "myplugin");
        assert_eq!(j["version"], "1.0.0");
        assert_eq!(j["author"], "Bob");
        assert!(j["capabilities"].as_array().unwrap().len() == 1);
    }

    // -- PluginStatus --

    #[test]
    fn test_status_is_active() {
        assert!(PluginStatus::Active.is_active());
        assert!(!PluginStatus::Loaded.is_active());
        assert!(!PluginStatus::Registered.is_active());
        assert!(!PluginStatus::Disabled.is_active());
        assert!(!PluginStatus::Error("err".into()).is_active());
    }

    #[test]
    fn test_status_is_error() {
        assert!(PluginStatus::Error("oops".into()).is_error());
        assert!(!PluginStatus::Active.is_error());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(PluginStatus::Registered.to_string(), "Registered");
        assert_eq!(PluginStatus::Loaded.to_string(), "Loaded");
        assert_eq!(PluginStatus::Active.to_string(), "Active");
        assert_eq!(PluginStatus::Disabled.to_string(), "Disabled");
        assert_eq!(
            PluginStatus::Error("bad".into()).to_string(),
            "Error(bad)"
        );
    }

    // -- Plugin trait lifecycle via PluginInstance --

    #[test]
    fn test_instance_initial_status() {
        let inst = PluginInstance::new(simple("test"));
        assert_eq!(*inst.status(), PluginStatus::Registered);
        assert_eq!(inst.name(), "test");
    }

    #[test]
    fn test_instance_load() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Loaded);
    }

    #[test]
    fn test_instance_full_lifecycle() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        inst.activate().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Active);
        inst.deactivate().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Disabled);
        inst.unload().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Registered);
    }

    #[test]
    fn test_instance_reactivate_from_disabled() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        inst.activate().unwrap();
        inst.deactivate().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Disabled);
        inst.activate().unwrap();
        assert_eq!(*inst.status(), PluginStatus::Active);
    }

    // -- Invalid transitions --

    #[test]
    fn test_instance_activate_from_registered_fails() {
        let mut inst = PluginInstance::new(simple("test"));
        let err = inst.activate();
        assert!(err.is_err());
        assert!(inst.status().is_error());
    }

    #[test]
    fn test_instance_deactivate_from_loaded_fails() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        let err = inst.deactivate();
        assert!(err.is_err());
        assert!(inst.status().is_error());
    }

    #[test]
    fn test_instance_load_twice_fails() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        let err = inst.load();
        assert!(err.is_err());
    }

    #[test]
    fn test_instance_unload_from_active_fails() {
        let mut inst = PluginInstance::new(simple("test"));
        inst.load().unwrap();
        inst.activate().unwrap();
        let err = inst.unload();
        assert!(err.is_err());
    }

    // -- PluginRegistry CRUD --

    #[test]
    fn test_registry_new_is_empty() {
        let reg = PluginRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let reg = PluginRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("alpha")).unwrap();
        assert_eq!(reg.len(), 1);
        let inst = reg.get("alpha").unwrap();
        assert_eq!(inst.name(), "alpha");
        // register auto-loads
        assert_eq!(*inst.status(), PluginStatus::Loaded);
    }

    #[test]
    fn test_registry_duplicate_registration_fails() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("dup")).unwrap();
        let err = reg.register(simple("dup"));
        assert!(err.is_err());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = PluginRegistry::new();
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("rem")).unwrap();
        reg.unregister("rem").unwrap();
        assert!(reg.is_empty());
        assert!(reg.get("rem").is_none());
    }

    #[test]
    fn test_registry_unregister_missing_fails() {
        let mut reg = PluginRegistry::new();
        assert!(reg.unregister("ghost").is_err());
    }

    #[test]
    fn test_registry_unregister_active_fails() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("act")).unwrap();
        reg.activate("act").unwrap();
        let err = reg.unregister("act");
        assert!(err.is_err());
        // Still registered.
        assert_eq!(reg.len(), 1);
    }

    // -- Activate / deactivate workflows --

    #[test]
    fn test_registry_activate_and_deactivate() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("p")).unwrap();
        reg.activate("p").unwrap();
        assert!(reg.get("p").unwrap().status().is_active());
        reg.deactivate("p").unwrap();
        assert!(!reg.get("p").unwrap().status().is_active());
    }

    #[test]
    fn test_registry_activate_missing_fails() {
        let mut reg = PluginRegistry::new();
        assert!(reg.activate("none").is_err());
    }

    #[test]
    fn test_registry_activate_already_active_fails() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("p")).unwrap();
        reg.activate("p").unwrap();
        let err = reg.activate("p");
        assert!(err.is_err());
    }

    #[test]
    fn test_registry_deactivate_missing_fails() {
        let mut reg = PluginRegistry::new();
        assert!(reg.deactivate("none").is_err());
    }

    // -- active_plugins --

    #[test]
    fn test_registry_active_plugins() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("a")).unwrap();
        reg.register(simple("b")).unwrap();
        reg.register(simple("c")).unwrap();
        reg.activate("a").unwrap();
        reg.activate("c").unwrap();
        let active = reg.active_plugins();
        assert_eq!(active.len(), 2);
        let names: Vec<&str> = active.iter().map(|i| i.name()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"c"));
    }

    // -- Capability filtering --

    #[test]
    fn test_registry_plugins_with_capability() {
        let mut reg = PluginRegistry::new();
        reg.register(simple_with_caps("tools1", vec![PluginCapability::ToolProvider]))
            .unwrap();
        reg.register(simple_with_caps("mw1", vec![PluginCapability::MiddlewareProvider]))
            .unwrap();
        reg.register(simple_with_caps(
            "both",
            vec![PluginCapability::ToolProvider, PluginCapability::MiddlewareProvider],
        ))
        .unwrap();

        let tool_plugins = reg.plugins_with_capability(&PluginCapability::ToolProvider);
        assert_eq!(tool_plugins.len(), 2);

        let mw_plugins = reg.plugins_with_capability(&PluginCapability::MiddlewareProvider);
        assert_eq!(mw_plugins.len(), 2);

        let event_plugins = reg.plugins_with_capability(&PluginCapability::EventHandler);
        assert_eq!(event_plugins.len(), 0);
    }

    // -- Dependency resolution --

    #[test]
    fn test_resolve_dependencies_no_deps() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("leaf")).unwrap();
        let order = reg.resolve_dependencies("leaf").unwrap();
        assert_eq!(order, vec!["leaf"]);
    }

    #[test]
    fn test_resolve_dependencies_simple_chain() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("a")).unwrap();
        reg.register(simple_with_deps("b", vec!["a"])).unwrap();
        reg.register(simple_with_deps("c", vec!["b"])).unwrap();
        let order = reg.resolve_dependencies("c").unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_resolve_dependencies_diamond() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("base")).unwrap();
        reg.register(simple_with_deps("left", vec!["base"])).unwrap();
        reg.register(simple_with_deps("right", vec!["base"])).unwrap();
        reg.register(simple_with_deps("top", vec!["left", "right"]))
            .unwrap();
        let order = reg.resolve_dependencies("top").unwrap();
        // base must appear before left and right; left and right before top.
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap();
        assert!(pos("base") < pos("left"));
        assert!(pos("base") < pos("right"));
        assert!(pos("left") < pos("top"));
        assert!(pos("right") < pos("top"));
    }

    #[test]
    fn test_resolve_dependencies_missing_dep() {
        let mut reg = PluginRegistry::new();
        reg.register(simple_with_deps("orphan", vec!["missing"]))
            .unwrap();
        let err = reg.resolve_dependencies("orphan");
        assert!(err.is_err());
    }

    // -- SimplePlugin tracking --

    #[test]
    fn test_simple_plugin_tracks_calls() {
        let mut p = SimplePlugin::new("tracker", "0.1.0");
        assert_eq!(p.load_count, 0);
        p.on_load().unwrap();
        assert_eq!(p.load_count, 1);
        p.on_activate().unwrap();
        assert_eq!(p.activate_count, 1);
        p.on_deactivate().unwrap();
        assert_eq!(p.deactivate_count, 1);
        p.on_unload().unwrap();
        assert_eq!(p.unload_count, 1);
        // Call again
        p.on_load().unwrap();
        assert_eq!(p.load_count, 2);
    }

    #[test]
    fn test_simple_plugin_with_builder() {
        let p = SimplePlugin::new("ext", "1.0.0")
            .with_capability(PluginCapability::ToolProvider)
            .with_dependency("core")
            .with_author("Dev")
            .with_description("An extension");
        let m = p.metadata();
        assert_eq!(m.name, "ext");
        assert_eq!(m.author.as_deref(), Some("Dev"));
        assert_eq!(m.description.as_deref(), Some("An extension"));
        assert!(m.has_capability(&PluginCapability::ToolProvider));
        assert_eq!(m.dependencies, vec!["core"]);
    }

    // -- Registry list --

    #[test]
    fn test_registry_list() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("x")).unwrap();
        reg.register(simple("y")).unwrap();
        let items = reg.list();
        assert_eq!(items.len(), 2);
        // Each item should have a status field.
        for item in &items {
            assert!(item.get("status").is_some());
            assert!(item.get("name").is_some());
        }
    }

    // -- Unregister after deactivate --

    #[test]
    fn test_unregister_after_deactivate() {
        let mut reg = PluginRegistry::new();
        reg.register(simple("p")).unwrap();
        reg.activate("p").unwrap();
        reg.deactivate("p").unwrap();
        reg.unregister("p").unwrap();
        assert!(reg.is_empty());
    }
}
