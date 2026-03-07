//! Plugin System Example
//!
//! Demonstrates the DeepAgents plugin system for extending agent capabilities
//! at runtime. Plugins provide a structured way to bundle and manage extensions
//! that add tools, middleware, state transformers, or event handlers.
//!
//! Features shown:
//! - SimplePlugin creation with capabilities and metadata
//! - PluginRegistry for managing plugin lifecycles
//! - Plugin activation and deactivation
//! - Capability filtering to find plugins by feature
//! - Dependency resolution with topological sorting
//! - Plugin listing and inspection
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example plugin_system`

use deepagents::plugins::{
    Plugin, PluginCapability, PluginRegistry, SimplePlugin,
};

fn main() {
    println!("=== Plugin System Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating Plugins with Capabilities
    // -----------------------------------------------------------------------
    println!("--- 1. Creating Plugins ---");
    println!("Plugins are created with metadata, capabilities, and optional dependencies.\n");

    let tool_plugin = SimplePlugin::new("web-search", "1.0.0")
        .with_capability(PluginCapability::ToolProvider)
        .with_author("RustChain Team")
        .with_description("Provides web search tools for information retrieval");

    let middleware_plugin = SimplePlugin::new("rate-limiter", "0.5.0")
        .with_capability(PluginCapability::MiddlewareProvider)
        .with_author("RustChain Team")
        .with_description("Rate limiting middleware for API calls");

    let multi_plugin = SimplePlugin::new("observability", "2.0.0")
        .with_capability(PluginCapability::EventHandler)
        .with_capability(PluginCapability::MiddlewareProvider)
        .with_capability(PluginCapability::Custom("telemetry".to_string()))
        .with_author("Monitoring Team")
        .with_description("Observability plugin with tracing, metrics, and event logging");

    println!("Created plugins:");
    println!("  - {} v{}", tool_plugin.metadata().name, tool_plugin.metadata().version);
    println!("  - {} v{}", middleware_plugin.metadata().name, middleware_plugin.metadata().version);
    println!("  - {} v{}", multi_plugin.metadata().name, multi_plugin.metadata().version);

    // Show metadata JSON
    println!("\nPlugin metadata (JSON):");
    println!(
        "  {}",
        serde_json::to_string_pretty(&multi_plugin.metadata().to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 2. Plugin Registry — Registration and Lifecycle
    // -----------------------------------------------------------------------
    println!("\n--- 2. Plugin Registry ---");
    println!("The registry manages plugin registration, loading, and lifecycle.\n");

    let mut registry = PluginRegistry::new();

    // Register plugins (auto-loads them)
    registry.register(Box::new(tool_plugin)).unwrap();
    registry.register(Box::new(middleware_plugin)).unwrap();
    registry.register(Box::new(multi_plugin)).unwrap();

    println!(
        "Registered {} plugins in the registry",
        registry.len()
    );

    // Check initial status
    let web_search = registry.get("web-search").unwrap();
    println!(
        "  'web-search' status: {} (auto-loaded on register)",
        web_search.status()
    );

    // -----------------------------------------------------------------------
    // 3. Activating and Deactivating Plugins
    // -----------------------------------------------------------------------
    println!("\n--- 3. Plugin Activation ---");
    println!("Plugins follow a lifecycle: Registered -> Loaded -> Active -> Disabled.\n");

    // Activate plugins
    registry.activate("web-search").unwrap();
    registry.activate("observability").unwrap();
    println!("Activated 'web-search' and 'observability'");

    // Check active plugins
    let active = registry.active_plugins();
    println!(
        "\nActive plugins ({}):",
        active.len()
    );
    for plugin in &active {
        println!(
            "  - {} (status: {})",
            plugin.name(),
            plugin.status()
        );
    }

    // rate-limiter is loaded but not active
    let rl = registry.get("rate-limiter").unwrap();
    println!(
        "\n'rate-limiter' status: {} (registered but not activated)",
        rl.status()
    );

    // Deactivate a plugin
    registry.deactivate("web-search").unwrap();
    println!("\nDeactivated 'web-search'");
    let ws = registry.get("web-search").unwrap();
    println!("  'web-search' status: {}", ws.status());

    // Reactivate
    registry.activate("web-search").unwrap();
    println!("  Reactivated 'web-search': {}", registry.get("web-search").unwrap().status());

    // -----------------------------------------------------------------------
    // 4. Capability Filtering
    // -----------------------------------------------------------------------
    println!("\n--- 4. Capability Filtering ---");
    println!("Find plugins that provide specific capabilities.\n");

    let tool_providers = registry.plugins_with_capability(&PluginCapability::ToolProvider);
    println!(
        "ToolProvider plugins ({}):",
        tool_providers.len()
    );
    for p in &tool_providers {
        println!("  - {}", p.name());
    }

    let mw_providers = registry.plugins_with_capability(&PluginCapability::MiddlewareProvider);
    println!(
        "\nMiddlewareProvider plugins ({}):",
        mw_providers.len()
    );
    for p in &mw_providers {
        println!("  - {}", p.name());
    }

    let event_handlers = registry.plugins_with_capability(&PluginCapability::EventHandler);
    println!(
        "\nEventHandler plugins ({}):",
        event_handlers.len()
    );
    for p in &event_handlers {
        println!("  - {}", p.name());
    }

    let custom = registry.plugins_with_capability(&PluginCapability::Custom("telemetry".to_string()));
    println!(
        "\nCustom('telemetry') plugins ({}):",
        custom.len()
    );
    for p in &custom {
        println!("  - {}", p.name());
    }

    let state_transformers = registry.plugins_with_capability(&PluginCapability::StateTransformer);
    println!(
        "\nStateTransformer plugins ({}): (none registered)",
        state_transformers.len()
    );

    // -----------------------------------------------------------------------
    // 5. Dependency Resolution
    // -----------------------------------------------------------------------
    println!("\n--- 5. Dependency Resolution ---");
    println!("Resolve plugin dependencies in topological order.\n");

    // Create a fresh registry with dependency chains
    let mut dep_registry = PluginRegistry::new();

    let base = SimplePlugin::new("core-runtime", "1.0.0")
        .with_capability(PluginCapability::StateTransformer)
        .with_description("Core runtime providing basic agent state management");

    let llm_provider = SimplePlugin::new("llm-provider", "1.0.0")
        .with_capability(PluginCapability::ToolProvider)
        .with_dependency("core-runtime")
        .with_description("LLM provider depending on core runtime");

    let agent_tools = SimplePlugin::new("agent-tools", "1.0.0")
        .with_capability(PluginCapability::ToolProvider)
        .with_dependency("core-runtime")
        .with_dependency("llm-provider")
        .with_description("Agent tools depending on both core runtime and LLM provider");

    dep_registry.register(Box::new(base)).unwrap();
    dep_registry.register(Box::new(llm_provider)).unwrap();
    dep_registry.register(Box::new(agent_tools)).unwrap();

    // Resolve dependencies for agent-tools
    let order = dep_registry.resolve_dependencies("agent-tools").unwrap();
    println!("Dependency resolution for 'agent-tools':");
    println!("  Activation order: {:?}", order);
    println!("  (dependencies are listed before their dependents)");

    // Resolve for a plugin with no dependencies
    let order = dep_registry.resolve_dependencies("core-runtime").unwrap();
    println!("\nDependency resolution for 'core-runtime':");
    println!("  Activation order: {:?} (no dependencies)", order);

    // -----------------------------------------------------------------------
    // 6. Listing All Plugins
    // -----------------------------------------------------------------------
    println!("\n--- 6. Plugin Listing ---");
    println!("List all registered plugins with their metadata and status.\n");

    let items = registry.list();
    println!("All plugins in main registry:");
    for item in &items {
        println!(
            "  - {} v{} [{}] - {}",
            item["name"].as_str().unwrap_or("?"),
            item["version"].as_str().unwrap_or("?"),
            item["status"].as_str().unwrap_or("?"),
            item["description"].as_str().unwrap_or("(no description)"),
        );
        if let Some(caps) = item["capabilities"].as_array() {
            let cap_strs: Vec<&str> = caps.iter().filter_map(|c| c.as_str()).collect();
            println!("    capabilities: {:?}", cap_strs);
        }
    }

    // -----------------------------------------------------------------------
    // 7. Unregistering a Plugin
    // -----------------------------------------------------------------------
    println!("\n--- 7. Unregistering Plugins ---");
    println!("Plugins must be deactivated before they can be unregistered.\n");

    // Deactivate first, then unregister
    registry.deactivate("web-search").unwrap();
    registry.unregister("web-search").unwrap();
    println!("Unregistered 'web-search'");
    println!(
        "Registry now has {} plugins",
        registry.len()
    );
    println!(
        "'web-search' found: {}",
        registry.get("web-search").is_some()
    );

    println!("\n=== Plugin System Example Complete ===");
}
