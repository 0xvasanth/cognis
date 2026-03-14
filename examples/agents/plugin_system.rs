//! Plugin System Example
//!
//! Demonstrates SimplePlugin creation, PluginRegistry lifecycle management,
//! capability filtering, and dependency resolution.

#[path = "../shared.rs"]
mod shared;

use cognisagent::plugins::{Plugin, PluginCapability, PluginRegistry, SimplePlugin};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Plugin System ===\n");

    // Create plugins with capabilities
    let tool_plugin = SimplePlugin::new("web-search", "1.0.0")
        .with_capability(PluginCapability::ToolProvider)
        .with_description("Web search tools for information retrieval");

    let mw_plugin = SimplePlugin::new("rate-limiter", "0.5.0")
        .with_capability(PluginCapability::MiddlewareProvider)
        .with_description("Rate limiting middleware for API calls");

    let multi_plugin = SimplePlugin::new("observability", "2.0.0")
        .with_capability(PluginCapability::EventHandler)
        .with_capability(PluginCapability::MiddlewareProvider)
        .with_capability(PluginCapability::Custom("telemetry".into()))
        .with_description("Tracing, metrics, and event logging");

    // Register plugins
    let mut registry = PluginRegistry::new();
    registry.register(Box::new(tool_plugin)).unwrap();
    registry.register(Box::new(mw_plugin)).unwrap();
    registry.register(Box::new(multi_plugin)).unwrap();
    println!("Registered {} plugins", registry.len());

    // Activate and check status
    registry.activate("web-search").unwrap();
    registry.activate("observability").unwrap();
    let active = registry.active_plugins();
    println!(
        "Active: {}",
        active
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Deactivate and reactivate
    registry.deactivate("web-search").unwrap();
    println!(
        "web-search: {}",
        registry.get("web-search").unwrap().status()
    );
    registry.activate("web-search").unwrap();

    // Capability filtering
    for cap in &[
        PluginCapability::ToolProvider,
        PluginCapability::MiddlewareProvider,
        PluginCapability::EventHandler,
    ] {
        let plugins = registry.plugins_with_capability(cap);
        let names: Vec<_> = plugins.iter().map(|p| p.name()).collect();
        println!("{:?}: {:?}", cap, names);
    }

    // Dependency resolution
    let mut dep_registry = PluginRegistry::new();
    dep_registry
        .register(Box::new(
            SimplePlugin::new("core-runtime", "1.0.0")
                .with_capability(PluginCapability::StateTransformer),
        ))
        .unwrap();
    dep_registry
        .register(Box::new(
            SimplePlugin::new("llm-provider", "1.0.0").with_dependency("core-runtime"),
        ))
        .unwrap();
    dep_registry
        .register(Box::new(
            SimplePlugin::new("agent-tools", "1.0.0")
                .with_dependency("core-runtime")
                .with_dependency("llm-provider"),
        ))
        .unwrap();

    let order = dep_registry.resolve_dependencies("agent-tools").unwrap();
    println!("\nDependency order for 'agent-tools': {:?}", order);

    // Plugin listing
    println!("\nAll plugins:");
    for item in &registry.list() {
        println!(
            "  {} v{} [{}]",
            item["name"].as_str().unwrap_or("?"),
            item["version"].as_str().unwrap_or("?"),
            item["status"].as_str().unwrap_or("?"),
        );
    }

    // Unregister
    registry.deactivate("web-search").unwrap();
    registry.unregister("web-search").unwrap();
    println!("\nAfter unregister: {} plugins", registry.len());

    // LLM demo
    let model = shared::get_chat_model(vec![
        "Plugins extend agents by providing modular tools, middleware, and event handlers at runtime.".into(),
    ]);
    let result = model
        ._generate(
            &[cognis_core::messages::Message::human(
                "How do plugins extend AI agent capabilities?",
            )],
            None,
        )
        .await?;
    if let Some(gen) = result.generations.first() {
        println!("\nLLM: {}", gen.message.content().text());
    }

    Ok(())
}
