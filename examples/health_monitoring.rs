//! Health Monitoring Example
//!
//! Demonstrates the DeepAgents health monitoring system:
//! - Creating health checks (`DiskSpaceCheck`, `MemoryHealthCheck`, `BackendHealthCheck`)
//! - Building a `HealthMonitor` with the builder pattern
//! - Running all checks and inspecting the `HealthReport`
//! - Using `HealthEndpoint` for HTTP-ready health responses
//! - Checking individual components by name
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example health_monitoring`

mod shared;
use std::sync::Arc;
use std::time::Duration;

use cognisagent::health::{
    BackendHealthCheck, ComponentHealth, DiskSpaceCheck, HealthCheck, HealthEndpoint,
    HealthMonitor, HealthStatus, MemoryHealthCheck, ToolHealthCheck,
};

// ---------------------------------------------------------------------------
// Custom health check implementation
// ---------------------------------------------------------------------------

/// A custom health check that verifies configuration is loaded.
struct ConfigHealthCheck {
    config_loaded: bool,
}

impl ConfigHealthCheck {
    fn new(config_loaded: bool) -> Self {
        Self { config_loaded }
    }
}

#[async_trait::async_trait]
impl HealthCheck for ConfigHealthCheck {
    async fn check(&self) -> ComponentHealth {
        if self.config_loaded {
            ComponentHealth::healthy("config")
        } else {
            ComponentHealth::unhealthy("config", "configuration file not found")
        }
    }

    fn component_name(&self) -> &str {
        "config"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Health Monitoring Example ===\n");

    // -----------------------------------------------------------------------
    // 1. DiskSpaceCheck
    // -----------------------------------------------------------------------
    println!("--- 1. DiskSpaceCheck ---");
    println!("Checks available disk space against a threshold.\n");

    // Create a DiskSpaceCheck with a custom space function for reproducible output
    let disk_check = DiskSpaceCheck::with_space_fn("/", 10_000_000_000, || {
        // Simulate: 50 GB available out of 500 GB total
        Ok((50_000_000_000, 500_000_000_000))
    });

    let result = disk_check.check().await;
    println!("  Component: {}", result.name);
    println!(
        "  Status: {} ({})",
        result.status.label(),
        status_detail(&result.status)
    );
    if let Some(latency) = result.latency {
        println!("  Latency: {:?}", latency);
    }
    for (key, value) in &result.details {
        println!("  {}: {}", key, value);
    }
    println!();

    // Disk check with low space (degraded)
    let low_disk = DiskSpaceCheck::with_space_fn("/data", 100_000_000_000, || {
        // Simulate: 60 GB available but threshold is 100 GB
        Ok((60_000_000_000, 500_000_000_000))
    });

    let low_result = low_disk.check().await;
    println!(
        "  Low disk status: {} - {}",
        low_result.status.label(),
        status_detail(&low_result.status)
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. MemoryHealthCheck
    // -----------------------------------------------------------------------
    println!("--- 2. MemoryHealthCheck ---");
    println!("Verifies memory storage is readable and writable.\n");

    let memory_check = MemoryHealthCheck::always_healthy();
    let mem_result = memory_check.check().await;
    println!("  Component: {}", mem_result.name);
    println!("  Status: {}\n", mem_result.status.label());

    // -----------------------------------------------------------------------
    // 3. BackendHealthCheck
    // -----------------------------------------------------------------------
    println!("--- 3. BackendHealthCheck ---");
    println!("Checks backend connectivity.\n");

    let backend_check = BackendHealthCheck::new("state_backend", || async {
        // Simulate a successful backend check
        Ok(())
    });

    let backend_result = backend_check.check().await;
    println!("  Component: {}", backend_result.name);
    println!("  Status: {}", backend_result.status.label());
    if let Some(latency) = backend_result.latency {
        println!("  Latency: {:?}", latency);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. ToolHealthCheck
    // -----------------------------------------------------------------------
    println!("--- 4. ToolHealthCheck ---");
    println!("Validates that registered tools are callable.\n");

    let tool_check = ToolHealthCheck::with_sync_checker(
        vec![
            "search".to_string(),
            "calculator".to_string(),
            "file_reader".to_string(),
        ],
        |tool_name| {
            // Simulate: all tools are available
            if tool_name == "file_reader" {
                Err("permission denied".to_string())
            } else {
                Ok(())
            }
        },
    );

    let tool_result = tool_check.check().await;
    println!("  Component: {}", tool_result.name);
    println!(
        "  Status: {} - {}",
        tool_result.status.label(),
        status_detail(&tool_result.status)
    );
    for (key, value) in &tool_result.details {
        println!("  {}: {}", key, value);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. Custom health check
    // -----------------------------------------------------------------------
    println!("--- 5. Custom HealthCheck ---");
    println!("Implement the HealthCheck trait for custom checks.\n");

    let config_ok = ConfigHealthCheck::new(true);
    let config_fail = ConfigHealthCheck::new(false);

    let ok_result = config_ok.check().await;
    let fail_result = config_fail.check().await;

    println!("  Config loaded: {}", ok_result.status.label());
    println!(
        "  Config missing: {} - {}\n",
        fail_result.status.label(),
        status_detail(&fail_result.status)
    );

    // -----------------------------------------------------------------------
    // 6. HealthMonitor with builder
    // -----------------------------------------------------------------------
    println!("--- 6. HealthMonitor ---");
    println!("Aggregates multiple checks into a unified report.\n");

    let monitor = HealthMonitor::builder()
        .with_check(Arc::new(DiskSpaceCheck::with_space_fn(
            "/",
            10_000_000_000,
            || Ok((80_000_000_000, 500_000_000_000)),
        )))
        .with_check(Arc::new(MemoryHealthCheck::always_healthy()))
        .with_check(Arc::new(BackendHealthCheck::new("database", || async {
            Ok(())
        })))
        .with_check(Arc::new(ConfigHealthCheck::new(true)))
        .with_timeout(Duration::from_secs(5))
        .build();

    println!("Monitor has {} registered checks", monitor.check_count());

    let report = monitor.check_all().await;

    println!("Overall status: {}", report.overall_status.label());
    println!("Is healthy: {}", report.is_healthy());
    println!("Total check duration: {:?}", report.duration);
    println!("Components:");
    for component in &report.components {
        println!(
            "  - {} : {} (latency: {:?})",
            component.name,
            component.status.label(),
            component.latency,
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 7. HealthEndpoint for HTTP responses
    // -----------------------------------------------------------------------
    println!("--- 7. HealthEndpoint ---");
    println!("Format health reports for HTTP API consumption.\n");

    let status_code = HealthEndpoint::status_code(&report);
    let json_response = HealthEndpoint::to_json(&report);

    println!("  HTTP Status Code: {}", status_code);
    println!(
        "  JSON Response:\n  {}",
        serde_json::to_string_pretty(&json_response)?
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. Check individual component
    // -----------------------------------------------------------------------
    println!("--- 8. Individual Component Check ---");
    println!("Query health of a specific component by name.\n");

    if let Some(disk_health) = monitor.check_component("disk_space").await {
        println!("  disk_space: {}", disk_health.status.label());
    } else {
        println!("  disk_space: not found");
    }

    if let Some(mem_health) = monitor.check_component("memory").await {
        println!("  memory: {}", mem_health.status.label());
    }

    if let Some(missing) = monitor.check_component("nonexistent").await {
        println!("  nonexistent: {}", missing.status.label());
    } else {
        println!("  nonexistent: check not registered");
    }

    // -----------------------------------------------------------------------
    // 9. Unhealthy scenario
    // -----------------------------------------------------------------------
    println!("\n--- 9. Unhealthy Scenario ---");
    println!("Demonstrate a report with failing components.\n");

    let unhealthy_monitor = HealthMonitor::builder()
        .with_check(Arc::new(MemoryHealthCheck::always_healthy()))
        .with_check(Arc::new(BackendHealthCheck::new("failing_db", || async {
            Err("connection refused".to_string())
        })))
        .build();

    let unhealthy_report = unhealthy_monitor.check_all().await;
    println!(
        "  Overall: {} - {}",
        unhealthy_report.overall_status.label(),
        status_detail(&unhealthy_report.overall_status)
    );
    println!(
        "  HTTP status: {}",
        HealthEndpoint::status_code(&unhealthy_report)
    );

    let unhealthy_components = unhealthy_report.unhealthy_components();
    println!("  Unhealthy components: {}", unhealthy_components.len());
    for c in unhealthy_components {
        println!("    - {}: {}", c.name, status_detail(&c.status));
    }

    // -----------------------------------------------------------------------
    // 10. Real LLM Demo — LLM health check
    // -----------------------------------------------------------------------
    println!("\n--- 10. Real LLM Health Check Demo ---");
    println!("Verify LLM connectivity by sending a simple prompt.\n");

    let model = shared::get_chat_model(vec![
        "LLM health check passed. Model is responsive and generating coherent output.".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Respond with a brief health status confirmation.",
    )];
    let llm_result = model._generate(&messages, None).await?;
    if let Some(gen) = llm_result.generations.first() {
        let response = gen.message.content().text();
        println!(
            "  LLM Health Check: {}",
            if response.is_empty() {
                "UNHEALTHY (empty response)"
            } else {
                "HEALTHY"
            }
        );
        println!("  Response: {}", response);
    }

    println!("\n=== Health Monitoring Example Complete ===");
    Ok(())
}

/// Helper to extract the status detail message.
fn status_detail(status: &HealthStatus) -> String {
    match status.message() {
        Some(msg) => msg.to_string(),
        None => "OK".to_string(),
    }
}
