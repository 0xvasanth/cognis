//! Health Monitoring Example
//!
//! Sets up health checks for an agent system (disk, memory, backend, LLM)
//! and prints a unified health report.
//!
//! Run with: `cargo run -p cognis-examples --example health_monitoring`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;
use std::time::Duration;

use cognisagent::health::{
    BackendHealthCheck, DiskSpaceCheck, HealthCheck, HealthEndpoint, HealthMonitor, HealthStatus,
    MemoryHealthCheck, ToolHealthCheck,
};

/// Custom check that verifies LLM connectivity.
struct LlmHealthCheck {
    model: Arc<dyn cognis_core::language_models::chat_model::BaseChatModel>,
}

#[async_trait::async_trait]
impl HealthCheck for LlmHealthCheck {
    async fn check(&self) -> cognisagent::health::ComponentHealth {
        let msgs = vec![cognis_core::messages::Message::human("ping")];
        match self.model._generate(&msgs, None).await {
            Ok(r)
                if r.generations
                    .first()
                    .map_or(true, |g| g.message.content().text().is_empty()) =>
            {
                cognisagent::health::ComponentHealth::unhealthy("llm", "empty response")
            }
            Ok(_) => cognisagent::health::ComponentHealth::healthy("llm"),
            Err(e) => cognisagent::health::ComponentHealth::unhealthy("llm", &e.to_string()),
        }
    }

    fn component_name(&self) -> &str {
        "llm"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Health Monitoring ===\n");

    // Build a health monitor with checks for every subsystem
    let model = shared::get_chat_model(vec!["pong".into()]);

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
        .with_check(Arc::new(ToolHealthCheck::with_sync_checker(
            vec!["search".into(), "calculator".into()],
            |_| Ok(()),
        )))
        .with_check(Arc::new(LlmHealthCheck { model }))
        .with_timeout(Duration::from_secs(5))
        .build();

    // Run all checks
    let report = monitor.check_all().await;

    println!(
        "Status: {}  ({} checks, {:?})",
        report.overall_status.label(),
        report.components.len(),
        report.duration
    );
    for c in &report.components {
        let detail = match c.status.message() {
            Some(msg) => format!(" — {msg}"),
            None => String::new(),
        };
        println!("  {:12} {}{}", c.name, c.status.label(), detail);
    }

    // HTTP-ready response
    println!("\nHTTP status: {}", HealthEndpoint::status_code(&report));

    // Show what happens when a component fails
    println!("\n--- Failure scenario ---");
    let failing = HealthMonitor::builder()
        .with_check(Arc::new(MemoryHealthCheck::always_healthy()))
        .with_check(Arc::new(BackendHealthCheck::new("cache", || async {
            Err("connection refused".into())
        })))
        .build();

    let bad_report = failing.check_all().await;
    println!(
        "Status: {} (HTTP {})",
        bad_report.overall_status.label(),
        HealthEndpoint::status_code(&bad_report)
    );
    for c in bad_report.unhealthy_components() {
        let msg = c.status.message().unwrap_or("unknown");
        println!("  FAIL {:12} {}", c.name, msg);
    }

    Ok(())
}
