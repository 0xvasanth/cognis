//! What you'll learn:
//!   How `ToolOrchestrator` topo-sorts a declared DAG of tool calls
//!   into batches and runs each batch concurrently — independent
//!   steps overlap, dependent steps wait.
//!
//! Why this matters:
//!   When an agent decides "fetch a price from each of three vendors
//!   and pick the cheapest", you don't want them to run sequentially.
//!   The orchestrator lets you describe the dependency graph once
//!   and get the parallelism for free, with a single
//!   `max_concurrency` knob to keep the third-party API happy.
//!
//! Scenario:
//!   A price-comparison flow. Three "vendor" stubs each fetch a
//!   price for the same SKU; a fourth step depends on all three
//!   completing and picks the lowest. With a sequential plan the
//!   total wait would be ~240ms — the orchestrator runs the three
//!   fetches in parallel so the elapsed time is closer to ~100ms.
//!
//! Run with:
//!   cargo run -p cognis-examples --example tools_orchestrator
//!
//! Sample output (against ollama / llama3.1):
//!   succeeded:  true
//!   elapsed:    102.560083ms  (sequential would be ~300ms)
//!
//!   === per-step results ===
//!     decide: Content(Object {"chosen_vendor": String("acme")})
//!     p_acme: Content(Object {"price_cents": Number(1299), "sku": String("SKU-42"), "vendor": String("acme")})
//!     p_globex: Content(Object {"price_cents": Number(1499), "sku": String("SKU-42"), "vendor": String("globex")})
//!     p_initech: Content(Object {"price_cents": Number(1399), "sku": String("SKU-42"), "vendor": String("initech")})

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis::{ExecutionPlan, ToolOrchestrator, ToolStep};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};
use serde_json::json;

/// Stand-in for a real "fetch price from vendor" call. The latency
/// is what makes parallelism visible in the elapsed-time print.
struct VendorPrice {
    vendor: &'static str,
    latency_ms: u64,
    price_cents: u32,
}

#[async_trait]
impl Tool for VendorPrice {
    fn name(&self) -> &str {
        self.vendor
    }
    fn description(&self) -> &str {
        "Fetch the current price from a vendor."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        None
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        tokio::time::sleep(std::time::Duration::from_millis(self.latency_ms)).await;
        Ok(ToolOutput::Content(json!({
            "vendor": self.vendor,
            "sku": input.into_json(),
            "price_cents": self.price_cents,
        })))
    }
}

/// Aggregator step — runs after the three vendor fetches and picks
/// the cheapest. In real code this would consume the prior outputs;
/// here it's a fast sentinel so the timing shows the join.
struct PickCheapest;

#[async_trait]
impl Tool for PickCheapest {
    fn name(&self) -> &str {
        "pick_cheapest"
    }
    fn description(&self) -> &str {
        "Compare vendor prices and pick the lowest."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        None
    }
    async fn _run(&self, _input: ToolInput) -> Result<ToolOutput> {
        Ok(ToolOutput::Content(json!({"chosen_vendor": "acme"})))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let orch = ToolOrchestrator::new()
        .register(Arc::new(VendorPrice {
            vendor: "acme",
            latency_ms: 100,
            price_cents: 1299,
        }))
        .register(Arc::new(VendorPrice {
            vendor: "globex",
            latency_ms: 100,
            price_cents: 1499,
        }))
        .register(Arc::new(VendorPrice {
            vendor: "initech",
            latency_ms: 100,
            price_cents: 1399,
        }))
        .register(Arc::new(PickCheapest))
        .with_max_concurrency(4);

    // Declare the DAG: three independent fetches, then a join.
    let plan =
        ExecutionPlan::new()
            .step(ToolStep::new(
                "p_acme",
                "acme",
                ToolInput::Text("SKU-42".into()),
            ))
            .step(ToolStep::new(
                "p_globex",
                "globex",
                ToolInput::Text("SKU-42".into()),
            ))
            .step(ToolStep::new(
                "p_initech",
                "initech",
                ToolInput::Text("SKU-42".into()),
            ))
            .step(
                ToolStep::new("decide", "pick_cheapest", ToolInput::Text("compare".into()))
                    .after(["p_acme", "p_globex", "p_initech"]),
            );

    let t0 = Instant::now();
    let result = orch.run(plan).await?;
    let elapsed = t0.elapsed();

    println!("succeeded:  {}", result.fully_succeeded());
    println!("elapsed:    {elapsed:?}  (sequential would be ~300ms)");
    println!("\n=== per-step results ===");
    let mut ids: Vec<_> = result.results.keys().collect();
    ids.sort();
    for id in ids {
        println!("  {id}: {:?}", result.results[id]);
    }
    Ok(())
}
