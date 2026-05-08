//! ToolOrchestrator — declare a DAG of tool calls; the orchestrator
//! topo-sorts into batches and runs each batch concurrently.
//!
//! Diamond: fetch_a + fetch_b run in parallel, then merge runs once
//! both finish.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use cognis::prelude::*;
use cognis::{ExecutionPlan, ToolOrchestrator, ToolStep};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};
use serde_json::json;

struct Slow(&'static str, u64);
#[async_trait]
impl Tool for Slow {
    fn name(&self) -> &str { self.0 }
    fn description(&self) -> &str { "slow stub tool" }
    fn args_schema(&self) -> Option<serde_json::Value> { None }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        tokio::time::sleep(std::time::Duration::from_millis(self.1)).await;
        Ok(ToolOutput::Content(json!({
            "tool": self.0,
            "input": input.into_json(),
        })))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let orch = ToolOrchestrator::new()
        .register(Arc::new(Slow("fetch_a", 80)))
        .register(Arc::new(Slow("fetch_b", 80)))
        .register(Arc::new(Slow("merge", 20)))
        .with_max_concurrency(4);

    let plan = ExecutionPlan::new()
        .step(ToolStep::new("a", "fetch_a", ToolInput::Text("doc-1".into())))
        .step(ToolStep::new("b", "fetch_b", ToolInput::Text("doc-2".into())))
        .step(
            ToolStep::new("m", "merge", ToolInput::Text("combine results".into()))
                .after(["a", "b"]),
        );

    let t0 = Instant::now();
    let result = orch.run(plan).await?;
    let elapsed = t0.elapsed();

    println!("succeeded: {}", result.fully_succeeded());
    println!("elapsed:   {elapsed:?}  (sequential would be ~180ms)");
    println!("\n=== per-step results ===");
    let mut ids: Vec<_> = result.results.keys().collect();
    ids.sort();
    for id in ids {
        println!("  {id}: {:?}", result.results[id]);
    }
    Ok(())
}
