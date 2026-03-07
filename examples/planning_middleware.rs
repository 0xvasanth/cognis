//! Planning Middleware Example
//!
//! Demonstrates the planning middleware from deepagents: creating plans with
//! steps and dependencies, tracking progress, updating step status, finding
//! ready steps, and using the SimplePlanningStrategy.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example planning_middleware`

use std::sync::Arc;

use serde_json::json;

use deepagents::middleware::planning::{
    Plan, PlanStep, PlanStepStatus, PlanningMiddleware, SimplePlanningStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Planning Middleware Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating a plan manually
    // -----------------------------------------------------------------------
    println!("--- 1. Creating a Plan Manually ---\n");

    let mut plan = Plan::new("build-website", "Build and deploy a personal website");

    let step0_id = plan.add_step(PlanStep::new(0, "Design the page layout"));
    let step1_id = plan.add_step(PlanStep::new(0, "Write HTML and CSS").with_dependency(step0_id));
    let step2_id = plan.add_step(PlanStep::new(0, "Add JavaScript interactivity").with_dependency(step1_id));
    let _step3_id = plan.add_step(
        PlanStep::new(0, "Deploy to production").with_dependencies(vec![step1_id, step2_id]),
    );

    println!("  Plan ID: {}", plan.id);
    println!("  Goal: {}", plan.goal);
    println!("  Total steps: {}", plan.steps.len());
    println!("  Progress: {:.0}%\n", plan.progress_percentage());

    println!("  Status summary:\n");
    for line in plan.status_summary().lines() {
        println!("    {line}");
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. Finding ready steps and dependencies
    // -----------------------------------------------------------------------
    println!("--- 2. Finding Ready Steps ---\n");

    let ready = plan.get_ready_steps();
    println!("  Ready steps (no unmet dependencies):");
    for step in &ready {
        println!("    Step {}: {}", step.id, step.description);
    }
    println!();

    if let Some(next) = plan.next_actionable_step() {
        println!("  Next actionable step: Step {} - {}\n", next.id, next.description);
    }

    // -----------------------------------------------------------------------
    // 3. Executing steps and tracking progress
    // -----------------------------------------------------------------------
    println!("--- 3. Executing Steps and Tracking Progress ---\n");

    // Complete step 0
    println!("  Completing Step 0: \"Design the page layout\"...");
    plan.update_step_status(
        step0_id,
        PlanStepStatus::Completed,
        Some("Wireframe created with Figma".to_string()),
    );
    println!("    Progress: {:.0}%", plan.progress_percentage());
    let ready = plan.get_ready_steps();
    println!("    Newly ready steps: {:?}\n", ready.iter().map(|s| s.id).collect::<Vec<_>>());

    // Start step 1 (set to in-progress)
    println!("  Starting Step 1: \"Write HTML and CSS\"...");
    plan.update_step_status(step1_id, PlanStepStatus::InProgress, None);
    println!("    Step 1 status: {}", plan.get_step(step1_id).unwrap().status);
    println!("    Step 1 is terminal: {}\n", plan.get_step(step1_id).unwrap().is_terminal());

    // Complete step 1
    println!("  Completing Step 1...");
    plan.update_step_status(
        step1_id,
        PlanStepStatus::Completed,
        Some("Pages built with Tailwind CSS".to_string()),
    );
    println!("    Progress: {:.0}%", plan.progress_percentage());
    let ready = plan.get_ready_steps();
    println!("    Newly ready steps: {:?}\n", ready.iter().map(|s| s.id).collect::<Vec<_>>());

    // Fail step 2
    println!("  Failing Step 2: \"Add JavaScript interactivity\"...");
    plan.update_step_status(
        step2_id,
        PlanStepStatus::Failed,
        Some("JS build errors encountered".to_string()),
    );
    println!("    Progress: {:.0}%", plan.progress_percentage());

    // Check if step 3 is ready (it depends on both step 1 and step 2)
    let ready = plan.get_ready_steps();
    println!("    Ready steps: {:?} (step 3 blocked because step 2 failed, not completed)\n",
        ready.iter().map(|s| s.id).collect::<Vec<_>>());

    // Skip step 3 since step 2 failed
    println!("  Skipping Step 3 since dependency failed...");
    plan.update_step_status(_step3_id, PlanStepStatus::Skipped, None);
    println!("    Plan complete: {}", plan.is_complete());
    println!("    Final progress: {:.0}%\n", plan.progress_percentage());

    println!("  Final status:\n");
    for line in plan.status_summary().lines() {
        println!("    {line}");
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. Using SimplePlanningStrategy
    // -----------------------------------------------------------------------
    println!("--- 4. Using SimplePlanningStrategy ---\n");

    let strategy = SimplePlanningStrategy::new();

    // Create a plan from a structured goal
    let structured_goal = "\
Build an API server:
1. Define the data models
2. Implement the REST endpoints
3. Write integration tests
4. Set up CI/CD pipeline
5. Deploy to staging";

    println!("  Goal:\n    {}\n", structured_goal.replace('\n', "\n    "));

    use deepagents::middleware::planning::PlanningStrategy;
    let auto_plan = strategy.create_plan(structured_goal, &json!({})).await
        .map_err(|e| format!("{e}"))?;

    println!("  Auto-generated plan ({} steps):", auto_plan.steps.len());
    for step in &auto_plan.steps {
        println!(
            "    Step {}: {} (deps: {:?})",
            step.id, step.description, step.dependencies
        );
    }
    println!();

    // Create a plan from an unstructured goal (single step)
    let simple_goal = "Fix the login bug on the homepage";
    let simple_plan = strategy.create_plan(simple_goal, &json!({})).await
        .map_err(|e| format!("{e}"))?;
    println!("  Unstructured goal: \"{simple_goal}\"");
    println!("  Generated {} step(s): \"{}\"\n", simple_plan.steps.len(), simple_plan.steps[0].description);

    // -----------------------------------------------------------------------
    // 5. Using PlanningMiddleware
    // -----------------------------------------------------------------------
    println!("--- 5. Using PlanningMiddleware ---\n");

    let strategy = Arc::new(SimplePlanningStrategy::new());
    let middleware = PlanningMiddleware::new(strategy);

    println!("  Middleware name: {}", deepagents::middleware::Middleware::name(&middleware));
    println!("  Current plan: {:?}\n", middleware.get_plan().await.map(|p| p.id));

    // Create a plan through the middleware
    let mw_plan = middleware
        .create_plan("1. Research\n2. Prototype\n3. Test", &json!({}))
        .await
        .map_err(|e| format!("{e}"))?;
    println!("  Created plan with {} steps via middleware", mw_plan.steps.len());
    println!("  Plan stored: {}\n", middleware.get_plan().await.is_some());

    // Update steps through middleware
    println!("  Updating step 0 to Completed via middleware...");
    let updated = middleware
        .update_step(0, PlanStepStatus::Completed, Some("Done".into()))
        .await;
    println!("  Update successful: {updated}");

    let current = middleware.get_plan().await.unwrap();
    println!("  Step 0 status: {}", current.get_step(0).unwrap().status);
    println!("  Plan progress: {:.0}%\n", current.progress_percentage());

    // -----------------------------------------------------------------------
    // 6. Plan serialization
    // -----------------------------------------------------------------------
    println!("--- 6. Plan Serialization ---\n");

    let mut serialize_plan = Plan::new("serialize-demo", "Demonstrate JSON roundtrip");
    serialize_plan.add_step(PlanStep::new(0, "Step A"));
    serialize_plan.add_step(PlanStep::new(0, "Step B").with_dependency(0));
    serialize_plan.update_step_status(0, PlanStepStatus::Completed, Some("done".into()));

    let json_str = serde_json::to_string_pretty(&serialize_plan)?;
    println!("  Serialized plan (JSON):\n");
    for line in json_str.lines() {
        println!("    {line}");
    }

    let deserialized: Plan = serde_json::from_str(&json_str)?;
    println!("\n  Deserialized: id={}, goal={}, steps={}", deserialized.id, deserialized.goal, deserialized.steps.len());
    println!("  Step 0 status after roundtrip: {}\n", deserialized.get_step(0).unwrap().status);

    println!("=== Done ===");
    Ok(())
}
