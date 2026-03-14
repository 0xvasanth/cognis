//! Planning Middleware Example
//!
//! Demonstrates creating plans with steps and dependencies, tracking progress,
//! using SimplePlanningStrategy, and PlanningMiddleware.
//!
//! Run with: `cargo run -p cognis-examples --example planning_middleware`

#[path = "../shared.rs"]
mod shared;
use cognisagent::middleware::planning::{
    Plan, PlanStep, PlanStepStatus, PlanningMiddleware, PlanningStrategy, SimplePlanningStrategy,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a plan manually with dependencies
    let mut plan = Plan::new("build-website", "Build and deploy a personal website");
    let step0_id = plan.add_step(PlanStep::new(0, "Design the page layout"));
    let step1_id = plan.add_step(PlanStep::new(0, "Write HTML and CSS").with_dependency(step0_id));
    let step2_id =
        plan.add_step(PlanStep::new(0, "Add JavaScript interactivity").with_dependency(step1_id));
    let step3_id = plan.add_step(
        PlanStep::new(0, "Deploy to production").with_dependencies(vec![step1_id, step2_id]),
    );

    println!(
        "Plan: {} ({} steps, {:.0}%)",
        plan.goal,
        plan.steps.len(),
        plan.progress_percentage()
    );

    // Find and execute ready steps
    let ready = plan.get_ready_steps();
    println!(
        "Ready: {:?}",
        ready.iter().map(|s| s.id).collect::<Vec<_>>()
    );

    // Execute steps
    plan.update_step_status(
        step0_id,
        PlanStepStatus::Completed,
        Some("Wireframe done".into()),
    );
    plan.update_step_status(
        step1_id,
        PlanStepStatus::Completed,
        Some("Pages built".into()),
    );
    plan.update_step_status(
        step2_id,
        PlanStepStatus::Failed,
        Some("JS build errors".into()),
    );
    plan.update_step_status(step3_id, PlanStepStatus::Skipped, None);

    println!(
        "Complete: {}, progress: {:.0}%",
        plan.is_complete(),
        plan.progress_percentage()
    );

    // SimplePlanningStrategy - auto-generate from structured goal
    let strategy = SimplePlanningStrategy::new();
    let structured_goal = "Build an API server:\n1. Define data models\n2. Implement REST endpoints\n3. Write tests\n4. Set up CI/CD\n5. Deploy to staging";
    let auto_plan = strategy
        .create_plan(structured_goal, &json!({}))
        .await
        .map_err(|e| format!("{e}"))?;
    println!("Auto-plan: {} steps", auto_plan.steps.len());
    for step in &auto_plan.steps {
        println!("  Step {}: {}", step.id, step.description);
    }

    // PlanningMiddleware
    let strategy = Arc::new(SimplePlanningStrategy::new());
    let middleware = PlanningMiddleware::new(strategy);

    let mw_plan = middleware
        .create_plan("1. Research\n2. Prototype\n3. Test", &json!({}))
        .await
        .map_err(|e| format!("{e}"))?;
    println!("Middleware plan: {} steps", mw_plan.steps.len());

    middleware
        .update_step(0, PlanStepStatus::Completed, Some("Done".into()))
        .await;
    let current = middleware.get_plan().await.unwrap();
    println!(
        "Step 0: {}, progress: {:.0}%",
        current.get_step(0).unwrap().status,
        current.progress_percentage()
    );

    // Plan serialization roundtrip
    let mut ser_plan = Plan::new("demo", "JSON roundtrip");
    ser_plan.add_step(PlanStep::new(0, "Step A"));
    ser_plan.add_step(PlanStep::new(0, "Step B").with_dependency(0));
    ser_plan.update_step_status(0, PlanStepStatus::Completed, Some("done".into()));

    let json_str = serde_json::to_string(&ser_plan)?;
    let deserialized: Plan = serde_json::from_str(&json_str)?;
    println!(
        "Roundtrip: {} steps, step 0={}",
        deserialized.steps.len(),
        deserialized.get_step(0).unwrap().status
    );

    // LLM demo
    let model = shared::get_chat_model(vec![
        "1. Define data models\n2. Implement CRUD endpoints\n3. Add auth\n4. Write tests\n5. Deploy with CI/CD".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Break down 'Build a REST API server' into actionable steps.",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
