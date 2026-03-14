//! Planning System Demo
//!
//! Demonstrates PlanBuilder for sequential/parallel steps, PlanExecutor
//! for driving execution, PlanProgress tracking, and Replanner for
//! modifying plans mid-execution.
//!
//! Run with: `cargo run -p cognis-examples --example planning_demo`

#[path = "../shared.rs"]
mod shared;
use cognisagent::planning::{PlanBuilder, PlanExecutor, PlanStep, Replanner};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a plan with sequential and parallel steps
    let plan = PlanBuilder::new("Deploy ML Pipeline")
        .step("Gather training data")
        .step("Clean and preprocess data")
        .parallel(vec![
            "Train model A (random forest)".into(),
            "Train model B (neural network)".into(),
            "Train model C (gradient boosting)".into(),
        ])
        .step("Evaluate and select best model")
        .sequential(vec![
            "Package model as API".into(),
            "Deploy to staging".into(),
            "Run integration tests".into(),
            "Deploy to production".into(),
        ])
        .build();

    println!("Plan: {}, {} steps", plan.name, plan.len());

    // Execute steps with PlanExecutor
    let mut executor = PlanExecutor::new(plan);

    // Complete first two sequential steps
    for _ in 0..2 {
        let step_id = executor.start_next().unwrap();
        let desc = executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
            .clone();
        executor.complete_step(&step_id, Some(json!({"status": "done"})));
        println!("  Completed: {}", desc);
    }

    // Complete three parallel training steps
    let ready = executor.current_plan().ready_steps();
    println!("  {} parallel steps ready", ready.len());
    for i in 0..3 {
        let step_id = executor.start_next().unwrap();
        let desc = executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
            .clone();
        executor.complete_step(
            &step_id,
            Some(json!({"accuracy": 0.85 + (i as f64) * 0.03})),
        );
        println!("  Completed: {}", desc);
    }

    // Complete evaluation step
    let step_id = executor.start_next().unwrap();
    executor.complete_step(&step_id, Some(json!({"selected": "gradient boosting"})));

    // Track progress
    let progress = executor.current_plan().progress();
    println!(
        "Progress: {}/{} ({:.0}%)",
        progress.completed,
        progress.total,
        progress.completion_rate() * 100.0
    );

    // Demonstrate failure and replanning
    let step_id = executor.start_next().unwrap();
    executor.fail_step(&step_id, "Docker build failed");
    println!(
        "Step failed, plan failed: {}",
        executor.current_plan().is_failed()
    );

    let mut replanner = Replanner::new();
    let recovery_step = PlanStep::new("Fix Docker dependencies and rebuild");
    replanner.add_steps(executor.current_plan_mut(), vec![recovery_step]);
    println!(
        "After replan: {} total steps",
        executor.current_plan().len()
    );

    // LLM demo
    let model = shared::get_chat_model(vec![
        "1. Run tests\n2. Build Docker image\n3. Deploy to staging\n4. Deploy to production with canary rollout\n5. Monitor metrics".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Create a step-by-step plan for deploying an ML model to production safely.",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM plan: {}", gen.message.content().text());
    }

    Ok(())
}
