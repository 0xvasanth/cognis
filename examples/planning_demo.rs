//! Planning System Demo
//!
//! Demonstrates the planning system from `cognisagent::planning`:
//! - PlanBuilder for creating plans with sequential and parallel steps
//! - PlanExecutor for driving execution (start, complete, fail, skip)
//! - PlanProgress for tracking progress
//! - Replanner for modifying a plan mid-execution
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example planning_demo`

mod shared;
use cognisagent::planning::{PlanBuilder, PlanExecutor, PlanStep, Replanner};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Planning System Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Build a plan with sequential and parallel steps
    // -----------------------------------------------------------------------
    println!("--- 1. Building a Plan with PlanBuilder ---\n");

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

    println!("  Plan: {}", plan.name);
    println!("  Total steps: {}", plan.len());
    println!();

    println!("  Steps:");
    for step in plan.steps() {
        let deps = if step.dependencies.is_empty() {
            "none".to_string()
        } else {
            step.dependencies
                .iter()
                .map(|d| &d[..8])
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "    [{}] {} (deps: {})",
            &step.id[..8],
            step.description,
            deps
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. Execute the plan with PlanExecutor
    // -----------------------------------------------------------------------
    println!("--- 2. Executing Steps with PlanExecutor ---\n");

    let mut executor = PlanExecutor::new(plan);

    // Start and complete the first step: "Gather training data"
    let step_id = executor.start_next().expect("should have a ready step");
    println!(
        "  Started: {}",
        executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
    );
    executor.complete_step(&step_id, Some(json!({"rows": 50000})));
    println!("  Completed with result: {{\"rows\": 50000}}");

    // Start and complete the second step: "Clean and preprocess data"
    let step_id = executor.start_next().expect("should have a ready step");
    println!(
        "  Started: {}",
        executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
    );
    executor.complete_step(&step_id, Some(json!({"rows_after_cleaning": 48500})));
    println!("  Completed with result: {{\"rows_after_cleaning\": 48500}}");

    // Now the three parallel training steps should all be ready
    let ready = executor.current_plan().ready_steps();
    println!(
        "\n  Ready steps (parallel): {} steps available",
        ready.len()
    );
    for s in &ready {
        println!("    - {}", s.description);
    }

    // Start and complete all three parallel steps
    for i in 0..3 {
        let step_id = executor.start_next().expect("should have a ready step");
        let desc = executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
            .clone();
        let accuracy = 0.85 + (i as f64) * 0.03;
        executor.complete_step(&step_id, Some(json!({"accuracy": accuracy})));
        println!("  Completed: {} (accuracy: {:.2})", desc, accuracy);
    }

    // Complete the evaluation step
    let step_id = executor.start_next().expect("should have a ready step");
    println!(
        "\n  Started: {}",
        executor
            .current_plan()
            .get_step(&step_id)
            .unwrap()
            .description
    );
    executor.complete_step(
        &step_id,
        Some(json!({"selected": "gradient boosting", "accuracy": 0.91})),
    );
    println!("  Completed: selected gradient boosting");

    println!();

    // -----------------------------------------------------------------------
    // 3. Track progress with PlanProgress
    // -----------------------------------------------------------------------
    println!("--- 3. Tracking Progress ---\n");

    let progress = executor.current_plan().progress();
    println!("  Total steps:     {}", progress.total);
    println!("  Completed:       {}", progress.completed);
    println!("  Failed:          {}", progress.failed);
    println!("  Skipped:         {}", progress.skipped);
    println!("  In progress:     {}", progress.in_progress);
    println!("  Pending:         {}", progress.pending);
    println!(
        "  Completion rate: {:.0}%",
        progress.completion_rate() * 100.0
    );
    println!("  Success rate:    {:.0}%", progress.success_rate() * 100.0);
    println!();

    // -----------------------------------------------------------------------
    // 4. Demonstrate failure and skip
    // -----------------------------------------------------------------------
    println!("--- 4. Handling Failure and Skip ---\n");

    // Start the next sequential step: "Package model as API"
    let step_id = executor.start_next().expect("should have a ready step");
    let desc = executor
        .current_plan()
        .get_step(&step_id)
        .unwrap()
        .description
        .clone();
    println!("  Started: {}", desc);
    executor.fail_step(&step_id, "Docker build failed: missing dependency");
    println!("  Failed: Docker build failed: missing dependency");

    let progress = executor.current_plan().progress();
    println!(
        "\n  After failure: {} completed, {} failed, {} pending",
        progress.completed, progress.failed, progress.pending
    );
    println!("  Plan failed: {}", executor.current_plan().is_failed());
    println!();

    // -----------------------------------------------------------------------
    // 5. Use Replanner to modify the plan mid-execution
    // -----------------------------------------------------------------------
    println!("--- 5. Replanning Mid-Execution ---\n");

    let mut replanner = Replanner::new();

    // Add a new recovery step
    let recovery_step = PlanStep::new("Fix Docker dependencies and rebuild");
    let plan = executor.current_plan_mut();
    replanner.add_steps(plan, vec![recovery_step]);
    println!("  Added recovery step. Total steps now: {}", plan.len());

    // Find the failed step and skip it since we added a recovery step
    let failed_step_id = plan
        .steps()
        .iter()
        .find(|s| matches!(s.status, cognisagent::planning::StepStatus::Failed(_)))
        .map(|s| s.id.clone());

    if let Some(id) = failed_step_id {
        // Insert a fix step right after the failed one
        let fix_step = PlanStep::new("Retry packaging with updated Dockerfile");
        let new_id = replanner.insert_after(executor.current_plan_mut(), &id, fix_step);
        println!("  Inserted fix step after failed step: {}...", &new_id[..8]);
    }

    println!("  Replan count: {}", replanner.replan_count());
    println!("  Updated plan has {} steps", executor.current_plan().len());
    println!();

    // -----------------------------------------------------------------------
    // 6. Review execution log
    // -----------------------------------------------------------------------
    println!("--- 6. Execution Log ---\n");

    for event in executor.execution_log() {
        let step_info = if event.step_id.is_empty() {
            "(plan-level)".to_string()
        } else {
            format!("{}...", &event.step_id[..8])
        };
        println!(
            "  [{}] {} - {}",
            event.timestamp, event.event_type, step_info
        );
    }

    // -----------------------------------------------------------------------
    // 7. Final progress
    // -----------------------------------------------------------------------
    println!("\n--- 7. Final Progress ---\n");

    let progress = executor.current_plan().progress();
    println!(
        "  {}",
        serde_json::to_string_pretty(&progress.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 8. Real LLM Demo — LLM-powered planning
    // -----------------------------------------------------------------------
    println!("\n--- 8. Real LLM Demo ---");
    println!("Ask the LLM to create a deployment plan.\n");

    let model = shared::get_chat_model(vec![
        "Here is a deployment plan:\n1. Run unit tests and integration tests\n2. Build Docker image with optimized settings\n3. Deploy to staging and run smoke tests\n4. Deploy to production with canary rollout\n5. Monitor metrics and rollback if error rate exceeds 1%".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Create a step-by-step plan for deploying a machine learning model to production safely.",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  LLM Plan:\n  {}", gen.message.content().text());
    }

    println!("\n=== Planning Demo Complete ===");
    Ok(())
}
