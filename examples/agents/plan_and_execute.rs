//! Plan-and-Execute Agent Example
//!
//! Demonstrates SimplePlanner, TemplatePlanner, manual step iteration,
//! and PlanAndExecuteAgent with builder pattern.
//!
//! Run with: `cargo run -p cognis-examples --example plan_and_execute`

#[path = "../shared.rs"]
mod shared;
use cognis::agents::plan_and_execute::{
    PlanAndExecuteAgent, PlanStepStatus, Planner, SimplePlanner, TemplatePlanner, ToolStepExecutor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // SimplePlanner - parse numbered steps
    let planner = SimplePlanner::new();
    let goal = "Build a web scraper:\n\
                1. Research target website structure\n\
                2. Implement HTTP client with proper headers\n\
                3. Parse HTML content with CSS selectors\n\
                4. Store extracted data in JSON format\n\
                5. Add error handling and retry logic";

    let plan = planner.create_plan(goal)?;
    println!("Plan: {} steps", plan.steps.len());
    for step in &plan.steps {
        println!("  {}: {}", step.index, step.description);
    }

    // TemplatePlanner with generator
    let gen_planner = TemplatePlanner::new("Plan: {goal}").with_generator(|_prompt| {
        Ok("1. Gather data\n2. Preprocess\n3. Train model\n4. Evaluate".to_string())
    });
    let gen_plan = gen_planner.create_plan("train an ML model")?;
    println!("Generated plan: {} steps", gen_plan.steps.len());

    // Manual step iteration
    let mut plan =
        planner.create_plan("1. Initialize project\n2. Add dependencies\n3. Build and test")?;
    while let Some(step) = plan.next_step() {
        step.status = PlanStepStatus::Completed;
        step.result = Some(format!("Step {} done", step.index));
    }
    let (done, total) = plan.progress();
    println!("Progress: {}/{}", done, total);

    // PlanAndExecuteAgent with builder
    let executor = ToolStepExecutor::new(vec![]);
    let agent = PlanAndExecuteAgent::builder()
        .planner(SimplePlanner::new())
        .executor(executor)
        .max_replans(2)
        .build()?;

    let result = agent
        .run_with_callback(
            "1. Search for Rust docs\n2. Summarize key concepts\n3. Generate cheat sheet",
            |step| println!("  Executing step {}: {}", step.index, step.description),
        )
        .await?;

    println!(
        "Result: {} steps, {} replans",
        result.total_steps, result.replans
    );

    // LLM-powered planning
    let model = shared::get_chat_model(vec![
        "1. Define CLI arguments using clap\n2. Implement input parsing\n3. Build core logic\n4. Add error handling\n5. Write tests".to_string(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Create a numbered plan (5 steps) to build a CLI tool in Rust.",
    )];
    let ai_response = model.invoke_messages(&messages, None).await?;
    let response_text = ai_response.base.content.text();

    let llm_plan = SimplePlanner::new().create_plan(&response_text)?;
    println!("LLM plan: {} steps", llm_plan.steps.len());
    for step in &llm_plan.steps {
        println!("  {}: {}", step.index, step.description);
    }

    Ok(())
}
