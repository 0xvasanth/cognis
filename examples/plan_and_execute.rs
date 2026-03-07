//! Plan-and-Execute Agent Example
//!
//! Demonstrates the Plan-and-Execute agent pattern:
//! - Creating a `SimplePlanner` and generating a plan from a goal
//! - Iterating over plan steps and inspecting their status
//! - Using the `PlanAndExecuteAgent` builder pattern
//! - Running the agent with a callback to observe step execution
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example plan_and_execute`

use rustchain::agents::plan_and_execute::{
    PlanAndExecuteAgent, PlanStep, PlanStepStatus, Planner, SimplePlanner, TemplatePlanner,
    ToolStepExecutor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Plan-and-Execute Agent Example ===\n");

    // -----------------------------------------------------------------------
    // 1. SimplePlanner: parse numbered steps from a goal string
    // -----------------------------------------------------------------------
    println!("--- 1. SimplePlanner ---");
    println!("Parses numbered steps from a goal description.\n");

    let planner = SimplePlanner::new();

    let goal = "Build a web scraper:\n\
                1. Research target website structure\n\
                2. Implement HTTP client with proper headers\n\
                3. Parse HTML content with CSS selectors\n\
                4. Store extracted data in JSON format\n\
                5. Add error handling and retry logic";

    let plan = planner.create_plan(goal)?;

    println!("Goal: {}", plan.goal);
    println!("Steps ({} total):", plan.steps.len());
    for step in &plan.steps {
        println!(
            "  [{}] Step {}: {} (status: {})",
            if step.status == PlanStepStatus::Pending {
                " "
            } else {
                "x"
            },
            step.index,
            step.description,
            step.status,
        );
    }

    let (done, total) = plan.progress();
    println!("Progress: {}/{}", done, total);
    println!("Complete: {}\n", plan.is_complete());

    // -----------------------------------------------------------------------
    // 2. Single-step fallback when no numbered steps are found
    // -----------------------------------------------------------------------
    println!("--- 2. Single-step Fallback ---");
    println!("When no numbered steps are detected, the entire goal becomes one step.\n");

    let simple_goal = "Deploy the application to production";
    let simple_plan = planner.create_plan(simple_goal)?;

    println!("Goal: {}", simple_plan.goal);
    println!("Steps: {}", simple_plan.steps.len());
    println!("  Step 0: {}\n", simple_plan.steps[0].description);

    // -----------------------------------------------------------------------
    // 3. TemplatePlanner: use a prompt template for plan generation
    // -----------------------------------------------------------------------
    println!("--- 3. TemplatePlanner ---");
    println!("Uses a configurable template with a {{goal}} placeholder.\n");

    let template_planner = TemplatePlanner::new(
        "Create a step-by-step plan to {goal}:\n\
         1. Analyze requirements\n\
         2. Design architecture\n\
         3. Implement solution\n\
         4. Write tests\n\
         5. Deploy and monitor",
    );

    let template_plan = template_planner.create_plan("build a REST API")?;

    println!("Goal: {}", template_plan.goal);
    println!("Steps:");
    for step in &template_plan.steps {
        println!("  {}: {}", step.index, step.description);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. TemplatePlanner with a custom generator function
    // -----------------------------------------------------------------------
    println!("--- 4. TemplatePlanner with Generator ---");
    println!("A generator function can produce plan text dynamically.\n");

    let gen_planner = TemplatePlanner::new("Plan: {goal}").with_generator(|_prompt| {
        Ok("1. Gather data from all sources\n\
                2. Clean and preprocess data\n\
                3. Train the model\n\
                4. Evaluate results"
            .to_string())
    });

    let gen_plan = gen_planner.create_plan("train an ML model")?;
    println!("Generated plan with {} steps:", gen_plan.steps.len());
    for step in &gen_plan.steps {
        println!("  {}: {}", step.index, step.description);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. Manual step manipulation
    // -----------------------------------------------------------------------
    println!("--- 5. Manual Step Iteration ---");
    println!("Demonstrating next_step() and status transitions.\n");

    let mut plan = planner.create_plan(
        "1. Initialize project\n\
         2. Add dependencies\n\
         3. Build and test",
    )?;

    while let Some(step) = plan.next_step() {
        println!("  Executing step {}: '{}'", step.index, step.description);
        step.status = PlanStepStatus::Completed;
        step.result = Some(format!("Step {} done successfully", step.index));
    }

    let (done, total) = plan.progress();
    println!("Progress: {}/{}", done, total);
    println!("Complete: {}\n", plan.is_complete());

    // -----------------------------------------------------------------------
    // 6. PlanAndExecuteAgent with builder pattern
    // -----------------------------------------------------------------------
    println!("--- 6. PlanAndExecuteAgent Builder ---");
    println!("Build and run the agent with a planner and executor.\n");

    // Create a ToolStepExecutor with no tools (passthrough mode)
    let executor = ToolStepExecutor::new(vec![]);

    let agent = PlanAndExecuteAgent::builder()
        .planner(SimplePlanner::new())
        .executor(executor)
        .max_replans(2)
        .build()?;

    let goal = "1. Search for Rust documentation\n\
                2. Summarize key concepts\n\
                3. Generate a cheat sheet";

    let result = agent
        .run_with_callback(goal, |step| {
            println!(
                "  [callback] About to execute step {}: {}",
                step.index, step.description
            );
        })
        .await?;

    println!("\nAgent result:");
    println!("  Total steps executed: {}", result.total_steps);
    println!("  Replans: {}", result.replans);
    println!("  Final output: {}", result.output);
    println!("  Step results:");
    for (desc, res) in &result.step_results {
        println!("    - {}: {}", desc, res);
    }

    println!("\n=== Plan-and-Execute Example Complete ===");
    Ok(())
}
