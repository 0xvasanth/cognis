//! Agent Communication Example
//!
//! Shows how to set up inter-agent messaging using the CommunicationHub.
//! A planner agent sends tasks to an executor agent, which replies with results.
//!
//! Run with: `cargo run -p cognis-examples --example agent_communication`

#[path = "../shared.rs"]
mod shared;
use cognis_core::messages::Message;
use cognisagent::communication::{AgentMessage, CommunicationHub, MessagePriority};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Communication ===\n");

    // --- Set up the hub and register agents ---
    let mut hub = CommunicationHub::new();
    hub.register_agent("planner");
    hub.register_agent("executor");
    println!("Registered {} agents in the hub", hub.agent_count());

    // --- Step 1: Planner sends a task to executor ---
    println!("\n--- Step 1: Planner assigns a task ---");
    let task = AgentMessage::new(
        "planner",
        "executor",
        "research async patterns",
        json!({"query": "Rust async/await best practices", "max_results": 10}),
    )
    .with_priority(MessagePriority::Urgent)
    .with_metadata("plan_version", json!(1));

    let task_id = task.id.clone();
    hub.send(task).unwrap();
    println!("Planner sent urgent task to executor");

    // Share the plan via shared state
    hub.shared_state_mut().set(
        "plan",
        json!({"goal": "Summarize Rust async patterns", "steps": ["search", "analyze", "report"]}),
        "planner",
    );

    // --- Step 2: Executor reads the task ---
    println!("\n--- Step 2: Executor processes the task ---");
    let inbox = hub.get_mailbox("executor").unwrap().read();
    let received = &inbox[0];
    println!(
        "Received: '{}' from '{}' [priority: {}]",
        received.subject, received.from, received.priority
    );

    // --- Step 3: Executor replies with results ---
    println!("\n--- Step 3: Executor replies with results ---");
    let reply = AgentMessage::new(
        "executor",
        "planner",
        "research complete",
        json!({"found": 47, "top": ["tokio", "async-std", "futures"]}),
    )
    .with_reply_to(&task_id);
    hub.send(reply).unwrap();

    // Update shared state with results
    hub.shared_state_mut().set(
        "search_results",
        json!({"count": 47, "status": "complete"}),
        "executor",
    );
    println!("Executor replied and updated shared state");

    // --- Step 4: Planner reads the reply ---
    println!("\n--- Step 4: Planner reads the reply ---");
    let replies = hub.get_mailbox("planner").unwrap().read();
    let result = &replies[0];
    println!(
        "Got reply: '{}' (is_reply: {})",
        result.subject,
        result.is_reply()
    );
    println!(
        "Results: {}",
        hub.shared_state().get("search_results").unwrap()
    );

    // --- Step 5: Broadcast completion via channel ---
    println!("\n--- Step 5: Broadcast completion ---");
    hub.create_channel("updates");
    hub.subscribe("updates", "planner");
    hub.subscribe("updates", "executor");

    let delivered = hub.broadcast(
        "updates",
        "planner",
        "workflow complete",
        json!({"status": "success", "results_count": 47}),
    )?;
    println!(
        "Broadcast sent to {} agents on 'updates' channel",
        delivered
    );

    // --- LLM Demo: Generate a task with a real model ---
    println!("\n--- LLM Demo ---");
    let model = shared::get_chat_model(vec![
        "Research the top 3 Rust async runtime crates and compare their performance.".into(),
    ]);
    let messages = vec![
        Message::system("You are a planner agent. Generate a brief task description."),
        Message::human("Create a task about researching Rust async patterns."),
    ];
    let llm_result = model._generate(&messages, None).await?;
    if let Some(gen) = llm_result.generations.first() {
        println!("LLM-generated task: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
