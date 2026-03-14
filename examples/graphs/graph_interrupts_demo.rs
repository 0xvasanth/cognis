//! Graph Interrupts Demo
//!
//! Shows a human-in-the-loop approval workflow using graph interrupts.
//! An LLM reviews a proposed action, then the interrupt system gates execution
//! until a human approves or rejects.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_interrupts_demo`

#[path = "../shared.rs"]
mod shared;

use cognisgraph::graph::interrupts::{
    InterruptPolicy, InterruptQueue, InterruptRequest, InterruptResponse, InterruptType,
};
use serde_json::json;

fn main() {
    println!("=== Graph Interrupts: Human-in-the-Loop Approval ===\n");

    // 1. Define which nodes require human approval
    let policy = InterruptPolicy::new()
        .interrupt_before("send_email_node")
        .interrupt_after("draft_node");

    println!(
        "Policy: interrupt before send_email_node={}, after draft_node={}",
        policy.should_interrupt_before("send_email_node"),
        policy.should_interrupt_after("draft_node"),
    );

    // 2. Simulate: the graph reaches send_email_node and raises an interrupt
    let email_action = json!({
        "action": "send_email",
        "to": "team@example.com",
        "subject": "Quarterly Report",
        "body_preview": "Please find attached the Q4 results..."
    });

    let request = InterruptRequest::new(
        "send_email_node",
        InterruptType::Review {
            data: email_action.clone(),
        },
    )
    .with_context("reason", json!("Outbound email requires approval"));

    let request_id = request.id.clone();

    // 3. Enqueue the interrupt — graph execution pauses here
    let mut queue = InterruptQueue::new();
    queue.enqueue(request);
    println!(
        "\nGraph paused. Pending approvals: {}",
        queue.pending_count()
    );

    // 4. Ask the LLM to pre-screen the action
    let model = shared::get_chat_model(vec![
        "APPROVE — the email content is professional and the recipient is valid.".into(),
    ]);

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a security reviewer. Respond APPROVE or REJECT with a one-line reason.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(&format!(
            "Review this action:\n{}",
            serde_json::to_string_pretty(&email_action).unwrap()
        ))),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { model.invoke_messages(&messages, None).await });

    match result {
        Ok(response) => {
            let review = response.base.content.text();
            println!("LLM review: {}", review);

            // 5. Use the LLM recommendation to respond to the interrupt
            let approved = review.to_lowercase().contains("approve");
            let resp = if approved {
                InterruptResponse::approve_with_data(
                    &request_id,
                    json!({ "reviewer": "llm-prescreener", "recommendation": review }),
                )
            } else {
                InterruptResponse::reject(&request_id)
            };

            queue.respond(&request_id, resp).expect("valid request id");
            println!(
                "Interrupt resolved: {} (pending: {})",
                if approved { "APPROVED" } else { "REJECTED" },
                queue.pending_count(),
            );
        }
        Err(e) => println!("LLM error: {}", e),
    }

    println!("\n=== Done ===");
}
