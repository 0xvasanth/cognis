//! Graph Interrupts Demo
//!
//! Demonstrates the interrupt/resume system from `langgraph::graph::interrupts`
//! for human-in-the-loop workflows.
//!
//! Features shown:
//! - Creating an InterruptPolicy with before/after/always rules
//! - Creating InterruptRequests of different types (Approval, Input, Review)
//! - Using InterruptQueue to manage pending requests
//! - Responding to requests (approve, reject, approve with data)
//! - Using InterruptLog for audit trail
//! - Using TimeoutPolicy for auto-resolution
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example graph_interrupts_demo`

use langgraph::graph::interrupts::{
    InterruptLog, InterruptPolicy, InterruptQueue, InterruptRequest, InterruptResponse,
    InterruptType, TimeoutPolicy,
};
use serde_json::json;

fn main() {
    println!("=== Graph Interrupts Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create an InterruptPolicy with various rules
    // -----------------------------------------------------------------------
    let policy = InterruptPolicy::new()
        .interrupt_before("human_review_node")
        .interrupt_after("dangerous_action_node")
        .always_interrupt("sensitive_data_node");

    println!("1. Created InterruptPolicy:");
    println!(
        "   Policy JSON: {}\n",
        serde_json::to_string_pretty(&policy.to_json()).unwrap()
    );

    // Check which nodes trigger interrupts
    println!("   Checking policy rules:");
    println!(
        "   - human_review_node: before={}, after={}",
        policy.should_interrupt_before("human_review_node"),
        policy.should_interrupt_after("human_review_node"),
    );
    println!(
        "   - dangerous_action_node: before={}, after={}",
        policy.should_interrupt_before("dangerous_action_node"),
        policy.should_interrupt_after("dangerous_action_node"),
    );
    println!(
        "   - sensitive_data_node: before={}, after={}",
        policy.should_interrupt_before("sensitive_data_node"),
        policy.should_interrupt_after("sensitive_data_node"),
    );
    println!(
        "   - normal_node: before={}, after={}\n",
        policy.should_interrupt_before("normal_node"),
        policy.should_interrupt_after("normal_node"),
    );

    // -----------------------------------------------------------------------
    // 2. Create InterruptRequests of different types
    // -----------------------------------------------------------------------
    println!("2. Creating InterruptRequests:\n");

    let approval_request = InterruptRequest::new("human_review_node", InterruptType::Approval)
        .with_context("reason", json!("Model wants to delete a file"));

    println!(
        "   Approval request: id={}, node={}, type={}",
        approval_request.id, approval_request.node_id, approval_request.interrupt_type
    );

    let input_request = InterruptRequest::new(
        "user_input_node",
        InterruptType::Input {
            prompt: "Please provide the target directory path".to_string(),
        },
    );

    println!(
        "   Input request: id={}, node={}, type={}",
        input_request.id, input_request.node_id, input_request.interrupt_type
    );

    let review_request = InterruptRequest::new(
        "sensitive_data_node",
        InterruptType::Review {
            data: json!({
                "action": "send_email",
                "to": "user@example.com",
                "subject": "Important Update",
                "body_preview": "Your account has been..."
            }),
        },
    )
    .with_context("urgency", json!("high"));

    println!(
        "   Review request: id={}, node={}, type={}\n",
        review_request.id, review_request.node_id, review_request.interrupt_type
    );

    // -----------------------------------------------------------------------
    // 3. Use InterruptQueue to manage pending requests
    // -----------------------------------------------------------------------
    println!("3. Managing requests with InterruptQueue:\n");

    let mut queue = InterruptQueue::new();

    // Save IDs before moving requests into the queue
    let approval_id = approval_request.id.clone();
    let input_id = input_request.id.clone();
    let review_id = review_request.id.clone();

    queue.enqueue(approval_request);
    queue.enqueue(input_request);
    queue.enqueue(review_request);

    println!("   Enqueued 3 requests");
    println!("   Pending count: {}", queue.pending_count());
    println!("   Queue length: {}", queue.len());
    println!("   Is empty: {}\n", queue.is_empty());

    // Peek at the front of the queue
    if let Some(front) = queue.peek() {
        println!("   Front of queue: {} ({})", front.id, front.interrupt_type);
    }

    // -----------------------------------------------------------------------
    // 4. Respond to requests
    // -----------------------------------------------------------------------
    println!("\n4. Responding to interrupt requests:\n");

    // Approve the first request
    let approve_response = InterruptResponse::approve(&approval_id);
    queue
        .respond(&approval_id, approve_response)
        .expect("should succeed");
    println!("   Approved request: {}", approval_id);

    // Reject the second request
    let reject_response = InterruptResponse::reject(&input_id);
    queue
        .respond(&input_id, reject_response)
        .expect("should succeed");
    println!("   Rejected request: {}", input_id);

    // Approve with data for the third request
    let data_response = InterruptResponse::approve_with_data(
        &review_id,
        json!({
            "reviewer": "admin",
            "notes": "Looks good, proceed with sending"
        }),
    );
    queue
        .respond(&review_id, data_response)
        .expect("should succeed");
    println!("   Approved with data: {}", review_id);

    println!("\n   After responses:");
    println!("   Responded count: {}", queue.responded_count());
    println!("   Pending (unresponded): {}", queue.pending_count());

    // Check individual responses
    if let Some(resp) = queue.get_response(&approval_id) {
        println!(
            "\n   Response for {}: approved={}, data={:?}",
            approval_id, resp.approved, resp.data
        );
    }
    if let Some(resp) = queue.get_response(&review_id) {
        println!(
            "   Response for {}: approved={}, data={}",
            review_id,
            resp.approved,
            resp.data
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default()
        );
    }

    // -----------------------------------------------------------------------
    // 5. Use InterruptLog for audit trail
    // -----------------------------------------------------------------------
    println!("\n5. Audit trail with InterruptLog:\n");

    let mut log = InterruptLog::new();

    // Create fresh requests for logging (since the originals were moved)
    let req1 = InterruptRequest::new("node_a", InterruptType::Approval);
    let req1_id = req1.id.clone();
    log.record_request(&req1);

    let req2 = InterruptRequest::new(
        "node_b",
        InterruptType::Input {
            prompt: "Confirm action".to_string(),
        },
    );
    let req2_id = req2.id.clone();
    log.record_request(&req2);

    // Record responses
    let resp1 = InterruptResponse::approve(&req1_id);
    log.record_response(&req1_id, &resp1);

    let resp2 = InterruptResponse::reject(&req2_id);
    log.record_response(&req2_id, &resp2);

    println!("   Log entries: {}", log.len());
    println!("   Is empty: {}", log.is_empty());

    // Filter entries by node
    let node_a_entries = log.filter_by_node("node_a");
    println!("   Entries for 'node_a': {}", node_a_entries.len());

    // Display all log entries
    println!("\n   Full audit log:");
    for entry in log.entries() {
        println!(
            "   - event={}, request_id={}, node_id={}",
            entry.event, entry.request_id, entry.node_id
        );
    }

    // Export log as JSON
    println!(
        "\n   Log JSON:\n   {}",
        serde_json::to_string_pretty(&log.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 6. Use TimeoutPolicy for auto-resolution
    // -----------------------------------------------------------------------
    println!("\n6. TimeoutPolicy for auto-resolution:\n");

    let mut timeout_policy = TimeoutPolicy::new(300); // 5 minutes default
    timeout_policy.set_timeout("fast_node", 30);
    timeout_policy.set_timeout("instant_node", 0); // immediate expiry

    println!(
        "   Default timeout: {}s",
        timeout_policy.get_timeout("any_node")
    );
    println!(
        "   fast_node timeout: {}s",
        timeout_policy.get_timeout("fast_node")
    );
    println!(
        "   instant_node timeout: {}s",
        timeout_policy.get_timeout("instant_node")
    );

    // Check expiration
    let fast_request = InterruptRequest::new("fast_node", InterruptType::Approval);
    let instant_request = InterruptRequest::new("instant_node", InterruptType::Approval);
    let normal_request = InterruptRequest::new("normal_node", InterruptType::Approval);

    println!(
        "\n   Is fast_node request expired? {}",
        timeout_policy.is_expired(&fast_request)
    );
    println!(
        "   Is instant_node request expired? {}",
        timeout_policy.is_expired(&instant_request)
    );
    println!(
        "   Is normal_node request expired? {}",
        timeout_policy.is_expired(&normal_request)
    );

    // -----------------------------------------------------------------------
    // 7. Demonstrate error handling for invalid responses
    // -----------------------------------------------------------------------
    println!("\n7. Error handling:\n");

    let mut error_queue = InterruptQueue::new();
    let bad_response = InterruptResponse::approve("nonexistent-request");
    match error_queue.respond("nonexistent-request", bad_response) {
        Ok(()) => println!("   Unexpected success"),
        Err(e) => println!("   Expected error for invalid request: {}", e),
    }

    // -----------------------------------------------------------------------
    // 8. Dequeue and process requests
    // -----------------------------------------------------------------------
    println!("\n8. Dequeue and process:\n");

    let mut process_queue = InterruptQueue::new();
    process_queue.enqueue(InterruptRequest::new("step_1", InterruptType::Approval));
    process_queue.enqueue(InterruptRequest::new(
        "step_2",
        InterruptType::Input {
            prompt: "Enter value".to_string(),
        },
    ));

    println!("   Queue size before dequeue: {}", process_queue.len());

    while let Some(req) = process_queue.dequeue() {
        println!(
            "   Processing: id={}, node={}, type={}",
            req.id, req.node_id, req.interrupt_type
        );
    }

    println!("   Queue size after dequeue: {}", process_queue.len());
    println!("   Queue is empty: {}", process_queue.is_empty());

    println!("\n=== Graph Interrupts Demo Complete ===");
}
