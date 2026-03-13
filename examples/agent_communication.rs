//! Agent Communication Example
//!
//! Demonstrates the DeepAgents inter-agent communication system for message
//! passing, shared state, pub/sub channels, and centralized routing through
//! the CommunicationHub.
//!
//! Features shown:
//! - Creating AgentMessages with priority and metadata
//! - Mailbox deliver and read
//! - MessageFilter for filtering messages
//! - SharedState for key-value sharing between agents
//! - Channel pub/sub with subscribe and broadcast
//! - CommunicationHub routing messages between agents
//! - End-to-end conversation flow
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example agent_communication`

mod shared;
use cognisagent::communication::{
    AgentMessage, Channel, CommunicationHub, Mailbox, MessageFilter, MessagePriority, SharedState,
};
use serde_json::json;

fn main() {
    println!("=== Agent Communication Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating AgentMessages with Priority and Metadata
    // -----------------------------------------------------------------------
    println!("--- 1. Creating AgentMessages ---");
    println!("Messages carry sender, recipient, subject, body, priority, and metadata.\n");

    let basic_msg = AgentMessage::new(
        "planner",
        "executor",
        "run task",
        json!({"task": "search", "query": "Rust async patterns"}),
    );
    println!("Basic message:");
    println!("  id:       {}", basic_msg.id);
    println!("  from:     {}", basic_msg.from);
    println!("  to:       {}", basic_msg.to);
    println!("  subject:  {}", basic_msg.subject);
    println!("  priority: {}", basic_msg.priority);
    println!("  is_reply: {}", basic_msg.is_reply());

    let urgent_msg = AgentMessage::new(
        "monitor",
        "planner",
        "alert: rate limit",
        json!({"remaining": 5, "reset_in_secs": 60}),
    )
    .with_priority(MessagePriority::Urgent)
    .with_metadata("source", json!("api_monitor"))
    .with_metadata("severity", json!("high"));

    println!("\nUrgent message with metadata:");
    println!("  priority: {}", urgent_msg.priority);
    println!("  metadata: {:?}", urgent_msg.metadata);

    let reply_msg = AgentMessage::new(
        "executor",
        "planner",
        "task complete",
        json!({"status": "success", "results": 42}),
    )
    .with_reply_to(&basic_msg.id)
    .with_priority(MessagePriority::Normal);

    println!("\nReply message:");
    println!("  is_reply:  {}", reply_msg.is_reply());
    println!("  reply_to:  {:?}", reply_msg.reply_to);

    println!("\nMessage as JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&urgent_msg.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 2. Mailbox Deliver and Read
    // -----------------------------------------------------------------------
    println!("\n--- 2. Mailbox ---");
    println!("Each agent has a mailbox to receive and consume messages.\n");

    let mut mailbox = Mailbox::new("executor");
    println!("Mailbox owner: '{}'", mailbox.owner());
    println!(
        "  Empty: {}, unread: {}",
        mailbox.is_empty(),
        mailbox.unread_count()
    );

    // Deliver messages
    mailbox.deliver(AgentMessage::new(
        "planner",
        "executor",
        "task 1",
        json!({"id": 1}),
    ));
    mailbox.deliver(
        AgentMessage::new("planner", "executor", "urgent task 2", json!({"id": 2}))
            .with_priority(MessagePriority::Urgent),
    );
    mailbox.deliver(AgentMessage::new(
        "monitor",
        "executor",
        "status check",
        json!({}),
    ));

    println!("\nAfter delivering 3 messages:");
    println!(
        "  Empty: {}, unread: {}",
        mailbox.is_empty(),
        mailbox.unread_count()
    );

    // Peek without consuming
    let peeked = mailbox.peek();
    println!("\nPeek (does not consume):");
    for msg in &peeked {
        println!(
            "  - [{}] from '{}': '{}'",
            msg.priority, msg.from, msg.subject
        );
    }
    println!("  Still unread: {}", mailbox.unread_count());

    // Read and consume all
    let messages = mailbox.read();
    println!("\nRead (consumes all):");
    for msg in &messages {
        println!("  - from '{}': '{}'", msg.from, msg.subject);
    }
    println!("  Now unread: {} (inbox drained)", mailbox.unread_count());

    // -----------------------------------------------------------------------
    // 3. MessageFilter
    // -----------------------------------------------------------------------
    println!("\n--- 3. MessageFilter ---");
    println!("Declarative filters select messages by sender, recipient, subject, or priority.\n");

    // Deliver fresh messages for filtering
    mailbox.deliver(
        AgentMessage::new("planner", "executor", "urgent task", json!({}))
            .with_priority(MessagePriority::Urgent),
    );
    mailbox.deliver(AgentMessage::new(
        "planner",
        "executor",
        "normal task",
        json!({}),
    ));
    mailbox.deliver(
        AgentMessage::new("monitor", "executor", "urgent alert", json!({}))
            .with_priority(MessagePriority::Urgent),
    );
    mailbox.deliver(
        AgentMessage::new("reviewer", "executor", "review request", json!({}))
            .with_priority(MessagePriority::Low),
    );

    println!("Mailbox has {} messages", mailbox.unread_count());

    // Filter by priority
    let urgent_filter = MessageFilter::new().priority(MessagePriority::Urgent);
    let urgent_msgs = mailbox.read_filtered(&urgent_filter);
    println!("\nFilter: priority=Urgent -> {} matches", urgent_msgs.len());
    for msg in &urgent_msgs {
        println!("  - from '{}': '{}'", msg.from, msg.subject);
    }
    println!("  Remaining in mailbox: {}", mailbox.unread_count());

    // Filter by sender
    let from_planner = MessageFilter::new().from_agent("planner");
    let planner_msgs = mailbox.read_filtered(&from_planner);
    println!("\nFilter: from='planner' -> {} matches", planner_msgs.len());
    for msg in &planner_msgs {
        println!("  - '{}' (priority: {})", msg.subject, msg.priority);
    }

    // Filter by subject
    let review_filter = MessageFilter::new().subject_contains("review");
    let review_msgs = mailbox.read_filtered(&review_filter);
    println!(
        "\nFilter: subject_contains='review' -> {} matches",
        review_msgs.len()
    );
    for msg in &review_msgs {
        println!("  - from '{}': '{}'", msg.from, msg.subject);
    }

    // Combined filter
    println!("\nCombined filter (from='monitor' AND priority=Urgent):");
    let combo = MessageFilter::new()
        .from_agent("monitor")
        .priority(MessagePriority::Urgent);
    let test_msg = AgentMessage::new("monitor", "x", "alert", json!({}))
        .with_priority(MessagePriority::Urgent);
    let non_match =
        AgentMessage::new("planner", "x", "task", json!({})).with_priority(MessagePriority::Urgent);
    println!(
        "  monitor+urgent message matches: {}",
        combo.matches(&test_msg)
    );
    println!(
        "  planner+urgent message matches: {}",
        combo.matches(&non_match)
    );

    // -----------------------------------------------------------------------
    // 4. SharedState
    // -----------------------------------------------------------------------
    println!("\n--- 4. SharedState ---");
    println!("Key-value store with ownership tracking for inter-agent data sharing.\n");

    let mut state = SharedState::new();
    println!(
        "Empty state: len={}, is_empty={}",
        state.len(),
        state.is_empty()
    );

    // Agents write state
    state.set(
        "plan",
        json!({"steps": ["search", "analyze", "report"]}),
        "planner",
    );
    state.set(
        "search_results",
        json!({"count": 15, "relevant": 7}),
        "executor",
    );
    state.set("status", json!("in_progress"), "planner");
    state.set("memory_usage_mb", json!(256), "monitor");

    println!("\nAfter agents write state:");
    println!("  Entries: {}", state.len());

    let mut keys = state.keys();
    keys.sort();
    println!("  Keys: {:?}", keys);

    // Read values
    println!("\nReading values:");
    println!("  plan:           {}", state.get("plan").unwrap());
    println!("  plan owner:     {}", state.get_owner("plan").unwrap());
    println!("  search_results: {}", state.get("search_results").unwrap());
    println!("  nonexistent:    {:?}", state.get("missing_key"));

    // Entries by owner
    let planner_entries = state.entries_by_owner("planner");
    println!("\nEntries owned by 'planner' ({}):", planner_entries.len());
    for (key, value) in &planner_entries {
        println!("  {} = {}", key, value);
    }

    let monitor_entries = state.entries_by_owner("monitor");
    println!("\nEntries owned by 'monitor' ({}):", monitor_entries.len());
    for (key, value) in &monitor_entries {
        println!("  {} = {}", key, value);
    }

    // Overwrite changes ownership
    state.set("status", json!("completed"), "executor");
    println!("\nAfter executor overwrites 'status':");
    println!("  value: {}", state.get("status").unwrap());
    println!(
        "  owner: {} (was 'planner')",
        state.get_owner("status").unwrap()
    );

    // Remove
    let removed = state.remove("memory_usage_mb");
    println!("\nRemoved 'memory_usage_mb': {:?}", removed);
    println!("  Entries remaining: {}", state.len());

    // -----------------------------------------------------------------------
    // 5. Channel Pub/Sub
    // -----------------------------------------------------------------------
    println!("\n--- 5. Channel Pub/Sub ---");
    println!("Named channels for broadcasting messages to subscribed agents.\n");

    let mut updates_ch = Channel::new("task-updates");
    println!(
        "Channel '{}' created, subscribers: {}",
        updates_ch.name(),
        updates_ch.subscriber_count()
    );

    // Subscribe agents
    updates_ch.subscribe("planner");
    updates_ch.subscribe("executor");
    updates_ch.subscribe("reviewer");
    updates_ch.subscribe("monitor");
    println!("\nSubscribed 4 agents: {:?}", updates_ch.subscribers());

    // Idempotent subscribe
    updates_ch.subscribe("planner");
    println!(
        "After re-subscribing 'planner': {} subscribers (idempotent)",
        updates_ch.subscriber_count()
    );

    // Broadcast from executor (excluded from own broadcast)
    let broadcast_msgs = updates_ch.broadcast(
        "executor",
        "task completed",
        json!({"task_id": "T-42", "result": "success"}),
    );
    println!("\nBroadcast from 'executor':");
    println!(
        "  Messages generated: {} (sender excluded)",
        broadcast_msgs.len()
    );
    for msg in &broadcast_msgs {
        println!("  -> to '{}': '{}'", msg.to, msg.subject);
    }

    // Unsubscribe
    updates_ch.unsubscribe("monitor");
    println!(
        "\nAfter unsubscribing 'monitor': {:?}",
        updates_ch.subscribers()
    );

    // -----------------------------------------------------------------------
    // 6. CommunicationHub
    // -----------------------------------------------------------------------
    println!("\n--- 6. CommunicationHub ---");
    println!("Central router managing mailboxes, channels, and shared state.\n");

    let mut hub = CommunicationHub::new();

    // Register agents
    hub.register_agent("planner");
    hub.register_agent("executor");
    hub.register_agent("reviewer");
    println!("Registered {} agents", hub.agent_count());

    // Direct messaging
    let task_msg = AgentMessage::new(
        "planner",
        "executor",
        "execute search",
        json!({"query": "Rust trait objects"}),
    )
    .with_priority(MessagePriority::Urgent);
    hub.send(task_msg).unwrap();
    println!("\nPlanner sent urgent task to executor");

    let info_msg = AgentMessage::new("planner", "reviewer", "review pending", json!({"items": 3}));
    hub.send(info_msg).unwrap();
    println!("Planner sent review request to reviewer");

    // Check mailboxes
    println!("\nMailbox status:");
    println!(
        "  executor: {} unread",
        hub.get_mailbox("executor").unwrap().unread_count()
    );
    println!(
        "  reviewer: {} unread",
        hub.get_mailbox("reviewer").unwrap().unread_count()
    );
    println!(
        "  planner:  {} unread",
        hub.get_mailbox("planner").unwrap().unread_count()
    );

    // Send to unknown agent
    let bad_msg = AgentMessage::new("planner", "nonexistent", "hello", json!({}));
    let result = hub.send(bad_msg);
    println!("\nSend to unregistered agent: {:?}", result.err().unwrap());

    // Channel broadcast through hub
    hub.create_channel("announcements");
    hub.subscribe("announcements", "planner");
    hub.subscribe("announcements", "executor");
    hub.subscribe("announcements", "reviewer");
    println!(
        "\nCreated 'announcements' channel, {} channels total",
        hub.channel_count()
    );

    let count = hub
        .broadcast(
            "announcements",
            "planner",
            "sprint started",
            json!({"sprint": 5}),
        )
        .unwrap();
    println!(
        "Planner broadcast on 'announcements': {} recipients (excludes sender)",
        count
    );

    println!("\nMailbox status after broadcast:");
    println!(
        "  executor: {} unread",
        hub.get_mailbox("executor").unwrap().unread_count()
    );
    println!(
        "  reviewer: {} unread",
        hub.get_mailbox("reviewer").unwrap().unread_count()
    );
    println!(
        "  planner:  {} unread (not self-delivered)",
        hub.get_mailbox("planner").unwrap().unread_count()
    );

    // Shared state through hub
    hub.shared_state_mut()
        .set("current_sprint", json!(5), "planner");
    hub.shared_state_mut()
        .set("sprint_goal", json!("ship v2.0"), "planner");
    println!("\nShared state via hub:");
    println!(
        "  current_sprint: {}",
        hub.shared_state().get("current_sprint").unwrap()
    );
    println!(
        "  sprint_goal:    {}",
        hub.shared_state().get("sprint_goal").unwrap()
    );

    // -----------------------------------------------------------------------
    // 7. End-to-End Conversation Flow
    // -----------------------------------------------------------------------
    println!("\n--- 7. End-to-End Conversation Flow ---");
    println!("Simulating a multi-agent workflow: plan -> execute -> review -> report.\n");

    let mut hub = CommunicationHub::new();
    hub.register_agent("planner");
    hub.register_agent("executor");
    hub.register_agent("reviewer");

    // Set up a results channel
    hub.create_channel("results");
    hub.subscribe("results", "planner");
    hub.subscribe("results", "executor");
    hub.subscribe("results", "reviewer");

    // Step 1: Planner creates a plan and shares it
    println!("Step 1: Planner creates plan");
    hub.shared_state_mut().set(
        "plan",
        json!({
            "goal": "Summarize Rust async patterns",
            "steps": ["search", "analyze", "summarize"]
        }),
        "planner",
    );

    let task = AgentMessage::new(
        "planner",
        "executor",
        "execute plan",
        json!({"step": "search", "query": "Rust async"}),
    )
    .with_priority(MessagePriority::Urgent)
    .with_metadata("plan_version", json!(1));
    let task_id = task.id.clone();
    hub.send(task).unwrap();
    println!("  Planner sent task to executor (id: {}...)", &task_id[..8]);

    // Step 2: Executor reads task and executes
    println!("\nStep 2: Executor processes task");
    let executor_msgs = hub.get_mailbox("executor").unwrap().read();
    let received = &executor_msgs[0];
    println!(
        "  Received: '{}' from '{}' (priority: {})",
        received.subject, received.from, received.priority
    );
    println!(
        "  Plan version: {:?}",
        received.metadata.get("plan_version")
    );

    // Executor replies with result
    let result = AgentMessage::new(
        "executor",
        "planner",
        "search complete",
        json!({"found": 47, "top_results": ["tokio", "async-std", "futures"]}),
    )
    .with_reply_to(&task_id);
    hub.send(result).unwrap();
    println!("  Executor replied with search results");

    // Executor updates shared state
    hub.shared_state_mut().set(
        "search_results",
        json!({"count": 47, "status": "complete"}),
        "executor",
    );

    // Step 3: Planner reads reply and sends to reviewer
    println!("\nStep 3: Planner forwards to reviewer");
    let planner_msgs = hub.get_mailbox("planner").unwrap().read();
    let reply = &planner_msgs[0];
    println!(
        "  Got reply: '{}' (is_reply: {})",
        reply.subject,
        reply.is_reply()
    );

    let review_req = AgentMessage::new(
        "planner",
        "reviewer",
        "review search results",
        json!({"results_key": "search_results"}),
    )
    .with_metadata("step", json!("review"));
    hub.send(review_req).unwrap();
    println!("  Sent review request to reviewer");

    // Step 4: Reviewer reads shared state and approves
    println!("\nStep 4: Reviewer checks shared state and approves");
    let reviewer_msgs = hub.get_mailbox("reviewer").unwrap().read();
    let review = &reviewer_msgs[0];
    println!("  Received: '{}' from '{}'", review.subject, review.from);

    // Reviewer reads shared state
    let results = hub.shared_state().get("search_results").unwrap();
    println!("  Shared state 'search_results': {}", results);

    hub.shared_state_mut()
        .set("review_status", json!("approved"), "reviewer");
    println!("  Set review_status = 'approved'");

    // Step 5: Broadcast final result to all agents
    println!("\nStep 5: Broadcast final result");
    let count = hub
        .broadcast(
            "results",
            "planner",
            "workflow complete",
            json!({
                "goal": "Summarize Rust async patterns",
                "status": "success",
                "review": "approved"
            }),
        )
        .unwrap();
    println!("  Broadcast to {} agents on 'results' channel", count);

    // Final state
    println!("\nFinal shared state:");
    let state = hub.shared_state();
    for key in &["plan", "search_results", "review_status"] {
        if let Some(val) = state.get(key) {
            println!(
                "  {} = {} (owner: '{}')",
                key,
                val,
                state.get_owner(key).unwrap_or("?")
            );
        }
    }

    // Final mailbox status
    println!("\nFinal mailbox status:");
    println!(
        "  executor: {} unread",
        hub.get_mailbox("executor").unwrap().unread_count()
    );
    println!(
        "  reviewer: {} unread",
        hub.get_mailbox("reviewer").unwrap().unread_count()
    );
    println!(
        "  planner:  {} unread",
        hub.get_mailbox("planner").unwrap().unread_count()
    );

    println!("\n=== Agent Communication Example Complete ===");
}
