//! Agent Lifecycle Demo
//!
//! Demonstrates the `cognisagent::lifecycle` module:
//! - Creating an `AgentLifecycle` and walking through state transitions
//! - Handling invalid transition errors
//! - Using `HealthCheck` to monitor agent health
//! - Using `RestartPolicy` variants (never, always, on_failure) with backoff
//! - Using `GracefulShutdown` with cleanup callbacks
//! - Using `LifecycleMonitor` to track multiple agents
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example agent_lifecycle_demo`

mod shared;
use cognis_core::messages::Message;
use cognisagent::lifecycle::{
    AgentLifecycle, GracefulShutdown, HealthCheck, LifecycleMonitor, RestartPolicy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Lifecycle Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. AgentLifecycle — state transitions
    // -----------------------------------------------------------------------
    println!("--- 1. AgentLifecycle State Transitions ---");

    let mut lifecycle = AgentLifecycle::new("agent-001".into());
    println!(
        "Created agent '{}' in state: {}",
        lifecycle.agent_id(),
        lifecycle.state()
    );

    // Start the agent (Initializing -> Ready -> Running)
    lifecycle.start().unwrap();
    println!("After start(): {}", lifecycle.state());

    // Record a few processing steps
    lifecycle.record_step();
    lifecycle.record_step();
    lifecycle.record_step();
    println!("Steps completed: {}", lifecycle.uptime_steps());

    // Pause the agent
    lifecycle.pause().unwrap();
    println!("After pause(): {}", lifecycle.state());

    // Resume the agent
    lifecycle.resume().unwrap();
    println!("After resume(): {}", lifecycle.state());

    // Stop the agent (Running -> Stopping -> Stopped)
    lifecycle.stop().unwrap();
    println!("After stop(): {}", lifecycle.state());

    // Print transition history
    println!(
        "\nTransition history ({} entries):",
        lifecycle.history().len()
    );
    for t in lifecycle.history() {
        let reason = t.reason.as_deref().unwrap_or("-");
        println!("  {} -> {} (reason: {})", t.from, t.to, reason);
    }

    // -----------------------------------------------------------------------
    // 2. Invalid transition errors
    // -----------------------------------------------------------------------
    println!("\n--- 2. Invalid Transition Errors ---");

    let mut lc2 = AgentLifecycle::new("agent-002".into());
    lc2.start().unwrap();

    // Attempt to start an already-running agent
    match lc2.start() {
        Err(e) => println!("Expected error (start while running): {}", e),
        Ok(_) => println!("Unexpected success"),
    }

    // Attempt to resume when not paused
    match lc2.resume() {
        Err(e) => println!("Expected error (resume while running): {}", e),
        Ok(_) => println!("Unexpected success"),
    }

    // Pause, then attempt to pause again
    lc2.pause().unwrap();
    match lc2.pause() {
        Err(e) => println!("Expected error (pause while paused): {}", e),
        Ok(_) => println!("Unexpected success"),
    }

    // Stop the agent, then try to stop again
    lc2.resume().unwrap();
    lc2.stop().unwrap();
    match lc2.stop() {
        Err(e) => println!("Expected error (stop while stopped): {}", e),
        Ok(_) => println!("Unexpected success"),
    }

    // Attempt to start a stopped agent
    match lc2.start() {
        Err(e) => println!("Expected error (start while stopped): {}", e),
        Ok(_) => println!("Unexpected success"),
    }

    // -----------------------------------------------------------------------
    // 3. AgentState — fail transition
    // -----------------------------------------------------------------------
    println!("\n--- 3. Agent Failure ---");

    let mut lc3 = AgentLifecycle::new("agent-003".into());
    lc3.start().unwrap();
    println!("Agent state before failure: {}", lc3.state());

    lc3.fail("out of memory".into());
    println!("Agent state after failure: {}", lc3.state());
    println!("Is terminal? {}", lc3.state().is_terminal());
    println!("Is active? {}", lc3.state().is_active());

    // -----------------------------------------------------------------------
    // 4. HealthCheck
    // -----------------------------------------------------------------------
    println!("\n--- 4. HealthCheck ---");

    let mut health = HealthCheck::new("agent-004".into());
    println!("Initial health: is_healthy={}", health.is_healthy());

    // Record heartbeats and steps
    health.record_heartbeat();
    health.record_step_completion();
    health.record_step_completion();
    println!(
        "After heartbeat + 2 steps: is_healthy={}, steps={}",
        health.is_healthy(),
        health.steps_completed()
    );

    // Record some errors
    health.record_error("connection timeout".into());
    health.record_error("rate limit exceeded".into());
    println!(
        "After 2 errors: consecutive_errors={}, error_rate={:.2}",
        health.consecutive_errors(),
        health.error_rate()
    );

    // A heartbeat resets the consecutive error counter
    health.record_heartbeat();
    println!(
        "After another heartbeat: consecutive_errors={}, is_healthy={}",
        health.consecutive_errors(),
        health.is_healthy()
    );

    // Trigger unhealthy state with 3 consecutive errors
    health.record_error("err1".into());
    health.record_error("err2".into());
    health.record_error("err3".into());
    println!(
        "After 3 consecutive errors: is_healthy={}, consecutive_errors={}",
        health.is_healthy(),
        health.consecutive_errors()
    );

    println!(
        "Health JSON: {}",
        serde_json::to_string_pretty(&health.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 5. RestartPolicy
    // -----------------------------------------------------------------------
    println!("\n--- 5. RestartPolicy ---");

    // Never restart
    let never = RestartPolicy::never();
    println!(
        "Never policy: should_restart(0)={}, should_restart(1)={}",
        never.should_restart(0),
        never.should_restart(1)
    );
    println!("  Config: {}", never.to_json());

    // Always restart (up to 3 times)
    let always = RestartPolicy::always(3);
    println!("Always(3) policy:");
    for i in 0..=3 {
        println!(
            "  failure_count={}: should_restart={}, delay={}ms",
            i,
            always.should_restart(i),
            always.delay_ms(i)
        );
    }

    // On failure with exponential backoff
    let on_failure = RestartPolicy::on_failure(5, 100);
    println!("OnFailure(max=5, backoff=100ms) policy:");
    for i in 0..=5 {
        println!(
            "  failure_count={}: should_restart={}, delay={}ms",
            i,
            on_failure.should_restart(i),
            on_failure.delay_ms(i)
        );
    }

    // -----------------------------------------------------------------------
    // 6. GracefulShutdown
    // -----------------------------------------------------------------------
    println!("\n--- 6. GracefulShutdown ---");

    let mut shutdown = GracefulShutdown::new(30);
    println!("Shutdown initiated? {}", shutdown.is_shutting_down());

    // Register cleanup callbacks
    shutdown.register_cleanup(
        "flush_logs".into(),
        Box::new(|| println!("  [cleanup] Flushing logs...")),
    );
    shutdown.register_cleanup(
        "close_connections".into(),
        Box::new(|| println!("  [cleanup] Closing connections...")),
    );
    shutdown.register_cleanup(
        "save_state".into(),
        Box::new(|| println!("  [cleanup] Saving agent state...")),
    );

    // Initiate shutdown
    shutdown.initiate();
    println!("Shutdown initiated? {}", shutdown.is_shutting_down());
    println!("Timed out? {}", shutdown.is_timed_out());

    // Run cleanup callbacks
    println!("Running cleanups:");
    let completed = shutdown.run_cleanups();
    println!("Completed cleanups: {:?}", completed);

    // -----------------------------------------------------------------------
    // 7. LifecycleMonitor — tracking multiple agents
    // -----------------------------------------------------------------------
    println!("\n--- 7. LifecycleMonitor ---");

    let mut monitor = LifecycleMonitor::new();

    // Create and register several agents in different states
    let mut a1 = AgentLifecycle::new("worker-1".into());
    a1.start().unwrap();
    a1.record_step();
    a1.record_step();

    let mut a2 = AgentLifecycle::new("worker-2".into());
    a2.start().unwrap();
    a2.pause().unwrap();

    let mut a3 = AgentLifecycle::new("worker-3".into());
    a3.start().unwrap();
    a3.fail("segfault".into());

    let mut a4 = AgentLifecycle::new("worker-4".into());
    a4.start().unwrap();
    a4.stop().unwrap();

    let mut a5 = AgentLifecycle::new("worker-5".into());
    a5.start().unwrap();
    a5.record_step();

    monitor.register(a1);
    monitor.register(a2);
    monitor.register(a3);
    monitor.register(a4);
    monitor.register(a5);

    println!("Total agents: {}", monitor.agent_count());
    println!("Active agents: {:?}", monitor.active_agents());
    println!("Failed agents: {:?}", monitor.failed_agents());

    // Look up a specific agent
    if let Some(agent) = monitor.get("worker-1") {
        println!(
            "worker-1 state: {}, steps: {}",
            agent.state(),
            agent.uptime_steps()
        );
    }

    println!("\nMonitor summary:");
    println!(
        "{}",
        serde_json::to_string_pretty(&monitor.to_json()).unwrap()
    );

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Using an LLM to diagnose a failed agent's error.\n");

    let model = shared::get_chat_model(vec![
        "The 'segfault' error in worker-3 likely indicates a memory access violation. Recommended action: check for unsafe code blocks and restart with memory sanitizer enabled.".into(),
    ]);
    let messages = vec![
        Message::system("You are an agent lifecycle monitor. Diagnose agent failures concisely."),
        Message::human("Agent 'worker-3' failed with error: 'segfault'. What happened and what should we do?"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM diagnosis: {}", gen.message.content().text());
    }

    println!("\n=== Agent Lifecycle Demo Complete ===");
    Ok(())
}
