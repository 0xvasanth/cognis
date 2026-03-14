//! Agent Lifecycle Demo
//!
//! Demonstrates AgentLifecycle state transitions, HealthCheck, RestartPolicy,
//! GracefulShutdown, and LifecycleMonitor.
//!
//! Run with: `cargo run -p cognis-examples --example agent_lifecycle_demo`

#[path = "../shared.rs"]
mod shared;
use cognis_core::messages::Message;
use cognisagent::lifecycle::{
    AgentLifecycle, GracefulShutdown, HealthCheck, LifecycleMonitor, RestartPolicy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // State transitions
    let mut lifecycle = AgentLifecycle::new("agent-001".into());
    lifecycle.start().unwrap();
    lifecycle.record_step();
    lifecycle.record_step();
    println!(
        "State: {}, steps: {}",
        lifecycle.state(),
        lifecycle.uptime_steps()
    );

    lifecycle.pause().unwrap();
    lifecycle.resume().unwrap();
    lifecycle.stop().unwrap();
    println!("After stop: {}", lifecycle.state());

    for t in lifecycle.history() {
        println!(
            "  {} -> {} ({})",
            t.from,
            t.to,
            t.reason.as_deref().unwrap_or("-")
        );
    }

    // Health check
    let mut health = HealthCheck::new("agent-004".into());
    health.record_heartbeat();
    health.record_step_completion();
    health.record_step_completion();
    health.record_error("connection timeout".into());
    health.record_error("rate limit exceeded".into());
    println!(
        "Health: healthy={}, errors={}, rate={:.2}",
        health.is_healthy(),
        health.consecutive_errors(),
        health.error_rate()
    );

    // Restart policies
    let on_failure = RestartPolicy::on_failure(5, 100);
    println!(
        "OnFailure policy: restart at 0={}, delay={}ms",
        on_failure.should_restart(0),
        on_failure.delay_ms(0)
    );

    // Graceful shutdown
    let mut shutdown = GracefulShutdown::new(30);
    shutdown.register_cleanup(
        "flush_logs".into(),
        Box::new(|| println!("  Flushing logs...")),
    );
    shutdown.register_cleanup(
        "save_state".into(),
        Box::new(|| println!("  Saving state...")),
    );
    shutdown.initiate();
    let completed = shutdown.run_cleanups();
    println!("Cleanups completed: {:?}", completed);

    // Lifecycle monitor - multiple agents
    let mut monitor = LifecycleMonitor::new();

    let mut a1 = AgentLifecycle::new("worker-1".into());
    a1.start().unwrap();
    a1.record_step();

    let mut a2 = AgentLifecycle::new("worker-2".into());
    a2.start().unwrap();
    a2.pause().unwrap();

    let mut a3 = AgentLifecycle::new("worker-3".into());
    a3.start().unwrap();
    a3.fail("segfault".into());

    monitor.register(a1);
    monitor.register(a2);
    monitor.register(a3);

    println!(
        "Agents: {}, active: {:?}, failed: {:?}",
        monitor.agent_count(),
        monitor.active_agents(),
        monitor.failed_agents()
    );

    // LLM demo
    let model = shared::get_chat_model(vec![
        "The 'segfault' error indicates a memory access violation. Check for unsafe code and restart with memory sanitizer.".into(),
    ]);
    let messages = vec![
        Message::system("Diagnose agent failures concisely."),
        Message::human("Agent 'worker-3' failed with error: 'segfault'. What happened?"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
