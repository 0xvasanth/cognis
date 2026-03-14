//! Graph Debug Demo
//!
//! Demonstrates the graph debugging tools from `cognisgraph::graph::debug`:
//!
//! - **DebugSession** — manage breakpoints and record execution steps
//! - **Breakpoint** with **BreakCondition** (Always, OnStateChange, OnValue, OnIteration)
//! - **DebugStep** — record and complete individual execution steps
//! - **StateInspector** — inspect state reports and compute diffs between snapshots
//! - **ExecutionProfiler** — profile node execution times and find the slowest node
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_debug_demo`

mod shared;
use cognisgraph::graph::debug::{
    BreakCondition, Breakpoint, DebugSession, ExecutionProfiler, StateInspector,
};
use serde_json::json;

fn main() {
    println!("=== Graph Debug Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. DebugSession — breakpoints
    // -----------------------------------------------------------------------
    println!("--- 1. DebugSession with Breakpoints ---\n");

    let mut session = DebugSession::new();

    // Add breakpoints with different conditions
    let bp_always = Breakpoint::new("process").with_condition(BreakCondition::Always);
    let bp_state_change = Breakpoint::new("enrich").with_condition(BreakCondition::OnStateChange {
        key: "data".to_string(),
    });
    let bp_on_value = Breakpoint::new("validate").with_condition(BreakCondition::OnValue {
        key: "status".to_string(),
        expected: json!("error"),
    });
    let bp_iteration = Breakpoint::new("retry").with_condition(BreakCondition::OnIteration(3));

    session.add_breakpoint(bp_always);
    session.add_breakpoint(bp_state_change);
    session.add_breakpoint(bp_on_value);
    session.add_breakpoint(bp_iteration);

    println!(
        "  Added {} breakpoint(s):",
        session.list_breakpoints().len()
    );
    for bp in session.list_breakpoints() {
        println!("    {} — {}", bp.node_id, bp.to_json());
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. Check breakpoints against state
    // -----------------------------------------------------------------------
    println!("--- 2. Checking breakpoints against state ---\n");

    let state_ok = json!({"status": "ok", "data": "hello"});
    let state_err = json!({"status": "error", "data": "hello"});

    // "process" has Always condition — should always trigger
    let hit_process = session.is_breakpoint("process", &state_ok);
    println!(
        "  is_breakpoint(\"process\", status=ok)     = {}",
        hit_process
    );

    // "validate" has OnValue(status == "error") — should not trigger for "ok"
    let hit_validate_ok = session.is_breakpoint("validate", &state_ok);
    println!(
        "  is_breakpoint(\"validate\", status=ok)    = {}",
        hit_validate_ok
    );

    // "validate" should trigger for "error"
    let hit_validate_err = session.is_breakpoint("validate", &state_err);
    println!(
        "  is_breakpoint(\"validate\", status=error) = {}",
        hit_validate_err
    );

    // "enrich" has OnStateChange(data) — triggers when "data" key is present
    let hit_enrich = session.is_breakpoint("enrich", &state_ok);
    println!(
        "  is_breakpoint(\"enrich\", data present)   = {}",
        hit_enrich
    );

    // A node with no breakpoint
    let hit_unknown = session.is_breakpoint("unknown_node", &state_ok);
    println!(
        "  is_breakpoint(\"unknown_node\")           = {}",
        hit_unknown
    );
    println!();

    // -----------------------------------------------------------------------
    // 3. Recording execution steps
    // -----------------------------------------------------------------------
    println!("--- 3. Recording execution steps ---\n");

    let state_step1 = json!({"messages": [], "iteration": 0});
    let idx1 = session.record_step("fetch", state_step1);
    println!("  Recorded step {} for node \"fetch\"", idx1);

    let state_step2 = json!({"messages": ["fetched data"], "iteration": 1});
    let idx2 = session.record_step("process", state_step2);
    println!("  Recorded step {} for node \"process\"", idx2);

    let state_step3 = json!({"messages": ["fetched data", "processed"], "iteration": 2});
    let idx3 = session.record_step("enrich", state_step3);
    println!("  Recorded step {} for node \"enrich\"", idx3);

    println!("\n  Total steps recorded: {}", session.step_count());

    // Complete a step (simulate finishing execution)
    let history = session.execution_history();
    println!("\n  Execution history:");
    for step in history {
        println!(
            "    Step {}: node=\"{}\", timestamp={}",
            step.step_number, step.node_id, step.timestamp
        );
        println!("      state_before = {}", step.state_before);
        println!("      state_after  = {:?}", step.state_after);
        println!("      duration_ms  = {:?}", step.duration_ms);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. StateInspector — inspect and diff state
    // -----------------------------------------------------------------------
    println!("--- 4. StateInspector — state inspection and diffs ---\n");

    let mut inspector = StateInspector::new();

    let state_before = json!({
        "messages": ["hello"],
        "count": 1,
        "config": {"model": "gpt-4", "temperature": 0.7}
    });

    // Inspect the state
    let report = inspector.inspect(&state_before);
    println!("  State report:");
    println!("    keys          = {:?}", report.keys);
    println!("    total_fields  = {}", report.total_fields);
    println!("    nested_depth  = {}", report.nested_depth);
    println!("    estimated_size = {} bytes", report.estimated_size);
    println!();

    // Set up watched keys
    inspector.watch_key("messages");
    inspector.watch_key("count");

    let watched = inspector.watched_values(&state_before);
    println!("  Watched values:");
    for (key, val) in &watched {
        println!("    {} = {}", key, val);
    }
    println!();

    // Compute diff between two state snapshots
    let state_after = json!({
        "messages": ["hello", "world"],
        "count": 2,
        "result": "done"
    });

    let deltas = inspector.diff(&state_before, &state_after);
    println!("  State diff (before -> after):");
    for delta in &deltas {
        println!("    {}", delta);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. ExecutionProfiler — profile node execution
    // -----------------------------------------------------------------------
    println!("--- 5. ExecutionProfiler — profiling node execution ---\n");

    let mut profiler = ExecutionProfiler::new();

    // Simulate profiling three nodes with different execution times
    profiler.start_node("fetch");
    // Simulate work with a busy loop
    std::hint::black_box(simulate_work(2_000_000));
    profiler.end_node("fetch");

    profiler.start_node("process");
    std::hint::black_box(simulate_work(5_000_000));
    profiler.end_node("process");

    profiler.start_node("store");
    std::hint::black_box(simulate_work(1_000_000));
    profiler.end_node("store");

    // Run "process" a second time to see averaging
    profiler.start_node("process");
    std::hint::black_box(simulate_work(3_000_000));
    profiler.end_node("process");

    println!("  Node execution times:");
    for (node_id, times) in &profiler.node_times() {
        let avg = profiler.average_time(node_id).unwrap_or(0.0);
        println!(
            "    {} — runs: {:?} ms, average: {:.2} ms",
            node_id, times, avg
        );
    }

    println!(
        "\n  Total execution time: {} ms",
        profiler.total_execution_ms()
    );

    if let Some((slowest, avg)) = profiler.slowest_node() {
        println!("  Slowest node: {} (avg {:.2} ms)", slowest, avg);
    }

    println!("\n  Profiler JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&profiler.to_json()).unwrap()
    );
    println!();

    // -----------------------------------------------------------------------
    // 6. Real LLM Demo — LLM call in debug demo graph
    // -----------------------------------------------------------------------
    println!("--- 6. Real LLM Demo (LLM in Debug Graph) ---\n");

    let model = shared::get_chat_model(vec![
        "The profiler shows that 'process' is the slowest node, which is expected as it performs the most computation. Consider optimizing this node or parallelizing its workload.".into(),
    ]);

    // Record a debug step for the LLM call
    let llm_state =
        json!({"query": "Analyze profiler results", "profiler_data": profiler.to_json()});
    session.record_step("llm_analysis", llm_state);

    // Profile the LLM call
    profiler.start_node("llm_analysis");

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a performance analysis expert. Analyze profiling data concisely.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            &format!(
                "Analyze these node execution times and suggest which node to optimize first. Answer in 1-2 sentences.\nProfiler data: {}",
                serde_json::to_string(&profiler.to_json()).unwrap()
            ),
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let ai_msg = rt.block_on(async { model.invoke_messages(&messages, None).await });

    profiler.end_node("llm_analysis");

    match ai_msg {
        Ok(response) => {
            println!("  LLM analysis: {}", response.base.content.text());
            if let Some(avg) = profiler.average_time("llm_analysis") {
                println!("  LLM node execution time: {:.2} ms", avg);
            }
        }
        Err(e) => println!("  LLM error: {}", e),
    }

    println!("  Total debug steps: {}", session.step_count());

    println!("\n=== Graph Debug Demo Complete ===");
}

/// Simulate work by doing some computation. Returns a value to prevent
/// the compiler from optimizing the loop away.
fn simulate_work(iterations: u64) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..iterations {
        acc = acc.wrapping_add(i);
    }
    acc
}
