//! State Machine Example
//!
//! Demonstrates the finite state machine layer provided by `cognisgraph::graph::state_machine`
//! for modeling agent workflows as explicit state transitions with guards and actions.
//!
//! Features shown:
//! - StateId creation and identity
//! - Transition creation with optional guards and actions
//! - StateMachine stepping through states one at a time
//! - StateMachine running to completion
//! - StateHistory recording and inspecting transitions
//! - StateMachineBuilder fluent API for ergonomic construction
//! - StateMachineValidator for static correctness checks
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example state_machine`

#[path = "../shared.rs"]
mod shared;
use cognis_core::messages::Message;
use cognisgraph::graph::{
    StateHistory, StateId, StateMachine, StateMachineBuilder, StateMachineValidator, Transition,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== State Machine Example ===\n");

    // -----------------------------------------------------------------------
    // 1. StateId Creation
    // -----------------------------------------------------------------------
    println!("--- 1. StateId Creation ---");
    println!("StateId is a lightweight, hashable identifier for states.\n");

    let idle = StateId::new("idle");
    let processing = StateId::new("processing");
    let done = StateId::new("done");

    println!("Created states: {}, {}, {}", idle, processing, done);
    println!(
        "Equality check: idle == idle? {}",
        idle == StateId::new("idle")
    );
    println!("Equality check: idle == done? {}", idle == done);

    // -----------------------------------------------------------------------
    // 2. Transition Creation with Guards and Actions
    // -----------------------------------------------------------------------
    println!("\n--- 2. Transitions ---");
    println!("Transitions connect states with optional guard predicates and actions.\n");

    // Simple unguarded transition
    let simple = Transition::new(StateId::new("a"), StateId::new("b"), "move");
    println!("Simple transition: {:?}", simple);
    println!(
        "  Can fire with empty context? {}",
        simple.can_transition(&json!({}))
    );

    // Guarded transition: only fires when context has "approved" == true
    let guarded = Transition::new(StateId::new("review"), StateId::new("approved"), "approve")
        .with_guard(|ctx| {
            ctx.get("approved")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });

    println!("\nGuarded transition: {:?}", guarded);
    println!(
        "  Fires with {{}}?                {}",
        guarded.can_transition(&json!({}))
    );
    println!(
        "  Fires with {{approved: true}}?  {}",
        guarded.can_transition(&json!({"approved": true}))
    );
    println!(
        "  Fires with {{approved: false}}? {}",
        guarded.can_transition(&json!({"approved": false}))
    );

    // Transition with an action that mutates context
    let with_action = Transition::new(StateId::new("init"), StateId::new("running"), "start")
        .with_action(|ctx| {
            ctx.as_object_mut()
                .unwrap()
                .insert("started".into(), json!(true));
        });
    println!("\nTransition with action: {:?}", with_action);

    // -----------------------------------------------------------------------
    // 3. StateMachine — Stepping Through States
    // -----------------------------------------------------------------------
    println!("\n--- 3. StateMachine Stepping ---");
    println!("Step through a machine one transition at a time.\n");

    let mut machine = StateMachine::new(StateId::new("draft"));
    machine.add_state(StateId::new("review"));
    machine.add_state(StateId::new("approved"));
    machine.add_final_state(StateId::new("published"));

    machine.add_transition(Transition::new(
        StateId::new("draft"),
        StateId::new("review"),
        "submit",
    ));
    machine.add_transition(Transition::new(
        StateId::new("review"),
        StateId::new("approved"),
        "approve",
    ));
    machine.add_transition(Transition::new(
        StateId::new("approved"),
        StateId::new("published"),
        "publish",
    ));

    let mut ctx = json!({"title": "My Article"});

    println!("Initial state: {}", machine.current());

    // Step 1: draft -> review
    let next = machine.step(&mut ctx).unwrap();
    println!(
        "After step 1:  {} (transitioned to {:?})",
        machine.current(),
        next
    );

    // Step 2: review -> approved
    let next = machine.step(&mut ctx).unwrap();
    println!(
        "After step 2:  {} (transitioned to {:?})",
        machine.current(),
        next
    );

    // Step 3: approved -> published
    let next = machine.step(&mut ctx).unwrap();
    println!(
        "After step 3:  {} (transitioned to {:?})",
        machine.current(),
        next
    );

    println!("Is finished:   {}", machine.is_finished());

    // Stepping again at a final state returns None
    let next = machine.step(&mut ctx).unwrap();
    println!("Step at final: {:?} (no transition)", next);

    // -----------------------------------------------------------------------
    // 4. StateMachine — Run to Completion
    // -----------------------------------------------------------------------
    println!("\n--- 4. Run to Completion ---");
    println!("Run a machine with guards and actions until it reaches a final state.\n");

    // A retry loop: process -> check -> (if retries < 3, retry and loop; else done)
    let mut retry_machine = StateMachine::new(StateId::new("process"));
    retry_machine.add_state(StateId::new("check"));
    retry_machine.add_state(StateId::new("retry"));
    retry_machine.add_final_state(StateId::new("done"));

    retry_machine.add_transition(Transition::new(
        StateId::new("process"),
        StateId::new("check"),
        "evaluate",
    ));

    // If retries >= 3, move to done
    retry_machine.add_transition(
        Transition::new(StateId::new("check"), StateId::new("done"), "succeed")
            .with_guard(|ctx| ctx.get("retries").and_then(|v| v.as_i64()).unwrap_or(0) >= 3),
    );

    // Otherwise, retry and increment counter
    retry_machine.add_transition(
        Transition::new(StateId::new("check"), StateId::new("retry"), "retry").with_action(|ctx| {
            let retries = ctx.get("retries").and_then(|v| v.as_i64()).unwrap_or(0);
            ctx.as_object_mut()
                .unwrap()
                .insert("retries".into(), json!(retries + 1));
        }),
    );

    retry_machine.add_transition(Transition::new(
        StateId::new("retry"),
        StateId::new("process"),
        "reprocess",
    ));

    retry_machine.add_transition(Transition::new(
        StateId::new("process"),
        StateId::new("check"),
        "evaluate-again",
    ));

    let mut retry_ctx = json!({"retries": 0});
    let path = retry_machine.run(&mut retry_ctx).unwrap();

    let state_names: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
    println!("Path taken: {:?}", state_names);
    println!("Final state: {}", retry_machine.current());
    println!("Is finished: {}", retry_machine.is_finished());
    println!("Final context: {}", retry_ctx);

    // Reset and run again
    retry_machine.reset();
    println!("\nAfter reset: {}", retry_machine.current());

    // -----------------------------------------------------------------------
    // 5. StateHistory — Recording Transitions
    // -----------------------------------------------------------------------
    println!("\n--- 5. StateHistory ---");
    println!("Record and inspect a log of state transitions.\n");

    let mut history = StateHistory::new();
    println!(
        "Empty history: len={}, is_empty={}",
        history.len(),
        history.is_empty()
    );

    // Simulate recording transitions from a workflow
    history.record(
        &StateId::new("idle"),
        &StateId::new("loading"),
        json!({"progress": 0}),
    );
    history.record(
        &StateId::new("loading"),
        &StateId::new("processing"),
        json!({"progress": 50}),
    );
    history.record(
        &StateId::new("processing"),
        &StateId::new("complete"),
        json!({"progress": 100}),
    );

    println!("Recorded {} transitions", history.len());
    println!("State path: {:?}", history.path());

    // Inspect individual entries
    for entry in history.entries() {
        println!(
            "  step {}: {} -> {} (context: {})",
            entry.step, entry.from, entry.to, entry.context
        );
    }

    // Serialize to JSON
    println!(
        "\nHistory as JSON:\n{}",
        serde_json::to_string_pretty(&history.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 6. StateMachineBuilder — Fluent API
    // -----------------------------------------------------------------------
    println!("\n--- 6. StateMachineBuilder ---");
    println!("Construct state machines ergonomically with the builder pattern.\n");

    let mut order_machine = StateMachineBuilder::new("pending")
        .state("confirmed")
        .state("shipped")
        .final_state("delivered")
        .final_state("cancelled")
        .transition("pending", "confirmed", "confirm_order")
        .transition("confirmed", "shipped", "ship_order")
        .transition("shipped", "delivered", "deliver_order")
        .guarded_transition("pending", "cancelled", "cancel_order", |ctx| {
            ctx.get("cancel_requested")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        })
        .build()
        .unwrap();

    println!("Built order machine: {:?}", order_machine);
    println!("Initial state: {}", order_machine.current());

    // Run the happy path
    let mut order_ctx = json!({"order_id": "ORD-001"});
    let order_path = order_machine.run(&mut order_ctx).unwrap();
    let order_names: Vec<&str> = order_path.iter().map(|s| s.as_str()).collect();
    println!("Happy path: {:?}", order_names);
    println!("Finished: {}", order_machine.is_finished());

    // Reset and try the cancellation path
    order_machine.reset();
    let mut cancel_ctx = json!({"order_id": "ORD-002", "cancel_requested": true});
    let cancel_path = order_machine.run(&mut cancel_ctx).unwrap();
    let cancel_names: Vec<&str> = cancel_path.iter().map(|s| s.as_str()).collect();
    println!("\nCancellation path: {:?}", cancel_names);
    println!("Finished: {}", order_machine.is_finished());

    // -----------------------------------------------------------------------
    // 7. StateMachineValidator — Validation
    // -----------------------------------------------------------------------
    println!("\n--- 7. StateMachineValidator ---");
    println!("Validate state machine correctness before execution.\n");

    // Valid machine
    let valid = StateMachineBuilder::new("start")
        .state("middle")
        .final_state("end")
        .transition("start", "middle", "go")
        .transition("middle", "end", "finish")
        .build()
        .unwrap();

    match StateMachineValidator::validate(&valid) {
        Ok(()) => println!("Valid machine: passed all checks"),
        Err(errors) => println!("Valid machine errors: {:?}", errors),
    }

    // Machine with no final states
    let no_final = StateMachineBuilder::new("start")
        .state("middle")
        .transition("start", "middle", "go")
        .build()
        .unwrap();

    match StateMachineValidator::validate(&no_final) {
        Ok(()) => println!("No-final machine: passed (unexpected)"),
        Err(errors) => {
            println!("No-final machine errors:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // Machine with unreachable state
    let mut unreachable = StateMachine::new(StateId::new("start"));
    unreachable.add_state(StateId::new("reachable"));
    unreachable.add_state(StateId::new("orphan"));
    unreachable.add_final_state(StateId::new("end"));
    unreachable.add_transition(Transition::new(
        StateId::new("start"),
        StateId::new("reachable"),
        "go",
    ));
    unreachable.add_transition(Transition::new(
        StateId::new("reachable"),
        StateId::new("end"),
        "finish",
    ));

    match StateMachineValidator::validate(&unreachable) {
        Ok(()) => println!("\nUnreachable machine: passed (unexpected)"),
        Err(errors) => {
            println!("\nUnreachable state machine errors:");
            for err in &errors {
                println!("  - {}", err);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 8. Real LLM Demo — LLM-guided state transition
    // -----------------------------------------------------------------------
    println!("\n--- 8. Real LLM Demo ---");
    println!("Ask an LLM to decide the next state transition.\n");

    let model = shared::get_chat_model(vec![
        "The order should transition to 'confirmed' because all required fields are present and valid.".into(),
    ]);
    let messages = vec![
        Message::system("You are a state machine advisor. Given a state and context, suggest the next transition."),
        Message::human("Current state: 'pending', context: {\"order_id\": \"ORD-003\", \"items\": 3, \"payment\": \"verified\"}. What transition should fire next?"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM transition advice: {}", gen.message.content().text());
    }

    println!("\n=== State Machine Example Complete ===");
    Ok(())
}
