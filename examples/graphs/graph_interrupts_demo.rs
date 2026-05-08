//! What you'll learn:
//!   How `with_interrupt_before` pauses a graph at a named node,
//!   surfaces a `GraphInterrupted` error, and lets you peek at the
//!   saved state before deciding whether to resume — the building
//!   block for any HITL approval flow.
//!
//! Why this matters:
//!   Human-in-the-loop is built on this primitive. The graph
//!   serialises its state to the checkpointer, raises a typed
//!   interrupt, and waits — a UI can show the reviewer the pending
//!   draft, collect approval (or edits), and resume from where the
//!   graph paused. Same pattern whether it's email, code commits,
//!   or production deploys.
//!
//! Scenario:
//!   The agent has just drafted a customer email. The graph would
//!   next call the `send` node, but we want a human to approve the
//!   draft first. `with_interrupt_before(["send"])` pauses there;
//!   we inspect the saved state, "approve", and resume against a
//!   no-interrupt clone of the graph — at which point the email
//!   actually goes out.
//!
//! Run with:
//!   cargo run -p cognis-examples --example graphs_interrupts
//!
//! Sample output (against ollama / llama3.1):
//!   [draft] composed email: "Hi Maya, your refund of $42.00 has been processed."
//!   [host] paused before at step 1, awaiting approval
//!   [host] saved state: draft = "Hi Maya, your refund of $42.00 has been processed."
//!   [host] reviewer approved — resuming
//!   [send] dispatching email: "Hi Maya, your refund of $42.00 has been processed."
//!   [host] final: sent = true

use std::sync::Arc;

use cognis::prelude::*;
use cognis::CompiledGraph;
use cognis_core::CognisError;

#[derive(Default, Clone, Debug)]
struct State {
    draft: String,
    sent: bool,
}
#[derive(Default, Clone)]
struct Update {
    draft: Option<String>,
    sent: Option<bool>,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(d) = u.draft {
            self.draft = d;
        }
        if let Some(s) = u.sent {
            self.sent = s;
        }
    }
}

fn build_graph(
    cp: Arc<dyn Checkpointer<State>>,
    interrupts: bool,
) -> Result<CompiledGraph<State>> {
    // Node 1: write a draft email. In real code this is an LLM call.
    let draft = node_fn::<State, _, _>("draft", |_, _| async {
        let body = "Hi Maya, your refund of $42.00 has been processed.";
        println!("[draft] composed email: {body:?}");
        Ok(NodeOut {
            update: Update {
                draft: Some(body.into()),
                sent: None,
            },
            goto: Goto::node("send"),
        })
    });

    // Node 2: actually send. We pause BEFORE this node so a human
    // can review the draft.
    let send = node_fn::<State, _, _>("send", |s, _| {
        let d = s.draft.clone();
        async move {
            println!("[send] dispatching email: {d:?}");
            Ok(NodeOut {
                update: Update {
                    draft: None,
                    sent: Some(true),
                },
                goto: Goto::end(),
            })
        }
    });

    let compiled = Graph::<State>::new()
        .node("draft", draft)
        .node("send", send)
        .start_at("draft")
        .compile()?
        .with_checkpointer(cp);

    Ok(if interrupts {
        compiled.with_interrupt_before(["send"])
    } else {
        compiled
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp: Arc<dyn Checkpointer<State>> = Arc::new(InMemoryCheckpointer::<State>::new());

    // Run #1 — interrupts ON. Pauses before `send`.
    let with_pause = build_graph(cp.clone(), true)?;
    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;

    let pause_step = match with_pause.invoke(State::default(), cfg.clone()).await {
        Err(CognisError::GraphInterrupted { kind, step, .. }) => {
            println!("[host] paused {kind} at step {step}, awaiting approval");
            step
        }
        other => {
            println!("[host] unexpected: {other:?}");
            return Ok(());
        }
    };

    // Show the reviewer the saved draft. The UI would render this and
    // wait for an approve / edit / reject click.
    let saved = with_pause.get_state(run_id).await?.unwrap_or_default();
    println!("[host] saved state: draft = {:?}", saved.draft);
    println!("[host] reviewer approved — resuming");

    // Run #2 — interrupts OFF, same checkpointer + same run_id.
    // `resume` reloads the active node set from the checkpoint and
    // continues past the original pause point.
    let no_pause = build_graph(cp, false)?;
    let final_state = no_pause.resume(run_id, pause_step, saved, cfg).await?;
    println!("[host] final: sent = {}", final_state.sent);
    Ok(())
}
