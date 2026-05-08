//! What you'll learn:
//!   How attaching `InMemoryCheckpointer` to a multi-step graph
//!   captures one snapshot per step, and how to walk that history
//!   back to confirm exactly which steps ran — the foundation for
//!   "resume from where we crashed".
//!
//! Why this matters:
//!   The same `Checkpointer` trait that swaps to sqlite or postgres
//!   in production powers crash recovery, time-travel debugging, and
//!   human-in-the-loop pause/resume. The in-memory variant is what
//!   you'll use in tests and local dev; the API is identical.
//!
//! Scenario:
//!   A 5-step ETL pipeline (extract -> validate -> transform -> load
//!   -> notify). Each step writes its name to state and advances. If
//!   step 3 had crashed, the saved checkpoints from steps 1 and 2
//!   would be enough to resume at step 3 without re-running the
//!   earlier work. Here we run the whole pipeline, then list every
//!   checkpoint to confirm the per-step granularity.
//!
//! Run with:
//!   cargo run -p cognis-examples --example graphs_with_checkpoints
//!
//! Sample output (against ollama / llama3.1):
//!   [step] extract
//!   [step] validate
//!   [step] transform
//!   [step] load
//!   [step] notify
//!
//!   final completed: ["extract", "validate", "transform", "load", "notify"]
//!
//!   ...
//!     step 2: completed = ["extract", "validate", "transform"]
//!     step 3: completed = ["extract", "validate", "transform", "load"]
//!     step 4: completed = ["extract", "validate", "transform", "load", "notify"]

use std::sync::Arc;

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State {
    completed: Vec<String>,
}
#[derive(Default, Clone)]
struct Update {
    add: Option<String>,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(name) = u.add {
            self.completed.push(name);
        }
    }
}

/// Build a node that records its name and routes to `next`. In a
/// real ETL pipeline each of these would be doing the actual work.
fn step(name: &'static str, next: Goto) -> impl Node<State> {
    node_fn::<State, _, _>(name, move |_, _| {
        let next = next.clone();
        async move {
            println!("[step] {name}");
            Ok(NodeOut { update: Update { add: Some(name.into()) }, goto: next })
        }
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp: Arc<dyn Checkpointer<State>> = Arc::new(InMemoryCheckpointer::<State>::new());

    let g = Graph::<State>::new()
        .node("extract",   step("extract",   Goto::node("validate")))
        .node("validate",  step("validate",  Goto::node("transform")))
        .node("transform", step("transform", Goto::node("load")))
        .node("load",      step("load",      Goto::node("notify")))
        .node("notify",    step("notify",    Goto::end()))
        .start_at("extract")
        .compile()?
        .with_checkpointer(cp.clone());

    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let final_state = g.invoke(State::default(), cfg).await?;

    println!("\nfinal completed: {:?}", final_state.completed);
    println!("\n=== saved checkpoints (one per superstep) ===");
    for s in cp.list(run_id).await? {
        let snap = cp.load(run_id, Some(s)).await?.unwrap();
        println!("  step {s}: completed = {:?}", snap.completed);
    }
    Ok(())
}
