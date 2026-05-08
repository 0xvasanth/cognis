//! What you'll learn:
//!   How to attach a checkpointer to a graph, capture state at every
//!   step, and time-travel by loading any saved snapshot back.
//!
//! Why this matters:
//!   Checkpointers are the foundation for crash recovery,
//!   human-in-the-loop pauses, and debugging stuck runs. The same
//!   `Checkpointer` trait swaps between in-memory, sqlite, and
//!   postgres backends — your graph code is unchanged.
//!
//! Scenario:
//!   A counter graph with a `Checkpointer` attached. After the run, list
//!   every saved superstep — the shape you reach for when you need
//!   time-travel debugging or HITL workflows.
//!
//! Run with:
//!   cargo run -p cognis-examples --example 05_checkpoint_resume
//!
//! Sample output (against ollama / llama3.1):
//!   final count: 5
//!   checkpoints saved: [0, 1, 2, 3, 4, 5]
//!   step 0: count = 1
//!   step 1: count = 2
//!   step 2: count = 3
//!   step 3: count = 4
//!   step 4: count = 5
//!   step 5: count = 5

use std::sync::Arc;

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State {
    count: u32,
}

#[derive(Default, Clone)]
struct StateUpdate {
    count: u32,
}

impl GraphState for State {
    type Update = StateUpdate;
    fn apply(&mut self, u: Self::Update) {
        self.count += u.count;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp: Arc<dyn Checkpointer<State>> = Arc::new(InMemoryCheckpointer::<State>::new());

    let tick = node_fn::<State, _, _>("tick", |s, _| {
        let cur = s.count;
        async move {
            if cur >= 5 {
                Ok(NodeOut {
                    update: StateUpdate { count: 0 },
                    goto: Goto::end(),
                })
            } else {
                Ok(NodeOut {
                    update: StateUpdate { count: 1 },
                    goto: Goto::node("tick"),
                })
            }
        }
    });

    let graph = Graph::<State>::new()
        .node("tick", tick)
        .start_at("tick")
        .compile()?
        .with_checkpointer(cp.clone());

    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let final_state = graph.invoke(State::default(), cfg).await?;
    println!("final count: {}", final_state.count);

    let steps = cp.list(run_id).await?;
    println!("checkpoints saved: {:?}", steps);

    // Time-travel: load count at each saved step.
    for s in &steps {
        if let Some(snapshot) = cp.load(run_id, Some(*s)).await? {
            println!("step {}: count = {}", s, snapshot.count);
        }
    }
    Ok(())
}
