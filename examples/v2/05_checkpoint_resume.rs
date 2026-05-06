//! Checkpoint + time-travel demo. Builds a small graph, runs it with a
//! checkpointer attached, then loads each saved step.

use std::sync::Arc;

use cognis2::prelude::*;

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
