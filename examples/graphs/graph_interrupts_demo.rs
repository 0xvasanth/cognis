//! Interrupt before a node runs, edit the saved state, then resume.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::CognisError;

#[derive(Default, Clone, Debug)]
struct State {
    count: u32,
}
#[derive(Default, Clone)]
struct Update {
    count: u32,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        self.count += u.count;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp: Arc<dyn Checkpointer<State>> = Arc::new(InMemoryCheckpointer::<State>::new());
    let bump = node_fn::<State, _, _>("bump", |s, _| {
        let cur = s.count;
        async move {
            if cur >= 4 {
                Ok(NodeOut {
                    update: Update { count: 0 },
                    goto: Goto::end(),
                })
            } else {
                Ok(NodeOut {
                    update: Update { count: 1 },
                    goto: Goto::node("bump"),
                })
            }
        }
    });
    let g = Graph::<State>::new()
        .node("bump", bump)
        .start_at("bump")
        .compile()?
        .with_checkpointer(cp)
        .with_interrupt_before(["bump"]);

    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let res = g.invoke(State::default(), cfg.clone()).await;
    match res {
        Err(CognisError::GraphInterrupted { kind, step, .. }) => {
            println!("paused {kind} bump at step {step}");
        }
        other => println!("unexpected: {other:?}"),
    }
    let saved = g.get_state(run_id).await?.unwrap_or_default();
    println!("saved state: count={}", saved.count);
    Ok(())
}
