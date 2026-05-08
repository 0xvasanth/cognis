//! Graph + InMemoryCheckpointer. Inspect each saved step after the run.

use std::sync::Arc;

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State { count: u32 }
#[derive(Default, Clone)]
struct Update { count: u32 }
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) { self.count += u.count; }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cp: Arc<dyn Checkpointer<State>> = Arc::new(InMemoryCheckpointer::<State>::new());
    let tick = node_fn::<State, _, _>("tick", |s, _| {
        let cur = s.count;
        async move {
            if cur >= 3 {
                Ok(NodeOut { update: Update { count: 0 }, goto: Goto::end() })
            } else {
                Ok(NodeOut { update: Update { count: 1 }, goto: Goto::node("tick") })
            }
        }
    });
    let g = Graph::<State>::new()
        .node("tick", tick)
        .start_at("tick")
        .compile()?
        .with_checkpointer(cp.clone());
    let cfg = RunnableConfig::default();
    let run_id = cfg.run_id;
    let _ = g.invoke(State::default(), cfg).await?;
    for s in cp.list(run_id).await? {
        let snap = cp.load(run_id, Some(s)).await?.unwrap();
        println!("step {s}: count={}", snap.count);
    }
    Ok(())
}
