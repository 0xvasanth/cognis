//! ProfilingObserver records per-node timings.

use std::sync::Arc;

use cognis::prelude::*;
use cognis_graph::{ProfilingObserver};

#[derive(Default, Clone, Debug)]
struct State { ticks: u32 }
#[derive(Default, Clone)]
struct Update { ticks: u32 }
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) { self.ticks += u.ticks; }
}

#[tokio::main]
async fn main() -> Result<()> {
    let bump = node_fn::<State, _, _>("bump", |s, _| {
        let cur = s.ticks;
        async move {
            if cur >= 3 { Ok(NodeOut { update: Update { ticks: 0 }, goto: Goto::end() }) }
            else {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                Ok(NodeOut { update: Update { ticks: 1 }, goto: Goto::node("bump") })
            }
        }
    });
    let g = Graph::<State>::new().node("bump", bump).start_at("bump").compile()?;
    let prof = Arc::new(ProfilingObserver::default());
    let mut cfg = RunnableConfig::default();
    cfg.observers.push(prof.clone());
    let _ = g.invoke(State::default(), cfg).await?;
    for (name, t) in prof.snapshot() {
        println!("{name:>10}: {} runs, total {} ns", t.count, t.total_ns);
    }
    Ok(())
}
