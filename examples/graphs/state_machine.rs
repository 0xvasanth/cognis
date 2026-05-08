//! Tiny state machine: a counter that increments until it hits 5.

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
    let tick = node_fn::<State, _, _>("tick", |s, _| {
        let cur = s.count;
        async move {
            if cur >= 5 {
                Ok(NodeOut { update: Update { count: 0 }, goto: Goto::end() })
            } else {
                Ok(NodeOut { update: Update { count: 1 }, goto: Goto::node("tick") })
            }
        }
    });
    let graph = Graph::<State>::new().node("tick", tick).start_at("tick").compile()?;
    let final_state = graph.invoke(State::default(), Default::default()).await?;
    println!("final count: {}", final_state.count);
    Ok(())
}
