//! State updates flow through `apply()`. A node returns a partial
//! `Update`; the engine merges it into the running state.

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State { items: Vec<String>, total: u32 }
#[derive(Default, Clone)]
struct Update { add_item: Option<String>, add_total: u32 }
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(s) = u.add_item { self.items.push(s); }
        self.total += u.add_total;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let push = node_fn::<State, _, _>("push", |s, _| {
        let n = s.items.len();
        async move {
            if n >= 3 {
                Ok(NodeOut { update: Update::default(), goto: Goto::end() })
            } else {
                Ok(NodeOut {
                    update: Update { add_item: Some(format!("item{n}")), add_total: 10 },
                    goto: Goto::node("push"),
                })
            }
        }
    });
    let g = Graph::<State>::new().node("push", push).start_at("push").compile()?;
    let f = g.invoke(State::default(), Default::default()).await?;
    println!("items: {:?}, total: {}", f.items, f.total);
    Ok(())
}
