//! Channels — multi-producer/single-consumer reducers attached to graph
//! state. Useful for fan-in patterns. Here we accumulate strings via a
//! `Vec<String>` field with a custom `apply` that appends.

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State {
    topics: Vec<String>,
}
#[derive(Default, Clone)]
struct Update {
    add: Option<String>,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(t) = u.add {
            self.topics.push(t);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let producer = node_fn::<State, _, _>("produce", |s, _| {
        let n = s.topics.len();
        async move {
            if n >= 3 {
                Ok(NodeOut {
                    update: Update::default(),
                    goto: Goto::end(),
                })
            } else {
                Ok(NodeOut {
                    update: Update {
                        add: Some(format!("topic-{n}")),
                    },
                    goto: Goto::node("produce"),
                })
            }
        }
    });
    let g = Graph::<State>::new()
        .node("produce", producer)
        .start_at("produce")
        .compile()?;
    let f = g.invoke(State::default(), Default::default()).await?;
    println!("collected: {:?}", f.topics);
    Ok(())
}
