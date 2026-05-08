//! Two nodes, deterministic transition: `start` → `finish` → end.

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State {
    trail: Vec<String>,
}
#[derive(Default, Clone)]
struct Update {
    push: Option<String>,
}
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) {
        if let Some(s) = u.push {
            self.trail.push(s);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let start = node_fn::<State, _, _>("start", |_s, _| async {
        Ok(NodeOut {
            update: Update {
                push: Some("start".into()),
            },
            goto: Goto::node("finish"),
        })
    });
    let finish = node_fn::<State, _, _>("finish", |_s, _| async {
        Ok(NodeOut {
            update: Update {
                push: Some("finish".into()),
            },
            goto: Goto::end(),
        })
    });
    let g = Graph::<State>::new()
        .node("start", start)
        .node("finish", finish)
        .start_at("start")
        .compile()?;
    let f = g.invoke(State::default(), Default::default()).await?;
    println!("trail: {:?}", f.trail);
    Ok(())
}
