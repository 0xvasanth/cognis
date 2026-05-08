//! Conditional routing — node returns a `Goto` based on state content.

use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State { input: String, route: String }
#[derive(Default, Clone)]
struct Update { route: Option<String> }
impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) { if let Some(r) = u.route { self.route = r; } }
}

#[tokio::main]
async fn main() -> Result<()> {
    let classify = node_fn::<State, _, _>("classify", |s, _| {
        let input = s.input.clone();
        async move {
            let route = if input.contains('?') { "qa" } else { "echo" };
            Ok(NodeOut {
                update: Update { route: Some(route.into()) },
                goto: Goto::node(route),
            })
        }
    });
    let qa = node_fn::<State, _, _>("qa", |_, _| async {
        println!("→ QA path"); Ok(NodeOut { update: Update::default(), goto: Goto::end() })
    });
    let echo = node_fn::<State, _, _>("echo", |_, _| async {
        println!("→ Echo path"); Ok(NodeOut { update: Update::default(), goto: Goto::end() })
    });

    let g = Graph::<State>::new()
        .node("classify", classify).node("qa", qa).node("echo", echo)
        .start_at("classify").compile()?;

    for input in ["What is 2+2?", "hello there"] {
        let f = g.invoke(State { input: input.into(), route: String::new() }, Default::default()).await?;
        println!("{input:?} → {}", f.route);
    }
    Ok(())
}
