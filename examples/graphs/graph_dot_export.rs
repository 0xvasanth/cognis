//! Render a CompiledGraph as GraphViz DOT — pipe through `dot -Tsvg`
//! or paste into an online viewer.

use cognis::prelude::*;

#[derive(Default, Clone)]
struct S;
impl GraphState for S {
    type Update = ();
    fn apply(&mut self, _: ()) {}
}

#[tokio::main]
async fn main() -> Result<()> {
    let plan = node_fn::<S, _, _>("plan", |_, _| async {
        Ok(NodeOut {
            update: (),
            goto: Goto::node("execute"),
        })
    });
    let execute = node_fn::<S, _, _>("execute", |_, _| async {
        Ok(NodeOut {
            update: (),
            goto: Goto::node("review"),
        })
    });
    let review = node_fn::<S, _, _>("review", |_, _| async {
        Ok(NodeOut {
            update: (),
            goto: Goto::end(),
        })
    });

    let g = Graph::<S>::new()
        .node("plan", plan)
        .node("execute", execute)
        .node("review", review)
        .edge("plan", "execute")
        .edge("execute", "review")
        .start_at("plan")
        .compile()?;

    println!("--- DOT ---");
    println!("{}", g.to_dot());
    println!("--- Mermaid ---");
    println!("{}", g.to_mermaid());
    println!("--- ASCII ---");
    println!("{}", g.to_ascii());
    Ok(())
}
