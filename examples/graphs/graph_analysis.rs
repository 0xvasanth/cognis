//! CompiledGraph exposes `node_names()` — handy for rendering simple
//! ASCII / DOT diagrams without a separate visitor.

use cognis::prelude::*;

#[derive(Default, Clone)]
struct S {}
impl GraphState for S {
    type Update = ();
    fn apply(&mut self, _: ()) {}
}

fn main() -> Result<()> {
    let a = node_fn::<S, _, _>("a", |_, _| async { Ok(NodeOut { update: (), goto: Goto::node("b") }) });
    let b = node_fn::<S, _, _>("b", |_, _| async { Ok(NodeOut { update: (), goto: Goto::end() }) });
    let g = Graph::<S>::new().node("a", a).node("b", b).start_at("a").compile()?;
    println!("digraph G {{");
    for n in g.node_names() { println!("  \"{n}\";"); }
    println!("}}");
    println!("nodes: {}", g.node_count());
    Ok(())
}
