//! `Graph::compile` validates the graph before producing a CompiledGraph.
//! Missing `start_at`, dangling edges, or unknown node names all error.

use cognis::prelude::*;

#[derive(Default, Clone)]
struct S {}
impl GraphState for S {
    type Update = ();
    fn apply(&mut self, _: ()) {}
}

fn main() {
    let r: Result<_> = Graph::<S>::new().compile();
    match r {
        Ok(_) => println!("(unexpected) compile succeeded"),
        Err(e) => println!("compile rejected: {e}"),
    }
}
