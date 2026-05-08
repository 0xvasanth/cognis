//! Linear pipeline — three nodes wired in order.

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let a = node_fn::<(), _, _>("a", |_, _| async {
        println!("a");
        Ok(NodeOut {
            update: (),
            goto: Goto::node("b"),
        })
    });
    let b = node_fn::<(), _, _>("b", |_, _| async {
        println!("b");
        Ok(NodeOut {
            update: (),
            goto: Goto::node("c"),
        })
    });
    let c = node_fn::<(), _, _>("c", |_, _| async {
        println!("c");
        Ok(NodeOut {
            update: (),
            goto: Goto::end(),
        })
    });
    let g = Graph::<()>::new()
        .node("a", a)
        .node("b", b)
        .node("c", c)
        .start_at("a")
        .compile()?;
    let _ = g.invoke((), Default::default()).await?;
    Ok(())
}
