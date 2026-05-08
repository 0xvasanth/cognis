//! LongContextReorder — push the most-/least-relevant docs to the
//! ends of the list (where LLMs read best). The middle gets the noise.

use cognis_rag::{Document, LongContextReorder};

fn main() {
    let docs = vec![
        Document::new("most relevant"),
        Document::new("a bit relevant"),
        Document::new("less relevant"),
        Document::new("least relevant"),
    ];
    let reordered = LongContextReorder::reorder(docs.clone());
    println!(
        "original  : {:?}",
        docs.iter().map(|d| &d.content).collect::<Vec<_>>()
    );
    println!(
        "reordered : {:?}",
        reordered.iter().map(|d| &d.content).collect::<Vec<_>>()
    );
}
