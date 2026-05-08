//! Splitter walkthrough — Character / Recursive / Markdown / Sentence.

use cognis_rag::{
    CharacterSplitter, Document, MarkdownSplitter, RecursiveCharSplitter, SentenceSplitter,
    TextSplitter,
};

fn main() {
    let prose = Document::new(
        "Rust is a systems language. It is fast and safe.\n\nIt has \
         no GC. It uses ownership.",
    );

    let cs = CharacterSplitter::new().with_chunk_size(40).with_overlap(0);
    let rs = RecursiveCharSplitter::new()
        .with_chunk_size(40)
        .with_overlap(0);
    let ss = SentenceSplitter::new().with_chunk_size(2);

    println!("character → {}", cs.split(&prose).len());
    println!("recursive → {}", rs.split(&prose).len());
    println!("sentence  → {}", ss.split(&prose).len());

    let md = Document::new("# Title\n\nIntro paragraph.\n\n## Section\n\nBody one.\n\nBody two.");
    let ms = MarkdownSplitter::new().with_chunk_size(50);
    for (i, c) in ms.split(&md).into_iter().enumerate() {
        println!(
            "md[{i}] ({} chars): {}",
            c.content.len(),
            c.content.replace('\n', " ")
        );
    }
}
