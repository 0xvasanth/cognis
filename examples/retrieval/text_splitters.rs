//! What you'll learn:
//!   How `RecursiveCharSplitter`, `MarkdownSplitter`, and
//!   `SentenceSplitter` carve the *same* document differently — and
//!   why splitter choice is one of the highest-leverage knobs in a
//!   RAG pipeline.
//!
//! Why this matters:
//!   Pick the wrong splitter and your chunks tear sentences in half
//!   or merge unrelated sections; pick the right one and embeddings
//!   actually reflect the document's structure. For a Markdown blog
//!   post the right answer is "split on headings"; for free-form
//!   prose, sentence boundaries; for prose with no structure,
//!   recursive descent.
//!
//! Scenario:
//!   We take a short Markdown blog post (intro + two H2 sections)
//!   and run all three splitters over it. The output makes the
//!   trade-offs visible: recursive cuts on character count regardless
//!   of structure; sentence groups by sentence; markdown follows
//!   `##` headings.
//!
//! Run with:
//!   cargo run -p cognis-examples --example retrieval_text_splitters
//!
//! Sample output (against ollama / llama3.1):
//!
//!   === RecursiveCharSplitter(200) -> 6 chunks ===
//!   first chunk (26 chars): # Why Rust matters in 2026
//!
//!   === SentenceSplitter(2 sentences) -> 7 chunks ===
//!   first chunk (121 chars): # Why Rust matters in 2026 |  | Rust keeps showing up in the places we least expected: kernels, browsers, even web front
//!
//!   === MarkdownSplitter(headings) -> 3 chunks ===
//!   first chunk (183 chars): Rust keeps showing up in the places we least expected: kernels, browsers, even web frontends. The reason is simple — it

use cognis_rag::{
    Document, MarkdownSplitter, RecursiveCharSplitter, SentenceSplitter, TextSplitter,
};

const POST: &str = "\
# Why Rust matters in 2026

Rust keeps showing up in the places we least expected: kernels, \
browsers, even web frontends. The reason is simple — it pays off \
where memory safety used to mean garbage collection.

## Performance without a runtime

Zero-cost abstractions are not a marketing claim. The compiler \
generates the same machine code you would write by hand. Async \
tasks compile down to small state machines instead of OS threads.

## Safety without overhead

The borrow checker is the famous part. The quieter win is that \
Rust's type system catches whole categories of bugs at compile \
time — null pointers, data races, use-after-free.
";

fn show(label: &str, chunks: Vec<Document>) {
    println!("\n=== {label} -> {} chunks ===", chunks.len());
    if let Some(first) = chunks.first() {
        let body = first.content.replace('\n', " | ");
        let trimmed: String = body.chars().take(120).collect();
        println!("first chunk ({} chars): {trimmed}", first.content.len());
    }
}

fn main() {
    let doc = Document::new(POST);

    // Recursive char splitter: cuts on the longest separator that fits.
    // Use this for messy free-form text where no structural signal is
    // reliable.
    let recursive = RecursiveCharSplitter::new()
        .with_chunk_size(200)
        .with_overlap(0);
    show("RecursiveCharSplitter(200)", recursive.split(&doc));

    // Sentence splitter: groups N sentences per chunk. Use this for
    // prose where sentence boundaries should never be torn.
    let sentence = SentenceSplitter::new().with_chunk_size(2);
    show("SentenceSplitter(2 sentences)", sentence.split(&doc));

    // Markdown splitter: respects `#` / `##` boundaries. Use this on
    // anything with reliable heading structure (docs, READMEs, blog
    // posts) so each section becomes its own chunk.
    let markdown = MarkdownSplitter::new().with_chunk_size(500);
    show("MarkdownSplitter(headings)", markdown.split(&doc));
}
