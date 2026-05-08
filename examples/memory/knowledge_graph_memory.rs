//! V2 doesn't ship a built-in KnowledgeGraphMemory. The Memory trait
//! is the extension point: implement it with whatever graph / DB you
//! want. Below is a tiny in-memory triple store wrapped as `Memory`.

use cognis::prelude::*;
use cognis::Memory;

#[derive(Default)]
struct TripleMemory {
    triples: Vec<(String, String, String)>, // (subject, predicate, object)
    history: Vec<Message>,
}

impl TripleMemory {
    fn record_fact(&mut self, s: &str, p: &str, o: &str) {
        self.triples.push((s.into(), p.into(), o.into()));
    }
    fn facts(&self) -> Vec<String> {
        self.triples.iter().map(|(s, p, o)| format!("{s} {p} {o}.")).collect()
    }
}

impl Memory for TripleMemory {
    fn read(&self) -> &[Message] { &self.history }
    fn write(&mut self, msg: Message) { self.history.push(msg); }
    fn clear(&mut self) { self.history.clear(); self.triples.clear(); }
    fn seed(&self) -> Vec<Message> {
        let kb = format!("Known facts:\n{}", self.facts().join("\n"));
        let mut out = vec![Message::system(kb)];
        out.extend(self.history.iter().cloned());
        out
    }
}

fn main() {
    let mut m = TripleMemory::default();
    m.record_fact("Rust", "released_in", "2010");
    m.record_fact("Rust", "creator", "Mozilla");
    m.write(Message::human("Tell me about Rust."));
    let seed = m.seed();
    println!("seed has {} messages; first one is the synthesized KB:\n{}", seed.len(), seed[0].content());
}
