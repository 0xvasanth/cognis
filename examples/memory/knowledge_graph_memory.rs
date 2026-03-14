//! Knowledge Graph Memory Example
//!
//! Demonstrates knowledge graph memory: manual triples, regex extraction,
//! querying, merging, and KnowledgeGraphMemory with conversations.
//!
//! Run with: `cargo run -p cognis-examples --example knowledge_graph_memory`

#[path = "../shared.rs"]
mod shared;
use cognis::memory::knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphMemory, KnowledgeTriple, RegexTripleExtractor, TripleExtractor,
};
use cognis::memory::BaseMemory;
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a knowledge graph manually
    let mut graph = KnowledgeGraph::new();
    graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp"));
    graph.add_triple(KnowledgeTriple::new("Bob", "works at", "TechStart"));
    graph.add_triple(KnowledgeTriple::new("Alice", "knows", "Bob"));
    graph.add_triple(KnowledgeTriple::new("Alice", "lives in", "San Francisco"));
    graph.add_triple(KnowledgeTriple::new("Bob", "manages", "Project Alpha").with_confidence(0.9));

    println!("Graph: {} triples", graph.len());
    println!("{}", graph.to_natural_language());

    // Query by entity
    let alice_triples = graph.get_triples_for_entity("Alice");
    println!("Alice triples: {}", alice_triples.len());

    let related = graph.get_related_entities("Alice");
    println!("Related to Alice: {:?}", related);

    // Regex triple extraction
    let extractor = RegexTripleExtractor::new();
    let text = "Alice is a software engineer. Bob works at Google. Charlie lives in New York.";
    let extracted = extractor.extract_triples(text);
    println!("Extracted {} triples from text", extracted.len());
    for triple in &extracted {
        println!(
            "  {} --[{}]--> {}",
            triple.subject, triple.predicate, triple.object
        );
    }

    // Graph merging with deduplication
    let mut graph_a = KnowledgeGraph::new();
    graph_a.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp"));
    graph_a.add_triple(KnowledgeTriple::new("Bob", "lives in", "Berlin"));

    let mut graph_b = KnowledgeGraph::new();
    graph_b.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp"));
    graph_b.add_triple(KnowledgeTriple::new("Charlie", "teaches", "Rust"));

    graph_a.merge(&graph_b);
    println!("Merged: {} triples (deduplicated)", graph_a.len());

    // KnowledgeGraphMemory with conversation
    let memory = KnowledgeGraphMemory::builder()
        .memory_key("chat_history")
        .knowledge_key("knowledge_context")
        .initial_triples(vec![KnowledgeTriple::new(
            "Cognis",
            "is",
            "an LLM framework",
        )])
        .build();

    memory
        .save_context(
            &Message::human("Alice works at Google and she lives in London."),
            &Message::ai("Interesting!"),
        )
        .await?;

    memory
        .save_context(
            &Message::human("Bob knows Alice and he created a new startup."),
            &Message::ai("Great to hear!"),
        )
        .await?;

    println!(
        "After conversation: {} triples",
        memory.triple_count().await
    );
    println!(
        "Alice: {}",
        memory.get_knowledge_for(&["Alice".to_string()]).await
    );

    // LLM demo
    let model = shared::get_chat_model(vec![
        "Triples:\n- (Elon Musk, founded, SpaceX)\n- (SpaceX, is, aerospace company)\n- (SpaceX, headquartered in, Hawthorne California)".into(),
    ]);
    let messages = vec![Message::human(
        "Extract knowledge triples from: 'Elon Musk founded SpaceX, an aerospace company headquartered in Hawthorne, California.'",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
