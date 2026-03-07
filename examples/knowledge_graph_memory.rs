//! Knowledge Graph Memory Example
//!
//! Demonstrates the knowledge graph memory system:
//! - Creating a `KnowledgeGraph` and adding triples manually
//! - Using `RegexTripleExtractor` to extract triples from natural language
//! - Querying the graph by entity, relationship, and search
//! - Using `KnowledgeGraphMemory` with the builder pattern
//! - Merging graphs and natural language output
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example knowledge_graph_memory`

use rustchain::memory::knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphMemory, KnowledgeTriple, RegexTripleExtractor, TripleExtractor,
};
use rustchain::memory::BaseMemory;
use rustchain_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Knowledge Graph Memory Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a KnowledgeGraph and add triples manually
    // -----------------------------------------------------------------------
    println!("--- 1. Manual Knowledge Graph ---");
    println!("Build a graph by adding subject-predicate-object triples.\n");

    let mut graph = KnowledgeGraph::new();

    graph.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp"));
    graph.add_triple(KnowledgeTriple::new("Bob", "works at", "TechStart"));
    graph.add_triple(KnowledgeTriple::new("Alice", "knows", "Bob"));
    graph.add_triple(KnowledgeTriple::new("Alice", "lives in", "San Francisco"));
    graph.add_triple(KnowledgeTriple::new("Bob", "manages", "Project Alpha").with_confidence(0.9));
    graph.add_triple(
        KnowledgeTriple::new("Charlie", "created", "RustChain")
            .with_confidence(0.95)
            .with_source("Internal docs"),
    );

    println!("Graph has {} triples", graph.len());
    println!("All triples as natural language:");
    println!("  {}\n", graph.to_natural_language());

    // -----------------------------------------------------------------------
    // 2. Query by entity
    // -----------------------------------------------------------------------
    println!("--- 2. Query by Entity ---");
    println!("Find all triples involving a specific entity.\n");

    let alice_triples = graph.get_triples_for_entity("Alice");
    println!("Triples involving Alice ({} found):", alice_triples.len());
    for triple in &alice_triples {
        println!(
            "  {} {} {}",
            triple.subject, triple.predicate, triple.object
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. Related entities
    // -----------------------------------------------------------------------
    println!("--- 3. Related Entities ---");
    println!("Find all entities connected to a given entity.\n");

    let related = graph.get_related_entities("Alice");
    println!("Entities related to Alice:");
    for entity in &related {
        println!("  - {}", entity);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. Search triples
    // -----------------------------------------------------------------------
    println!("--- 4. Search Triples ---");
    println!("Fuzzy search across subjects, predicates, and objects.\n");

    let work_results = graph.search_triples("works");
    println!("Search for 'works' ({} results):", work_results.len());
    for triple in &work_results {
        println!("  {}", triple.to_natural_language());
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. RegexTripleExtractor
    // -----------------------------------------------------------------------
    println!("--- 5. RegexTripleExtractor ---");
    println!("Automatically extract triples from natural language text.\n");

    let extractor = RegexTripleExtractor::new();
    println!(
        "Extractor loaded with {} default patterns",
        extractor.pattern_count()
    );

    let text = "Alice is a software engineer. Bob works at Google. \
                Charlie lives in New York. Diana knows Charlie. \
                Eve created a machine learning framework.";

    println!("Input text: \"{}\"\n", text);

    let extracted = extractor.extract_triples(text);
    println!("Extracted {} triples:", extracted.len());
    for triple in &extracted {
        println!(
            "  {} --[{}]--> {} (confidence: {:.1})",
            triple.subject, triple.predicate, triple.object, triple.confidence
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Graph merging
    // -----------------------------------------------------------------------
    println!("--- 6. Graph Merging ---");
    println!("Merge two graphs with automatic deduplication.\n");

    let mut graph_a = KnowledgeGraph::new();
    graph_a.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp"));
    graph_a.add_triple(KnowledgeTriple::new("Bob", "lives in", "Berlin"));

    let mut graph_b = KnowledgeGraph::new();
    graph_b.add_triple(KnowledgeTriple::new("Alice", "works at", "Acme Corp")); // duplicate
    graph_b.add_triple(KnowledgeTriple::new("Charlie", "teaches", "Rust"));

    println!("Graph A: {} triples", graph_a.len());
    println!("Graph B: {} triples", graph_b.len());

    graph_a.merge(&graph_b);
    println!("After merge: {} triples (duplicate removed)", graph_a.len());
    println!("  {}\n", graph_a.to_natural_language());

    // -----------------------------------------------------------------------
    // 7. KnowledgeGraphMemory with conversation
    // -----------------------------------------------------------------------
    println!("--- 7. KnowledgeGraphMemory ---");
    println!("Automatically extracts and stores knowledge from conversations.\n");

    let memory = KnowledgeGraphMemory::builder()
        .memory_key("chat_history")
        .knowledge_key("knowledge_context")
        .initial_triples(vec![KnowledgeTriple::new(
            "RustChain",
            "is",
            "an LLM framework",
        )])
        .build();

    println!("Initial triple count: {}", memory.triple_count().await);

    // Simulate a conversation
    memory
        .save_context(
            &Message::human("Alice works at Google and she lives in London."),
            &Message::ai("That's interesting! Alice must enjoy the tech scene there."),
        )
        .await?;

    memory
        .save_context(
            &Message::human("Bob knows Alice and he created a new startup."),
            &Message::ai("Great to hear about Bob's entrepreneurial journey!"),
        )
        .await?;

    println!(
        "After conversation: {} triples extracted",
        memory.triple_count().await
    );

    // Query knowledge for specific entities
    let alice_knowledge = memory.get_knowledge_for(&["Alice".to_string()]).await;
    println!("\nKnowledge about Alice:");
    println!("  {}", alice_knowledge);

    let bob_knowledge = memory.get_knowledge_for(&["Bob".to_string()]).await;
    println!("\nKnowledge about Bob:");
    println!("  {}", bob_knowledge);

    // Load memory variables
    let vars = memory.load_memory_variables().await?;
    println!("\nMemory variables:");
    println!("  Keys: {:?}", vars.keys().collect::<Vec<_>>());

    if let Some(knowledge) = vars.get("knowledge_context") {
        println!(
            "  Knowledge context: {}",
            knowledge.as_str().unwrap_or("N/A")
        );
    }

    // Get a graph snapshot
    let snapshot = memory.graph_snapshot().await;
    println!("\nGraph snapshot: {} triples", snapshot.len());
    for triple in snapshot.triples() {
        println!("  {}", triple.to_natural_language());
    }

    println!("\n=== Knowledge Graph Memory Example Complete ===");
    Ok(())
}
