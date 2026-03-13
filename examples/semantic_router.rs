//! Semantic Router Example
//!
//! Demonstrates how to use SemanticRouter and RouterChain to route queries
//! to different prompt chains based on embedding similarity. Queries are
//! matched to the closest route description and handled with route-specific
//! prompt templates.
//!
//! No API keys required -- uses DeterministicFakeEmbedding and FakeListChatModel.
//!
//! Run with: cargo run -p cognis-examples --example semantic_router

mod shared;

use std::sync::Arc;

use cognis::chains::router::{Route, RouterChain, SemanticRouter};
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Semantic Router Example ===\n");

    // Step 1: Define routes with descriptive text for semantic matching.
    //
    // Each route has a name and a description that the router uses to match
    // incoming queries via embedding similarity.
    let routes = vec![
        Route::new(
            "math",
            "mathematics calculations arithmetic algebra geometry numbers equations formulas",
        )
        .with_prompt_template("You are a math tutor. Solve this step by step: {query}"),
        Route::new(
            "science",
            "physics chemistry biology experiments atoms molecules forces energy",
        )
        .with_prompt_template("You are a science expert. Explain clearly: {query}"),
        Route::new(
            "history",
            "historical events dates civilizations wars leaders empires ancient medieval",
        )
        .with_prompt_template("You are a history professor. Provide context: {query}"),
        Route::new(
            "programming",
            "code software programming languages algorithms data structures debugging",
        )
        .with_prompt_template("You are a senior developer. Explain with examples: {query}"),
    ];

    println!("Defined {} routes:", routes.len());
    for route in &routes {
        println!("  - {} : {}", route.name, &route.description[..50]);
    }
    println!();

    // Step 2: Create the SemanticRouter with fake embeddings.
    //
    // DeterministicFakeEmbedding produces consistent hash-based vectors,
    // so identical text always matches with similarity 1.0.
    let embedding: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(128));
    let router = SemanticRouter::new(embedding, routes).await?;

    println!("Step 2: SemanticRouter created (embedding dim=128)\n");

    // Step 3: Route several queries and show which route was selected.
    println!("--- Routing Queries (SemanticRouter only) ---\n");

    let queries = [
        "What is the quadratic formula for solving equations?",
        "How do atoms bond together in chemistry?",
        "When did the Roman Empire fall?",
        "How do I implement a binary search tree?",
        "What are the laws of thermodynamics?",
    ];

    for query in &queries {
        let (route, score) = router.route_with_score(query).await?;
        println!("  Query: \"{query}\"");
        println!("  Route: {} (score: {:.4})\n", route.name, score);
    }

    // Step 4: Create a full RouterChain that also calls an LLM.
    //
    // The RouterChain combines routing with LLM invocation: it selects the
    // best route, picks the appropriate prompt template, and sends the
    // formatted prompt to the model.
    println!("--- RouterChain (Route + LLM) ---\n");

    let llm = shared::get_chat_model(vec![
        "The quadratic formula is x = (-b +/- sqrt(b^2 - 4ac)) / 2a. This solves any equation of the form ax^2 + bx + c = 0.".into(),
        "Atoms bond through ionic bonds (electron transfer), covalent bonds (electron sharing), and metallic bonds. The type depends on electronegativity differences.".into(),
        "The Western Roman Empire fell in 476 AD when Odoacer deposed the last emperor, Romulus Augustulus. Contributing factors included economic decline, military overextension, and barbarian invasions.".into(),
    ]);

    let chain = RouterChain::new(router, llm);

    let chain_queries = [
        "What is the quadratic formula for solving equations?",
        "How do atoms bond together in chemistry?",
        "When did the Roman Empire fall?",
    ];

    for query in &chain_queries {
        let result = chain.call(query).await?;
        println!("  Query: \"{query}\"");
        println!(
            "  Route: {} (confidence: {:.4})",
            result.route_name, result.confidence
        );
        println!("  Answer: {}\n", result.answer);
    }

    println!("Done!");
    Ok(())
}
