//! Graph Runtime Example
//!
//! Shows how to build a Runtime with context injection, a shared store,
//! and stream writing, then use it to scope an LLM call — the pattern
//! every graph node follows at execution time.
//!
//! Run with: `cargo run -p cognis-examples --example graph_runtime`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use cognis_core::messages::Message;
use cognisgraph::config::{InMemoryStore, Store};
use cognisgraph::runtime::{RuntimeBuilder, RuntimeConfig, RuntimeScope, StreamWriter};
use serde_json::json;

/// Application-specific context injected into every graph node.
#[derive(Debug, Clone, Default)]
struct RequestContext {
    user_id: String,
    session_id: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -- 1. Prepare a shared store with a system prompt ----------------------
    let store = Arc::new(InMemoryStore::new());
    store.put(
        &["prompts"],
        "system",
        json!("You are a concise technical assistant."),
    )?;

    // -- 2. Set up a stream writer to capture node events --------------------
    let events: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_handle = events.clone();
    let writer: StreamWriter = Arc::new(move |v| {
        events_handle.lock().unwrap().push(v);
    });

    // -- 3. Build the runtime ------------------------------------------------
    let config = RuntimeConfig::new()
        .with_run_id("run-001")
        .with_thread_id("thread-42")
        .with_tag("demo")
        .with_metadata("model_name", json!("llama3.2"));

    let ctx = RequestContext {
        user_id: "user-123".into(),
        session_id: "sess-abc".into(),
    };

    let rt = RuntimeBuilder::new()
        .with_context(ctx)
        .with_store(store.clone())
        .with_config(config)
        .with_stream_writer(writer)
        .build();

    // -- 4. Enter a scope and invoke the LLM ---------------------------------
    let scope = RuntimeScope::new(rt);
    println!("Runtime ready — user={}", scope.runtime().context().user_id);

    // Emit a stream event (nodes do this to report progress)
    scope
        .runtime()
        .write_to_stream(json!({"event": "node_start", "node": "llm"}));

    // Retrieve the system prompt from the store
    let system_text = scope
        .runtime()
        .store()
        .unwrap()
        .get(&["prompts"], "system")?
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "You are a helpful assistant.".into());

    let model = shared::get_chat_model(vec![
        "A runtime scope bundles run metadata, store access, and stream \
         writing into a single injectable context for graph nodes."
            .into(),
    ]);

    let messages = vec![
        Message::system(&system_text),
        Message::human(
            "What does a runtime scope do in a graph execution framework? One sentence.",
        ),
    ];

    let response = model.invoke_messages(&messages, None).await?;
    println!("LLM: {}", response.base.content.text());

    // Close the stream
    scope
        .runtime()
        .write_to_stream(json!({"event": "node_end", "node": "llm"}));

    // -- 5. Capture result as "previous" for downstream nodes ----------------
    let mut rt = scope.into_inner();
    rt.set_previous(Some(json!({
        "response": response.base.content.text(),
        "run_id": rt.config().run_id,
    })));

    println!(
        "Captured {} stream events, run complete.",
        events.lock().unwrap().len()
    );

    Ok(())
}
