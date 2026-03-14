//! Graph Runtime Example
//!
//! Demonstrates the Runtime context injection system from cognisgraph,
//! which bundles run-scoped context, store access, stream writing, and
//! configuration into a single injectable object for graph nodes.
//!
//! Features shown:
//! - RuntimeConfig (run metadata, tags, thread IDs)
//! - RuntimeBuilder (fluent construction API)
//! - Runtime with custom context types
//! - RuntimeScope (RAII lifecycle management)
//! - NoContext (default placeholder)
//! - Stream writing
//! - LLM invocation within a runtime scope
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example graph_runtime`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use cognis_core::messages::Message;
use cognisgraph::config::{InMemoryStore, Store};
use cognisgraph::runtime::{
    NoContext, Runtime, RuntimeBuilder, RuntimeConfig, RuntimeProvider, RuntimeScope, StreamWriter,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Runtime Example ===\n");

    // -----------------------------------------------------------------------
    // 1. RuntimeConfig — run metadata
    // -----------------------------------------------------------------------
    println!("--- 1. RuntimeConfig: run metadata ---");
    let config = RuntimeConfig::new()
        .with_run_id("run-001")
        .with_thread_id("thread-42")
        .with_tag("production")
        .with_tag("v2")
        .with_tags(vec!["gpu", "batch"])
        .with_metadata("model", json!("llama3.2"))
        .with_metadata("temperature", json!(0.7));

    println!("  Run ID:    {:?}", config.run_id);
    println!("  Thread ID: {:?}", config.thread_id);
    println!("  Tags:      {:?}", config.tags);
    println!("  Metadata:  {:?}", config.metadata);

    // Serialization roundtrip
    let json_str = serde_json::to_string_pretty(&config)?;
    println!("  Serialized config:\n{}", textwrap(&json_str, "    "));
    let deserialized: RuntimeConfig = serde_json::from_str(&json_str)?;
    assert_eq!(config, deserialized);
    println!("  Roundtrip: OK");
    println!();

    // -----------------------------------------------------------------------
    // 2. RuntimeBuilder — fluent construction
    // -----------------------------------------------------------------------
    println!("--- 2. RuntimeBuilder: building runtimes ---");

    // Minimal runtime with NoContext
    let rt_minimal: Runtime<NoContext> = RuntimeBuilder::new().build();
    println!("  Minimal runtime context: {:?}", rt_minimal.context());
    println!("  Has store: {}", rt_minimal.store().is_some());
    println!("  Has previous: {}", rt_minimal.previous().is_some());

    // Runtime with InMemoryStore
    let store = Arc::new(InMemoryStore::new());
    store.put(&["users"], "alice", json!({"role": "admin"}))?;

    let rt_with_store = RuntimeBuilder::new()
        .with_store(store.clone())
        .with_config(RuntimeConfig::new().with_run_id("run-002"))
        .with_previous(json!({"step": "initialized"}))
        .build();

    println!(
        "  Runtime with store — has store: {}",
        rt_with_store.store().is_some()
    );
    let alice = rt_with_store.store().unwrap().get(&["users"], "alice")?;
    println!("  Store lookup users/alice: {:?}", alice);
    println!("  Previous value: {:?}", rt_with_store.previous());
    println!("  Config run_id: {:?}", rt_with_store.config().run_id);
    println!();

    // -----------------------------------------------------------------------
    // 3. Runtime with custom context
    // -----------------------------------------------------------------------
    println!("--- 3. Runtime with custom context ---");

    #[derive(Debug, Clone, Default)]
    struct AppContext {
        user_id: String,
        session_token: String,
        request_count: u32,
    }

    let ctx = AppContext {
        user_id: "user-123".into(),
        session_token: "tok-abc-secret".into(),
        request_count: 0,
    };

    let mut rt_custom = RuntimeBuilder::new()
        .with_context(ctx)
        .with_config(
            RuntimeConfig::new()
                .with_run_id("run-003")
                .with_tag("custom"),
        )
        .build();

    println!("  Context user_id: {}", rt_custom.context().user_id);
    println!("  Context session: {}", rt_custom.context().session_token);
    println!("  Request count: {}", rt_custom.context().request_count);

    // Mutate context
    rt_custom.context_mut().request_count += 1;
    println!("  After increment: {}", rt_custom.context().request_count);

    // Map context to a different type
    let rt_mapped =
        rt_custom.map_context(|ctx| format!("user={}, reqs={}", ctx.user_id, ctx.request_count));
    println!("  Mapped context: {}", rt_mapped.context());

    // Replace context entirely
    let rt_replaced = rt_mapped.with_context(42u32);
    println!("  Replaced context: {}", rt_replaced.context());
    println!();

    // -----------------------------------------------------------------------
    // 4. Stream writing
    // -----------------------------------------------------------------------
    println!("--- 4. Stream writing ---");
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cap = captured.clone();
    let writer: StreamWriter = Arc::new(move |v| {
        cap.lock().unwrap().push(v);
    });

    let rt_stream = RuntimeBuilder::new()
        .with_stream_writer(writer)
        .with_config(RuntimeConfig::new().with_run_id("run-004"))
        .build();

    rt_stream.write_to_stream(json!({"event": "node_start", "node": "agent"}));
    rt_stream.write_to_stream(json!({"event": "node_end", "node": "agent", "output": "hello"}));
    rt_stream.write_to_stream(json!({"event": "graph_end"}));

    let events = captured.lock().unwrap();
    println!("  Captured {} stream events:", events.len());
    for (i, event) in events.iter().enumerate() {
        println!("    [{}] {}", i, event);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. RuntimeScope — RAII lifecycle management
    // -----------------------------------------------------------------------
    println!("--- 5. RuntimeScope: RAII lifecycle ---");

    // Basic scope
    {
        let rt = Runtime::<NoContext>::new();
        let scope = RuntimeScope::new(rt);
        println!("  Scope created, context: {:?}", scope.runtime().context());
        println!("  Store present: {}", scope.runtime().store().is_some());
    }
    println!("  Scope dropped (cleanup runs automatically)");

    // Scope with cleanup callback
    let cleanup_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = cleanup_ran.clone();
    {
        let rt = RuntimeBuilder::new()
            .with_config(RuntimeConfig::new().with_run_id("scoped-run"))
            .build();
        let _scope = RuntimeScope::with_cleanup(rt, move || {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        println!("  Scoped run_id: {:?}", _scope.runtime().config().run_id);
    }
    println!(
        "  Cleanup callback ran: {}",
        cleanup_ran.load(std::sync::atomic::Ordering::SeqCst)
    );

    // Scope with mutable access
    let rt = RuntimeBuilder::new()
        .with_context(String::from("initial"))
        .build();
    let mut scope = RuntimeScope::new(rt);
    scope.runtime_mut().context_mut().push_str("_modified");
    println!("  Mutable scope context: {}", scope.runtime().context());

    // RuntimeProvider trait
    fn show_provider<P: RuntimeProvider<String>>(provider: &P) {
        println!("  Provider context: {}", provider.runtime().context());
    }
    show_provider(&scope);

    // into_inner skips cleanup
    let rt_back = scope.into_inner();
    println!("  Recovered runtime context: {}", rt_back.context());

    println!();

    // -----------------------------------------------------------------------
    // 6. NoContext — the default placeholder
    // -----------------------------------------------------------------------
    println!("--- 6. NoContext: default placeholder ---");
    let nc = NoContext;
    println!("  NoContext debug: {:?}", nc);
    println!("  NoContext clone: {:?}", nc);
    let nc_json = serde_json::to_string(&nc)?;
    println!("  NoContext serialized: {}", nc_json);
    let nc_back: NoContext = serde_json::from_str(&nc_json)?;
    println!(
        "  NoContext roundtrip: {:?} (equal={})",
        nc_back,
        nc == nc_back
    );

    let rt_default = Runtime::<NoContext>::new();
    println!("  Default runtime context: {:?}", rt_default.context());
    println!();

    // -----------------------------------------------------------------------
    // 7. Runtime merge
    // -----------------------------------------------------------------------
    println!("--- 7. Runtime merge ---");
    let base = RuntimeBuilder::new()
        .with_context(String::from("base"))
        .with_store(Arc::new(InMemoryStore::new()) as Arc<_>)
        .with_config(
            RuntimeConfig::new()
                .with_run_id("base-run")
                .with_tag("base-tag"),
        )
        .with_previous(json!("base_prev"))
        .build();

    let overlay = RuntimeBuilder::new()
        .with_context(String::from("overlay"))
        .with_config(RuntimeConfig::new().with_run_id("overlay-run"))
        .build();

    let merged = base.merge(overlay);
    println!("  Merged context: {} (from overlay)", merged.context());
    println!(
        "  Merged run_id: {:?} (from overlay)",
        merged.config().run_id
    );
    println!(
        "  Merged store present: {} (falls back to base)",
        merged.store().is_some()
    );
    println!(
        "  Merged tags: {:?} (falls back to base since overlay empty)",
        merged.config().tags
    );
    println!(
        "  Merged previous: {:?} (falls back to base)",
        merged.previous()
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. LLM invocation within a runtime scope
    // -----------------------------------------------------------------------
    println!("--- 8. LLM call within a runtime scope ---");

    let model = shared::get_chat_model(vec![
        "The runtime scope provides context injection for graph nodes, enabling \
         stateful execution with configuration metadata, store access, and stream \
         writing capabilities."
            .to_string(),
    ]);

    // Create a runtime scope that simulates a graph execution context
    let store = Arc::new(InMemoryStore::new());
    store.put(
        &["prompts"],
        "system",
        json!("You are a concise technical assistant."),
    )?;

    let rt = RuntimeBuilder::new()
        .with_context(String::from("llm-demo"))
        .with_store(store.clone())
        .with_config(
            RuntimeConfig::new()
                .with_run_id("llm-run-001")
                .with_thread_id("thread-llm")
                .with_tag("demo")
                .with_metadata("model_name", json!("llama3.2")),
        )
        .build();

    let scope = RuntimeScope::new(rt);
    println!("  Runtime run_id: {:?}", scope.runtime().config().run_id);
    println!(
        "  Runtime thread_id: {:?}",
        scope.runtime().config().thread_id
    );
    println!("  Runtime context: {}", scope.runtime().context());

    // Retrieve system prompt from store
    let system_prompt = scope
        .runtime()
        .store()
        .unwrap()
        .get(&["prompts"], "system")?
        .unwrap_or(json!("You are a helpful assistant."));
    let system_text = system_prompt
        .as_str()
        .unwrap_or("You are a helpful assistant.");

    let messages = vec![
        Message::system(system_text),
        Message::human(
            "Explain what a runtime scope does in a graph execution framework, in one sentence.",
        ),
    ];

    let ai_response = model.invoke_messages(&messages, None).await?;
    println!("  LLM response: {}", ai_response.base.content.text());

    // Record the response as the previous value
    let mut rt_after = scope.into_inner();
    rt_after.set_previous(Some(json!({
        "response": ai_response.base.content.text(),
        "run_id": rt_after.config().run_id,
    })));
    println!("  Previous value recorded: {:?}", rt_after.previous());
    println!();

    println!("=== Graph Runtime Example Complete ===");
    Ok(())
}

/// Indent each line of text with the given prefix.
fn textwrap(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}
