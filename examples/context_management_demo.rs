//! Context Management Demo
//!
//! Demonstrates context management from `cognisagent::context`:
//! - ContextWindow with a token budget
//! - Pushing entries with different roles (System, User, Assistant, Tool)
//! - Checking utilization and remaining tokens
//! - ContextPolicy and ContextCompressor for compression
//! - ContextFilter for finding entries by role
//! - ContextStats for analyzing usage
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example context_management_demo`

mod shared;
use cognis_core::messages::Message;
use cognisagent::context::{
    ContextCompressor, ContextEntry, ContextFilter, ContextPolicy, ContextRole, ContextSnapshot,
    ContextStats, ContextWindow,
};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Context Management Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a ContextWindow with a token budget
    // -----------------------------------------------------------------------
    println!("--- 1. Creating a ContextWindow ---\n");

    let mut window = ContextWindow::new(500);
    println!("  Max tokens: {}", window.max_tokens());
    println!("  Current entries: {}", window.len());
    println!("  Total tokens: {}", window.total_tokens());
    println!("  Remaining tokens: {}", window.remaining_tokens());
    println!("  Utilization: {:.1}%", window.utilization() * 100.0);
    println!();

    // -----------------------------------------------------------------------
    // 2. Push entries with different roles
    // -----------------------------------------------------------------------
    println!("--- 2. Adding Entries with Different Roles ---\n");

    // System message
    window.push(ContextEntry::new(
        ContextRole::System,
        "You are a helpful coding assistant. Always provide examples.",
    ));
    println!("  Added: System message");

    // User message
    window.push(ContextEntry::new(
        ContextRole::User,
        "How do I read a file in Rust?",
    ));
    println!("  Added: User message");

    // Assistant response
    window.push(ContextEntry::new(
        ContextRole::Assistant,
        "You can use std::fs::read_to_string to read a file into a String. \
         Here is an example: let content = std::fs::read_to_string(\"path.txt\")?;",
    ));
    println!("  Added: Assistant response");

    // Tool output
    window.push(
        ContextEntry::new(
            ContextRole::Tool("file_reader".into()),
            "File contents: fn main() { println!(\"Hello, world!\"); }",
        )
        .with_metadata("path", Value::String("/tmp/example.rs".into())),
    );
    println!("  Added: Tool output (file_reader)");

    // Another user message
    window.push(ContextEntry::new(
        ContextRole::User,
        "Can you also show me how to write to a file?",
    ));
    println!("  Added: User follow-up");

    // Another assistant response
    window.push(ContextEntry::new(
        ContextRole::Assistant,
        "Sure! Use std::fs::write for simple cases: std::fs::write(\"output.txt\", \"data\")?; \
         For more control, use File::create with a BufWriter.",
    ));
    println!("  Added: Assistant response");

    println!();

    // -----------------------------------------------------------------------
    // 3. Check utilization and remaining tokens
    // -----------------------------------------------------------------------
    println!("--- 3. Token Budget Status ---\n");

    println!("  Total entries: {}", window.len());
    println!("  Total tokens: {}", window.total_tokens());
    println!("  Max tokens: {}", window.max_tokens());
    println!("  Remaining tokens: {}", window.remaining_tokens());
    println!("  Utilization: {:.1}%", window.utilization() * 100.0);
    println!();

    // List entries by role
    println!("  Entries by role:");
    for entry in window.entries() {
        println!(
            "    [{:>9}] {} tokens: \"{}\"",
            entry.role,
            entry.token_estimate,
            if entry.content.len() > 50 {
                format!("{}...", &entry.content[..50])
            } else {
                entry.content.clone()
            }
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. Use ContextFilter to find entries by role
    // -----------------------------------------------------------------------
    println!("--- 4. Filtering Entries ---\n");

    // Filter by User role
    let user_filter = ContextFilter::new().by_role(ContextRole::User);
    let user_entries = user_filter.apply(&window);
    println!("  User messages: {}", user_entries.len());
    for entry in &user_entries {
        println!("    - \"{}\"", entry.content);
    }

    // Filter by Assistant role
    let assistant_filter = ContextFilter::new().by_role(ContextRole::Assistant);
    let assistant_entries = assistant_filter.apply(&window);
    println!("  Assistant messages: {}", assistant_entries.len());
    for entry in &assistant_entries {
        println!(
            "    - \"{}\"",
            if entry.content.len() > 60 {
                format!("{}...", &entry.content[..60])
            } else {
                entry.content.clone()
            }
        );
    }

    // Filter by Tool role
    let tool_filter = ContextFilter::new().by_role(ContextRole::Tool("file_reader".into()));
    let tool_entries = tool_filter.apply(&window);
    println!("  Tool (file_reader) messages: {}", tool_entries.len());
    for entry in &tool_entries {
        let path = entry
            .metadata
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        println!("    - path={}, content=\"{}\"", path, entry.content);
    }

    // Filter by metadata key
    let meta_filter = ContextFilter::new().by_metadata_key("path");
    let meta_entries = meta_filter.apply(&window);
    println!("  Entries with 'path' metadata: {}", meta_entries.len());

    // Filter by token range
    let large_filter = ContextFilter::new().by_token_range(20, 100);
    let large_entries = large_filter.apply(&window);
    println!("  Entries with 20-100 tokens: {}", large_entries.len());
    println!();

    // -----------------------------------------------------------------------
    // 5. Use ContextStats to analyze usage
    // -----------------------------------------------------------------------
    println!("--- 5. Context Statistics ---\n");

    let stats = ContextStats::from_window(&window);
    println!("  Total entries:    {}", stats.total_entries);
    println!("  Total tokens:     {}", stats.total_tokens);
    println!("  Max tokens:       {}", stats.max_tokens);
    println!("  Utilization:      {:.1}%", stats.utilization * 100.0);
    println!("  Avg entry size:   {} tokens", stats.average_entry_size());
    println!();

    println!("  Tokens by role:");
    let mut roles: Vec<_> = stats.tokens_by_role().iter().collect();
    roles.sort_by(|a, b| b.1.cmp(a.1));
    for (role, tokens) in &roles {
        println!("    {:<12} {} tokens", role, tokens);
    }

    println!("\n  Entries by role:");
    let mut counts: Vec<_> = stats.entry_count_by_role().iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    for (role, count) in &counts {
        println!("    {:<12} {} entries", role, count);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. Use ContextPolicy and ContextCompressor
    // -----------------------------------------------------------------------
    println!("--- 6. Context Compression ---\n");

    // Create a policy with a tight budget to trigger compression
    let policy = ContextPolicy::new(100).with_response_reserve(20);
    println!("  Policy max tokens: {}", policy.max_tokens);
    println!(
        "  Policy effective limit: {} (after reserving {} for response)",
        policy.effective_limit(),
        policy.reserve_for_response
    );
    println!("  Needs compression: {}", policy.needs_compression(&window));

    // Take a snapshot before compression
    let snapshot = ContextSnapshot::capture(&window);
    println!("\n  Snapshot captured: {} entries", snapshot.entry_count());

    let tokens_before = window.total_tokens();
    println!("  Tokens before compression: {}", tokens_before);

    // Compress the context
    let compressor = ContextCompressor::new(policy);
    compressor.compress(&mut window);

    let tokens_after = window.total_tokens();
    println!("  Tokens after compression:  {}", tokens_after);
    println!(
        "  Compression ratio: {:.2}",
        ContextCompressor::compression_ratio(tokens_before, tokens_after)
    );
    println!("  Entries after compression: {}", window.len());

    println!("\n  Entries after compression:");
    for entry in window.entries() {
        println!(
            "    [{:>9}] {} tokens: \"{}\"",
            entry.role,
            entry.token_estimate,
            if entry.content.len() > 60 {
                format!("{}...", &entry.content[..60])
            } else {
                entry.content.clone()
            }
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 7. Restore from snapshot
    // -----------------------------------------------------------------------
    println!("--- 7. Restoring from Snapshot ---\n");

    let restored = snapshot.restore();
    println!("  Restored window: {} entries", restored.len());
    println!("  Restored tokens: {}", restored.total_tokens());
    println!(
        "  Matches original entry count: {}",
        restored.len() == snapshot.entry_count()
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. Window operations: find_by_role and last_n
    // -----------------------------------------------------------------------
    println!("--- 8. Window Operations ---\n");

    let system_entries = restored.find_by_role(&ContextRole::System);
    println!(
        "  System entries in restored window: {}",
        system_entries.len()
    );

    let last_two = restored.last_n(2);
    println!("  Last 2 entries:");
    for entry in last_two {
        println!(
            "    [{:>9}] \"{}\"",
            entry.role,
            &entry.content[..entry.content.len().min(50)]
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 9. Truncation
    // -----------------------------------------------------------------------
    println!("--- 9. Truncation ---\n");

    let mut truncation_window = snapshot.restore();
    println!("  Before truncation: {} entries", truncation_window.len());

    truncation_window.truncate_oldest(2);
    println!(
        "  After truncate_oldest(2): {} entries",
        truncation_window.len()
    );

    // System messages should still be present
    let system_count = truncation_window.find_by_role(&ContextRole::System).len();
    println!(
        "  System messages preserved: {} (kept: {})",
        system_count > 0,
        system_count
    );

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Using an LLM within a managed context window.\n");

    let mut llm_window = ContextWindow::new(500);
    llm_window.push(ContextEntry::new(
        ContextRole::System,
        "You are a concise coding assistant.",
    ));
    llm_window.push(ContextEntry::new(
        ContextRole::User,
        "How do I read a file in Rust?",
    ));

    println!("Context window: {} entries, {} tokens used, {} remaining",
        llm_window.len(), llm_window.total_tokens(), llm_window.remaining_tokens());

    let model = shared::get_chat_model(vec![
        "Use std::fs::read_to_string(\"path\") to read a file into a String in Rust.".into(),
    ]);

    // Build messages from context window entries
    let messages: Vec<Message> = llm_window.entries().iter().map(|e| {
        match &e.role {
            ContextRole::System => Message::system(&e.content),
            ContextRole::User => Message::human(&e.content),
            ContextRole::Assistant => Message::ai(&e.content),
            ContextRole::Tool(_) | ContextRole::Summary => Message::human(&e.content),
        }
    }).collect();

    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let reply = gen.message.content().text();
        println!("LLM Response: {}", reply);

        // Add the response back into the context window
        llm_window.push(ContextEntry::new(ContextRole::Assistant, &reply));
        println!("Context window after LLM reply: {} entries, {} tokens, {:.1}% utilization",
            llm_window.len(), llm_window.total_tokens(), llm_window.utilization() * 100.0);
    }

    println!("\n=== Context Management Demo Complete ===");
    Ok(())
}
