//! Message Graph Example
//!
//! Demonstrates the LangGraph message graph system for building message-based
//! workflows. Message graphs are specialized state graphs where state is a
//! conversation (a list of `GraphMessage`s) that accumulates as nodes execute.
//!
//! Features shown:
//! - MessageRole variants and display
//! - GraphMessage creation with metadata
//! - MessageState operations (add, filter, last_human, last_ai)
//! - MessageNode creation with handler functions
//! - MessageGraph building with nodes and edges
//! - MessageGraph execution with linear flow
//! - MessageGraph with conditional edges (routing based on message content)
//! - MessageGraphBuilder fluent API
//! - Graph validation
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example message_graph`

mod shared;
use cognisgraph::graph::{
    GraphMessage, MessageGraph, MessageGraphBuilder, MessageNode, MessageRole, MessageState,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Message Graph Example ===\n");

    // -----------------------------------------------------------------------
    // 1. MessageRole Variants and Display
    // -----------------------------------------------------------------------
    println!("--- 1. MessageRole Variants ---");
    println!("MessageRole represents the author of a message in a conversation.\n");

    let roles = vec![
        MessageRole::System,
        MessageRole::Human,
        MessageRole::Ai,
        MessageRole::Tool,
        MessageRole::Function,
        MessageRole::Custom("moderator".to_string()),
    ];

    for role in &roles {
        println!("  Role: {:<12} as_str: \"{}\"", format!("{:?}", role), role);
    }

    // Conversion from string slices
    let from_str: MessageRole = "assistant".into(); // maps to Ai
    println!(
        "\n  \"assistant\".into() => {:?} (display: {})",
        from_str, from_str
    );
    let from_str: MessageRole = "user".into(); // maps to Human
    println!(
        "  \"user\".into()      => {:?} (display: {})",
        from_str, from_str
    );

    // -----------------------------------------------------------------------
    // 2. GraphMessage Creation with Metadata
    // -----------------------------------------------------------------------
    println!("\n--- 2. GraphMessage Creation ---");
    println!("Messages carry role, content, optional name, and arbitrary metadata.\n");

    let system_msg = GraphMessage::new(MessageRole::System, "You are a helpful assistant.");
    println!("  System message: \"{}\"", system_msg.content);

    let user_msg = GraphMessage::new(MessageRole::Human, "What is Rust?")
        .with_name("alice")
        .with_metadata("source", serde_json::json!("cli"))
        .with_metadata("priority", serde_json::json!(1));
    println!(
        "  Human message: \"{}\" (name: {:?})",
        user_msg.content, user_msg.name
    );
    println!("  Metadata: {:?}", user_msg.metadata);

    let tool_msg = GraphMessage::new(
        MessageRole::Tool,
        "search result: Rust is a systems language",
    )
    .with_tool_call_id("call_001");
    println!("  Tool message: tool_call_id={:?}", tool_msg.tool_call_id);

    // Serialize to JSON
    let json = user_msg.to_json();
    println!("\n  User message as JSON:");
    println!("  {}", serde_json::to_string_pretty(&json).unwrap());

    // Role checks
    println!(
        "\n  is_human: {}, is_ai: {}, is_system: {}, is_tool: {}",
        user_msg.is_human(),
        user_msg.is_ai(),
        system_msg.is_system(),
        tool_msg.is_tool()
    );

    // -----------------------------------------------------------------------
    // 3. MessageState Operations
    // -----------------------------------------------------------------------
    println!("\n--- 3. MessageState Operations ---");
    println!("MessageState holds an ordered list of messages representing the conversation.\n");

    let mut state = MessageState::new();
    assert!(state.is_empty());
    println!(
        "  Empty state: len={}, is_empty={}",
        state.len(),
        state.is_empty()
    );

    state.add(GraphMessage::new(MessageRole::System, "Be concise."));
    state.add(GraphMessage::new(MessageRole::Human, "Hello!"));
    state.add(GraphMessage::new(MessageRole::Ai, "Hi there!"));
    state.add(GraphMessage::new(MessageRole::Human, "What is 2+2?"));
    state.add(GraphMessage::new(MessageRole::Ai, "4"));
    println!("  After adding 5 messages: len={}", state.len());

    // Filter by role
    let human_msgs = state.filter_by_role(&MessageRole::Human);
    println!(
        "  Human messages: {} (\"{}\" and \"{}\")",
        human_msgs.len(),
        human_msgs[0].content,
        human_msgs[1].content
    );

    let ai_msgs = state.filter_by_role(&MessageRole::Ai);
    println!("  AI messages: {}", ai_msgs.len());

    // Last human and last AI
    let last_human = state.last_human().unwrap();
    println!("  Last human message: \"{}\"", last_human.content);

    let last_ai = state.last_ai().unwrap();
    println!("  Last AI message: \"{}\"", last_ai.content);

    // Last message overall
    let last = state.last().unwrap();
    println!(
        "  Last message (any role): \"{}\" [{}]",
        last.content, last.role
    );

    // Serialize state to JSON
    let state_json = state.to_json();
    println!(
        "  State as JSON array: {} messages",
        state_json.as_array().unwrap().len()
    );

    // -----------------------------------------------------------------------
    // 4. MessageNode Creation
    // -----------------------------------------------------------------------
    println!("\n--- 4. MessageNode Creation ---");
    println!("Nodes are functions that receive the state and return new messages.\n");

    let greeter = MessageNode::new("greeter", |_state: &MessageState| {
        vec![GraphMessage::new(
            MessageRole::Ai,
            "Welcome! How can I help you?",
        )]
    });
    println!("  Created node: {:?}", greeter);

    // Execute the node against a state
    let test_state = MessageState::new();
    let output = greeter.execute(&test_state);
    println!("  Greeter output: \"{}\"", output[0].content);

    let echo = MessageNode::new("echo", |state: &MessageState| {
        if let Some(last) = state.last_human() {
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("You said: {}", last.content),
            )]
        } else {
            vec![GraphMessage::new(
                MessageRole::Ai,
                "I did not hear anything.",
            )]
        }
    });
    println!("  Created node: {:?}", echo);

    // -----------------------------------------------------------------------
    // 5. MessageGraph with Linear Flow
    // -----------------------------------------------------------------------
    println!("\n--- 5. Linear MessageGraph ---");
    println!("Build and execute a graph with a simple linear chain of nodes.\n");

    let mut graph = MessageGraph::new();

    graph.add_node(MessageNode::new("intake", |_state: &MessageState| {
        vec![GraphMessage::new(
            MessageRole::Ai,
            "I received your request. Let me process it.",
        )]
    }));

    graph.add_node(MessageNode::new("process", |state: &MessageState| {
        let user_query = state
            .last_human()
            .map(|m| m.content.as_str())
            .unwrap_or("(none)");
        vec![GraphMessage::new(
            MessageRole::Ai,
            format!("Processing query: '{}'", user_query),
        )]
    }));

    graph.add_node(MessageNode::new("respond", |_state: &MessageState| {
        vec![GraphMessage::new(
            MessageRole::Ai,
            "Here is your response. Thank you!",
        )]
    }));

    graph.add_edge("intake", "process");
    graph.add_edge("process", "respond");
    graph.set_entry_point("intake");
    graph.set_finish_point("respond");

    println!(
        "  Graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Execute with initial user message
    let mut initial_state = MessageState::new();
    initial_state.add(GraphMessage::new(MessageRole::Human, "Tell me about Rust."));

    let result = graph.execute(initial_state).unwrap();
    println!("  Execution produced {} messages:", result.len());
    for msg in result.messages() {
        println!("    [{}] {}", msg.role, msg.content);
    }

    // -----------------------------------------------------------------------
    // 6. MessageGraph with Conditional Edges
    // -----------------------------------------------------------------------
    println!("\n--- 6. Conditional Edges ---");
    println!("Route to different nodes based on message content at runtime.\n");

    let mut graph = MessageGraph::new();

    graph.add_node(MessageNode::new("classifier", |state: &MessageState| {
        let query = state.last_human().map(|m| m.content.as_str()).unwrap_or("");
        let label = if query.contains("weather") {
            "weather"
        } else if query.contains("math") || query.contains('+') || query.contains("calculate") {
            "math"
        } else {
            "general"
        };
        vec![GraphMessage::new(
            MessageRole::Ai,
            format!("category:{}", label),
        )]
    }));

    graph.add_node(MessageNode::new(
        "weather_handler",
        |_state: &MessageState| {
            vec![GraphMessage::new(
                MessageRole::Ai,
                "It is sunny and 25C today.",
            )]
        },
    ));

    graph.add_node(MessageNode::new("math_handler", |_state: &MessageState| {
        vec![GraphMessage::new(MessageRole::Ai, "The answer is 42.")]
    }));

    graph.add_node(MessageNode::new(
        "general_handler",
        |_state: &MessageState| {
            vec![GraphMessage::new(
                MessageRole::Ai,
                "I can help with general questions.",
            )]
        },
    ));

    graph.set_entry_point("classifier");
    graph.add_conditional_edge("classifier", |state: &MessageState| {
        let last = state.last_ai().map(|m| m.content.as_str()).unwrap_or("");
        if last.contains("category:weather") {
            "weather_handler".to_string()
        } else if last.contains("category:math") {
            "math_handler".to_string()
        } else {
            "general_handler".to_string()
        }
    });
    // All handlers are finish points — set weather_handler as finish
    // but since there is no outgoing edge from any handler, execution stops naturally.
    graph.set_finish_point("weather_handler");

    // Test different inputs
    let queries = vec![
        "What is the weather like?",
        "Please calculate 2+2",
        "Tell me a joke",
    ];

    for query in queries {
        let mut state = MessageState::new();
        state.add(GraphMessage::new(MessageRole::Human, query));
        let result = graph.execute(state).unwrap();
        let response = result.last_ai().unwrap();
        println!("  Query: \"{}\"", query);
        println!("  Response: \"{}\"\n", response.content);
    }

    // -----------------------------------------------------------------------
    // 7. MessageGraphBuilder Fluent API
    // -----------------------------------------------------------------------
    println!("--- 7. MessageGraphBuilder ---");
    println!("Build graphs using a chainable fluent API.\n");

    let graph = MessageGraphBuilder::new()
        .node("start", |_state: &MessageState| {
            vec![GraphMessage::new(MessageRole::Ai, "Starting pipeline...")]
        })
        .node("transform", |state: &MessageState| {
            let count = state.len();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Transformed. Messages so far: {}", count),
            )]
        })
        .node("finalize", |state: &MessageState| {
            let total = state.len();
            vec![GraphMessage::new(
                MessageRole::Ai,
                format!("Done. Total messages: {}", total + 1),
            )]
        })
        .edge("start", "transform")
        .edge("transform", "finalize")
        .entry("start")
        .finish("finalize")
        .build()
        .expect("Graph validation should pass");

    println!(
        "  Built graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    let mut state = MessageState::new();
    state.add(GraphMessage::new(MessageRole::Human, "Go!"));
    let result = graph.execute(state).unwrap();
    println!("  Final messages:");
    for msg in result.messages() {
        println!("    [{}] {}", msg.role, msg.content);
    }

    // -----------------------------------------------------------------------
    // 8. Graph Validation
    // -----------------------------------------------------------------------
    println!("\n--- 8. Graph Validation ---");
    println!("Validate graph structure before execution.\n");

    // Valid graph
    let mut valid_graph = MessageGraph::new();
    valid_graph.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
    valid_graph.add_node(MessageNode::new("b", |_: &MessageState| vec![]));
    valid_graph.add_edge("a", "b");
    valid_graph.set_entry_point("a");
    valid_graph.set_finish_point("b");

    match valid_graph.validate() {
        Ok(()) => println!("  Valid graph: validation passed"),
        Err(e) => println!("  Valid graph: unexpected error: {}", e),
    }

    // Missing entry point
    let no_entry = MessageGraph::new();
    match no_entry.validate() {
        Ok(()) => println!("  No entry point: unexpected success"),
        Err(e) => println!("  No entry point: {}", e),
    }

    // Entry point references non-existent node
    let mut bad_entry = MessageGraph::new();
    bad_entry.set_entry_point("ghost");
    match bad_entry.validate() {
        Ok(()) => println!("  Bad entry: unexpected success"),
        Err(e) => println!("  Bad entry: {}", e),
    }

    // Edge target does not exist
    let mut bad_edge = MessageGraph::new();
    bad_edge.add_node(MessageNode::new("a", |_: &MessageState| vec![]));
    bad_edge.set_entry_point("a");
    bad_edge.add_edge("a", "nonexistent");
    match bad_edge.validate() {
        Ok(()) => println!("  Bad edge: unexpected success"),
        Err(e) => println!("  Bad edge target: {}", e),
    }

    // Builder validation catches errors
    let result = MessageGraphBuilder::new()
        .node("only_node", |_: &MessageState| vec![])
        .entry("missing_node")
        .build();
    match result {
        Ok(_) => println!("  Builder: unexpected success"),
        Err(e) => println!("  Builder validation: {}", e),
    }

    // -----------------------------------------------------------------------
    // 9. Real LLM Demo — LLM in message graph context
    // -----------------------------------------------------------------------
    println!("\n--- 9. Real LLM Demo ---");
    println!("Use an LLM to generate a response in a message graph context.\n");

    let model = shared::get_chat_model(vec![
        "In a message graph, nodes process messages sequentially or conditionally, allowing you to build conversational workflows with routing logic.".into(),
    ]);
    let messages = vec![cognis_core::messages::Message::human(
        "Explain how message graphs help build conversational AI workflows.",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  LLM Response: {}", gen.message.content().text());
    }

    println!("\n=== Message Graph Example Complete ===");
    Ok(())
}
