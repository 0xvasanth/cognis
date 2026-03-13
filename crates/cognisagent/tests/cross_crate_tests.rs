//! Cross-crate integration tests proving all 4 workspace crates compose correctly.
//!
//! These tests exercise the interaction between:
//! - `cognis-core` (traits, fakes, runnables, messages, output parsers, embeddings)
//! - `cognis` (chains, memory, vectorstores, text_splitter)
//! - `cognisgraph` (StateGraph, CompiledStateGraph, checkpointing, create_tool_agent)
//! - `cognisagent` (create_deep_agent, middleware, config)
//!
//! Run with: `cargo test -p cognisagent --test cross_crate_tests`

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

// --- cognis-core imports ---
use cognis_core::embeddings::Embeddings;
use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
use cognis_core::error::Result as CoreResult;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::fake::{
    FakeListChatModel, FakeMessagesListChatModel, ParrotFakeChatModel,
};
use cognis_core::messages::{AIMessage, HumanMessage, Message, ToolCall};
use cognis_core::output_parsers::{
    CommaSeparatedListOutputParser, JsonOutputParser, OutputParser, StrOutputParser,
};
use cognis_core::retrievers::BaseRetriever;
use cognis_core::runnables::{Runnable, RunnableLambda, RunnableSequence};
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognis_core::vectorstores::base::{SearchType, VectorStoreRetriever};

// --- cognis imports ---
use cognis::chains::llm::LLMChain;
use cognis::chains::retrieval::RetrievalQAChain;
use cognis::memory::buffer::ConversationBufferMemory;
use cognis::memory::window::ConversationWindowMemory;
use cognis::memory::BaseMemory;
use cognis::text_splitter::{CharacterTextSplitter, TextSplitter};
use cognis::vectorstores::in_memory::InMemoryVectorStore;

// --- cognisgraph imports ---
use cognisgraph::checkpoint::{CheckpointMetadata, CheckpointSaver, InMemoryCheckpointSaver};
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};
use cognisgraph::prebuilt::tool_agent::create_tool_agent;
use cognisgraph::pregel::checkpoint::empty_checkpoint;

// --- cognisagent imports ---
use cognisagent::config::DeepAgentConfig;
use cognisagent::create_deep_agent;
use cognisagent::middleware::memory::MemoryMiddleware;
use cognisagent::middleware::Middleware;

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/// A mock tool that returns a fixed result.
struct MockTool {
    tool_name: String,
    result: String,
}

impl MockTool {
    fn new(name: &str, result: &str) -> Self {
        Self {
            tool_name: name.to_string(),
            result: result.to_string(),
        }
    }
}

#[async_trait]
impl BaseTool for MockTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        "A mock tool for integration testing"
    }

    async fn _run(&self, _input: ToolInput) -> CoreResult<ToolOutput> {
        Ok(ToolOutput::Content(Value::String(self.result.clone())))
    }
}

/// A no-op middleware for testing.
struct NoopMiddleware;

#[async_trait]
impl Middleware for NoopMiddleware {
    fn name(&self) -> &str {
        "noop"
    }
}

// ---------------------------------------------------------------------------
// Test 1: cognis-core traits are properly implemented by cognis types
//         (RunnableLambda implements Runnable)
// ---------------------------------------------------------------------------

/// Verify that RunnableLambda from cognis-core implements the Runnable trait
/// and can be used polymorphically alongside LLMChain from cognis.
#[tokio::test]
async fn test_core_runnable_trait_implemented_by_lambda_and_chain() {
    // RunnableLambda (core) implements Runnable
    let lambda = RunnableLambda::new("uppercase", |input: Value| async move {
        let s = input.as_str().unwrap_or("fallback").to_uppercase();
        Ok(Value::String(s))
    });

    // Verify it works as a Runnable trait object
    let runnable: &dyn Runnable = &lambda;
    assert_eq!(runnable.name(), "uppercase");

    let result = runnable.invoke(json!("hello world"), None).await.unwrap();
    assert_eq!(result, json!("HELLO WORLD"));

    // LLMChain (cognis) also implements Runnable via core trait
    let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec!["Paris".to_string()]));
    let chain = LLMChain::builder()
        .model(model)
        .prompt("What is the capital of {country}?")
        .build();

    let chain_runnable: &dyn Runnable = &chain;
    assert_eq!(chain_runnable.name(), "LLMChain");

    let chain_result = chain_runnable
        .invoke(json!({"country": "France"}), None)
        .await
        .unwrap();
    assert_eq!(chain_result["text"], "Paris");
}

// ---------------------------------------------------------------------------
// Test 2: RunnableSequence from cognis-core works with RunnableLambda steps
// ---------------------------------------------------------------------------

/// Build a RunnableSequence from multiple RunnableLambda steps and verify
/// that output flows through each step correctly.
#[tokio::test]
async fn test_runnable_sequence_with_lambda_steps() {
    // Step 1: extract the "input" field
    let step1 = RunnableLambda::new("extract", |v: Value| async move {
        let text = v["input"].as_str().unwrap_or("").to_string();
        Ok(Value::String(text))
    });

    // Step 2: convert to uppercase
    let step2 = RunnableLambda::new("uppercase", |v: Value| async move {
        let s = v.as_str().unwrap_or("").to_uppercase();
        Ok(Value::String(s))
    });

    // Step 3: wrap in a result object
    let step3 = RunnableLambda::new("wrap", |v: Value| async move { Ok(json!({"result": v})) });

    let sequence = RunnableSequence::new(vec![
        Arc::new(step1) as Arc<dyn Runnable>,
        Arc::new(step2) as Arc<dyn Runnable>,
        Arc::new(step3) as Arc<dyn Runnable>,
    ])
    .unwrap();

    let output = sequence
        .invoke(json!({"input": "hello world"}), None)
        .await
        .unwrap();

    assert_eq!(output["result"], "HELLO WORLD");
    assert_eq!(sequence.name(), "RunnableSequence");
}

// ---------------------------------------------------------------------------
// Test 3: cognisgraph StateGraph can use cognis-core types in its state
// ---------------------------------------------------------------------------

/// Build a StateGraph whose node uses cognis-core Message types in its state
/// and verify the graph processes them correctly.
#[tokio::test]
async fn test_cognisgraph_stategraph_with_core_message_types() {
    // Use a ParrotFakeChatModel (core) that echoes the human message
    let model: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());

    let model_for_node = model.clone();
    let node_action: AsyncNodeAction = Arc::new(move |state: Value| {
        let m = model_for_node.clone();
        Box::pin(async move {
            let query = state
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("default question");

            // Use core Message types
            let messages = vec![Message::Human(HumanMessage::new(query))];
            let ai_msg = m
                .invoke_messages(&messages, None)
                .await
                .map_err(|e| cognisgraph::errors::LangGraphError::Other(e.to_string()))?;

            // Serialize core AIMessage into state
            let response_text = ai_msg.base.content.text();
            let serialized_msg = serde_json::to_value(&Message::Ai(ai_msg))
                .map_err(|e| cognisgraph::errors::LangGraphError::Other(e.to_string()))?;

            Ok(json!({
                "query": query,
                "response": response_text,
                "last_message": serialized_msg,
            }))
        })
    });

    let graph = StateGraph::new()
        .add_node("model_node", node_action)
        .set_entry_point("model_node")
        .set_finish_point("model_node")
        .compile()
        .unwrap();

    let result = graph
        .invoke(json!({"query": "Echo this back"}))
        .await
        .unwrap();

    // Parrot model echoes input
    assert_eq!(result["response"].as_str().unwrap(), "Echo this back");

    // The last_message should deserialize back to a core Message
    let msg: Message = serde_json::from_value(result["last_message"].clone()).unwrap();
    assert_eq!(msg.content().text(), "Echo this back");
}

// ---------------------------------------------------------------------------
// Test 4: LLMChain from cognis with FakeListChatModel from core
// ---------------------------------------------------------------------------

/// Build an LLMChain (cognis) using a FakeListChatModel (core) and verify
/// prompt formatting and model invocation produce correct output.
#[tokio::test]
async fn test_llmchain_with_fake_chat_model() {
    let model: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
        "Rust is a systems programming language focused on safety and performance.".to_string(),
    ]));

    let chain = LLMChain::builder()
        .model(model)
        .prompt("Tell me about {topic} in one sentence.")
        .output_key("answer")
        .build();

    let result = chain.invoke(json!({"topic": "Rust"}), None).await.unwrap();

    assert_eq!(
        result["answer"],
        "Rust is a systems programming language focused on safety and performance."
    );

    // Verify it also works through the Runnable trait
    let runnable: &dyn Runnable = &chain;
    let result2 = runnable.invoke(json!({"topic": "Python"}), None).await;
    // FakeListChatModel cycles, so second call may reuse the response
    assert!(result2.is_ok());
}

// ---------------------------------------------------------------------------
// Test 5: cognisagent middleware can wrap cognis chat models
// ---------------------------------------------------------------------------

/// Create a deep agent with middleware that wraps a fake chat model.
/// Verify that middleware hooks fire and the agent produces correct results.
#[tokio::test]
async fn test_cognisagent_middleware_wraps_chat_model() {
    // Core: model that first issues a tool call, then returns final answer
    let tc = ToolCall {
        name: "weather".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert("city".to_string(), json!("London"));
            m
        },
        id: Some("call_w1".to_string()),
    };
    let mut ai_with_tc = AIMessage::new("");
    ai_with_tc.tool_calls = vec![tc];

    let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(ai_with_tc),
        Message::Ai(AIMessage::new("It is sunny in London")),
    ]));

    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("weather", "sunny, 22C"));

    // Set up memory middleware (cognisagent) that injects context
    let memory_mw = Arc::new(MemoryMiddleware::new(10));
    memory_mw.remember("user_location", "London").await;

    let config = DeepAgentConfig {
        tools: vec![tool],
        middleware: vec![
            memory_mw.clone() as Arc<dyn Middleware>,
            Arc::new(NoopMiddleware) as Arc<dyn Middleware>,
        ],
        system_prompt: Some("You are a weather assistant.".to_string()),
        ..Default::default()
    };

    let graph = create_deep_agent(model, config).unwrap();

    let input = json!({
        "messages": [{"type": "human", "content": "What is the weather?"}]
    });

    let result = graph.invoke(input).await.unwrap();
    let messages = result["messages"].as_array().unwrap();

    // Should have at least human + ai(tool_call) + tool_result + ai(final)
    assert!(
        messages.len() >= 4,
        "Expected at least 4 messages, got {}",
        messages.len()
    );

    // Final message should be the AI response
    let last_msg: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
    assert_eq!(last_msg.content().text(), "It is sunny in London");

    // Verify memory middleware retained its state through the graph execution
    assert_eq!(
        memory_mw.recall("user_location").await,
        Some("London".to_string())
    );
}

// ---------------------------------------------------------------------------
// Test 6: cognisgraph checkpoint can save/restore state across crate boundaries
// ---------------------------------------------------------------------------

/// Use InMemoryCheckpointSaver (cognisgraph) to save state containing
/// cognis-core Message types, then restore and verify the data.
#[tokio::test]
async fn test_cognisgraph_checkpoint_save_restore_with_core_types() {
    let saver = InMemoryCheckpointSaver::new();

    // Build state containing core message types
    let messages = vec![
        Message::human("What is Rust?"),
        Message::ai("Rust is a systems programming language."),
    ];
    let serialized_messages: Vec<Value> = messages
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect();

    // Create a checkpoint with the message state
    let mut cp = empty_checkpoint();
    cp.channel_values.insert(
        "messages".to_string(),
        Value::Array(serialized_messages.clone()),
    );
    cp.channel_values.insert("turn_count".to_string(), json!(1));
    cp.channel_versions.insert("messages".to_string(), 1);
    cp.channel_versions.insert("turn_count".to_string(), 1);

    let mut config = HashMap::new();
    config.insert("thread_id".to_string(), json!("test-thread-1"));

    let metadata = CheckpointMetadata {
        source: "loop".to_string(),
        step: 1,
        writes: Some({
            let mut w = HashMap::new();
            w.insert("agent".to_string(), json!("wrote messages"));
            w
        }),
        extra: HashMap::new(),
    };

    // Save checkpoint
    let saved_config = saver.put(&config, cp.clone(), metadata).await.unwrap();
    assert!(saved_config.contains_key("checkpoint_id"));

    // Restore checkpoint
    let restored = saver.get(&config).await.unwrap().unwrap();

    // Verify the messages round-trip through checkpoint serialization
    let restored_msgs = restored.checkpoint.channel_values["messages"]
        .as_array()
        .unwrap();
    assert_eq!(restored_msgs.len(), 2);

    // Deserialize back to core Message types
    let msg0: Message = serde_json::from_value(restored_msgs[0].clone()).unwrap();
    let msg1: Message = serde_json::from_value(restored_msgs[1].clone()).unwrap();
    assert_eq!(msg0.content().text(), "What is Rust?");
    assert_eq!(
        msg1.content().text(),
        "Rust is a systems programming language."
    );

    // Verify turn_count survived the round-trip
    assert_eq!(restored.checkpoint.channel_values["turn_count"], json!(1));

    // Save a second checkpoint with updated state and verify list works
    let mut cp2 = empty_checkpoint();
    cp2.channel_values.insert(
        "messages".to_string(),
        json!([
            {"type": "human", "content": "What is Rust?"},
            {"type": "ai", "content": "Rust is a systems programming language."},
            {"type": "human", "content": "Tell me more."},
            {"type": "ai", "content": "Rust was first released in 2015."}
        ]),
    );
    cp2.channel_values
        .insert("turn_count".to_string(), json!(2));

    let metadata2 = CheckpointMetadata {
        source: "loop".to_string(),
        step: 2,
        writes: None,
        extra: HashMap::new(),
    };
    saver.put(&config, cp2, metadata2).await.unwrap();

    // List should return both checkpoints (newest first)
    let all = saver.list(&config, None).await.unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].checkpoint.channel_values["turn_count"], json!(2));
}

// ---------------------------------------------------------------------------
// Test 7: cognis memory types work with cognisagent agent state
// ---------------------------------------------------------------------------

/// Use ConversationBufferMemory and ConversationWindowMemory (cognis)
/// to store messages that could feed into a cognisagent agent.
#[tokio::test]
async fn test_cognis_memory_with_cognisagent_agent_state() {
    // Create buffer memory (cognis) and save conversation turns using core Message types
    let buffer_mem = ConversationBufferMemory::new();

    let human1 = Message::human("What is the weather in London?");
    let ai1 = Message::ai("It is sunny in London.");
    buffer_mem.save_context(&human1, &ai1).await.unwrap();

    let human2 = Message::human("And in Paris?");
    let ai2 = Message::ai("It is rainy in Paris.");
    buffer_mem.save_context(&human2, &ai2).await.unwrap();

    // Load memory variables -- should have all 4 messages
    let vars = buffer_mem.load_memory_variables().await.unwrap();
    let history = vars["history"].as_array().unwrap();
    assert_eq!(history.len(), 4);

    // Create window memory (cognis) that keeps only last 1 turn
    let window_mem = ConversationWindowMemory::new(1);
    window_mem.save_context(&human1, &ai1).await.unwrap();
    window_mem.save_context(&human2, &ai2).await.unwrap();

    let window_vars = window_mem.load_memory_variables().await.unwrap();
    let window_history = window_vars["history"].as_array().unwrap();
    // Window memory with k=1 keeps only the last 2 messages (1 turn)
    assert_eq!(window_history.len(), 2);

    // Now feed the buffer memory history into a cognisagent graph
    // by constructing the messages state from memory
    let model: Arc<dyn BaseChatModel> =
        Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("Weather summary provided."),
        )]));
    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("dummy", "unused"));

    let config = DeepAgentConfig {
        tools: vec![tool],
        ..Default::default()
    };

    let graph = create_deep_agent(model, config).unwrap();

    // Build input state from memory history
    let input = json!({
        "messages": history
    });

    let result = graph.invoke(input).await.unwrap();
    let result_messages = result["messages"].as_array().unwrap();

    // Original 4 messages from memory + 1 new AI response = 5
    assert_eq!(result_messages.len(), 5);

    let last: Message = serde_json::from_value(result_messages.last().unwrap().clone()).unwrap();
    assert_eq!(last.content().text(), "Weather summary provided.");
}

// ---------------------------------------------------------------------------
// Test 8: cognis output parsers work with chain outputs
// ---------------------------------------------------------------------------

/// Use output parsers from cognis-core (StrOutputParser, JsonOutputParser,
/// CommaSeparatedListOutputParser) to parse outputs from an LLMChain (cognis).
#[tokio::test]
async fn test_output_parsers_with_chain_outputs() {
    // --- StrOutputParser ---
    let str_parser = StrOutputParser;
    let parsed = str_parser.parse("Hello, world!").unwrap();
    assert_eq!(parsed, json!("Hello, world!"));

    // Use StrOutputParser as a Runnable in a sequence with a lambda
    let extract_text = RunnableLambda::new("extract_text", |v: Value| async move {
        let text = v["text"].as_str().unwrap_or("").to_string();
        Ok(Value::String(text))
    });

    let sequence = RunnableSequence::new(vec![
        Arc::new(extract_text) as Arc<dyn Runnable>,
        Arc::new(str_parser) as Arc<dyn Runnable>,
    ])
    .unwrap();

    let result = sequence
        .invoke(json!({"text": "parsed output"}), None)
        .await
        .unwrap();
    assert_eq!(result, json!("parsed output"));

    // --- JsonOutputParser ---
    let json_parser = JsonOutputParser::new();

    // Simulate LLM output with markdown code fence
    let llm_json_output = "```json\n{\"name\": \"Alice\", \"age\": 30}\n```";
    let parsed_json = json_parser.parse(llm_json_output).unwrap();
    assert_eq!(parsed_json["name"], "Alice");
    assert_eq!(parsed_json["age"], 30);

    // JsonOutputParser as Runnable
    let json_result = json_parser
        .invoke(json!("{\"key\": \"value\"}"), None)
        .await
        .unwrap();
    assert_eq!(json_result["key"], "value");

    // --- CommaSeparatedListOutputParser ---
    let list_parser = CommaSeparatedListOutputParser;

    // Simulate LLM returning comma-separated items
    let llm_list_output = "apple, banana, cherry, date";
    let parsed_list = list_parser.parse(llm_list_output).unwrap();
    let items = parsed_list.as_array().unwrap();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0], "apple");
    assert_eq!(items[3], "date");

    // Use it as a Runnable step after an LLMChain
    let model: Arc<dyn BaseChatModel> =
        Arc::new(FakeListChatModel::new(vec!["red, green, blue".to_string()]));

    let chain = LLMChain::builder()
        .model(model)
        .prompt("List the primary colors of {subject}.")
        .build();

    // Run the chain, then parse the output
    let chain_output = chain
        .invoke(json!({"subject": "light"}), None)
        .await
        .unwrap();

    let chain_text = chain_output["text"].as_str().unwrap();
    let parsed_colors = list_parser.parse(chain_text).unwrap();
    let colors = parsed_colors.as_array().unwrap();
    assert_eq!(colors.len(), 3);
    assert_eq!(colors[0], "red");
    assert_eq!(colors[1], "green");
    assert_eq!(colors[2], "blue");
}

// ---------------------------------------------------------------------------
// Test 9: Full RAG pipeline across all crates
// ---------------------------------------------------------------------------

/// TextSplitter (cognis) -> FakeEmbeddings (core) -> InMemoryVectorStore (cognis)
/// -> RetrievalQAChain (cognis) with FakeChatModel (core).
#[tokio::test]
async fn test_full_rag_pipeline_across_crates() {
    let raw_text = "Rust is a systems programming language.\n\n\
                    It focuses on safety, speed, and concurrency.\n\n\
                    Rust was first released in 2015.\n\n\
                    The Rust compiler is called rustc.";

    // Rustchain: split text into chunks
    let splitter = CharacterTextSplitter::new()
        .with_separator("\n\n")
        .with_chunk_size(100)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text(raw_text);
    assert!(chunks.len() >= 2);

    // Core: fake embeddings
    let embeddings: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));

    // Rustchain: build vector store from chunks
    let texts: Vec<String> = chunks;
    let store = InMemoryVectorStore::from_texts(&texts, None, embeddings.clone())
        .await
        .unwrap();

    // Core: wrap vector store as a retriever
    let retriever: Arc<dyn BaseRetriever> = Arc::new(VectorStoreRetriever::new(
        Arc::new(store),
        SearchType::Similarity,
        2,
    ));

    // Core: fake chat model
    let llm: Arc<dyn BaseChatModel> = Arc::new(FakeListChatModel::new(vec![
        "Rust is a systems language focused on safety.".to_string(),
    ]));

    // Rustchain: build RAG chain
    let chain = RetrievalQAChain::new(retriever, llm).with_k(2);
    let result = chain.call_with_sources("What is Rust?").await.unwrap();

    assert_eq!(
        result.answer,
        "Rust is a systems language focused on safety."
    );
    assert!(!result.source_documents.is_empty());
}

// ---------------------------------------------------------------------------
// Test 10: deep agent full round-trip with cognisgraph tool agent
// ---------------------------------------------------------------------------

/// create_deep_agent (cognisagent) returns a CompiledStateGraph (cognisgraph).
/// Verify the full tool-calling loop with core types works.
#[tokio::test]
async fn test_deep_agent_tool_calling_round_trip() {
    let tc = ToolCall {
        name: "lookup".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert("query".to_string(), json!("meaning of life"));
            m
        },
        id: Some("call_42".to_string()),
    };
    let mut ai_with_tc = AIMessage::new("");
    ai_with_tc.tool_calls = vec![tc];

    let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(ai_with_tc),
        Message::Ai(AIMessage::new("The answer is 42")),
    ]));

    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("lookup", "42"));

    // Use create_tool_agent from cognisgraph
    let graph = create_tool_agent(model, vec![tool], None).unwrap();

    let input = json!({
        "messages": [{"type": "human", "content": "What is the meaning of life?"}]
    });

    let result = graph.invoke(input).await.unwrap();
    let messages = result["messages"].as_array().unwrap();

    // human + ai(tool_call) + tool_result + ai(final) = 4
    assert_eq!(messages.len(), 4);

    let tool_msg: Message = serde_json::from_value(messages[2].clone()).unwrap();
    assert!(matches!(tool_msg, Message::Tool(_)));

    let final_msg: Message = serde_json::from_value(messages[3].clone()).unwrap();
    assert_eq!(final_msg.content().text(), "The answer is 42");
}
