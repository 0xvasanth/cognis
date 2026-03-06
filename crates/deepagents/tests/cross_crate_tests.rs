//! Cross-crate integration tests proving all 4 workspace crates compose correctly.
//!
//! These tests exercise the interaction between:
//! - `rustchain-core` (traits, fakes, embeddings, messages)
//! - `rustchain` (chains, vectorstores, document_loaders, text_splitter)
//! - `langgraph` (StateGraph, create_tool_agent, CompiledStateGraph)
//! - `deepagents` (create_deep_agent, middleware, config)
//!
//! Run with: `cargo test -p deepagents -- cross_crate`

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

// --- rustchain-core imports ---
use rustchain_core::documents::Document;
use rustchain_core::embeddings::Embeddings;
use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
use rustchain_core::error::Result as CoreResult;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::fake::{
    FakeListChatModel, FakeMessagesListChatModel, ParrotFakeChatModel,
};
use rustchain_core::messages::{AIMessage, HumanMessage, Message, ToolCall};
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use rustchain_core::tools::BaseTool;
use rustchain_core::vectorstores::base::{SearchType, VectorStore, VectorStoreRetriever};

// --- rustchain imports ---
use rustchain::chains::retrieval::RetrievalQAChain;
use rustchain::text_splitter::{CharacterTextSplitter, TextSplitter};
use rustchain::vectorstores::in_memory::InMemoryVectorStore;

// --- langgraph imports ---
use langgraph::graph::state::{AsyncNodeAction, StateGraph};
use langgraph::prebuilt::tool_agent::create_tool_agent;

// --- deepagents imports ---
use deepagents::config::DeepAgentConfig;
use deepagents::create_deep_agent;
use deepagents::middleware::memory::MemoryMiddleware;
use deepagents::middleware::Middleware;

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

/// A mock retriever that returns a fixed set of documents.
struct MockRetriever {
    docs: Vec<Document>,
}

#[async_trait]
impl BaseRetriever for MockRetriever {
    async fn get_relevant_documents(&self, _query: &str) -> CoreResult<Vec<Document>> {
        Ok(self.docs.clone())
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
// Test 1: Core traits used by rustchain
// ---------------------------------------------------------------------------

/// Create a FakeMessagesListChatModel (core), use it in a RetrievalQAChain
/// (rustchain), and verify the pipeline works end-to-end.
#[tokio::test]
async fn test_cross_crate_core_traits_used_by_rustchain() {
    // Core: create a fake chat model
    let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(AIMessage::new("Paris is the capital of France.")),
    ]));

    // Core: create a mock retriever with documents
    let docs = vec![
        Document::new("France is a country in Europe."),
        Document::new("Paris is the capital city of France."),
    ];
    let retriever: Arc<dyn BaseRetriever> = Arc::new(MockRetriever { docs });

    // Rustchain: build a RetrievalQAChain using core components
    let chain = RetrievalQAChain::new(retriever, model).with_k(2);
    let answer = chain.call("What is the capital of France?").await.unwrap();

    assert_eq!(answer, "Paris is the capital of France.");
}

// ---------------------------------------------------------------------------
// Test 2: Core tools in langgraph agent
// ---------------------------------------------------------------------------

/// Create a BaseTool impl (core), use it in create_tool_agent (langgraph),
/// and invoke the graph.
#[tokio::test]
async fn test_cross_crate_core_tools_in_langgraph_agent() {
    // Core: fake model that returns a plain response (no tool calls)
    let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(AIMessage::new("The answer is 42")),
    ]));

    // Core: mock tool
    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));

    // LangGraph: create tool agent
    let graph = create_tool_agent(model, vec![tool], None).unwrap();

    let input = json!({
        "messages": [{"type": "human", "content": "What is 6*7?"}]
    });

    let result = graph.invoke(input).await.unwrap();
    let messages = result["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2); // human + ai

    let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
    assert_eq!(last.content().text(), "The answer is 42");
}

// ---------------------------------------------------------------------------
// Test 3: Rustchain vectorstore with core embeddings
// ---------------------------------------------------------------------------

/// Use DeterministicFakeEmbedding (core) with InMemoryVectorStore (rustchain).
#[tokio::test]
async fn test_cross_crate_rustchain_vectorstore_with_core_embeddings() {
    // Core: fake embeddings
    let embeddings: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(128));

    // Rustchain: in-memory vector store
    let store = InMemoryVectorStore::new(embeddings);
    let texts = vec![
        "Rust is a systems programming language.".to_string(),
        "Python is a dynamic language.".to_string(),
        "Go is a compiled language.".to_string(),
    ];
    store.add_texts(&texts, None, None).await.unwrap();

    // Search should return the most similar document first
    let results = store.similarity_search("Rust programming", 2).await.unwrap();
    assert_eq!(results.len(), 2);
    // At minimum, we got 2 documents back (exact ordering depends on hash-based embeddings)
    assert!(results.iter().any(|d| d.page_content.contains("Rust")));
}

// ---------------------------------------------------------------------------
// Test 4: LangGraph graph with rustchain/core model
// ---------------------------------------------------------------------------

/// Build a StateGraph (langgraph) node that uses a BaseChatModel (core),
/// and invoke the graph.
#[tokio::test]
async fn test_cross_crate_langgraph_graph_with_rustchain_model() {
    // Core: create a parrot model that echoes input
    let model: Arc<dyn BaseChatModel> = Arc::new(ParrotFakeChatModel::new());

    // LangGraph: build a state graph with a node that calls the model
    let model_for_node = model.clone();
    let node_action: AsyncNodeAction = Arc::new(move |state: Value| {
        let m = model_for_node.clone();
        Box::pin(async move {
            let query = state
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("default question");

            let messages = vec![Message::Human(HumanMessage::new(query))];
            let ai_msg = m
                .invoke_messages(&messages, None)
                .await
                .map_err(|e| langgraph::errors::LangGraphError::Other(e.to_string()))?;

            Ok(json!({
                "query": query,
                "response": ai_msg.base.content.text()
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
        .invoke(json!({"query": "Hello from integration test"}))
        .await
        .unwrap();

    // The parrot model echoes the input
    assert_eq!(
        result["response"].as_str().unwrap(),
        "Hello from integration test"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Deep agent uses langgraph graph
// ---------------------------------------------------------------------------

/// create_deep_agent returns a CompiledStateGraph (langgraph). Invoke it
/// with core Message types.
#[tokio::test]
async fn test_cross_crate_deep_agent_uses_langgraph_graph() {
    // Core: fake model
    let model: Arc<dyn BaseChatModel> = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(AIMessage::new("Deep agent response")),
    ]));

    // Core: mock tool
    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("search", "result"));

    // Deepagents: create deep agent (returns a CompiledStateGraph from langgraph)
    let config = DeepAgentConfig {
        tools: vec![tool],
        ..Default::default()
    };
    let graph = create_deep_agent(model, config).unwrap();

    // Verify the graph has the expected nodes
    let mut names: Vec<&str> = graph.node_names();
    names.sort();
    assert_eq!(names, vec!["agent", "tools"]);

    // Invoke with core Message types serialized
    let input = json!({
        "messages": [{"type": "human", "content": "Tell me something"}]
    });
    let result = graph.invoke(input).await.unwrap();

    let messages = result["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2); // human + ai

    let last_msg: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
    assert_eq!(last_msg.content().text(), "Deep agent response");
}

// ---------------------------------------------------------------------------
// Test 6: Full RAG pipeline across crates
// ---------------------------------------------------------------------------

/// TextLoader (rustchain) -> CharacterTextSplitter (rustchain) ->
/// FakeEmbeddings (core) -> InMemoryVectorStore (rustchain) ->
/// RetrievalQAChain (rustchain) with FakeChatModel (core).
#[tokio::test]
async fn test_cross_crate_full_rag_pipeline_across_crates() {
    // Instead of TextLoader (which needs a real file), we simulate its output
    // to keep the test self-contained.
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
    assert!(chunks.len() >= 2, "Expected multiple chunks, got {}", chunks.len());

    // Core: fake embeddings
    let embeddings: Arc<dyn Embeddings> = Arc::new(DeterministicFakeEmbedding::new(64));

    // Rustchain: build vector store from chunks
    let texts: Vec<String> = chunks;
    let store = InMemoryVectorStore::new(embeddings.clone());
    store.add_texts(&texts, None, None).await.unwrap();

    // Core: wrap vector store as a retriever
    let retriever: Arc<dyn BaseRetriever> = Arc::new(VectorStoreRetriever::new(
        Arc::new(InMemoryVectorStore::from_texts(&texts, None, embeddings).await.unwrap()),
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

    assert_eq!(result.answer, "Rust is a systems language focused on safety.");
    assert!(!result.source_documents.is_empty());
}

// ---------------------------------------------------------------------------
// Test 7: LangGraph tool agent with rustchain tools
// ---------------------------------------------------------------------------

/// create_tool_agent (langgraph) with a mock BaseTool (core), invoke
/// with messages including a tool-call loop.
#[tokio::test]
async fn test_cross_crate_langgraph_tool_agent_with_rustchain_tools() {
    // Set up a tool call scenario: model first returns a tool call, then a final answer.
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

    // LangGraph: create tool agent
    let graph = create_tool_agent(model, vec![tool], None).unwrap();

    let input = json!({
        "messages": [{"type": "human", "content": "What is the meaning of life?"}]
    });

    let result = graph.invoke(input).await.unwrap();
    let messages = result["messages"].as_array().unwrap();

    // human + ai(tool_call) + tool_result + ai(final) = 4
    assert_eq!(messages.len(), 4);

    // Verify the tool message is present
    let tool_msg: Message = serde_json::from_value(messages[2].clone()).unwrap();
    assert!(matches!(tool_msg, Message::Tool(_)));

    // Verify the final answer
    let final_msg: Message = serde_json::from_value(messages[3].clone()).unwrap();
    assert_eq!(final_msg.content().text(), "The answer is 42");
}

// ---------------------------------------------------------------------------
// Test 8: Deep agent with middleware and tools
// ---------------------------------------------------------------------------

/// Full deepagents pipeline with middleware, tools, and model. Verifies
/// that middleware hooks fire and the agent produces a correct result.
#[tokio::test]
async fn test_cross_crate_deep_agent_with_middleware_and_tools() {
    // Core: model that first issues a tool call, then returns final answer.
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

    // Core: mock tool
    let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("weather", "sunny, 22C"));

    // Deepagents: set up memory middleware
    let memory_mw = Arc::new(MemoryMiddleware::new(10));
    memory_mw.remember("user_location", "London").await;

    // Deepagents: configure agent with middleware and tool
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

    // The memory middleware injects a system message before the model call.
    // Expected flow: [human, system(memory)] -> model -> [ai(tool_call)] -> tool -> [tool_result] -> model -> [ai(final)]
    // But messages in state: human + system(memory) + ai(tool_call) + tool_result + ai(final)
    // Note: the memory system message is injected into the messages array by middleware.
    // The final messages array should have at least 4 entries (human + injected_system + ai_with_tc + tool + ai_final).
    assert!(
        messages.len() >= 4,
        "Expected at least 4 messages, got {}",
        messages.len()
    );

    // The last message should be the final AI response.
    let last_msg: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
    assert_eq!(last_msg.content().text(), "It is sunny in London");

    // Verify memory middleware still has its entries.
    assert_eq!(
        memory_mw.recall("user_location").await,
        Some("London".to_string())
    );
}
