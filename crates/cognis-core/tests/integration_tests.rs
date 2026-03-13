use async_trait::async_trait;
use cognis_core::agents::{AgentAction, AgentFinish, AgentStep};
use cognis_core::caches::{BaseCache, InMemoryCache};
use cognis_core::callbacks::{CallbackHandler, CallbackManager};
use cognis_core::chat_history::{BaseChatMessageHistory, InMemoryChatMessageHistory};
use cognis_core::documents::BaseDocumentCompressor;
use cognis_core::documents::Document;
use cognis_core::error::{ErrorCode, CognisError};
use cognis_core::messages::*;
use cognis_core::outputs::{
    merge_chat_generation_chunks, ChatGeneration, ChatGenerationChunk, ChatResult, Generation,
    LLMResult, RunInfo,
};
use cognis_core::prompt_values::{PromptValue, StringPromptValue};
use cognis_core::stores::{BaseStore, InMemoryStore};
use cognis_core::utils::{generate_id, merge_dicts};
use serde_json::json;
use std::collections::HashMap;
use uuid::Uuid;

/// End-to-end: simulate an LLM call with tool use, caching, and history.
#[tokio::test]
async fn end_to_end_llm_with_tools() {
    // 1. Create a prompt
    let prompt = StringPromptValue::new("What is the weather in Paris?");
    let prompt_text = PromptValue::to_string(&prompt);
    assert_eq!(prompt_text, "What is the weather in Paris?");

    // 2. Simulate an AI response with a tool call
    let tool_call = ToolCall {
        name: "get_weather".into(),
        args: {
            let mut m = HashMap::new();
            m.insert("city".into(), json!("Paris"));
            m
        },
        id: Some("call_abc123".into()),
    };
    let ai_msg = AIMessage::new("Let me check the weather.")
        .with_tool_calls(vec![tool_call])
        .with_usage(UsageMetadata::new(15, 25, 40));

    assert_eq!(ai_msg.tool_calls.len(), 1);
    assert_eq!(ai_msg.tool_calls[0].name, "get_weather");

    // 3. Create a tool response
    let tool_msg = ToolMessage::new("Sunny, 22C", "call_abc123");
    assert_eq!(tool_msg.status, ToolStatus::Success);

    // 4. Build a chat history
    let history = InMemoryChatMessageHistory::new();
    history
        .add_messages(vec![
            Message::Human(HumanMessage::new("What is the weather in Paris?")),
            Message::Ai(ai_msg.clone()),
            Message::Tool(tool_msg),
        ])
        .await
        .unwrap();
    let msgs = history.messages().await.unwrap();
    assert_eq!(msgs.len(), 3);

    // 5. Final AI response
    let final_ai = AIMessage::new("It's sunny and 22C in Paris today!");
    history
        .add_messages(vec![Message::Ai(final_ai)])
        .await
        .unwrap();
    let msgs = history.messages().await.unwrap();
    assert_eq!(msgs.len(), 4);

    // 6. Cache the result
    let cache = InMemoryCache::new();
    let gen = Generation::new("It's sunny and 22C in Paris today!");
    cache
        .update(&prompt_text, "gpt-4", vec![gen])
        .await
        .unwrap();
    let cached = cache.lookup(&prompt_text, "gpt-4").await.unwrap();
    assert!(cached.is_some());
    assert_eq!(
        cached.unwrap()[0].text,
        "It's sunny and 22C in Paris today!"
    );
}

/// Test document storage and retrieval workflow.
#[tokio::test]
async fn document_store_workflow() {
    let store = InMemoryStore::new();

    let doc1 = Document::new("Rust is a systems programming language.")
        .with_id("doc1")
        .with_metadata({
            let mut m = HashMap::new();
            m.insert("source".into(), json!("wiki"));
            m
        });
    let doc2 = Document::new("Python is great for ML.").with_id("doc2");

    // Store documents as JSON
    store
        .mset(vec![
            ("doc1".into(), serde_json::to_value(&doc1).unwrap()),
            ("doc2".into(), serde_json::to_value(&doc2).unwrap()),
        ])
        .await
        .unwrap();

    let vals = store.mget(&["doc1".into(), "doc2".into()]).await.unwrap();
    assert!(vals[0].is_some());
    assert!(vals[1].is_some());

    let restored: Document = serde_json::from_value(vals[0].clone().unwrap()).unwrap();
    assert_eq!(
        restored.page_content,
        "Rust is a systems programming language."
    );
    assert_eq!(restored.id, Some("doc1".into()));
}

/// Test agent action/step workflow.
#[test]
fn agent_workflow() {
    let action = AgentAction::new("search", json!({"query": "Rust lang"}), "I need to search");
    let step = AgentStep::new(action.clone(), "Rust is a systems language.");
    assert_eq!(step.action.tool, "search");
    assert_eq!(step.observation, "Rust is a systems language.");

    let mut rv = HashMap::new();
    rv.insert(
        "output".into(),
        json!("Rust is a systems programming language."),
    );
    let finish = AgentFinish::new(rv, "Final Answer: Rust is a systems programming language.");
    assert!(finish.return_values.contains_key("output"));
}

/// Test LLMResult flatten across multiple prompts.
#[test]
fn llm_result_multi_prompt_flatten() {
    let result = LLMResult {
        generations: vec![
            vec![Generation::new("Paris"), Generation::new("Paris, France")],
            vec![Generation::new("Berlin")],
            vec![Generation::new("Tokyo")],
        ],
        llm_output: Some({
            let mut m = HashMap::new();
            m.insert("token_usage".into(), json!({"total": 100}));
            m.insert("model_name".into(), json!("gpt-4"));
            m
        }),
        run: Some(vec![RunInfo {
            run_id: Uuid::new_v4(),
        }]),
    };

    let flat = result.flatten();
    assert_eq!(flat.len(), 3);
    // First keeps full token usage
    assert_eq!(flat[0].generations[0].len(), 2);
    assert_eq!(
        flat[0].llm_output.as_ref().unwrap()["token_usage"],
        json!({"total": 100})
    );
    // Others have empty token usage
    assert_eq!(
        flat[1].llm_output.as_ref().unwrap()["token_usage"],
        json!({})
    );
    assert_eq!(
        flat[2].llm_output.as_ref().unwrap()["token_usage"],
        json!({})
    );
}

/// Test ChatGeneration and ChatResult.
#[test]
fn chat_generation_result() {
    let msg = AIMessage::new("Hello, how can I help?");
    let gen = ChatGeneration::new(msg);
    assert_eq!(gen.text, "Hello, how can I help?");

    let result = ChatResult {
        generations: vec![gen],
        llm_output: None,
    };
    assert_eq!(result.generations.len(), 1);
}

/// Test that merge_dicts works for combining response metadata.
#[test]
fn merge_response_metadata() {
    let meta1 = json!({"model": "gpt-4", "headers": {"x-request-id": "abc"}});
    let meta2 = json!({"headers": {"x-ratelimit": "100"}, "latency_ms": 42});
    let merged = merge_dicts(&meta1, &[&meta2]).unwrap();
    assert_eq!(merged["model"], json!("gpt-4"));
    assert_eq!(merged["headers"]["x-request-id"], json!("abc"));
    assert_eq!(merged["headers"]["x-ratelimit"], json!("100"));
    assert_eq!(merged["latency_ms"], json!(42));
}

/// Test usage metadata addition.
#[test]
fn usage_metadata_accumulation() {
    let u1 = UsageMetadata {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        input_token_details: Some(InputTokenDetails {
            audio: None,
            cache_creation: Some(20),
            cache_read: Some(30),
        }),
        output_token_details: Some(OutputTokenDetails {
            audio: Some(5),
            reasoning: Some(10),
        }),
    };
    let u2 = UsageMetadata {
        input_tokens: 200,
        output_tokens: 100,
        total_tokens: 300,
        input_token_details: Some(InputTokenDetails {
            audio: Some(15),
            cache_creation: None,
            cache_read: Some(40),
        }),
        output_token_details: Some(OutputTokenDetails {
            audio: None,
            reasoning: Some(20),
        }),
    };
    let sum = u1.add(&u2);
    assert_eq!(sum.input_tokens, 300);
    assert_eq!(sum.output_tokens, 150);
    assert_eq!(sum.total_tokens, 450);
    let itd = sum.input_token_details.unwrap();
    assert_eq!(itd.audio, Some(15));
    assert_eq!(itd.cache_creation, Some(20));
    assert_eq!(itd.cache_read, Some(70));
    let otd = sum.output_token_details.unwrap();
    assert_eq!(otd.audio, Some(5));
    assert_eq!(otd.reasoning, Some(30));
}

/// Test generate_id produces unique UUIDs.
#[test]
fn unique_ids() {
    let ids: Vec<String> = (0..100).map(|_| generate_id()).collect();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), unique.len());
}

/// Test error types.
#[test]
fn error_types_comprehensive() {
    let err = CognisError::OutputParserError {
        message: "bad".into(),
        observation: None,
        llm_output: Some("raw output".into()),
    };
    assert!(err.to_string().contains("bad"));

    let err = CognisError::NotImplemented("feature X".into());
    assert!(err.to_string().contains("feature X"));

    assert_eq!(ErrorCode::ModelRateLimit.as_str(), "MODEL_RATE_LIMIT");
}

// --- Integration tests for new features (Task 10) ---

/// Test chunk concatenation in a streaming simulation workflow.
#[test]
fn streaming_chunk_concatenation_workflow() {
    // Simulate receiving AI message chunks during streaming
    let chunk1 = AIMessageChunk::new("The capital ");
    let chunk2 = AIMessageChunk::new("of France ");
    let mut chunk3 = AIMessageChunk::new("is Paris.");
    chunk3.usage_metadata = Some(UsageMetadata::new(10, 15, 25));

    let combined = chunk1.add(chunk2).add(chunk3);
    assert_eq!(
        combined.base.content.text(),
        "The capital of France is Paris."
    );
    assert!(combined.usage_metadata.is_some());

    // Convert to regular message
    let msg = message_chunk_to_message(&Message::AiChunk(combined));
    assert_eq!(msg.message_type(), MessageType::Ai);
    assert_eq!(msg.content().text(), "The capital of France is Paris.");
}

/// Test ChatGenerationChunk in a streaming simulation.
#[test]
fn streaming_chat_generation_workflow() {
    let chunks = vec![
        ChatGenerationChunk::new(AIMessageChunk::new("Hello")),
        ChatGenerationChunk::new(AIMessageChunk::new(", ")),
        ChatGenerationChunk::new(AIMessageChunk::new("world!")),
    ];
    let merged = merge_chat_generation_chunks(chunks).unwrap();
    assert_eq!(merged.text, "Hello, world!");
    assert_eq!(merged.message.base.content.text(), "Hello, world!");
}

/// Test message utilities in a conversation management workflow.
#[test]
fn conversation_management_workflow() {
    // Build a conversation
    let messages = convert_to_messages(vec![
        ("system".into(), "You are a helpful assistant.".into()),
        ("human".into(), "What is Rust?".into()),
        (
            "assistant".into(),
            "Rust is a systems programming language.".into(),
        ),
        ("human".into(), "Tell me more.".into()),
        (
            "assistant".into(),
            "Rust focuses on safety and performance.".into(),
        ),
    ]);
    assert_eq!(messages.len(), 5);

    // Get buffer string
    let buffer = get_buffer_string(&messages, "User", "Bot");
    assert!(buffer.contains("User: What is Rust?"));
    assert!(buffer.contains("Bot: Rust is a systems programming language."));

    // Trim to last N tokens (word count)
    let counter = |s: &str| -> usize { s.split_whitespace().count() };
    let trimmed = trim_messages(&messages, 15, &counter, TrimStrategy::Last);
    assert!(!trimmed.is_empty());
    // Should include the more recent messages
    assert_eq!(
        trimmed.last().unwrap().content().text(),
        "Rust focuses on safety and performance."
    );

    // Filter out system messages
    let filtered = filter_messages(
        &messages,
        None,
        None,
        None,
        Some(&[MessageType::System]),
        None,
    );
    assert_eq!(filtered.len(), 4);
    assert!(filtered
        .iter()
        .all(|m| m.message_type() != MessageType::System));
}

/// Test CallbackManager in an end-to-end LLM simulation.
#[tokio::test]
async fn callback_manager_llm_simulation() {
    use std::sync::{Arc, Mutex};

    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let events_clone = events.clone();

    struct RecordingHandler {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl CallbackHandler for RecordingHandler {
        async fn on_llm_start(
            &self,
            _serialized: &serde_json::Value,
            _prompts: &[String],
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_start".into());
            Ok(())
        }

        async fn on_llm_new_token(
            &self,
            token: &str,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push(format!("token:{}", token));
            Ok(())
        }

        async fn on_llm_end(
            &self,
            _response: &LLMResult,
            _run_id: Uuid,
            _parent_run_id: Option<Uuid>,
        ) -> cognis_core::error::Result<()> {
            self.events.lock().unwrap().push("llm_end".into());
            Ok(())
        }
    }

    let mut manager = CallbackManager::new(vec![], None);
    manager.add_handler(
        Arc::new(RecordingHandler {
            events: events_clone,
        }),
        true,
    );

    let run_id = Uuid::new_v4();

    // Simulate LLM lifecycle
    manager
        .on_llm_start(&json!({}), &["What is Rust?".into()], run_id)
        .await
        .unwrap();
    for token in ["Rust", " is", " great"] {
        manager.on_llm_new_token(token, run_id).await.unwrap();
    }
    let result = LLMResult {
        generations: vec![vec![Generation::new("Rust is great")]],
        llm_output: None,
        run: None,
    };
    manager.on_llm_end(&result, run_id).await.unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded[0], "llm_start");
    assert_eq!(recorded[1], "token:Rust");
    assert_eq!(recorded[2], "token: is");
    assert_eq!(recorded[3], "token: great");
    assert_eq!(recorded[4], "llm_end");
}

/// Test document compressor in a RAG workflow simulation.
#[tokio::test]
async fn document_compressor_rag_workflow() {
    struct RelevanceCompressor;

    #[async_trait]
    impl BaseDocumentCompressor for RelevanceCompressor {
        async fn compress_documents(
            &self,
            documents: &[cognis_core::documents::Document],
            query: &str,
        ) -> cognis_core::error::Result<Vec<cognis_core::documents::Document>> {
            Ok(documents
                .iter()
                .filter(|d| {
                    query
                        .split_whitespace()
                        .any(|w| d.page_content.to_lowercase().contains(&w.to_lowercase()))
                })
                .cloned()
                .collect())
        }
    }

    let docs = vec![
        Document::new("Rust is a fast systems programming language"),
        Document::new("Python is great for machine learning"),
        Document::new("JavaScript runs in the browser"),
        Document::new("Rust's borrow checker ensures memory safety"),
    ];

    let compressor = RelevanceCompressor;
    let relevant = compressor.compress_documents(&docs, "Rust").await.unwrap();
    assert_eq!(relevant.len(), 2);
    assert!(relevant.iter().all(|d| d.page_content.contains("Rust")));
}

/// Test multimodal content blocks.
#[test]
fn multimodal_content() {
    let blocks = vec![
        ContentBlock::Text {
            text: "Look at this image:".into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        },
        ContentBlock::Image {
            id: None,
            url: None,
            base64: Some("base64data".into()),
            file_id: None,
            mime_type: Some("image/png".into()),
            index: None,
            image: None,
            source_type: Some("base64".into()),
            media_type: None,
            extras: None,
        },
        ContentBlock::Reasoning {
            reasoning: Some("I analyzed the image...".into()),
            id: None,
            index: None,
            extras: None,
        },
    ];
    let msg = HumanMessage::with_blocks(blocks);
    assert_eq!(msg.base.content.text(), "Look at this image:");

    // Roundtrip through serde
    let json = serde_json::to_value(&msg).unwrap();
    let back: HumanMessage = serde_json::from_value(json).unwrap();
    assert_eq!(msg, back);
}
