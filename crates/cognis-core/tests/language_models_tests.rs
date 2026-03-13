use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::*;
use cognis_core::messages::{AIMessage, HumanMessage, Message};
use cognis_core::outputs::*;
use cognis_core::runnables::Runnable;

// ─── Mock LLM ───

struct MockLLM {
    response: String,
}

#[async_trait]
impl BaseLanguageModel for MockLLM {
    async fn generate(&self, prompts: &[String]) -> Result<LLMResult> {
        Ok(LLMResult {
            generations: prompts
                .iter()
                .map(|_| vec![Generation::new(&self.response)])
                .collect(),
            llm_output: None,
            run: None,
        })
    }

    async fn generate_chat(&self, _messages: &[Vec<Message>]) -> Result<ChatResult> {
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&self.response))],
            llm_output: None,
        })
    }

    fn model_type(&self) -> &str {
        "mock_llm"
    }
}

#[async_trait]
impl BaseLLM for MockLLM {
    async fn _generate(&self, prompts: &[String], _stop: Option<&[String]>) -> Result<LLMResult> {
        self.generate(prompts).await
    }

    fn llm_type(&self) -> &str {
        "mock_llm"
    }
}

// ─── Mock Chat Model ───

struct MockChatModel {
    response: String,
}

#[async_trait]
impl BaseChatModel for MockChatModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&self.response))],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "mock_chat"
    }
}

// ─── BaseLLM Tests ───

#[tokio::test]
async fn test_base_llm_generate() {
    let llm = MockLLM {
        response: "42".into(),
    };
    let result = llm
        ._generate(&["What is the answer?".into()], None)
        .await
        .unwrap();
    assert_eq!(result.generations.len(), 1);
    assert_eq!(result.generations[0][0].text, "42");
}

#[tokio::test]
async fn test_base_llm_predict() {
    let llm = MockLLM {
        response: "Hello!".into(),
    };
    let result = llm.predict("Say hello").await.unwrap();
    assert_eq!(result, "Hello!");
}

#[tokio::test]
async fn test_base_llm_stream_not_implemented() {
    let llm = MockLLM {
        response: "test".into(),
    };
    let result = llm._stream("prompt", None).await;
    assert!(result.is_err());
    match result {
        Err(CognisError::NotImplemented(_)) => {}
        other => panic!("Expected NotImplemented, got {:?}", other.err()),
    }
}

#[test]
fn test_base_llm_type() {
    let llm = MockLLM {
        response: "".into(),
    };
    assert_eq!(llm.llm_type(), "mock_llm");
}

#[test]
fn test_base_llm_get_num_tokens() {
    let llm = MockLLM {
        response: "".into(),
    };
    assert_eq!(llm.get_num_tokens("Hello world!"), 3);
}

#[test]
fn test_base_llm_identifying_params() {
    let llm = MockLLM {
        response: "".into(),
    };
    assert!(llm.identifying_params().is_empty());
}

#[test]
fn test_extract_prompt_string() {
    let prompt = llm::extract_prompt(&json!("Hello"));
    assert_eq!(prompt, "Hello");
}

#[test]
fn test_extract_prompt_object() {
    let prompt = llm::extract_prompt(&json!({"prompt": "Hello"}));
    assert_eq!(prompt, "Hello");
}

#[test]
fn test_extract_prompt_number() {
    let prompt = llm::extract_prompt(&json!(42));
    assert_eq!(prompt, "42");
}

// ─── BaseChatModel Tests ───

#[tokio::test]
async fn test_chat_model_generate() {
    let model = MockChatModel {
        response: "I'm helpful!".into(),
    };
    let messages = vec![Message::Human(cognis_core::messages::HumanMessage::new(
        "Hello",
    ))];
    let result = model._generate(&messages, None).await.unwrap();
    assert_eq!(result.generations.len(), 1);
    assert_eq!(result.generations[0].text, "I'm helpful!");
}

#[tokio::test]
async fn test_chat_model_invoke_messages() {
    let model = MockChatModel {
        response: "Response".into(),
    };
    let messages = vec![Message::Human(cognis_core::messages::HumanMessage::new(
        "Query",
    ))];
    let ai_msg = model.invoke_messages(&messages, None).await.unwrap();
    assert_eq!(ai_msg.base.content.text(), "Response");
}

#[tokio::test]
async fn test_chat_model_generate_batch() {
    let model = MockChatModel {
        response: "Reply".into(),
    };
    let batches = vec![
        vec![Message::Human(cognis_core::messages::HumanMessage::new(
            "A",
        ))],
        vec![Message::Human(cognis_core::messages::HumanMessage::new(
            "B",
        ))],
    ];
    let results = model.generate(&batches, None).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_chat_model_stream_not_implemented() {
    let model = MockChatModel {
        response: "test".into(),
    };
    let messages = vec![Message::Human(cognis_core::messages::HumanMessage::new(
        "Hi",
    ))];
    let result = model._stream(&messages, None).await;
    assert!(result.is_err());
    match result {
        Err(CognisError::NotImplemented(_)) => {}
        other => panic!("Expected NotImplemented, got {:?}", other.err()),
    }
}

#[test]
fn test_chat_model_type() {
    let model = MockChatModel {
        response: "".into(),
    };
    assert_eq!(model.llm_type(), "mock_chat");
}

#[test]
fn test_chat_model_bind_tools_not_implemented() {
    let model = MockChatModel {
        response: "".into(),
    };
    let result = model.bind_tools(&[], None);
    assert!(result.is_err());
    match result {
        Err(CognisError::NotImplemented(_)) => {}
        other => panic!("Expected NotImplemented, got {:?}", other.err()),
    }
}

#[test]
fn test_chat_model_profile_defaults() {
    let model = MockChatModel {
        response: "".into(),
    };
    let profile = model.profile();
    assert_eq!(profile.tool_calling, None);
    assert_eq!(profile.structured_output, None);
    assert_eq!(profile.max_input_tokens, None);
}

#[test]
fn test_chat_model_num_tokens_from_messages() {
    let model = MockChatModel {
        response: "".into(),
    };
    let messages = vec![Message::Human(cognis_core::messages::HumanMessage::new(
        "Hello world",
    ))];
    let tokens = model.get_num_tokens_from_messages(&messages);
    assert_eq!(tokens, 2);
}

// ─── ToolChoice Tests ───

#[test]
fn test_tool_choice_variants() {
    let auto = ToolChoice::Auto;
    let any = ToolChoice::Any;
    let specific = ToolChoice::Tool("calculator".into());
    let none = ToolChoice::None;

    match auto {
        ToolChoice::Auto => {}
        _ => panic!("Expected Auto"),
    }
    match any {
        ToolChoice::Any => {}
        _ => panic!("Expected Any"),
    }
    match specific {
        ToolChoice::Tool(name) => assert_eq!(name, "calculator"),
        _ => panic!("Expected Tool"),
    }
    match none {
        ToolChoice::None => {}
        _ => panic!("Expected None"),
    }
}

// ─── StreamingMode Tests ───

#[test]
fn test_streaming_mode_variants() {
    assert_eq!(StreamingMode::Always, StreamingMode::Always);
    assert_ne!(StreamingMode::Always, StreamingMode::Never);
    assert_ne!(StreamingMode::Never, StreamingMode::SkipToolCalling);
}

// ─── ModelProfile Tests ───

#[test]
fn test_model_profile_default() {
    let profile = ModelProfile::default();
    assert_eq!(profile.max_input_tokens, None);
    assert_eq!(profile.max_output_tokens, None);
    assert_eq!(profile.text_inputs, None);
    assert_eq!(profile.image_inputs, None);
    assert_eq!(profile.image_url_inputs, None);
    assert_eq!(profile.pdf_inputs, None);
    assert_eq!(profile.audio_inputs, None);
    assert_eq!(profile.video_inputs, None);
    assert_eq!(profile.image_tool_message, None);
    assert_eq!(profile.pdf_tool_message, None);
    assert_eq!(profile.reasoning_output, None);
    assert_eq!(profile.text_outputs, None);
    assert_eq!(profile.image_outputs, None);
    assert_eq!(profile.audio_outputs, None);
    assert_eq!(profile.video_outputs, None);
    assert_eq!(profile.tool_calling, None);
    assert_eq!(profile.tool_choice, None);
    assert_eq!(profile.structured_output, None);
}

#[test]
fn test_model_profile_custom() {
    let profile = ModelProfile {
        max_input_tokens: Some(128000),
        max_output_tokens: Some(4096),
        text_inputs: Some(true),
        image_inputs: Some(true),
        audio_inputs: Some(false),
        tool_calling: Some(true),
        structured_output: Some(true),
        reasoning_output: Some(true),
        ..Default::default()
    };
    assert_eq!(profile.max_input_tokens, Some(128000));
    assert_eq!(profile.max_output_tokens, Some(4096));
    assert_eq!(profile.tool_calling, Some(true));
    assert_eq!(profile.structured_output, Some(true));
    assert_eq!(profile.reasoning_output, Some(true));
    assert_eq!(profile.image_inputs, Some(true));
    assert_eq!(profile.audio_inputs, Some(false));
    // Unset fields remain None
    assert_eq!(profile.video_inputs, None);
    assert_eq!(profile.pdf_inputs, None);
}

#[test]
fn test_model_profile_serialization() {
    let profile = ModelProfile {
        max_input_tokens: Some(200000),
        max_output_tokens: Some(8192),
        tool_calling: Some(true),
        text_inputs: Some(true),
        image_inputs: Some(true),
        reasoning_output: Some(true),
        ..Default::default()
    };

    let json = serde_json::to_value(&profile).unwrap();
    assert_eq!(json["max_input_tokens"], 200000);
    assert_eq!(json["max_output_tokens"], 8192);
    assert_eq!(json["tool_calling"], true);
    assert_eq!(json["text_inputs"], true);
    assert_eq!(json["image_inputs"], true);
    assert_eq!(json["reasoning_output"], true);
    // None fields should be skipped
    assert!(json.get("video_inputs").is_none());
    assert!(json.get("pdf_inputs").is_none());
    assert!(json.get("audio_outputs").is_none());
}

#[test]
fn test_model_profile_deserialization() {
    let json = serde_json::json!({
        "max_input_tokens": 128000,
        "tool_calling": true,
        "image_inputs": false
    });

    let profile: ModelProfile = serde_json::from_value(json).unwrap();
    assert_eq!(profile.max_input_tokens, Some(128000));
    assert_eq!(profile.tool_calling, Some(true));
    assert_eq!(profile.image_inputs, Some(false));
    // Missing fields deserialize as None
    assert_eq!(profile.max_output_tokens, None);
    assert_eq!(profile.video_inputs, None);
}

#[test]
fn test_model_profile_registry() {
    use std::collections::HashMap;

    let mut registry: cognis_core::language_models::ModelProfileRegistry = HashMap::new();
    registry.insert(
        "claude-sonnet-4".to_string(),
        ModelProfile {
            max_input_tokens: Some(200000),
            max_output_tokens: Some(8192),
            tool_calling: Some(true),
            structured_output: Some(true),
            reasoning_output: Some(true),
            text_inputs: Some(true),
            image_inputs: Some(true),
            ..Default::default()
        },
    );

    let profile = registry.get("claude-sonnet-4").unwrap();
    assert_eq!(profile.max_input_tokens, Some(200000));
    assert_eq!(profile.tool_calling, Some(true));
}

// ─── FakeListLLM Tests ───

#[tokio::test]
async fn test_fake_list_llm_cycles() {
    let llm = FakeListLLM::new(vec!["a".into(), "b".into(), "c".into()]);
    let r1 = llm._generate(&["q".into()], None).await.unwrap();
    assert_eq!(r1.generations[0][0].text, "a");
    let r2 = llm._generate(&["q".into()], None).await.unwrap();
    assert_eq!(r2.generations[0][0].text, "b");
    let r3 = llm._generate(&["q".into()], None).await.unwrap();
    assert_eq!(r3.generations[0][0].text, "c");
    // wraps around
    let r4 = llm._generate(&["q".into()], None).await.unwrap();
    assert_eq!(r4.generations[0][0].text, "a");
}

#[tokio::test]
async fn test_fake_list_llm_predict() {
    let llm = FakeListLLM::new(vec!["hello".into()]);
    let result = llm.predict("anything").await.unwrap();
    assert_eq!(result, "hello");
}

// ─── FakeListChatModel Tests ───

#[tokio::test]
async fn test_fake_list_chat_model_cycles() {
    let model = FakeListChatModel::new(vec!["x".into(), "y".into()]);
    let msgs = vec![Message::Human(HumanMessage::new("test"))];
    let r1 = model._generate(&msgs, None).await.unwrap();
    assert_eq!(r1.generations[0].text, "x");
    let r2 = model._generate(&msgs, None).await.unwrap();
    assert_eq!(r2.generations[0].text, "y");
    let r3 = model._generate(&msgs, None).await.unwrap();
    assert_eq!(r3.generations[0].text, "x");
}

#[tokio::test]
async fn test_fake_list_chat_model_stream() {
    let model = FakeListChatModel::new(vec!["hi".into()]);
    let msgs = vec![Message::Human(HumanMessage::new("test"))];
    let mut stream = model._stream(&msgs, None).await.unwrap();
    let mut collected = String::new();
    while let Some(chunk) = stream.next().await {
        collected.push_str(&chunk.unwrap().text);
    }
    assert_eq!(collected, "hi");
}

// ─── FakeMessagesListChatModel Tests ───

#[tokio::test]
async fn test_fake_messages_list_chat_model() {
    let model = FakeMessagesListChatModel::new(vec![
        Message::Ai(AIMessage::new("response1")),
        Message::Ai(AIMessage::new("response2")),
    ]);
    let msgs = vec![Message::Human(HumanMessage::new("test"))];
    let r1 = model._generate(&msgs, None).await.unwrap();
    assert_eq!(r1.generations[0].text, "response1");
    let r2 = model._generate(&msgs, None).await.unwrap();
    assert_eq!(r2.generations[0].text, "response2");
}

// ─── ParrotFakeChatModel Tests ───

#[tokio::test]
async fn test_parrot_echoes_last_message() {
    let model = ParrotFakeChatModel::new();
    let msgs = vec![
        Message::Human(HumanMessage::new("first")),
        Message::Human(HumanMessage::new("echo me")),
    ];
    let result = model._generate(&msgs, None).await.unwrap();
    assert_eq!(result.generations[0].text, "echo me");
}

#[tokio::test]
async fn test_parrot_empty_messages_error() {
    let model = ParrotFakeChatModel::new();
    let result = model._generate(&[], None).await;
    assert!(result.is_err());
}

// ─── ChatModelRunnable Tests ───

#[tokio::test]
async fn test_chat_model_runnable_string_input() {
    let model = Arc::new(FakeListChatModel::new(vec!["reply".into()]));
    let runnable = ChatModelRunnable::new(model);
    let result = runnable.invoke(json!("hello"), None).await.unwrap();
    // AIMessage serialized as JSON — extract content from the structure
    let ai_msg: AIMessage = serde_json::from_value(result).unwrap();
    assert_eq!(ai_msg.base.content.text(), "reply");
}

#[tokio::test]
async fn test_chat_model_runnable_messages_input() {
    let model = Arc::new(ParrotFakeChatModel::new());
    let runnable = ChatModelRunnable::new(model);
    // Serialize actual Message objects to get proper JSON format
    let msgs = vec![Message::Human(HumanMessage::new("parrot this"))];
    let messages = serde_json::to_value(&msgs).unwrap();
    let result = runnable.invoke(messages, None).await.unwrap();
    let ai_msg: AIMessage = serde_json::from_value(result).unwrap();
    assert_eq!(ai_msg.base.content.text(), "parrot this");
}

// ─── LLMRunnable Tests ───

#[tokio::test]
async fn test_llm_runnable_string_input() {
    let llm = Arc::new(FakeListLLM::new(vec!["42".into()]));
    let runnable = LLMRunnable::new(llm);
    let result = runnable.invoke(json!("question"), None).await.unwrap();
    assert_eq!(result.as_str().unwrap(), "42");
}

#[tokio::test]
async fn test_llm_runnable_object_input() {
    let llm = Arc::new(FakeListLLM::new(vec!["answer".into()]));
    let runnable = LLMRunnable::new(llm);
    let result = runnable
        .invoke(json!({"prompt": "question"}), None)
        .await
        .unwrap();
    assert_eq!(result.as_str().unwrap(), "answer");
}
