use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use uuid::Uuid;

use rustchain_core::callbacks::*;
use rustchain_core::outputs::LLMResult;

struct TestHandler;

#[async_trait]
impl CallbackHandler for TestHandler {}

#[test]
fn test_callback_handler_default_filters() {
    let h = TestHandler;
    assert!(!h.ignore_llm());
    assert!(!h.ignore_chain());
    assert!(!h.ignore_agent());
    assert!(!h.ignore_retriever());
    assert!(!h.ignore_chat_model());
    assert!(!h.ignore_retry());
    assert!(!h.ignore_custom_event());
    assert!(!h.raise_error());
}

#[test]
fn test_callback_manager_new() {
    let mgr = CallbackManager::new(vec![], None);
    assert!(mgr.handlers().is_empty());
    assert!(mgr.inheritable_handlers().is_empty());
    assert!(mgr.parent_run_id().is_none());
    assert!(mgr.tags().is_empty());
    assert!(mgr.metadata().is_empty());
}

#[test]
fn test_callback_manager_default() {
    let mgr = CallbackManager::default();
    assert!(mgr.handlers().is_empty());
}

#[test]
fn test_callback_manager_add_handler_inheritable() {
    let mut mgr = CallbackManager::new(vec![], None);
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    mgr.add_handler(h, true);
    assert_eq!(mgr.handlers().len(), 1);
    assert_eq!(mgr.inheritable_handlers().len(), 1);
}

#[test]
fn test_callback_manager_add_handler_not_inheritable() {
    let mut mgr = CallbackManager::new(vec![], None);
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    mgr.add_handler(h, false);
    assert_eq!(mgr.handlers().len(), 1);
    // Started with 0 inheritable, added non-inheritable, still 0
    assert_eq!(mgr.inheritable_handlers().len(), 0);
}

#[test]
fn test_callback_manager_with_initial_handlers() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let mgr = CallbackManager::new(vec![h], None);
    // Constructing with handlers makes them both regular and inheritable
    assert_eq!(mgr.handlers().len(), 1);
    assert_eq!(mgr.inheritable_handlers().len(), 1);
}

#[test]
fn test_callback_manager_tags() {
    let mut mgr = CallbackManager::new(vec![], None);
    mgr.add_tags(vec!["tag1".into()], true);
    mgr.add_tags(vec!["tag2".into()], false);
    assert_eq!(mgr.tags().len(), 2);
    assert_eq!(mgr.inheritable_tags(), &["tag1"]);
}

#[test]
fn test_callback_manager_metadata() {
    let mut mgr = CallbackManager::new(vec![], None);
    let mut m = HashMap::new();
    m.insert("key".into(), json!("value"));
    mgr.add_metadata(m.clone(), true);

    let mut m2 = HashMap::new();
    m2.insert("key2".into(), json!(42));
    mgr.add_metadata(m2, false);

    assert_eq!(mgr.metadata().get("key"), Some(&json!("value")));
    assert_eq!(mgr.metadata().get("key2"), Some(&json!(42)));
    assert_eq!(mgr.inheritable_metadata().get("key"), Some(&json!("value")));
    assert!(mgr.inheritable_metadata().get("key2").is_none());
}

#[test]
fn test_callback_manager_get_child() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let mut mgr = CallbackManager::new(vec![h.clone()], None);
    mgr.add_tags(vec!["parent_tag".into()], true);
    mgr.add_tags(vec!["non_inheritable_tag".into()], false);

    let mut meta = HashMap::new();
    meta.insert("env".into(), json!("test"));
    mgr.add_metadata(meta, true);

    let run_id = Uuid::new_v4();
    let child = mgr.get_child(run_id);

    assert_eq!(child.parent_run_id(), Some(run_id));
    assert_eq!(child.handlers().len(), 1);
    assert_eq!(child.tags(), &["parent_tag"]);
    assert_eq!(child.metadata().get("env"), Some(&json!("test")));
}

#[test]
fn test_callback_manager_with_parent_run_id() {
    let id = Uuid::new_v4();
    let mgr = CallbackManager::new(vec![], None).with_parent_run_id(id);
    assert_eq!(mgr.parent_run_id(), Some(id));
}

#[test]
fn test_callback_manager_remove_handler() {
    let mut mgr = CallbackManager::new(vec![], None);
    let h1 = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let h2 = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    mgr.add_handler(h1, false);
    mgr.add_handler(h2, false);
    assert_eq!(mgr.handlers().len(), 2);
    mgr.remove_handler(0);
    assert_eq!(mgr.handlers().len(), 1);
    // Out of bounds removal is a no-op
    mgr.remove_handler(10);
    assert_eq!(mgr.handlers().len(), 1);
}

#[tokio::test]
async fn test_run_manager_for_chain() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let run_id = Uuid::new_v4();
    let run_mgr = RunManagerForChain::new(
        run_id,
        vec![h.clone()],
        vec![h],
        None,
        vec!["tag1".into()],
        vec!["tag1".into()],
        Default::default(),
        Default::default(),
    );
    assert_eq!(run_mgr.run_id(), run_id);
    assert_eq!(run_mgr.tags(), &["tag1"]);

    let child = run_mgr.get_child();
    assert_eq!(child.parent_run_id(), Some(run_id));
    assert_eq!(child.handlers().len(), 1);
}

#[tokio::test]
async fn test_run_manager_for_chain_dispatches() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let run_id = Uuid::new_v4();
    let run_mgr = RunManagerForChain::new(
        run_id,
        vec![h.clone()],
        vec![h],
        None,
        vec![],
        vec![],
        Default::default(),
        Default::default(),
    );
    // These should all succeed with the no-op handler
    run_mgr.on_chain_end(&json!({"result": "ok"})).await.unwrap();
    run_mgr.on_chain_error("some error").await.unwrap();
    run_mgr.on_text("hello").await.unwrap();
}

#[tokio::test]
async fn test_run_manager_for_llm() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let run_id = Uuid::new_v4();
    let run_mgr = RunManagerForLlm::new(run_id, vec![h], None);
    assert_eq!(run_mgr.run_id(), run_id);

    run_mgr.on_llm_new_token("hello").await.unwrap();
    run_mgr.on_llm_error("error").await.unwrap();
    run_mgr.on_text("text").await.unwrap();
}

#[tokio::test]
async fn test_run_manager_for_tool() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let run_id = Uuid::new_v4();
    let run_mgr = RunManagerForTool::new(run_id, vec![h.clone()], vec![h], None);
    assert_eq!(run_mgr.run_id(), run_id);

    run_mgr.on_tool_end("output").await.unwrap();
    run_mgr.on_tool_error("error").await.unwrap();
    run_mgr.on_text("text").await.unwrap();

    let child = run_mgr.get_child();
    assert_eq!(child.parent_run_id(), Some(run_id));
}

#[tokio::test]
async fn test_run_manager_for_retriever() {
    let h = Arc::new(TestHandler) as Arc<dyn CallbackHandler>;
    let run_id = Uuid::new_v4();
    let run_mgr = RunManagerForRetriever::new(run_id, vec![h], None);
    assert_eq!(run_mgr.run_id(), run_id);

    run_mgr.on_retriever_end(&[]).await.unwrap();
    run_mgr.on_retriever_error("error").await.unwrap();
    run_mgr.on_text("text").await.unwrap();
}

#[test]
fn test_stdout_handler_create() {
    let _h = StdOutCallbackHandler;
}

#[test]
fn test_streaming_handler_create() {
    let h = StreamingStdOutCallbackHandler;
    assert!(!h.ignore_llm());
}

/// Test that a custom handler with ignore flags works correctly.
struct IgnoringHandler;

#[async_trait]
impl CallbackHandler for IgnoringHandler {
    fn ignore_llm(&self) -> bool {
        true
    }
    fn ignore_chain(&self) -> bool {
        true
    }
    fn ignore_agent(&self) -> bool {
        true
    }
    fn ignore_retriever(&self) -> bool {
        true
    }
}

#[test]
fn test_custom_ignore_flags() {
    let h = IgnoringHandler;
    assert!(h.ignore_llm());
    assert!(h.ignore_chain());
    assert!(h.ignore_agent());
    assert!(h.ignore_retriever());
    assert!(!h.ignore_chat_model());
    assert!(!h.ignore_retry());
    assert!(!h.ignore_custom_event());
}

#[tokio::test]
async fn test_callback_manager_dispatch_respects_filters() {
    let h = Arc::new(IgnoringHandler) as Arc<dyn CallbackHandler>;
    let mgr = CallbackManager::new(vec![h], None);
    let run_id = Uuid::new_v4();

    // These should succeed (handlers with ignore flags are skipped)
    mgr.on_llm_start(&json!({}), &["prompt".into()], run_id)
        .await
        .unwrap();
    mgr.on_chain_start(&json!({}), &json!({}), run_id)
        .await
        .unwrap();
    mgr.on_chain_end(&json!({}), run_id).await.unwrap();
    mgr.on_llm_error("err", run_id).await.unwrap();
}

// --- UsageMetadataCallbackHandler tests ---

#[test]
fn test_usage_handler_new() {
    let handler = UsageMetadataCallbackHandler::new();
    assert_eq!(handler.total_tokens(), 0);
    assert_eq!(handler.prompt_tokens(), 0);
    assert_eq!(handler.completion_tokens(), 0);
    assert_eq!(handler.call_count(), 0);
    assert!(handler.get_usage().is_empty());
    assert!(handler.usage_metadata().is_empty());
}

#[test]
fn test_usage_handler_default() {
    let handler = UsageMetadataCallbackHandler::default();
    assert_eq!(handler.total_tokens(), 0);
    assert_eq!(handler.call_count(), 0);
}

#[tokio::test]
async fn test_usage_handler_legacy_token_usage() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());
    let run_id = Uuid::new_v4();

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }),
    );
    llm_output.insert("model_name".into(), json!("gpt-4o-mini"));

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };

    handler.on_llm_end(&result, run_id, None).await.unwrap();

    assert_eq!(handler.prompt_tokens(), 10);
    assert_eq!(handler.completion_tokens(), 20);
    assert_eq!(handler.total_tokens(), 30);
    assert_eq!(handler.call_count(), 1);

    let usage_map = handler.usage_metadata();
    assert!(usage_map.contains_key("gpt-4o-mini"));
    let model_usage = &usage_map["gpt-4o-mini"];
    assert_eq!(model_usage.input_tokens, 10);
    assert_eq!(model_usage.output_tokens, 20);
    assert_eq!(model_usage.total_tokens, 30);
}

#[tokio::test]
async fn test_usage_handler_structured_usage_metadata() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());
    let run_id = Uuid::new_v4();

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "usage_metadata".into(),
        json!({
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": 40
        }),
    );
    llm_output.insert("model_name".into(), json!("claude-sonnet-4-20250514"));

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };

    handler.on_llm_end(&result, run_id, None).await.unwrap();

    assert_eq!(handler.prompt_tokens(), 15);
    assert_eq!(handler.completion_tokens(), 25);
    assert_eq!(handler.total_tokens(), 40);
    assert_eq!(handler.call_count(), 1);

    let usage_map = handler.usage_metadata();
    assert!(usage_map.contains_key("claude-sonnet-4-20250514"));
}

#[tokio::test]
async fn test_usage_handler_multiple_calls_accumulate() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    // First call
    let mut llm_output1 = HashMap::new();
    llm_output1.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }),
    );
    llm_output1.insert("model_name".into(), json!("gpt-4o"));

    let result1 = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output1),
        run: None,
    };
    handler
        .on_llm_end(&result1, Uuid::new_v4(), None)
        .await
        .unwrap();

    // Second call, same model
    let mut llm_output2 = HashMap::new();
    llm_output2.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20
        }),
    );
    llm_output2.insert("model_name".into(), json!("gpt-4o"));

    let result2 = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output2),
        run: None,
    };
    handler
        .on_llm_end(&result2, Uuid::new_v4(), None)
        .await
        .unwrap();

    assert_eq!(handler.prompt_tokens(), 15);
    assert_eq!(handler.completion_tokens(), 35);
    assert_eq!(handler.total_tokens(), 50);
    assert_eq!(handler.call_count(), 2);

    // Per-model usage should be accumulated
    let usage_map = handler.usage_metadata();
    assert_eq!(usage_map.len(), 1);
    let model_usage = &usage_map["gpt-4o"];
    assert_eq!(model_usage.input_tokens, 15);
    assert_eq!(model_usage.output_tokens, 35);
    assert_eq!(model_usage.total_tokens, 50);
}

#[tokio::test]
async fn test_usage_handler_multiple_models() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    // Call with model A
    let mut llm_output1 = HashMap::new();
    llm_output1.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }),
    );
    llm_output1.insert("model_name".into(), json!("gpt-4o"));

    let result1 = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output1),
        run: None,
    };
    handler
        .on_llm_end(&result1, Uuid::new_v4(), None)
        .await
        .unwrap();

    // Call with model B
    let mut llm_output2 = HashMap::new();
    llm_output2.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 8,
            "completion_tokens": 12,
            "total_tokens": 20
        }),
    );
    llm_output2.insert("model_name".into(), json!("claude-haiku"));

    let result2 = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output2),
        run: None,
    };
    handler
        .on_llm_end(&result2, Uuid::new_v4(), None)
        .await
        .unwrap();

    // Aggregate totals
    assert_eq!(handler.prompt_tokens(), 18);
    assert_eq!(handler.completion_tokens(), 32);
    assert_eq!(handler.total_tokens(), 50);
    assert_eq!(handler.call_count(), 2);

    // Per-model
    let usage_map = handler.usage_metadata();
    assert_eq!(usage_map.len(), 2);
    assert!(usage_map.contains_key("gpt-4o"));
    assert!(usage_map.contains_key("claude-haiku"));

    // get_usage returns all entries
    let all_usage = handler.get_usage();
    assert_eq!(all_usage.len(), 2);
}

#[tokio::test]
async fn test_usage_handler_reset() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }),
    );
    llm_output.insert("model_name".into(), json!("gpt-4o"));

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };
    handler
        .on_llm_end(&result, Uuid::new_v4(), None)
        .await
        .unwrap();

    assert_eq!(handler.call_count(), 1);
    assert_eq!(handler.total_tokens(), 30);

    handler.reset();

    assert_eq!(handler.call_count(), 0);
    assert_eq!(handler.total_tokens(), 0);
    assert_eq!(handler.prompt_tokens(), 0);
    assert_eq!(handler.completion_tokens(), 0);
    assert!(handler.get_usage().is_empty());
    assert!(handler.usage_metadata().is_empty());
}

#[tokio::test]
async fn test_usage_handler_no_llm_output() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    let result = LLMResult {
        generations: vec![],
        llm_output: None,
        run: None,
    };
    handler
        .on_llm_end(&result, Uuid::new_v4(), None)
        .await
        .unwrap();

    // Call count still increments
    assert_eq!(handler.call_count(), 1);
    // No tokens tracked
    assert_eq!(handler.total_tokens(), 0);
    assert_eq!(handler.prompt_tokens(), 0);
    assert_eq!(handler.completion_tokens(), 0);
}

#[tokio::test]
async fn test_usage_handler_get_summary() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }),
    );
    llm_output.insert("model_name".into(), json!("test-model"));

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };
    handler
        .on_llm_end(&result, Uuid::new_v4(), None)
        .await
        .unwrap();

    let summary = handler.get_summary();
    assert_eq!(summary.input_tokens, 100);
    assert_eq!(summary.output_tokens, 200);
    assert_eq!(summary.total_tokens, 300);
    assert_eq!(summary.call_count, 1);
}

#[tokio::test]
async fn test_usage_handler_via_callback_manager() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());
    let mgr = CallbackManager::new(vec![handler.clone() as Arc<dyn CallbackHandler>], None);

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 50,
            "completion_tokens": 75,
            "total_tokens": 125
        }),
    );
    llm_output.insert("model_name".into(), json!("gpt-4o"));

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };
    mgr.on_llm_end(&result, Uuid::new_v4()).await.unwrap();

    assert_eq!(handler.total_tokens(), 125);
    assert_eq!(handler.prompt_tokens(), 50);
    assert_eq!(handler.completion_tokens(), 75);
    assert_eq!(handler.call_count(), 1);
}

#[tokio::test]
async fn test_usage_handler_unknown_model_when_name_missing() {
    let handler = Arc::new(UsageMetadataCallbackHandler::new());

    let mut llm_output = HashMap::new();
    llm_output.insert(
        "token_usage".into(),
        json!({
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15
        }),
    );
    // No model_name key

    let result = LLMResult {
        generations: vec![],
        llm_output: Some(llm_output),
        run: None,
    };
    handler
        .on_llm_end(&result, Uuid::new_v4(), None)
        .await
        .unwrap();

    let usage_map = handler.usage_metadata();
    assert!(usage_map.contains_key("unknown"));
}
