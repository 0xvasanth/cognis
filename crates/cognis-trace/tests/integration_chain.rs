//! End-to-end integration: a mock chain emits CallbackHandler events;
//! TracingHandler builds the span tree; MockExporter receives it.

use std::sync::Arc;

use cognis_core::callbacks::CallbackHandler;
use cognis_trace::span::{SpanKind, TokenUsage};
use cognis_trace::{MockExporter, TraceMeta, TracingHandler};
use uuid::Uuid;

#[tokio::test]
async fn nested_chain_with_llm_produces_correct_tree() {
    let mock = MockExporter::new();
    let collected = mock.collected();
    let handler = TracingHandler::builder()
        .with_exporter(mock)
        .with_default_pricing()
        .build();
    let h = Arc::new(handler);

    cognis_trace::parent::scope(async {
        let outer = Uuid::new_v4();
        let llm = Uuid::new_v4();

        h.on_chain_start("outer", &serde_json::json!({"q": "hi"}), outer);
        h.on_llm_start("gpt-4o", &serde_json::json!({}), llm);
        h.on_llm_end(
            "gpt-4o",
            &serde_json::json!({
                "content": "hello!",
                "model": "gpt-4o-2024-08-06",
                "provider": "openai",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            }),
            llm,
        );
        h.on_chain_end("outer", &serde_json::json!({"a": "hello!"}), outer);
    })
    .await;

    // shutdown drains batchers
    Arc::try_unwrap(h)
        .ok()
        .expect("only one Arc")
        .shutdown()
        .await;

    let spans = collected.lock().unwrap();
    assert_eq!(spans.len(), 2, "{spans:#?}");
    let outer = spans.iter().find(|s| s.kind == SpanKind::Chain).unwrap();
    let llm = spans.iter().find(|s| s.kind == SpanKind::Generation).unwrap();
    assert_eq!(llm.parent_run_id, Some(outer.run_id));
    assert_eq!(llm.trace_id, outer.run_id);
    let g = llm.generation.as_ref().unwrap();
    assert_eq!(g.usage, TokenUsage { input: 10, output: 5, cache_read: 0, cache_write: 0 });
    assert!(g.cost.is_some(), "cost should compute via prefix-match on 'gpt-4o'");
}

#[test]
fn trace_meta_helpers_build_metadata_object() {
    let v = serde_json::Value::Null;
    let v = cognis_trace::meta::merge_into(v, TraceMeta::session("s1"));
    let v = cognis_trace::meta::merge_into(v, TraceMeta::user("u1"));
    let obj = v.as_object().unwrap();
    assert_eq!(obj["session_id"], "s1");
    assert_eq!(obj["user_id"], "u1");
}
