//! Integration: verify ChatOpenAI's extra_headers reach the wire.
//!
//! Relevant to [#10](https://github.com/0xvasanth/cognis/issues/10) — OpenRouter
//! and other OpenAI-compatible gateways require attribution headers on every
//! request. This test spins up a wiremock server, sends a single chat
//! completion, and asserts both the custom headers AND the standard
//! Authorization header are present as expected.

#![cfg(feature = "openai")]

use cognis::chat_models::openai::ChatOpenAI;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use serde_json::json;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn extra_headers_reach_the_wire_on_invoke() {
    let server = MockServer::start().await;

    // A minimal OpenAI-compatible chat completion response.
    let body = json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hi" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("HTTP-Referer", "https://mysite.com"))
        .and(header("X-Title", "cognis-test-app"))
        .and(header_exists("Authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&server)
        .await;

    let model = ChatOpenAI::builder()
        .model("gpt-4o")
        .api_key("sk-test-key")
        .base_url(server.uri())
        .extra_header("HTTP-Referer", "https://mysite.com")
        .extra_header("X-Title", "cognis-test-app")
        .build()
        .unwrap();

    let messages = vec![Message::Human(HumanMessage::new("hello"))];
    let result = model._generate(&messages, None).await;

    assert!(result.is_ok(), "generate returned error: {:?}", result);
    // wiremock's `.expect(1)` + `MockServer::drop` automatically assert the
    // request was received. If headers didn't match, the mock would not
    // intercept it and the request would 404 / 500.
}
