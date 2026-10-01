//! Typed streaming of structured output.
//!
//! [`Client::stream_array`] turns a streamed JSON array into a
//! [`RunnableStream`] of validated `T` values, yielding each element as soon
//! as it closes. [`Client::stream_object_partial_value`] and
//! [`Client::stream_object_partial`] instead stream one object as
//! progressively-filled snapshots. All three are a single, tool-less model
//! turn over the plain streaming path, and all three end with an `Err` item
//! rather than ending silently when the model never produced the JSON.

use serde::de::DeserializeOwned;

use cognis_core::{
    close_partial_json, CognisError, Message, Result, RunnableStream, StreamingJsonArray,
};
use futures::StreamExt;

use crate::Client;

/// Longest run of raw model output (or of a decode error quoting it) echoed
/// in an error message, in characters.
const PREVIEW_CHARS: usize = 200;

impl Client {
    /// Stream a JSON array as typed elements, each delivered when it closes.
    ///
    /// Different from [`Client::stream`], which yields raw text chunks: this
    /// parses the array incrementally and yields validated `T` values.
    ///
    /// The prompt must ask the model for a JSON array — no instruction or
    /// schema is injected. Text around the array (prose, a code fence) is
    /// tolerated; see [`StreamingJsonArray`] for what counts as its start.
    ///
    /// Errors, all [`CognisError::Serialization`] unless noted:
    /// - a malformed element yields one `Err` for that element and the stream
    ///   continues;
    /// - an element still open when the model stops is yielded as an `Err`;
    /// - if the model answered without any array, one `Err` ends the stream
    ///   (a real empty array `[]` yields nothing and no error);
    /// - a provider error is yielded as-is and ends the stream.
    pub async fn stream_array<T: DeserializeOwned + Send + 'static>(
        &self,
        messages: Vec<Message>,
    ) -> Result<RunnableStream<T>> {
        let chunks = self.stream(messages).await?;
        let out = async_stream::stream! {
            let mut chunks = chunks;
            let mut parser = StreamingJsonArray::new();
            let mut head = OutputHead::default();
            while let Some(item) = chunks.next().await {
                match item {
                    Ok(chunk) => {
                        head.push(&chunk.content);
                        for raw in parser.push_str(&chunk.content) {
                            yield decode::<T>(&raw);
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            for raw in parser.finish() {
                yield decode::<T>(&raw);
            }
            if !parser.has_started() && head.has_content {
                yield Err(CognisError::Serialization(format!(
                    "no JSON array in model output: {}",
                    head.preview()
                )));
            }
        };
        Ok(RunnableStream::new(out))
    }

    /// Stream a single JSON object as progressively-filled snapshots.
    ///
    /// Different from [`Client::stream_array`], which yields whole elements:
    /// each item is the object parsed so far, with open strings, arrays and
    /// objects closed. String fields may be prefixes of their final value
    /// until the stream ends; numbers and literals appear only once complete
    /// (see [`close_partial_json`]). Snapshots that parse to the same value as
    /// the previous one are skipped.
    ///
    /// The prompt must ask the model for a JSON object — no instruction or
    /// schema is injected. Text around the object (prose, a code fence) is
    /// tolerated.
    ///
    /// If the provider stream ends before the object closes, the last good
    /// snapshot is the final item. If the model answered without anything
    /// that parses as an object, one [`CognisError::Serialization`] ends the
    /// stream. A provider error is yielded as-is and ends the stream.
    pub async fn stream_object_partial_value(
        &self,
        messages: Vec<Message>,
    ) -> Result<RunnableStream<serde_json::Value>> {
        let chunks = self.stream(messages).await?;
        let out = async_stream::stream! {
            let mut chunks = chunks;
            let mut buf = String::new();
            let mut head = OutputHead::default();
            let mut last: Option<serde_json::Value> = None;
            while let Some(item) = chunks.next().await {
                match item {
                    Ok(chunk) => {
                        head.push(&chunk.content);
                        buf.push_str(&chunk.content);
                        let snapshot = close_partial_json(&buf)
                            .and_then(|closed| serde_json::from_str::<serde_json::Value>(&closed).ok());
                        if let Some(v) = snapshot {
                            if last.as_ref() != Some(&v) {
                                last = Some(v.clone());
                                yield Ok(v);
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            if last.is_none() && head.has_content {
                yield Err(CognisError::Serialization(format!(
                    "no JSON object in model output: {}",
                    head.preview()
                )));
            }
        };
        Ok(RunnableStream::new(out))
    }

    /// Stream a single JSON object as typed, progressively-filled `T::Partial`.
    ///
    /// Different from [`Client::stream_object_partial_value`], which yields raw
    /// `Value` snapshots: each snapshot is decoded into the all-`Option`
    /// mirror generated by `#[derive(Partial)]`. As there, string fields may
    /// be prefixes until the stream ends and numbers appear only once
    /// complete.
    ///
    /// The prompt must ask the model for a JSON object matching `T` — no
    /// instruction or schema is injected.
    ///
    /// A mid-stream snapshot that does not fit the mirror yet (e.g. an enum
    /// variant whose name is still arriving) is skipped and logged at
    /// `debug`. If the *last* snapshot does not fit — the finished object has
    /// a wrong-typed field — one [`CognisError::Serialization`] ends the
    /// stream, so a consumer never mistakes an earlier snapshot for the final
    /// object. Errors from [`Client::stream_object_partial_value`] pass
    /// through and end the stream.
    pub async fn stream_object_partial<T: cognis_core::Partial>(
        &self,
        messages: Vec<Message>,
    ) -> Result<RunnableStream<T::Partial>> {
        let values = self.stream_object_partial_value(messages).await?;
        let out = async_stream::stream! {
            let mut values = values;
            let mut last_misfit: Option<String> = None;
            while let Some(item) = values.next().await {
                match item {
                    Ok(v) => match <T::Partial as serde::Deserialize>::deserialize(&v) {
                        Ok(p) => {
                            last_misfit = None;
                            yield Ok(p);
                        }
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                "partial snapshot does not fit the mirror yet; skipped"
                            );
                            last_misfit = Some(format!(
                                "final snapshot does not fit {}: {}: {}",
                                std::any::type_name::<T::Partial>(),
                                preview(&e.to_string()),
                                preview(&v.to_string()),
                            ));
                        }
                    },
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            if let Some(message) = last_misfit {
                yield Err(CognisError::Serialization(message));
            }
        };
        Ok(RunnableStream::new(out))
    }
}

fn decode<T: DeserializeOwned>(raw: &str) -> Result<T> {
    serde_json::from_str(raw).map_err(|e| {
        CognisError::Serialization(format!(
            "element: {}: {}",
            preview(&e.to_string()),
            preview(raw)
        ))
    })
}

/// `text` cut to [`PREVIEW_CHARS`] characters, with an ellipsis when cut.
/// Keeps error messages bounded: model output can be arbitrarily long, and
/// serde errors quote the offending value verbatim.
fn preview(text: &str) -> String {
    let mut chars = text.chars();
    let mut out: String = chars.by_ref().take(PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

/// The first [`PREVIEW_CHARS`] characters of the raw model output, plus
/// whether anything other than whitespace arrived — what an end-of-stream
/// "no JSON found" error needs, without retaining the whole response.
#[derive(Default)]
struct OutputHead {
    text: String,
    kept: usize,
    truncated: bool,
    has_content: bool,
}

impl OutputHead {
    fn push(&mut self, fragment: &str) {
        for ch in fragment.chars() {
            let blank = ch.is_whitespace();
            self.has_content |= !blank;
            // Skip leading whitespace so the preview starts at the content.
            if self.kept == 0 && blank {
                continue;
            }
            if self.kept < PREVIEW_CHARS {
                self.text.push(ch);
                self.kept += 1;
            } else {
                self.truncated = true;
                if self.has_content {
                    return;
                }
            }
        }
    }

    fn preview(&self) -> String {
        if self.truncated {
            format!("{}…", self.text)
        } else {
            self.text.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chat::{ChatResponse, HealthStatus, StreamChunk};
    use crate::provider::{LLMProvider, Provider};
    use crate::{ChatOptions, Client, ToolDefinition};
    use async_trait::async_trait;
    use cognis_core::{CognisError, Message, Result, RunnableStream};
    use futures::StreamExt;
    use std::sync::Arc;

    #[derive(cognis_macros::Partial)]
    #[allow(dead_code)]
    struct Report {
        title: String,
        score: u32,
    }

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Step {
        id: u32,
    }

    #[derive(serde::Deserialize, Debug, PartialEq)]
    #[serde(rename_all = "lowercase")]
    enum Status {
        Active,
        Done,
    }

    #[derive(cognis_macros::Partial)]
    #[allow(dead_code)]
    struct Ticket {
        title: String,
        status: Status,
    }

    /// One scripted stream item: a text delta or a provider failure.
    #[derive(Clone, Copy)]
    enum Piece {
        Text(&'static str),
        Fail,
    }

    /// Streams the script over the plain (tool-less) streaming path. Every
    /// other entry point panics, so a typed stream that reaches for the
    /// tool-calling path — and its non-streaming fallback — fails the test.
    struct ArrProvider {
        script: Vec<Piece>,
    }

    impl ArrProvider {
        fn new(pieces: &[&'static str]) -> Self {
            Self {
                script: pieces.iter().map(|s| Piece::Text(s)).collect(),
            }
        }

        fn scripted(script: &[Piece]) -> Self {
            Self {
                script: script.to_vec(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for ArrProvider {
        fn name(&self) -> &str {
            "arr"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
            unreachable!("typed streams must not take the non-streaming path")
        }
        async fn chat_completion_stream(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            let chunks: Vec<Result<StreamChunk>> = self
                .script
                .iter()
                .map(|piece| match piece {
                    Piece::Text(s) => Ok(StreamChunk {
                        content: (*s).into(),
                        is_delta: true,
                        is_done: false,
                        finish_reason: None,
                        usage: None,
                        tool_calls_delta: vec![],
                    }),
                    Piece::Fail => Err(CognisError::Internal("boom".into())),
                })
                .collect();
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn chat_completion_stream_with_tools(
            &self,
            _m: Vec<Message>,
            _t: Vec<ToolDefinition>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unreachable!("typed streams involve no tools; use the plain streaming path")
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn serialization_message(item: &Result<impl std::fmt::Debug>) -> &str {
        match item {
            Err(CognisError::Serialization(m)) => m,
            other => panic!("expected Serialization error, got: {other:?}"),
        }
    }

    fn client(p: ArrProvider) -> Client {
        Client::new(Arc::new(p))
    }

    #[tokio::test]
    async fn stream_array_yields_typed_elements() {
        let c = client(ArrProvider::new(&["[{\"id\":1},", "{\"id\":2}]"]));
        let mut s = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap();
        let a = s.next().await.unwrap().unwrap();
        let b = s.next().await.unwrap().unwrap();
        assert_eq!(a, Step { id: 1 });
        assert_eq!(b, Step { id: 2 });
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_array_reassembles_element_split_across_chunks() {
        let c = client(ArrProvider::new(&["[{\"i", "d\":", "7}", ",{\"id\":8}]"]));
        let got: Vec<Step> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(got, vec![Step { id: 7 }, Step { id: 8 }]);
    }

    #[tokio::test]
    async fn stream_array_yields_err_for_bad_element_and_continues() {
        let c = client(ArrProvider::new(&[
            "[{\"id\":1},{\"id\":\"x\"},{\"id\":3}]",
        ]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 3, "got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &Step { id: 1 });
        assert!(
            matches!(got[1], Err(CognisError::Serialization(_))),
            "got: {:?}",
            got[1]
        );
        assert_eq!(got[2].as_ref().unwrap(), &Step { id: 3 });
    }

    #[tokio::test]
    async fn stream_array_yields_nothing_for_empty_array() {
        let c = client(ArrProvider::new(&["[]"]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert!(got.is_empty(), "got: {got:?}");
    }

    #[tokio::test]
    async fn stream_array_propagates_provider_error_and_stops() {
        let p = ArrProvider::scripted(&[
            Piece::Text("[{\"id\":1},"),
            Piece::Fail,
            Piece::Text("{\"id\":2}]"),
        ]);
        let got: Vec<Result<Step>> = client(p)
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "stream must end at the error, got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &Step { id: 1 });
        assert!(
            matches!(got[1], Err(CognisError::Internal(_))),
            "got: {:?}",
            got[1]
        );
    }

    #[tokio::test]
    async fn stream_array_ends_with_error_when_response_has_no_array() {
        let c = client(ArrProvider::new(&["Sorry, I can't ", "produce a plan."]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 1, "got: {got:?}");
        let msg = serialization_message(&got[0]);
        assert!(msg.contains("no JSON array"), "got: {msg}");
        assert!(msg.contains("Sorry, I can't produce a plan."), "got: {msg}");
    }

    #[tokio::test]
    async fn stream_array_yields_nothing_when_response_is_blank() {
        let c = client(ArrProvider::new(&["", " \n"]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert!(got.is_empty(), "got: {got:?}");
    }

    #[tokio::test]
    async fn stream_array_yields_ok_then_error_when_final_element_is_truncated() {
        let c = client(ArrProvider::new(&["[{\"id\":1},{\"id\":"]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &Step { id: 1 });
        serialization_message(&got[1]);
    }

    #[tokio::test]
    async fn stream_array_ignores_code_fence_around_array() {
        let c = client(ArrProvider::new(&[
            "```json\n[{\"id\":1},",
            "{\"id\":2}]",
            "\n```\n",
        ]));
        let got: Vec<Result<Step>> = c
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &Step { id: 1 });
        assert_eq!(got[1].as_ref().unwrap(), &Step { id: 2 });
    }

    #[tokio::test]
    async fn stream_array_error_message_truncates_long_raw_payload() {
        let long = "x".repeat(5_000);
        let piece: &'static str = Box::leak(format!("[\"{long}\"]").into_boxed_str());
        let got: Vec<Result<Step>> = client(ArrProvider::new(&[piece]))
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 1, "got: {got:?}");
        let msg = serialization_message(&got[0]);
        assert!(
            msg.len() < 500,
            "message must be bounded, len = {}",
            msg.len()
        );

        let prose: &'static str = Box::leak("no array here ".repeat(500).into_boxed_str());
        let got: Vec<Result<Step>> = client(ArrProvider::new(&[prose]))
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        let msg = serialization_message(&got[0]);
        assert!(
            msg.len() < 500,
            "message must be bounded, len = {}",
            msg.len()
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_fills_snapshots_progressively() {
        // A number is withheld until a delimiter proves it is complete: `1`
        // only shows once the `,` arrives, `2` once the `}` does.
        let c = client(ArrProvider::new(&["{\"a\":1", ",\"b\":2", "}"]));
        let got: Vec<serde_json::Value> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(
            got,
            vec![
                serde_json::json!({}),
                serde_json::json!({"a": 1}),
                serde_json::json!({"a": 1, "b": 2})
            ],
            "got: {got:?}"
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_grows_string_value_across_chunks() {
        let c = client(ArrProvider::new(&["{\"msg\":\"he", "llo wor", "ld\"}"]));
        let got: Vec<serde_json::Value> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(
            got,
            vec![
                serde_json::json!({"msg": "he"}),
                serde_json::json!({"msg": "hello wor"}),
                serde_json::json!({"msg": "hello world"}),
            ],
            "got: {got:?}"
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_skips_duplicate_snapshots() {
        // The second and third chunks add no new parseable content.
        let c = client(ArrProvider::new(&["{\"a\":1,", " ", "\"b\":", "2}"]));
        let got: Vec<serde_json::Value> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(
            got,
            vec![
                serde_json::json!({"a": 1}),
                serde_json::json!({"a": 1, "b": 2})
            ],
            "got: {got:?}"
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_ends_with_last_good_snapshot_when_never_closed() {
        let c = client(ArrProvider::new(&["{\"a\":1,", "\"b\":\"par"]));
        let got: Vec<serde_json::Value> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(
            got.last(),
            Some(&serde_json::json!({"a": 1, "b": "par"})),
            "got: {got:?}"
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_ends_with_error_when_response_has_no_object() {
        let c = client(ArrProvider::new(&["no json ", "here"]));
        let got: Vec<Result<serde_json::Value>> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 1, "got: {got:?}");
        let msg = serialization_message(&got[0]);
        assert!(msg.contains("no JSON object"), "got: {msg}");
        assert!(msg.contains("no json here"), "got: {msg}");
    }

    #[tokio::test]
    async fn stream_object_partial_value_yields_nothing_when_response_is_blank() {
        let c = client(ArrProvider::new(&[" ", "\n"]));
        let got: Vec<Result<serde_json::Value>> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert!(got.is_empty(), "got: {got:?}");
    }

    #[tokio::test]
    async fn stream_object_partial_value_keeps_final_snapshot_when_code_fence_follows() {
        let c = client(ArrProvider::new(&[
            "```json\n{\"title\":\"Q3\",",
            "\"score\":92}",
            "\n```\n",
        ]));
        let got: Vec<serde_json::Value> = c
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(
            got,
            vec![
                serde_json::json!({"title": "Q3"}),
                serde_json::json!({"title": "Q3", "score": 92}),
            ],
            "got: {got:?}"
        );
    }

    #[tokio::test]
    async fn stream_object_partial_value_propagates_provider_error_and_stops() {
        let p = ArrProvider::scripted(&[
            Piece::Text("{\"a\":1,"),
            Piece::Fail,
            Piece::Text("\"b\":2}"),
        ]);
        let got: Vec<Result<serde_json::Value>> = client(p)
            .stream_object_partial_value(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "stream must end at the error, got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &serde_json::json!({"a": 1}));
        assert!(
            matches!(got[1], Err(CognisError::Internal(_))),
            "got: {:?}",
            got[1]
        );
    }

    #[tokio::test]
    async fn stream_object_partial_fills_typed_mirror_progressively() {
        let c = client(ArrProvider::new(&["{\"title\":\"x\"", ",\"score\":5}"]));
        let got: Vec<ReportPartial> = c
            .stream_object_partial::<Report>(vec![Message::human("obj")])
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(got.len(), 2, "got: {got:?}");
        assert_eq!(got[0].title.as_deref(), Some("x"));
        assert_eq!(got[0].score, None);
        let last = got.last().unwrap();
        assert_eq!(last.title.as_deref(), Some("x"));
        assert_eq!(last.score, Some(5));
    }

    #[tokio::test]
    async fn stream_object_partial_yields_fitting_snapshot_after_skipping_non_fitting_one() {
        // `"act` is not a `Status` variant yet, so that snapshot cannot be
        // decoded and is skipped; once `"active"` completes, the snapshot fits.
        let c = client(ArrProvider::new(&[
            "{\"title\":\"x\",\"status\":\"act",
            "ive\"}",
        ]));
        let got: Vec<Result<TicketPartial>> = c
            .stream_object_partial::<Ticket>(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 1, "got: {got:?}");
        let ticket = got[0].as_ref().unwrap();
        assert_eq!(ticket.title.as_deref(), Some("x"));
        assert_eq!(ticket.status, Some(Status::Active));
    }

    #[tokio::test]
    async fn stream_object_partial_ends_with_error_when_final_object_has_wrong_typed_field() {
        // A string `score` never fits `Option<u32>`: the earlier snapshot
        // (title only) is delivered, then the stream reports the failure
        // instead of ending as if the object had been complete.
        let c = client(ArrProvider::new(&[
            "{\"title\":\"a\",",
            "\"score\":\"hi\"}",
        ]));
        let got: Vec<Result<ReportPartial>> = c
            .stream_object_partial::<Report>(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "got: {got:?}");
        let first = got[0].as_ref().unwrap();
        assert_eq!(first.title.as_deref(), Some("a"));
        assert_eq!(first.score, None);
        let msg = serialization_message(&got[1]);
        assert!(msg.contains("ReportPartial"), "got: {msg}");
    }

    #[tokio::test]
    async fn stream_object_partial_ends_with_error_when_response_has_no_object() {
        let c = client(ArrProvider::new(&["I'd rather not."]));
        let got: Vec<Result<ReportPartial>> = c
            .stream_object_partial::<Report>(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 1, "got: {got:?}");
        let msg = serialization_message(&got[0]);
        assert!(msg.contains("no JSON object"), "got: {msg}");
    }

    #[tokio::test]
    async fn stream_object_partial_propagates_provider_error_and_stops() {
        let p = ArrProvider::scripted(&[
            Piece::Text("{\"title\":\"x\""),
            Piece::Fail,
            Piece::Text(",\"score\":5}"),
        ]);
        let got: Vec<Result<ReportPartial>> = client(p)
            .stream_object_partial::<Report>(vec![Message::human("obj")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "stream must end at the error, got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap().title.as_deref(), Some("x"));
        assert!(
            matches!(got[1], Err(CognisError::Internal(_))),
            "got: {:?}",
            got[1]
        );
    }
}
