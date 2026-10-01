//! Stateful server-sent-events decoder for OpenAI-compatible chat streams.
//!
//! HTTP transport reads are not event-framed: one read may carry several
//! events, and one event (or one UTF-8 codepoint) may straddle two reads.
//! [`SseDecoder`] owns the byte buffer across reads so OpenAI and Azure share
//! one framing implementation; reach for [`decode_stream`] to adapt a byte
//! stream into a [`RunnableStream`] of [`StreamChunk`]s.

use std::fmt::Display;

use futures::{Stream, StreamExt};

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{StreamChunk, ToolCallDelta};

const DONE_SENTINEL: &str = "[DONE]";

/// Incremental decoder: feed it transport reads, get back the
/// [`StreamChunk`]s for every `data:` line completed by that read.
#[derive(Debug)]
pub(crate) struct SseDecoder {
    provider: &'static str,
    /// Bytes of the current, not-yet-terminated line. Kept as raw bytes
    /// because a read boundary may fall inside a multibyte codepoint.
    buf: Vec<u8>,
    done: bool,
}

impl SseDecoder {
    /// New decoder; `provider` labels decode errors.
    pub(crate) fn new(provider: &'static str) -> Self {
        Self {
            provider,
            buf: Vec::new(),
            done: false,
        }
    }

    /// True once `[DONE]` was seen or a line failed to decode. Further
    /// input is ignored.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// Consume one transport read and return the chunks of every line it
    /// completed, in order. A line that fails to decode yields one `Err`
    /// and ends the stream.
    pub(crate) fn feed(&mut self, bytes: &[u8]) -> Vec<Result<StreamChunk>> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        self.buf.extend_from_slice(bytes);
        let mut start = 0;
        while let Some(len) = self.buf[start..].iter().position(|b| *b == b'\n') {
            let line_end = start + len;
            let decoded = decode_line(self.provider, &self.buf[start..line_end]);
            start = line_end + 1;
            if self.record(decoded, &mut out) {
                break;
            }
        }
        if self.done {
            self.buf.clear();
        } else {
            self.buf.drain(..start);
        }
        out
    }

    /// Flush at end of transport: a final line the server did not terminate
    /// with a newline is decoded as if it had been, so a stream cut off
    /// mid-event surfaces as an error instead of silently losing its tail.
    pub(crate) fn finish(&mut self) -> Vec<Result<StreamChunk>> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        let rest = std::mem::take(&mut self.buf);
        let decoded = decode_line(self.provider, &rest);
        self.record(decoded, &mut out);
        self.done = true;
        out
    }

    /// Push a decoded line's outcome; returns true when the stream is over.
    fn record(&mut self, decoded: Result<Line>, out: &mut Vec<Result<StreamChunk>>) -> bool {
        match decoded {
            Ok(Line::Chunk(chunk)) => out.push(Ok(chunk)),
            Ok(Line::Ignored) => {}
            Ok(Line::Done) => self.done = true,
            Err(e) => {
                out.push(Err(e));
                self.done = true;
            }
        }
        self.done
    }
}

/// What one complete SSE line contributes to the stream.
enum Line {
    Chunk(StreamChunk),
    Done,
    /// Blank separators, `:` comments, and fields other than `data`.
    Ignored,
}

fn decode_line(provider: &'static str, line: &[u8]) -> Result<Line> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let Some(payload) = line.strip_prefix(b"data:") else {
        return Ok(Line::Ignored);
    };
    let payload = std::str::from_utf8(payload)
        .map_err(|e| CognisError::Provider {
            provider: provider.into(),
            message: format!("invalid UTF-8 in stream: {e}"),
        })?
        .trim();
    if payload.is_empty() {
        return Ok(Line::Ignored);
    }
    if payload == DONE_SENTINEL {
        return Ok(Line::Done);
    }
    parse_payload(provider, payload).map(Line::Chunk)
}

/// Adapt a transport byte stream into decoded [`StreamChunk`]s. The stream
/// ends at `[DONE]`, at the first error, or when the transport ends.
pub(crate) fn decode_stream<S, B, E>(
    provider: &'static str,
    bytes: S,
) -> RunnableStream<StreamChunk>
where
    S: Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send,
    E: Display + Send,
{
    RunnableStream::new(async_stream::stream! {
        let mut decoder = SseDecoder::new(provider);
        futures::pin_mut!(bytes);
        while let Some(read) = bytes.next().await {
            match read {
                Ok(b) => {
                    for item in decoder.feed(b.as_ref()) {
                        yield item;
                    }
                    if decoder.is_done() {
                        return;
                    }
                }
                Err(e) => {
                    yield Err(CognisError::Network {
                        status_code: None,
                        message: e.to_string(),
                    });
                    return;
                }
            }
        }
        for item in decoder.finish() {
            yield item;
        }
    })
}

/// Map one `data:` JSON payload to a [`StreamChunk`].
fn parse_payload(provider: &'static str, payload: &str) -> Result<StreamChunk> {
    let v: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| CognisError::Provider {
            provider: provider.into(),
            message: format!("stream parse: {e}"),
        })?;
    let delta = &v["choices"][0]["delta"];
    let content = delta["content"].as_str().unwrap_or("").to_string();
    let mut tool_calls_delta = Vec::new();
    if let Some(arr) = delta["tool_calls"].as_array() {
        for (i, t) in arr.iter().enumerate() {
            tool_calls_delta.push(ToolCallDelta {
                index: t["index"].as_u64().unwrap_or(i as u64) as u32,
                id: t["id"].as_str().map(|s| s.to_string()),
                name: t["function"]["name"].as_str().map(|s| s.to_string()),
                arguments_delta: t["function"]["arguments"].as_str().map(|s| s.to_string()),
            });
        }
    }
    let finish_reason = v["choices"][0]["finish_reason"]
        .as_str()
        .map(|s| s.to_string());
    let is_done = finish_reason.is_some();
    Ok(StreamChunk {
        content,
        is_delta: true,
        is_done,
        finish_reason,
        usage: None,
        tool_calls_delta,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::streaming::StreamAggregator;

    fn content_event(text: &str) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({"choices":[{"delta":{"content": text}}]})
        )
    }

    fn args_event(id: Option<&str>, name: Option<&str>, fragment: &str) -> String {
        let mut call = serde_json::json!({"index": 0, "function": {"arguments": fragment}});
        if let Some(id) = id {
            call["id"] = serde_json::json!(id);
        }
        if let Some(name) = name {
            call["function"]["name"] = serde_json::json!(name);
        }
        format!(
            "data: {}\n\n",
            serde_json::json!({"choices":[{"delta":{"tool_calls":[call]}}]})
        )
    }

    const FINISH_TOOL_CALLS: &str =
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";

    /// One-shot loopback HTTP server standing in for a provider endpoint:
    /// answers the first request with `writes` as an SSE body, one socket
    /// write per element, and hands back the JSON body it was sent. Lets
    /// provider tests exercise the real transport without a live API.
    pub(crate) async fn serve_sse_once(
        writes: Vec<Vec<u8>>,
    ) -> (String, tokio::task::JoinHandle<serde_json::Value>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let base = format!("http://{}", listener.local_addr().expect("local addr"));
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.expect("accept");
            let mut req = Vec::new();
            let mut tmp = [0u8; 4096];
            let (head_end, content_length) = loop {
                let n = sock.read(&mut tmp).await.expect("read request");
                assert!(n > 0, "client closed before sending a full request");
                req.extend_from_slice(&tmp[..n]);
                if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&req[..pos]).to_ascii_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .map(|v| v.trim().parse::<usize>().expect("content-length"))
                        .unwrap_or(0);
                    break (pos + 4, len);
                }
            };
            while req.len() < head_end + content_length {
                let n = sock.read(&mut tmp).await.expect("read body");
                assert!(n > 0, "client closed mid-body");
                req.extend_from_slice(&tmp[..n]);
            }
            let body = serde_json::from_slice(&req[head_end..head_end + content_length])
                .expect("request body is JSON");

            sock.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
            )
            .await
            .expect("write head");
            for w in writes {
                sock.write_all(&w).await.expect("write sse");
                sock.flush().await.expect("flush");
                // Give the client a chance to observe this write on its own.
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            sock.shutdown().await.expect("shutdown");
            body
        });
        (base, handle)
    }

    /// SSE wire bytes for a `weather({"city":"San Francisco"})` tool call
    /// whose arguments arrive in three fragments, cut into socket writes
    /// that coalesce two events and split the third mid-JSON.
    pub(crate) fn tool_call_writes() -> Vec<Vec<u8>> {
        let e1 = args_event(Some("call_1"), Some("weather"), "{\"city\":\"San");
        let e2 = args_event(None, None, " Fran");
        let e3 = args_event(None, None, "cisco\"}");
        let wire = format!("{e1}{e2}{e3}{FINISH_TOOL_CALLS}data: [DONE]\n\n");
        let bytes = wire.as_bytes();
        let cut = e1.len() + e2.len() + 30;
        vec![bytes[..cut].to_vec(), bytes[cut..].to_vec()]
    }

    fn feed_ok(d: &mut SseDecoder, bytes: &[u8]) -> Vec<StreamChunk> {
        d.feed(bytes)
            .into_iter()
            .map(|r| r.expect("chunk decodes"))
            .collect()
    }

    #[test]
    fn feed_emits_both_chunks_when_two_events_share_one_read() {
        let mut d = SseDecoder::new("openai");
        let read = format!("{}{}", content_event("Hel"), content_event("lo"));
        let chunks = feed_ok(&mut d, read.as_bytes());
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(texts, vec!["Hel", "lo"]);
    }

    #[test]
    fn feed_buffers_event_when_split_mid_json_across_reads() {
        let mut d = SseDecoder::new("openai");
        let event = content_event("hello");
        let (a, b) = event.as_bytes().split_at(25);
        assert!(
            feed_ok(&mut d, a).is_empty(),
            "an incomplete line must not be emitted or rejected"
        );
        let chunks = feed_ok(&mut d, b);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello");
    }

    #[test]
    fn feed_reassembles_multibyte_codepoint_when_split_across_reads() {
        let mut d = SseDecoder::new("openai");
        let event = content_event("caf\u{e9} \u{1f980}");
        let bytes = event.as_bytes();
        // Cut inside the 4-byte crab: one byte of it in the first read.
        let crab = event.find('\u{1f980}').expect("crab present");
        let (a, b) = bytes.split_at(crab + 1);
        assert!(
            std::str::from_utf8(a).is_err(),
            "cut must split a codepoint"
        );
        assert!(feed_ok(&mut d, a).is_empty());
        let chunks = feed_ok(&mut d, b);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "caf\u{e9} \u{1f980}");
    }

    #[test]
    fn feed_parses_payload_when_data_prefix_has_no_space() {
        let mut d = SseDecoder::new("openai");
        let chunks = feed_ok(
            &mut d,
            b"data:{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hi");
    }

    #[test]
    fn feed_emits_finish_chunk_then_stops_when_done_follows_in_same_read() {
        let mut d = SseDecoder::new("openai");
        let read = format!(
            "{FINISH_TOOL_CALLS}data: [DONE]\n\n{}",
            content_event("late")
        );
        let chunks = feed_ok(&mut d, read.as_bytes());
        assert_eq!(chunks.len(), 1, "got: {chunks:?}");
        assert!(chunks[0].is_done);
        assert_eq!(chunks[0].finish_reason.as_deref(), Some("tool_calls"));
        assert!(d.is_done());
        assert!(
            d.feed(content_event("later").as_bytes()).is_empty(),
            "nothing is decoded after [DONE]"
        );
    }

    #[test]
    fn feed_tolerates_crlf_line_endings_and_ignores_non_data_lines() {
        let mut d = SseDecoder::new("openai");
        let read = b": keep-alive\r\nevent: message\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n";
        let chunks = feed_ok(&mut d, read);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "a");
        assert!(d.is_done());
    }

    #[test]
    fn feed_yields_error_and_stops_when_a_complete_line_is_not_json() {
        let mut d = SseDecoder::new("azure");
        let read = format!(
            "{}data: {{not json\n\n{}",
            content_event("a"),
            content_event("b")
        );
        let out = d.feed(read.as_bytes());
        assert_eq!(out.len(), 2, "got: {out:?}");
        assert_eq!(out[0].as_ref().expect("first chunk").content, "a");
        match &out[1] {
            Err(CognisError::Provider { provider, message }) => {
                assert_eq!(provider, "azure");
                assert!(message.contains("stream parse"), "got: {message}");
            }
            other => panic!("expected Provider error, got: {other:?}"),
        }
        assert!(d.is_done());
    }

    #[test]
    fn finish_decodes_final_line_when_server_omits_trailing_newline() {
        let mut d = SseDecoder::new("openai");
        let event = content_event("tail");
        let unterminated = event.trim_end_matches('\n');
        assert!(feed_ok(&mut d, unterminated.as_bytes()).is_empty());
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().expect("tail chunk").content, "tail");
    }

    #[test]
    fn finish_reports_error_when_transport_ends_mid_event() {
        let mut d = SseDecoder::new("openai");
        let event = content_event("truncated");
        assert!(feed_ok(&mut d, &event.as_bytes()[..20]).is_empty());
        let out = d.finish();
        assert_eq!(out.len(), 1);
        assert!(
            matches!(out[0], Err(CognisError::Provider { .. })),
            "got: {out:?}"
        );
    }

    /// Feed `reads` through a fresh decoder + aggregator and return the
    /// single assembled tool call's arguments.
    fn aggregate_arguments(reads: &[&[u8]]) -> serde_json::Value {
        let mut d = SseDecoder::new("openai");
        let mut agg = StreamAggregator::new();
        for read in reads {
            for chunk in feed_ok(&mut d, read) {
                agg.push(chunk);
            }
        }
        let out = agg.finalize();
        let calls = out.message.tool_calls();
        assert_eq!(calls.len(), 1, "got: {calls:?}");
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "weather");
        assert_eq!(out.finish_reason.as_deref(), Some("tool_calls"));
        calls[0].arguments.clone()
    }

    #[test]
    fn feed_aggregates_exact_tool_arguments_when_fragments_are_coalesced_and_split() {
        let e1 = args_event(Some("call_1"), Some("weather"), "{\"city\":\"San");
        let e2 = args_event(None, None, " Fran");
        let e3 = args_event(None, None, "cisco\"}");
        let wire = format!("{e1}{e2}{e3}{FINISH_TOOL_CALLS}data: [DONE]\n\n");
        let bytes = wire.as_bytes();
        let expected = serde_json::json!({"city": "San Francisco"});

        // Everything in one read.
        assert_eq!(aggregate_arguments(&[bytes]), expected);

        // First two events coalesced, third split mid-JSON.
        let cut = e1.len() + e2.len() + 30;
        assert_eq!(
            aggregate_arguments(&[&bytes[..cut], &bytes[cut..]]),
            expected
        );

        // Every possible single cut point.
        for cut in 1..bytes.len() {
            assert_eq!(
                aggregate_arguments(&[&bytes[..cut], &bytes[cut..]]),
                expected,
                "cut at byte {cut}"
            );
        }

        // One byte per read.
        let singles: Vec<&[u8]> = bytes.chunks(1).collect();
        assert_eq!(aggregate_arguments(&singles), expected);

        // Uneven reads that straddle event boundaries.
        let sevens: Vec<&[u8]> = bytes.chunks(7).collect();
        assert_eq!(aggregate_arguments(&sevens), expected);
    }

    #[tokio::test]
    async fn decode_stream_ends_at_done_and_surfaces_transport_errors() {
        let reads: Vec<std::result::Result<Vec<u8>, String>> = vec![
            Ok(format!("{}{}", content_event("a"), content_event("b")).into_bytes()),
            Ok(b"data: [DONE]\n\n".to_vec()),
            Ok(content_event("after-done").into_bytes()),
        ];
        let got = decode_stream("openai", futures::stream::iter(reads))
            .collect_into_vec()
            .await
            .expect("no error");
        let texts: Vec<&str> = got.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(texts, vec!["a", "b"]);

        let reads: Vec<std::result::Result<Vec<u8>, String>> = vec![
            Ok(content_event("a").into_bytes()),
            Err("connection reset".into()),
            Ok(content_event("never").into_bytes()),
        ];
        let mut s = decode_stream("openai", futures::stream::iter(reads));
        assert_eq!(s.next().await.expect("first").expect("ok").content, "a");
        match s.next().await {
            Some(Err(CognisError::Network { message, .. })) => {
                assert!(message.contains("connection reset"), "got: {message}")
            }
            other => panic!("expected Network error, got: {other:?}"),
        }
        assert!(s.next().await.is_none(), "stream ends at the error");
    }
}
