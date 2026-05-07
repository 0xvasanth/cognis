//! Stream-chunk aggregation utilities.
//!
//! Streaming chat surfaces a sequence of [`StreamChunk`]s; agents and
//! callers often want the *complete* result reconstructed at the end.
//! [`StreamAggregator`] owns the accumulation: text deltas concatenate,
//! tool-call deltas merge by index, the final usage / finish_reason are
//! captured.

use std::collections::HashMap;

use serde_json::Value;

use cognis_core::{AiMessage, Message, ToolCall};

use crate::chat::{StreamChunk, ToolCallDelta, Usage};

/// Accumulates a streaming response into a final [`Message`] plus
/// aggregated metadata. Chunk ordering matters for text content and for
/// tool-call argument deltas — feed chunks in the order they arrive.
#[derive(Debug, Default, Clone)]
pub struct StreamAggregator {
    /// Concatenated text content.
    content: String,
    /// Per-tool-call accumulators, keyed by chunk `index`.
    tool_calls: HashMap<u32, ToolCallAccumulator>,
    /// Reason the stream terminated (last `is_done` chunk's value).
    finish_reason: Option<String>,
    /// Final usage stats (last reported).
    usage: Option<Usage>,
}

#[derive(Debug, Default, Clone)]
struct ToolCallAccumulator {
    /// First-seen `id` for this index — providers send it once.
    id: Option<String>,
    /// First-seen `name` — providers send it once.
    name: Option<String>,
    /// Concatenated argument fragments (typically a JSON-encoded string).
    arguments_raw: String,
}

impl StreamAggregator {
    /// Construct an empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a single chunk. Cheap; safe to call inline as chunks arrive.
    pub fn push(&mut self, chunk: StreamChunk) {
        if !chunk.content.is_empty() {
            self.content.push_str(&chunk.content);
        }
        for d in chunk.tool_calls_delta {
            self.merge_tool_delta(d);
        }
        if chunk.is_done {
            if chunk.finish_reason.is_some() {
                self.finish_reason = chunk.finish_reason;
            }
            if chunk.usage.is_some() {
                self.usage = chunk.usage;
            }
        }
    }

    /// Drain the aggregator into a finalized assistant message + metadata.
    pub fn finalize(self) -> Aggregated {
        let mut tool_calls = Vec::with_capacity(self.tool_calls.len());
        // Stable order: by index ascending.
        let mut keyed: Vec<(u32, ToolCallAccumulator)> = self.tool_calls.into_iter().collect();
        keyed.sort_by_key(|(i, _)| *i);
        for (_, acc) in keyed {
            let id = acc.id.unwrap_or_default();
            let name = acc.name.unwrap_or_default();
            let arguments: Value = if acc.arguments_raw.is_empty() {
                Value::Null
            } else {
                serde_json::from_str(&acc.arguments_raw).unwrap_or(Value::String(acc.arguments_raw))
            };
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
            });
        }
        Aggregated {
            message: Message::Ai(AiMessage {
                content: self.content,
                tool_calls,
                parts: Vec::new(),
            }),
            finish_reason: self.finish_reason,
            usage: self.usage,
        }
    }

    fn merge_tool_delta(&mut self, d: ToolCallDelta) {
        let entry = self.tool_calls.entry(d.index).or_default();
        if entry.id.is_none() {
            entry.id = d.id;
        }
        if entry.name.is_none() {
            entry.name = d.name;
        }
        if let Some(frag) = d.arguments_delta {
            entry.arguments_raw.push_str(&frag);
        }
    }
}

/// Output of [`StreamAggregator::finalize`].
#[derive(Debug, Clone)]
pub struct Aggregated {
    /// The reconstructed assistant message.
    pub message: Message,
    /// Reason the stream stopped, if any.
    pub finish_reason: Option<String>,
    /// Final usage stats, if reported.
    pub usage: Option<Usage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> StreamChunk {
        StreamChunk {
            content: s.into(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: Vec::new(),
        }
    }

    fn done(reason: &str) -> StreamChunk {
        StreamChunk {
            content: String::new(),
            is_delta: false,
            is_done: true,
            finish_reason: Some(reason.into()),
            usage: Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
            tool_calls_delta: Vec::new(),
        }
    }

    #[test]
    fn concatenates_text_chunks() {
        let mut a = StreamAggregator::new();
        a.push(text("hel"));
        a.push(text("lo "));
        a.push(text("world"));
        a.push(done("stop"));
        let out = a.finalize();
        assert_eq!(out.message.content(), "hello world");
        assert_eq!(out.finish_reason.as_deref(), Some("stop"));
        assert_eq!(out.usage.unwrap().total_tokens, 12);
    }

    #[test]
    fn merges_tool_call_deltas_by_index() {
        let mut a = StreamAggregator::new();
        a.push(StreamChunk {
            content: String::new(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: vec![ToolCallDelta {
                index: 0,
                id: Some("c1".into()),
                name: Some("search".into()),
                arguments_delta: Some(r#"{"q":"#.into()),
            }],
        });
        a.push(StreamChunk {
            content: String::new(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: vec![ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: Some(r#""rust"}"#.into()),
            }],
        });
        a.push(done("tool_calls"));
        let out = a.finalize();
        assert_eq!(out.message.tool_calls().len(), 1);
        let tc = &out.message.tool_calls()[0];
        assert_eq!(tc.id, "c1");
        assert_eq!(tc.name, "search");
        assert_eq!(tc.arguments["q"], "rust");
    }

    #[test]
    fn multiple_tool_calls_kept_in_order_by_index() {
        let mut a = StreamAggregator::new();
        a.push(StreamChunk {
            content: String::new(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: vec![
                ToolCallDelta {
                    index: 1,
                    id: Some("c2".into()),
                    name: Some("b_tool".into()),
                    arguments_delta: Some("{}".into()),
                },
                ToolCallDelta {
                    index: 0,
                    id: Some("c1".into()),
                    name: Some("a_tool".into()),
                    arguments_delta: Some("{}".into()),
                },
            ],
        });
        a.push(done("tool_calls"));
        let out = a.finalize();
        let calls = out.message.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a_tool");
        assert_eq!(calls[1].name, "b_tool");
    }
}
