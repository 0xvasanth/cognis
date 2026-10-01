//! Typed streaming of structured list output.
//!
//! [`Client::stream_array`] turns a streamed JSON array into a
//! [`RunnableStream`] of validated `T` values, yielding each element as soon
//! as it closes. Processing is ordinary `futures::StreamExt` iteration.

use serde::de::DeserializeOwned;

use cognis_core::{CognisError, Message, Result, RunnableStream, StreamingJsonArray};
use futures::StreamExt;

use crate::Client;

impl Client {
    /// Stream a JSON array as typed elements, each delivered when it closes.
    ///
    /// Different from [`Client::stream_with_tools`], which yields raw text
    /// chunks: this parses the array incrementally and yields validated `T`
    /// values. A malformed element yields one `Err` item for that element and
    /// the stream continues; a provider error is yielded and ends the stream.
    pub async fn stream_array<T: DeserializeOwned + Send + 'static>(
        &self,
        messages: Vec<Message>,
    ) -> Result<RunnableStream<T>> {
        let chunks = self.stream_with_tools(messages, Vec::new()).await?;
        let mut parser = StreamingJsonArray::new();
        let out = async_stream::stream! {
            let mut chunks = chunks;
            while let Some(item) = chunks.next().await {
                match item {
                    Ok(chunk) => {
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
        };
        Ok(RunnableStream::new(out))
    }
}

fn decode<T: DeserializeOwned>(raw: &str) -> Result<T> {
    serde_json::from_str(raw)
        .map_err(|e| CognisError::Serialization(format!("element: {e}: {raw}")))
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

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Step {
        id: u32,
    }

    /// Streams the scripted text pieces as delta chunks; optionally ends in an error.
    struct ArrProvider {
        pieces: Vec<&'static str>,
        fail_after: bool,
    }

    impl ArrProvider {
        fn new(pieces: &[&'static str]) -> Self {
            Self {
                pieces: pieces.to_vec(),
                fail_after: false,
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
            unreachable!()
        }
        async fn chat_completion_stream(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unreachable!()
        }
        async fn chat_completion_stream_with_tools(
            &self,
            _m: Vec<Message>,
            _t: Vec<ToolDefinition>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            let mut chunks: Vec<Result<StreamChunk>> = self
                .pieces
                .iter()
                .map(|s| {
                    Ok(StreamChunk {
                        content: (*s).into(),
                        is_delta: true,
                        is_done: false,
                        finish_reason: None,
                        usage: None,
                        tool_calls_delta: vec![],
                    })
                })
                .collect();
            if self.fail_after {
                chunks.push(Err(CognisError::Internal("boom".into())));
            }
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
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
        let mut p = ArrProvider::new(&["[{\"id\":1},"]);
        p.fail_after = true;
        let got: Vec<Result<Step>> = client(p)
            .stream_array::<Step>(vec![Message::human("plan")])
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(got.len(), 2, "got: {got:?}");
        assert_eq!(got[0].as_ref().unwrap(), &Step { id: 1 });
        assert!(got[1].is_err());
    }
}
