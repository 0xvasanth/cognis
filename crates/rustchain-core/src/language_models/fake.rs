use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;

use crate::error::{Result, RustChainError};
use crate::messages::{AIMessage, AIMessageChunk, Message};
use crate::outputs::{
    ChatGeneration, ChatGenerationChunk, ChatResult, Generation, LLMResult,
};

use super::base::BaseLanguageModel;
use super::chat_model::{BaseChatModel, ChatStream};
use super::llm::BaseLLM;

// ─── FakeListLLM ───

/// A fake LLM that cycles through a list of predefined responses.
pub struct FakeListLLM {
    pub responses: Vec<String>,
    pub sleep_ms: Option<u64>,
    index: AtomicUsize,
}

impl FakeListLLM {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            sleep_ms: None,
            index: AtomicUsize::new(0),
        }
    }

    pub fn with_sleep(mut self, ms: u64) -> Self {
        self.sleep_ms = Some(ms);
        self
    }

    fn next_response(&self) -> String {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        self.responses[idx].clone()
    }
}

#[async_trait]
impl BaseLanguageModel for FakeListLLM {
    async fn generate(&self, prompts: &[String]) -> Result<LLMResult> {
        self._generate(prompts, None).await
    }

    async fn generate_chat(&self, _messages: &[Vec<Message>]) -> Result<ChatResult> {
        let response = self.next_response();
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&response))],
            llm_output: None,
        })
    }

    fn model_type(&self) -> &str {
        "fake_list_llm"
    }
}

#[async_trait]
impl BaseLLM for FakeListLLM {
    async fn _generate(
        &self,
        prompts: &[String],
        _stop: Option<&[String]>,
    ) -> Result<LLMResult> {
        if let Some(ms) = self.sleep_ms {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        }
        let generations = prompts
            .iter()
            .map(|_| vec![Generation::new(self.next_response())])
            .collect();
        Ok(LLMResult {
            generations,
            llm_output: None,
            run: None,
        })
    }

    fn llm_type(&self) -> &str {
        "fake_list_llm"
    }
}

// ─── FakeStreamingListLLM ───

/// A fake streaming LLM that streams responses character by character.
///
/// Extends `FakeListLLM` with streaming support — each character
/// of the response is yielded as a separate chunk.
/// Optionally raises an error on a specified chunk number.
pub struct FakeStreamingListLLM {
    pub responses: Vec<String>,
    pub sleep_ms: Option<u64>,
    pub error_on_chunk_number: Option<usize>,
    index: AtomicUsize,
}

impl FakeStreamingListLLM {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            sleep_ms: None,
            error_on_chunk_number: None,
            index: AtomicUsize::new(0),
        }
    }

    pub fn with_sleep(mut self, ms: u64) -> Self {
        self.sleep_ms = Some(ms);
        self
    }

    pub fn with_error_on_chunk(mut self, chunk: usize) -> Self {
        self.error_on_chunk_number = Some(chunk);
        self
    }

    fn next_response(&self) -> String {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        self.responses[idx].clone()
    }
}

#[async_trait]
impl BaseLanguageModel for FakeStreamingListLLM {
    async fn generate(&self, prompts: &[String]) -> Result<LLMResult> {
        self._generate(prompts, None).await
    }

    async fn generate_chat(&self, _messages: &[Vec<Message>]) -> Result<ChatResult> {
        let response = self.next_response();
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&response))],
            llm_output: None,
        })
    }

    fn model_type(&self) -> &str {
        "fake_streaming_list_llm"
    }
}

#[async_trait]
impl BaseLLM for FakeStreamingListLLM {
    async fn _generate(
        &self,
        prompts: &[String],
        _stop: Option<&[String]>,
    ) -> Result<LLMResult> {
        if let Some(ms) = self.sleep_ms {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        }
        let generations = prompts
            .iter()
            .map(|_| vec![Generation::new(self.next_response())])
            .collect();
        Ok(LLMResult {
            generations,
            llm_output: None,
            run: None,
        })
    }

    async fn _stream(
        &self,
        prompt: &str,
        _stop: Option<&[String]>,
    ) -> Result<crate::runnables::RunnableStream> {
        let _ = prompt;
        let response = self.next_response();
        let error_on = self.error_on_chunk_number;

        let chunks: Vec<(usize, char)> = response.chars().enumerate().collect();
        let stream = stream::iter(chunks.into_iter().map(move |(i, c)| {
            if let Some(err_chunk) = error_on {
                if i == err_chunk {
                    return Err(RustChainError::Other(
                        "FakeStreamingListLLM error on chunk".into(),
                    ));
                }
            }
            Ok(serde_json::Value::String(c.to_string()))
        }));

        Ok(Box::pin(stream))
    }

    fn llm_type(&self) -> &str {
        "fake_streaming_list_llm"
    }
}

// ─── FakeListChatModel ───

/// A fake chat model that cycles through a list of predefined string responses.
pub struct FakeListChatModel {
    pub responses: Vec<String>,
    pub sleep_ms: Option<u64>,
    index: AtomicUsize,
}

impl FakeListChatModel {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            sleep_ms: None,
            index: AtomicUsize::new(0),
        }
    }

    pub fn with_sleep(mut self, ms: u64) -> Self {
        self.sleep_ms = Some(ms);
        self
    }

    fn next_response(&self) -> String {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        self.responses[idx].clone()
    }
}

#[async_trait]
impl BaseChatModel for FakeListChatModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        if let Some(ms) = self.sleep_ms {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        }
        let response = self.next_response();
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&response))],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "fake_list_chat_model"
    }

    async fn _stream(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        if let Some(ms) = self.sleep_ms {
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
        }
        let response = self.next_response();
        let chunks: Vec<Result<ChatGenerationChunk>> = response
            .chars()
            .map(|c| {
                Ok(ChatGenerationChunk::new(AIMessageChunk::new(
                    c.to_string(),
                )))
            })
            .collect();
        Ok(Box::pin(stream::iter(chunks)))
    }
}

// ─── FakeMessagesListChatModel ───

/// A fake chat model that cycles through a list of predefined `Message` responses.
pub struct FakeMessagesListChatModel {
    pub responses: Vec<Message>,
    index: AtomicUsize,
}

impl FakeMessagesListChatModel {
    pub fn new(responses: Vec<Message>) -> Self {
        Self {
            responses,
            index: AtomicUsize::new(0),
        }
    }

    fn next_response(&self) -> Message {
        let idx = self.index.fetch_add(1, Ordering::SeqCst) % self.responses.len();
        self.responses[idx].clone()
    }
}

#[async_trait]
impl BaseChatModel for FakeMessagesListChatModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let msg = self.next_response();
        let ai_msg = match msg {
            Message::Ai(ai) => ai,
            other => AIMessage::new(other.content().text()),
        };
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai_msg)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "fake_messages_list_chat_model"
    }
}

// ─── ParrotFakeChatModel ───

/// A fake chat model that echoes the last input message.
pub struct ParrotFakeChatModel;

impl ParrotFakeChatModel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ParrotFakeChatModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseChatModel for ParrotFakeChatModel {
    async fn _generate(
        &self,
        messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let last = messages
            .last()
            .ok_or_else(|| RustChainError::Other("No messages provided".into()))?;
        let text = last.content().text();
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(&text))],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "parrot_fake_chat_model"
    }
}

// ─── FakeChatModel ───

/// A fake chat model that always returns "fake response".
///
/// Useful as a minimal stub for testing pipelines that need a chat model
/// but do not care about the actual output content.
pub struct FakeChatModel;

impl FakeChatModel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FakeChatModel {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseChatModel for FakeChatModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new("fake response"))],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "fake_chat_model"
    }
}

// ─── GenericFakeChatModel ───

/// A generic fake chat model that draws responses from an iterator.
///
/// Each call to `_generate` pulls the next item from the internal iterator.
/// Items can be `AIMessage` values or plain strings (converted to `AIMessage`).
/// Streaming splits the response content on whitespace boundaries, preserving
/// whitespace tokens, mirroring the Python `GenericFakeChatModel` behaviour.
///
/// # Example
///
/// ```rust
/// use rustchain_core::language_models::fake::GenericFakeChatModel;
/// use rustchain_core::messages::AIMessage;
///
/// let model = GenericFakeChatModel::from_messages(vec![
///     AIMessage::new("Hello world"),
///     AIMessage::new("Goodbye"),
/// ]);
/// ```
pub struct GenericFakeChatModel {
    messages: Mutex<Box<dyn Iterator<Item = AIMessage> + Send>>,
}

impl GenericFakeChatModel {
    /// Create from a boxed iterator of `AIMessage`.
    pub fn new(iter: Box<dyn Iterator<Item = AIMessage> + Send>) -> Self {
        Self {
            messages: Mutex::new(iter),
        }
    }

    /// Create from a `Vec<AIMessage>`.
    pub fn from_messages(messages: Vec<AIMessage>) -> Self {
        Self::new(Box::new(messages.into_iter()))
    }

    /// Create from a `Vec<String>`, each converted to an `AIMessage`.
    pub fn from_strings(strings: Vec<String>) -> Self {
        Self::new(Box::new(strings.into_iter().map(AIMessage::new)))
    }

    /// Pull the next `AIMessage` from the iterator.
    fn next_message(&self) -> Result<AIMessage> {
        self.messages
            .lock()
            .map_err(|e| RustChainError::Other(format!("Lock poisoned: {e}")))?
            .next()
            .ok_or_else(|| RustChainError::Other("Iterator exhausted".into()))
    }
}

#[async_trait]
impl BaseChatModel for GenericFakeChatModel {
    async fn _generate(
        &self,
        _messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let ai_msg = self.next_message()?;
        Ok(ChatResult {
            generations: vec![ChatGeneration::new(ai_msg)],
            llm_output: None,
        })
    }

    fn llm_type(&self) -> &str {
        "generic_fake_chat_model"
    }

    /// Stream the next response, splitting on whitespace boundaries.
    ///
    /// Uses a regex-style split that preserves whitespace tokens as separate
    /// chunks (e.g. `"hello world"` becomes `["hello", " ", "world"]`).
    async fn _stream(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        let result = self._generate(messages, stop).await?;
        let message = result
            .generations
            .into_iter()
            .next()
            .ok_or_else(|| RustChainError::Other("No generations".into()))?
            .message;
        let content = message.content().text();

        let chunks: Vec<Result<ChatGenerationChunk>> = if content.is_empty() {
            Vec::new()
        } else {
            split_preserving_whitespace(&content)
                .into_iter()
                .map(|token| {
                    Ok(ChatGenerationChunk::new(AIMessageChunk::new(token)))
                })
                .collect()
        };

        Ok(Box::pin(stream::iter(chunks)))
    }
}

/// Split a string on whitespace boundaries, preserving the whitespace as
/// separate tokens. Equivalent to Python `re.split(r"(\s)", text)`.
///
/// Example: `"hello world"` -> `["hello", " ", "world"]`
fn split_preserving_whitespace(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_whitespace = false;

    for ch in s.chars() {
        let is_ws = ch.is_whitespace();
        if current.is_empty() {
            in_whitespace = is_ws;
            current.push(ch);
        } else if is_ws == in_whitespace {
            current.push(ch);
        } else {
            parts.push(std::mem::take(&mut current));
            in_whitespace = is_ws;
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{HumanMessage, Message, SystemMessage};
    use futures::StreamExt;

    fn human(text: &str) -> Message {
        Message::Human(HumanMessage::new(text))
    }

    // ── FakeListLLM ──

    #[tokio::test]
    async fn test_fake_list_llm_cycles() {
        let llm = FakeListLLM::new(vec!["a".into(), "b".into(), "c".into()]);
        let r1 = llm._generate(&["p".into()], None).await.unwrap();
        assert_eq!(r1.generations[0][0].text, "a");
        let r2 = llm._generate(&["p".into()], None).await.unwrap();
        assert_eq!(r2.generations[0][0].text, "b");
        let r3 = llm._generate(&["p".into()], None).await.unwrap();
        assert_eq!(r3.generations[0][0].text, "c");
        // wraps around
        let r4 = llm._generate(&["p".into()], None).await.unwrap();
        assert_eq!(r4.generations[0][0].text, "a");
    }

    // ── FakeStreamingListLLM ──

    #[tokio::test]
    async fn test_fake_streaming_llm_generate() {
        let llm = FakeStreamingListLLM::new(vec!["hello".into()]);
        let r = llm._generate(&["prompt".into()], None).await.unwrap();
        assert_eq!(r.generations[0][0].text, "hello");
    }

    #[tokio::test]
    async fn test_fake_streaming_llm_stream() {
        let llm = FakeStreamingListLLM::new(vec!["abc".into()]);
        let stream = llm._stream("prompt", None).await.unwrap();
        let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_ref().unwrap(), &serde_json::Value::String("a".into()));
        assert_eq!(chunks[1].as_ref().unwrap(), &serde_json::Value::String("b".into()));
        assert_eq!(chunks[2].as_ref().unwrap(), &serde_json::Value::String("c".into()));
    }

    #[tokio::test]
    async fn test_fake_streaming_llm_error_on_chunk() {
        let llm = FakeStreamingListLLM::new(vec!["abc".into()])
            .with_error_on_chunk(1);
        let stream = llm._stream("prompt", None).await.unwrap();
        let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
        assert!(chunks[0].is_ok());
        assert!(chunks[1].is_err());
        assert!(chunks[2].is_ok());
    }

    // ── FakeListChatModel ──

    #[tokio::test]
    async fn test_fake_list_chat_model_cycles() {
        let model = FakeListChatModel::new(vec!["x".into(), "y".into()]);
        let msgs = vec![human("hi")];
        let r1 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r1.generations[0].message.content().text(), "x");
        let r2 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r2.generations[0].message.content().text(), "y");
        // wraps
        let r3 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r3.generations[0].message.content().text(), "x");
    }

    #[tokio::test]
    async fn test_fake_list_chat_model_stream() {
        let model = FakeListChatModel::new(vec!["abc".into()]);
        let msgs = vec![human("hi")];
        let stream = model._stream(&msgs, None).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 3);
        let text: String = chunks
            .into_iter()
            .map(|r| {
                let chunk = r.unwrap();
                chunk.message.base.content.text()
            })
            .collect();
        assert_eq!(text, "abc");
    }

    #[tokio::test]
    async fn test_fake_list_chat_model_llm_type() {
        let model = FakeListChatModel::new(vec!["a".into()]);
        assert_eq!(model.llm_type(), "fake_list_chat_model");
    }

    // ── FakeMessagesListChatModel ──

    #[tokio::test]
    async fn test_fake_messages_list_cycles() {
        let responses = vec![
            Message::Ai(AIMessage::new("first")),
            Message::Ai(AIMessage::new("second")),
        ];
        let model = FakeMessagesListChatModel::new(responses);
        let msgs = vec![human("hello")];

        let r1 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r1.generations[0].message.content().text(), "first");
        let r2 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r2.generations[0].message.content().text(), "second");
        // wraps
        let r3 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r3.generations[0].message.content().text(), "first");
    }

    #[tokio::test]
    async fn test_fake_messages_list_converts_non_ai() {
        let responses = vec![Message::Human(HumanMessage::new("echoed"))];
        let model = FakeMessagesListChatModel::new(responses);
        let r = model._generate(&[human("hi")], None).await.unwrap();
        // Should convert the Human message text into an AIMessage
        assert_eq!(r.generations[0].message.content().text(), "echoed");
    }

    // ── ParrotFakeChatModel ──

    #[tokio::test]
    async fn test_parrot_echoes_last() {
        let model = ParrotFakeChatModel::new();
        let msgs = vec![
            Message::System(SystemMessage::new("system prompt")),
            human("repeat me"),
        ];
        let r = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r.generations[0].message.content().text(), "repeat me");
    }

    #[tokio::test]
    async fn test_parrot_empty_messages_error() {
        let model = ParrotFakeChatModel::new();
        let result = model._generate(&[], None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_parrot_default() {
        let model = ParrotFakeChatModel::default();
        let r = model._generate(&[human("test")], None).await.unwrap();
        assert_eq!(r.generations[0].message.content().text(), "test");
    }

    // ── FakeChatModel ──

    #[tokio::test]
    async fn test_fake_chat_model_always_returns_fake_response() {
        let model = FakeChatModel::new();
        let msgs = vec![human("anything")];
        let r1 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r1.generations[0].message.content().text(), "fake response");
        // Same output regardless of input
        let r2 = model
            ._generate(&[human("something else")], None)
            .await
            .unwrap();
        assert_eq!(r2.generations[0].message.content().text(), "fake response");
    }

    #[tokio::test]
    async fn test_fake_chat_model_llm_type() {
        let model = FakeChatModel::new();
        assert_eq!(model.llm_type(), "fake_chat_model");
    }

    #[tokio::test]
    async fn test_fake_chat_model_default() {
        let model = FakeChatModel::default();
        let r = model._generate(&[human("hi")], None).await.unwrap();
        assert_eq!(r.generations[0].message.content().text(), "fake response");
    }

    // ── GenericFakeChatModel ──

    #[tokio::test]
    async fn test_generic_fake_generate() {
        let model = GenericFakeChatModel::from_messages(vec![
            AIMessage::new("hello"),
            AIMessage::new("world"),
        ]);
        let msgs = vec![human("test")];
        let r1 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r1.generations[0].message.content().text(), "hello");
        let r2 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r2.generations[0].message.content().text(), "world");
    }

    #[tokio::test]
    async fn test_generic_fake_from_strings() {
        let model = GenericFakeChatModel::from_strings(vec![
            "alpha".into(),
            "beta".into(),
        ]);
        let msgs = vec![human("go")];
        let r1 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r1.generations[0].message.content().text(), "alpha");
        let r2 = model._generate(&msgs, None).await.unwrap();
        assert_eq!(r2.generations[0].message.content().text(), "beta");
    }

    #[tokio::test]
    async fn test_generic_fake_exhausted_iterator_errors() {
        let model = GenericFakeChatModel::from_messages(vec![AIMessage::new("only")]);
        let msgs = vec![human("hi")];
        let r1 = model._generate(&msgs, None).await;
        assert!(r1.is_ok());
        let r2 = model._generate(&msgs, None).await;
        assert!(r2.is_err());
    }

    #[tokio::test]
    async fn test_generic_fake_stream_splits_on_whitespace() {
        let model =
            GenericFakeChatModel::from_messages(vec![AIMessage::new("hello world foo")]);
        let msgs = vec![human("go")];
        let stream = model._stream(&msgs, None).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let tokens: Vec<String> = chunks
            .into_iter()
            .map(|r| r.unwrap().message.base.content.text())
            .collect();
        // "hello world foo" splits into ["hello", " ", "world", " ", "foo"]
        assert_eq!(tokens, vec!["hello", " ", "world", " ", "foo"]);
    }

    #[tokio::test]
    async fn test_generic_fake_stream_empty_content() {
        let model = GenericFakeChatModel::from_messages(vec![AIMessage::new("")]);
        let msgs = vec![human("go")];
        let stream = model._stream(&msgs, None).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn test_generic_fake_stream_single_word() {
        let model = GenericFakeChatModel::from_messages(vec![AIMessage::new("hello")]);
        let msgs = vec![human("go")];
        let stream = model._stream(&msgs, None).await.unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].as_ref().unwrap().message.base.content.text(), "hello");
    }

    #[tokio::test]
    async fn test_generic_fake_llm_type() {
        let model = GenericFakeChatModel::from_messages(vec![AIMessage::new("x")]);
        assert_eq!(model.llm_type(), "generic_fake_chat_model");
    }

    // ── split_preserving_whitespace ──

    #[test]
    fn test_split_preserving_whitespace_basic() {
        let result = split_preserving_whitespace("hello world");
        assert_eq!(result, vec!["hello", " ", "world"]);
    }

    #[test]
    fn test_split_preserving_whitespace_multiple_spaces() {
        let result = split_preserving_whitespace("a  b");
        assert_eq!(result, vec!["a", "  ", "b"]);
    }

    #[test]
    fn test_split_preserving_whitespace_leading_trailing() {
        let result = split_preserving_whitespace(" hi ");
        assert_eq!(result, vec![" ", "hi", " "]);
    }

    #[test]
    fn test_split_preserving_whitespace_empty() {
        let result = split_preserving_whitespace("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_preserving_whitespace_only_spaces() {
        let result = split_preserving_whitespace("   ");
        assert_eq!(result, vec!["   "]);
    }

    #[test]
    fn test_split_preserving_whitespace_tabs_and_newlines() {
        let result = split_preserving_whitespace("a\t\nb");
        assert_eq!(result, vec!["a", "\t\n", "b"]);
    }
}
