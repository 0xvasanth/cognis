//! Request/response interceptor middleware for chat models.
//!
//! Provides [`InterceptedChatModel`], a chat model wrapper that runs
//! configurable interceptor hooks before and after the inner model call.
//! Built-in interceptors cover common patterns like system message injection,
//! message trimming, content filtering, logging, response validation, and
//! token counting.
//!
//! Use [`InterceptedChatModelBuilder`] for ergonomic construction.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use regex::Regex;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::{Message, SystemMessage};
use cognis_core::outputs::ChatResult;
use cognis_core::tools::ToolSchema;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Interceptor that can inspect and modify a request before it reaches the model.
pub trait RequestInterceptor: Send + Sync {
    /// Mutate messages and/or stop sequences before the model call.
    fn intercept_request(
        &self,
        messages: &mut Vec<Message>,
        stop: &mut Option<Vec<String>>,
    ) -> Result<()>;

    /// Human-readable name for this interceptor.
    fn name(&self) -> &str;
}

/// Interceptor that can inspect and modify a response after the model returns.
pub trait ResponseInterceptor: Send + Sync {
    /// Mutate the model result before it is returned to the caller.
    fn intercept_response(&self, result: &mut ChatResult) -> Result<()>;

    /// Human-readable name for this interceptor.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// InterceptedChatModel
// ---------------------------------------------------------------------------

/// A chat model wrapper that applies request and response interceptors.
///
/// On each `_generate` call, the wrapper:
/// 1. Clones the input messages and stop sequences.
/// 2. Runs all request interceptors in order.
/// 3. Calls the inner model.
/// 4. Runs all response interceptors in order.
/// 5. Returns the (potentially modified) result.
pub struct InterceptedChatModel {
    inner: Box<dyn BaseChatModel>,
    request_interceptors: Vec<Box<dyn RequestInterceptor>>,
    response_interceptors: Vec<Box<dyn ResponseInterceptor>>,
}

impl InterceptedChatModel {
    /// Create a new intercepted chat model.
    pub fn new(
        inner: Box<dyn BaseChatModel>,
        request_interceptors: Vec<Box<dyn RequestInterceptor>>,
        response_interceptors: Vec<Box<dyn ResponseInterceptor>>,
    ) -> Self {
        Self {
            inner,
            request_interceptors,
            response_interceptors,
        }
    }
}

#[async_trait]
impl BaseChatModel for InterceptedChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        // Clone inputs so interceptors can mutate them.
        let mut msgs = messages.to_vec();
        let mut stop_owned: Option<Vec<String>> = stop.map(|s| s.to_vec());

        for interceptor in &self.request_interceptors {
            interceptor.intercept_request(&mut msgs, &mut stop_owned)?;
        }

        let stop_refs: Option<Vec<String>> = stop_owned;
        let stop_slice: Option<&[String]> = stop_refs.as_deref();

        let mut result = self.inner._generate(&msgs, stop_slice).await?;

        for interceptor in &self.response_interceptors {
            interceptor.intercept_response(&mut result)?;
        }

        Ok(result)
    }

    fn llm_type(&self) -> &str {
        let s = format!("intercepted({})", self.inner.llm_type());
        Box::leak(s.into_boxed_str())
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        self.inner._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        _tools: &[ToolSchema],
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        Err(CognisError::NotImplemented(
            "InterceptedChatModel does not support bind_tools".into(),
        ))
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

// ---------------------------------------------------------------------------
// Built-in request interceptors
// ---------------------------------------------------------------------------

/// Prepends or appends a system message to the conversation.
pub struct SystemMessageInjector {
    message: String,
    append: bool,
}

impl SystemMessageInjector {
    /// Create an injector that prepends a system message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            append: false,
        }
    }

    /// Create an injector that appends a system message.
    pub fn append(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            append: true,
        }
    }
}

impl RequestInterceptor for SystemMessageInjector {
    fn intercept_request(
        &self,
        messages: &mut Vec<Message>,
        _stop: &mut Option<Vec<String>>,
    ) -> Result<()> {
        let sys = Message::System(SystemMessage::new(&self.message));
        if self.append {
            messages.push(sys);
        } else {
            messages.insert(0, sys);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "SystemMessageInjector"
    }
}

/// Trims messages to keep at most `max_messages`, always retaining system
/// messages at the start and the most recent non-system messages.
pub struct MessageTrimmer {
    max_messages: usize,
}

impl MessageTrimmer {
    /// Create a trimmer that keeps at most `max_messages`.
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

impl RequestInterceptor for MessageTrimmer {
    fn intercept_request(
        &self,
        messages: &mut Vec<Message>,
        _stop: &mut Option<Vec<String>>,
    ) -> Result<()> {
        if messages.len() <= self.max_messages {
            return Ok(());
        }

        // Separate system messages at the front from the rest.
        let mut system_msgs = Vec::new();
        let mut other_msgs = Vec::new();
        for msg in messages.drain(..) {
            if matches!(msg, Message::System(_)) && other_msgs.is_empty() {
                system_msgs.push(msg);
            } else {
                other_msgs.push(msg);
            }
        }

        let budget = self.max_messages.saturating_sub(system_msgs.len());
        let skip = other_msgs.len().saturating_sub(budget);
        let kept: Vec<Message> = other_msgs.into_iter().skip(skip).collect();

        messages.extend(system_msgs);
        messages.extend(kept);
        Ok(())
    }

    fn name(&self) -> &str {
        "MessageTrimmer"
    }
}

/// Filters or replaces patterns in message content using regex.
pub struct ContentFilter {
    pattern: Regex,
    replacement: String,
}

impl ContentFilter {
    /// Create a filter that replaces occurrences of `pattern` with `replacement`.
    pub fn new(pattern: &str, replacement: impl Into<String>) -> Result<Self> {
        let re = Regex::new(pattern)
            .map_err(|e| CognisError::Other(format!("Invalid regex: {e}")))?;
        Ok(Self {
            pattern: re,
            replacement: replacement.into(),
        })
    }
}

impl RequestInterceptor for ContentFilter {
    fn intercept_request(
        &self,
        messages: &mut Vec<Message>,
        _stop: &mut Option<Vec<String>>,
    ) -> Result<()> {
        for msg in messages.iter_mut() {
            let text = msg.content().text();
            let replaced = self.pattern.replace_all(&text, self.replacement.as_str());
            if replaced != text {
                *msg = rebuild_message_with_text(msg, replaced.into_owned());
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ContentFilter"
    }
}

/// Logs all request messages for later inspection.
pub struct MessageLogger {
    log: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl MessageLogger {
    /// Create a new logger.
    pub fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return a handle to the internal log for assertions.
    pub fn log(&self) -> Arc<Mutex<Vec<Vec<Message>>>> {
        Arc::clone(&self.log)
    }
}

impl Default for MessageLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestInterceptor for MessageLogger {
    fn intercept_request(
        &self,
        messages: &mut Vec<Message>,
        _stop: &mut Option<Vec<String>>,
    ) -> Result<()> {
        self.log.lock().unwrap().push(messages.clone());
        Ok(())
    }

    fn name(&self) -> &str {
        "MessageLogger"
    }
}

// ---------------------------------------------------------------------------
// Built-in response interceptors
// ---------------------------------------------------------------------------

/// Validates that the response meets configurable criteria.
pub struct ResponseValidator {
    min_length: Option<usize>,
    required_keywords: Vec<String>,
}

impl ResponseValidator {
    /// Create a validator with optional minimum length and required keywords.
    pub fn new() -> Self {
        Self {
            min_length: None,
            required_keywords: Vec::new(),
        }
    }

    /// Set the minimum response text length.
    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_length = Some(len);
        self
    }

    /// Require that the response contains all of the given keywords.
    pub fn with_required_keywords(mut self, keywords: Vec<String>) -> Self {
        self.required_keywords = keywords;
        self
    }
}

impl Default for ResponseValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseInterceptor for ResponseValidator {
    fn intercept_response(&self, result: &mut ChatResult) -> Result<()> {
        for gen in &result.generations {
            let text = &gen.text;
            if let Some(min) = self.min_length {
                if text.len() < min {
                    return Err(CognisError::Other(format!(
                        "Response too short: {} < {} minimum",
                        text.len(),
                        min
                    )));
                }
            }
            for kw in &self.required_keywords {
                if !text.contains(kw.as_str()) {
                    return Err(CognisError::Other(format!(
                        "Response missing required keyword: {kw}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ResponseValidator"
    }
}

/// Applies text transformations to response content.
pub struct ResponseTransformer {
    strip_prefix: Option<String>,
    strip_suffix: Option<String>,
}

impl ResponseTransformer {
    /// Create a new transformer.
    pub fn new() -> Self {
        Self {
            strip_prefix: None,
            strip_suffix: None,
        }
    }

    /// Strip the given prefix from the response text.
    pub fn with_strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.strip_prefix = Some(prefix.into());
        self
    }

    /// Strip the given suffix from the response text.
    pub fn with_strip_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.strip_suffix = Some(suffix.into());
        self
    }
}

impl Default for ResponseTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseInterceptor for ResponseTransformer {
    fn intercept_response(&self, result: &mut ChatResult) -> Result<()> {
        for gen in result.generations.iter_mut() {
            if let Some(ref prefix) = self.strip_prefix {
                if gen.text.starts_with(prefix.as_str()) {
                    gen.text = gen.text[prefix.len()..].to_string();
                }
            }
            if let Some(ref suffix) = self.strip_suffix {
                if gen.text.ends_with(suffix.as_str()) {
                    let new_len = gen.text.len() - suffix.len();
                    gen.text.truncate(new_len);
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "ResponseTransformer"
    }
}

/// Counts approximate tokens in the response and stores in metadata.
pub struct TokenCounter {
    counts: Arc<Mutex<Vec<usize>>>,
}

impl TokenCounter {
    /// Create a new token counter.
    pub fn new() -> Self {
        Self {
            counts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return a handle to the recorded counts.
    pub fn counts(&self) -> Arc<Mutex<Vec<usize>>> {
        Arc::clone(&self.counts)
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseInterceptor for TokenCounter {
    fn intercept_response(&self, result: &mut ChatResult) -> Result<()> {
        for gen in &result.generations {
            // Rough token estimate: split on whitespace.
            let count = gen.text.split_whitespace().count();
            self.counts.lock().unwrap().push(count);

            // Store in llm_output metadata.
            let output = result
                .llm_output
                .get_or_insert_with(std::collections::HashMap::new);
            output.insert(
                "estimated_tokens".to_string(),
                serde_json::Value::Number(serde_json::Number::from(count)),
            );
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "TokenCounter"
    }
}

/// Logs all responses for later inspection.
pub struct ResponseLogger {
    log: Arc<Mutex<Vec<ChatResult>>>,
}

impl ResponseLogger {
    /// Create a new response logger.
    pub fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return a handle to the internal log.
    pub fn log(&self) -> Arc<Mutex<Vec<ChatResult>>> {
        Arc::clone(&self.log)
    }
}

impl Default for ResponseLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseInterceptor for ResponseLogger {
    fn intercept_response(&self, result: &mut ChatResult) -> Result<()> {
        self.log.lock().unwrap().push(result.clone());
        Ok(())
    }

    fn name(&self) -> &str {
        "ResponseLogger"
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing an [`InterceptedChatModel`] with fluent API.
pub struct InterceptedChatModelBuilder {
    inner: Option<Box<dyn BaseChatModel>>,
    request_interceptors: Vec<Box<dyn RequestInterceptor>>,
    response_interceptors: Vec<Box<dyn ResponseInterceptor>>,
}

impl InterceptedChatModelBuilder {
    /// Start building a new intercepted model.
    pub fn new() -> Self {
        Self {
            inner: None,
            request_interceptors: Vec::new(),
            response_interceptors: Vec::new(),
        }
    }

    /// Set the inner chat model.
    pub fn model(mut self, model: Box<dyn BaseChatModel>) -> Self {
        self.inner = Some(model);
        self
    }

    /// Add a request interceptor.
    pub fn add_request_interceptor(mut self, interceptor: Box<dyn RequestInterceptor>) -> Self {
        self.request_interceptors.push(interceptor);
        self
    }

    /// Add a response interceptor.
    pub fn add_response_interceptor(mut self, interceptor: Box<dyn ResponseInterceptor>) -> Self {
        self.response_interceptors.push(interceptor);
        self
    }

    /// Convenience: add a [`SystemMessageInjector`] that prepends the given message.
    pub fn with_system_message(self, msg: impl Into<String>) -> Self {
        self.add_request_interceptor(Box::new(SystemMessageInjector::new(msg)))
    }

    /// Convenience: add a [`MessageTrimmer`] with the given limit.
    pub fn with_message_trimmer(self, max_messages: usize) -> Self {
        self.add_request_interceptor(Box::new(MessageTrimmer::new(max_messages)))
    }

    /// Convenience: add a [`ContentFilter`] with the given pattern and replacement.
    pub fn with_content_filter(
        self,
        pattern: &str,
        replacement: impl Into<String>,
    ) -> Result<Self> {
        let filter = ContentFilter::new(pattern, replacement)?;
        Ok(self.add_request_interceptor(Box::new(filter)))
    }

    /// Build the [`InterceptedChatModel`].
    ///
    /// Returns an error if no inner model was set.
    pub fn build(self) -> Result<InterceptedChatModel> {
        let inner = self.inner.ok_or_else(|| {
            CognisError::Other("InterceptedChatModelBuilder: inner model is required".into())
        })?;
        Ok(InterceptedChatModel::new(
            inner,
            self.request_interceptors,
            self.response_interceptors,
        ))
    }
}

impl Default for InterceptedChatModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rebuild a message with new text content, preserving its variant.
fn rebuild_message_with_text(original: &Message, new_text: String) -> Message {
    use cognis_core::messages::*;
    match original {
        Message::Human(_) => Message::Human(HumanMessage::new(new_text)),
        Message::Ai(_) => Message::Ai(AIMessage::new(new_text)),
        Message::System(_) => Message::System(SystemMessage::new(new_text)),
        Message::Tool(t) => Message::Tool(ToolMessage::new(new_text, &t.tool_call_id)),
        Message::Chat(c) => Message::Chat(ChatMessage::new(new_text, c.role.clone())),
        // For chunk types, just rebuild as the base type.
        _ => Message::Human(HumanMessage::new(new_text)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::language_models::fake::FakeListChatModel;
    use cognis_core::messages::{HumanMessage, Message, SystemMessage};

    fn human(text: &str) -> Message {
        Message::Human(HumanMessage::new(text))
    }

    fn system(text: &str) -> Message {
        Message::System(SystemMessage::new(text))
    }

    fn fake_model(responses: Vec<&str>) -> Box<FakeListChatModel> {
        Box::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    // ── SystemMessageInjector ──

    #[tokio::test]
    async fn test_system_message_injector_prepends() {
        let injector = SystemMessageInjector::new("You are helpful.");
        let mut msgs = vec![human("hello")];
        let mut stop = None;
        injector.intercept_request(&mut msgs, &mut stop).unwrap();

        assert_eq!(msgs.len(), 2);
        assert!(matches!(&msgs[0], Message::System(_)));
        assert_eq!(msgs[0].content().text(), "You are helpful.");
        assert_eq!(msgs[1].content().text(), "hello");
    }

    #[tokio::test]
    async fn test_system_message_injector_appends() {
        let injector = SystemMessageInjector::append("Remember to be kind.");
        let mut msgs = vec![human("hello")];
        let mut stop = None;
        injector.intercept_request(&mut msgs, &mut stop).unwrap();

        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content().text(), "hello");
        assert!(matches!(&msgs[1], Message::System(_)));
        assert_eq!(msgs[1].content().text(), "Remember to be kind.");
    }

    // ── MessageTrimmer ──

    #[tokio::test]
    async fn test_message_trimmer_keeps_last_n() {
        let trimmer = MessageTrimmer::new(2);
        let mut msgs = vec![human("a"), human("b"), human("c"), human("d")];
        let mut stop = None;
        trimmer.intercept_request(&mut msgs, &mut stop).unwrap();

        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content().text(), "c");
        assert_eq!(msgs[1].content().text(), "d");
    }

    #[tokio::test]
    async fn test_message_trimmer_preserves_system() {
        let trimmer = MessageTrimmer::new(3);
        let mut msgs = vec![system("sys"), human("a"), human("b"), human("c")];
        let mut stop = None;
        trimmer.intercept_request(&mut msgs, &mut stop).unwrap();

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content().text(), "sys");
        assert_eq!(msgs[1].content().text(), "b");
        assert_eq!(msgs[2].content().text(), "c");
    }

    #[tokio::test]
    async fn test_message_trimmer_noop_when_under_limit() {
        let trimmer = MessageTrimmer::new(10);
        let mut msgs = vec![human("a"), human("b")];
        let mut stop = None;
        trimmer.intercept_request(&mut msgs, &mut stop).unwrap();
        assert_eq!(msgs.len(), 2);
    }

    // ── ContentFilter ──

    #[tokio::test]
    async fn test_content_filter_replaces_pattern() {
        let filter = ContentFilter::new(r"secret\d+", "[REDACTED]").unwrap();
        let mut msgs = vec![human("my secret123 code")];
        let mut stop = None;
        filter.intercept_request(&mut msgs, &mut stop).unwrap();

        assert_eq!(msgs[0].content().text(), "my [REDACTED] code");
    }

    #[tokio::test]
    async fn test_content_filter_no_match_unchanged() {
        let filter = ContentFilter::new(r"xyz", "abc").unwrap();
        let mut msgs = vec![human("hello world")];
        let mut stop = None;
        filter.intercept_request(&mut msgs, &mut stop).unwrap();
        assert_eq!(msgs[0].content().text(), "hello world");
    }

    // ── MessageLogger ──

    #[tokio::test]
    async fn test_message_logger_captures_messages() {
        let logger = MessageLogger::new();
        let log = logger.log();
        let mut msgs = vec![human("first"), human("second")];
        let mut stop = None;
        logger.intercept_request(&mut msgs, &mut stop).unwrap();

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].len(), 2);
        assert_eq!(captured[0][0].content().text(), "first");
    }

    // ── ResponseValidator ──

    #[tokio::test]
    async fn test_response_validator_min_length_pass() {
        let validator = ResponseValidator::new().with_min_length(3);
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("hello"),
            )],
            llm_output: None,
        };
        assert!(validator.intercept_response(&mut result).is_ok());
    }

    #[tokio::test]
    async fn test_response_validator_min_length_fail() {
        let validator = ResponseValidator::new().with_min_length(100);
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("hi"),
            )],
            llm_output: None,
        };
        assert!(validator.intercept_response(&mut result).is_err());
    }

    #[tokio::test]
    async fn test_response_validator_required_keyword_fail() {
        let validator = ResponseValidator::new().with_required_keywords(vec!["magic".to_string()]);
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("no keyword here"),
            )],
            llm_output: None,
        };
        assert!(validator.intercept_response(&mut result).is_err());
    }

    // ── ResponseTransformer ──

    #[tokio::test]
    async fn test_response_transformer_strips_prefix() {
        let transformer = ResponseTransformer::new().with_strip_prefix("AI: ");
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("AI: hello there"),
            )],
            llm_output: None,
        };
        transformer.intercept_response(&mut result).unwrap();
        assert_eq!(result.generations[0].text, "hello there");
    }

    // ── TokenCounter ──

    #[tokio::test]
    async fn test_token_counter_counts_tokens() {
        let counter = TokenCounter::new();
        let counts = counter.counts();
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("one two three four"),
            )],
            llm_output: None,
        };
        counter.intercept_response(&mut result).unwrap();

        let recorded = counts.lock().unwrap();
        assert_eq!(recorded[0], 4);

        // Also check metadata.
        let output = result.llm_output.unwrap();
        assert_eq!(output["estimated_tokens"], serde_json::json!(4));
    }

    // ── ResponseLogger ──

    #[tokio::test]
    async fn test_response_logger_captures_response() {
        let logger = ResponseLogger::new();
        let log = logger.log();
        let mut result = ChatResult {
            generations: vec![cognis_core::outputs::ChatGeneration::new(
                cognis_core::messages::AIMessage::new("captured"),
            )],
            llm_output: None,
        };
        logger.intercept_response(&mut result).unwrap();

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].generations[0].text, "captured");
    }

    // ── Interceptor chain ordering ──

    #[tokio::test]
    async fn test_request_interceptors_run_in_order() {
        // First: inject system message. Second: trim to 2 messages.
        let model = InterceptedChatModel::new(
            fake_model(vec!["ok"]),
            vec![
                Box::new(SystemMessageInjector::new("system")),
                Box::new(MessageTrimmer::new(2)),
            ],
            vec![],
        );

        // Input: 1 human message. After injector: sys + human (2). After trimmer: unchanged (2).
        let result = model._generate(&[human("hi")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "ok");
    }

    #[tokio::test]
    async fn test_response_interceptors_run_in_order() {
        // First transformer strips prefix, then validator checks length.
        let transformer = ResponseTransformer::new().with_strip_prefix("PREFIX:");
        let validator = ResponseValidator::new().with_min_length(1);

        let model = InterceptedChatModel::new(
            fake_model(vec!["PREFIX:hello"]),
            vec![],
            vec![Box::new(transformer), Box::new(validator)],
        );

        let result = model._generate(&[human("go")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "hello");
    }

    // ── Builder ──

    #[tokio::test]
    async fn test_builder_pattern() {
        let model = InterceptedChatModelBuilder::new()
            .model(fake_model(vec!["built"]))
            .with_system_message("be nice")
            .with_message_trimmer(10)
            .build()
            .unwrap();

        let result = model._generate(&[human("hi")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "built");
    }

    #[tokio::test]
    async fn test_builder_no_model_errors() {
        let result = InterceptedChatModelBuilder::new().build();
        assert!(result.is_err());
    }

    // ── InterceptedChatModel pass-through ──

    #[tokio::test]
    async fn test_intercepted_model_passthrough() {
        let model = InterceptedChatModel::new(fake_model(vec!["pass"]), vec![], vec![]);
        let result = model._generate(&[human("hi")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "pass");
    }

    // ── Multiple interceptors combined ──

    #[tokio::test]
    async fn test_multiple_interceptors_combined() {
        let msg_logger = MessageLogger::new();
        let msg_log = msg_logger.log();
        let resp_logger = ResponseLogger::new();
        let resp_log = resp_logger.log();

        let model = InterceptedChatModel::new(
            fake_model(vec!["AI: world"]),
            vec![
                Box::new(SystemMessageInjector::new("sys")),
                Box::new(msg_logger),
            ],
            vec![
                Box::new(ResponseTransformer::new().with_strip_prefix("AI: ")),
                Box::new(resp_logger),
            ],
        );

        let result = model._generate(&[human("hello")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "world");

        // Message logger should have captured messages after system injection.
        let msgs = msg_log.lock().unwrap();
        assert_eq!(msgs[0].len(), 2);
        assert!(matches!(&msgs[0][0], Message::System(_)));

        // Response logger should have captured the transformed response.
        let resps = resp_log.lock().unwrap();
        assert_eq!(resps[0].generations[0].text, "world");
    }

    // ── Empty interceptor lists ──

    #[tokio::test]
    async fn test_empty_interceptor_lists() {
        let model = InterceptedChatModel::new(fake_model(vec!["empty"]), vec![], vec![]);
        let result = model._generate(&[human("test")], None).await.unwrap();
        assert_eq!(result.generations[0].text, "empty");
    }

    // ── Interceptor names ──

    #[test]
    fn test_interceptor_names() {
        assert_eq!(
            SystemMessageInjector::new("x").name(),
            "SystemMessageInjector"
        );
        assert_eq!(MessageTrimmer::new(5).name(), "MessageTrimmer");
        assert_eq!(
            ContentFilter::new("a", "b").unwrap().name(),
            "ContentFilter"
        );
        assert_eq!(MessageLogger::new().name(), "MessageLogger");
        assert_eq!(ResponseValidator::new().name(), "ResponseValidator");
        assert_eq!(ResponseTransformer::new().name(), "ResponseTransformer");
        assert_eq!(TokenCounter::new().name(), "TokenCounter");
        assert_eq!(ResponseLogger::new().name(), "ResponseLogger");
    }

    // ── Builder with content filter ──

    #[tokio::test]
    async fn test_builder_with_content_filter() {
        let model = InterceptedChatModelBuilder::new()
            .model(fake_model(vec!["filtered"]))
            .with_content_filter(r"bad", "good")
            .unwrap()
            .build()
            .unwrap();

        let result = model
            ._generate(&[human("this is bad")], None)
            .await
            .unwrap();
        assert_eq!(result.generations[0].text, "filtered");
    }
}
