//! Graceful degradation wrapper for chat models.
//!
//! Provides [`GracefulChatModel`], a chat model wrapper that returns a
//! configurable fallback response when the inner model fails, instead of
//! propagating the error. Useful for non-critical paths where partial
//! failure is acceptable.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::{AIMessage, Message};
use cognis_core::outputs::{ChatGeneration, ChatResult};
use cognis_core::tools::ToolSchema;

/// A chat model that returns a fallback response when the inner model fails.
///
/// On `_generate` failure, returns a [`ChatResult`] containing the
/// `fallback_message`. Optionally calls an `on_error` callback for
/// logging or monitoring.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::chat_models::graceful::GracefulChatModel;
///
/// let graceful = GracefulChatModel::new(
///     Box::new(my_model),
///     "I'm sorry, I'm unable to process your request right now.".into(),
/// );
/// ```
pub struct GracefulChatModel {
    inner: Box<dyn BaseChatModel>,
    fallback_message: String,
    #[allow(clippy::type_complexity)]
    on_error: Option<Arc<dyn Fn(&CognisError) + Send + Sync>>,
}

impl GracefulChatModel {
    /// Create a new graceful chat model wrapper.
    ///
    /// # Arguments
    /// * `inner` - The chat model to wrap.
    /// * `fallback_message` - The message to return when the inner model fails.
    pub fn new(inner: Box<dyn BaseChatModel>, fallback_message: String) -> Self {
        Self {
            inner,
            fallback_message,
            on_error: None,
        }
    }

    /// Set an error callback for logging or monitoring.
    ///
    /// The callback is invoked with the error before the fallback response
    /// is returned.
    pub fn with_on_error<F>(mut self, callback: F) -> Self
    where
        F: Fn(&CognisError) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(callback));
        self
    }

    /// Build a fallback ChatResult from the configured fallback message.
    fn fallback_result(&self) -> ChatResult {
        ChatResult {
            generations: vec![ChatGeneration {
                text: self.fallback_message.clone(),
                message: Message::Ai(AIMessage::new(&self.fallback_message)),
                generation_info: None,
            }],
            llm_output: None,
        }
    }
}

#[async_trait]
impl BaseChatModel for GracefulChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        match self.inner._generate(messages, stop).await {
            Ok(result) => Ok(result),
            Err(e) => {
                if let Some(ref callback) = self.on_error {
                    callback(&e);
                }
                Ok(self.fallback_result())
            }
        }
    }

    fn llm_type(&self) -> &str {
        self.inner.llm_type()
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        match self.inner._stream(messages, stop).await {
            Ok(stream) => Ok(stream),
            Err(e) => {
                if let Some(ref callback) = self.on_error {
                    callback(&e);
                }
                // Return a single-item stream with the fallback
                use futures::stream;
                use cognis_core::messages::AIMessageChunk;
                use cognis_core::outputs::ChatGenerationChunk;
                let chunk = ChatGenerationChunk {
                    text: self.fallback_message.clone(),
                    message: AIMessageChunk::new(&self.fallback_message),
                    generation_info: None,
                };
                Ok(Box::pin(stream::once(async move { Ok(chunk) })))
            }
        }
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        self.inner.bind_tools(tools, tool_choice)
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::HumanMessage;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A mock chat model that always succeeds.
    struct SuccessModel;

    #[async_trait]
    impl BaseChatModel for SuccessModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: "Real response".into(),
                    message: Message::Ai(AIMessage::new("Real response")),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "success_mock"
        }
    }

    /// A mock chat model that always fails.
    struct FailModel;

    #[async_trait]
    impl BaseChatModel for FailModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Err(CognisError::HttpError {
                status: 500,
                body: "Internal Server Error".into(),
            })
        }

        fn llm_type(&self) -> &str {
            "fail_mock"
        }
    }

    #[tokio::test]
    async fn test_graceful_passes_through_on_success() {
        let model = GracefulChatModel::new(Box::new(SuccessModel), "Fallback message".into());

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];
        let result = model._generate(&msgs, None).await;
        assert!(result.is_ok());
        let chat_result = result.unwrap();
        assert_eq!(chat_result.generations[0].text, "Real response");
    }

    #[tokio::test]
    async fn test_graceful_returns_fallback_on_error() {
        let model =
            GracefulChatModel::new(Box::new(FailModel), "Sorry, service unavailable".into());

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];
        let result = model._generate(&msgs, None).await;
        assert!(result.is_ok());
        let chat_result = result.unwrap();
        assert_eq!(
            chat_result.generations[0].text,
            "Sorry, service unavailable"
        );
    }

    #[tokio::test]
    async fn test_graceful_calls_on_error_callback() {
        let error_logged = Arc::new(AtomicBool::new(false));
        let error_logged_clone = error_logged.clone();

        let model = GracefulChatModel::new(Box::new(FailModel), "Fallback".into()).with_on_error(
            move |_err| {
                error_logged_clone.store(true, Ordering::SeqCst);
            },
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];
        let result = model._generate(&msgs, None).await;
        assert!(result.is_ok());
        assert!(
            error_logged.load(Ordering::SeqCst),
            "on_error callback should have been called"
        );
    }
}
