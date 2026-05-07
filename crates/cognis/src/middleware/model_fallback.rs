//! Fall through to a backup `Client` if the primary errors.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;
use cognis_llm::Client;

use super::{Middleware, MiddlewareCtx, Next};

/// On any error from the primary chain, retries the call against `fallback`
/// (a different `Client`). For multi-tier fallback, stack multiple
/// `ModelFallback` layers.
pub struct ModelFallback {
    fallback: Client,
}

impl ModelFallback {
    /// Wrap with a fallback client.
    pub fn new(fallback: Client) -> Self {
        Self { fallback }
    }
}

#[async_trait]
impl Middleware for ModelFallback {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        match next.invoke(ctx.clone()).await {
            Ok(r) => Ok(r),
            Err(_) => {
                self.fallback
                    .provider()
                    .chat_completion_with_tools(ctx.messages, ctx.tool_defs, ctx.opts)
                    .await
            }
        }
    }

    fn name(&self) -> &str {
        "ModelFallback"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_core::{CognisError, Message};
    use cognis_llm::chat::ChatOptions;

    #[tokio::test]
    async fn falls_through_on_error() {
        let primary_provider = make_flaky_provider(|_| {
            Err(CognisError::Network {
                status_code: Some(500),
                message: "boom".into(),
            })
        });
        let primary = Client::new(primary_provider);
        let backup_provider = make_flaky_provider(|_| Ok("backup".into()));
        let backup = Client::new(backup_provider);
        let pipe = MiddlewarePipeline::new()
            .push(ModelFallback::new(backup))
            .build(primary);
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "backup");
    }

    #[tokio::test]
    async fn primary_wins_on_success() {
        let primary = Client::new(make_flaky_provider(|_| Ok("primary".into())));
        let backup = Client::new(make_flaky_provider(|_| Ok("backup".into())));
        let pipe = MiddlewarePipeline::new()
            .push(ModelFallback::new(backup))
            .build(primary);
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "primary");
    }
}
