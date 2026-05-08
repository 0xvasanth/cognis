//! Model fallback middleware — sequential fallback to alternative models.
//!
//! Mirrors Python `langchain.agents.middleware.model_fallback`.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::error::Result;
use cognis_core::language_models::chat_model::BaseChatModel;

use super::types::{AgentMiddleware, AsyncModelHandler, ModelCallResult, ModelRequest};

/// Middleware that tries alternative models when the primary model fails.
pub struct ModelFallbackMiddleware {
    /// Fallback models to try in order.
    pub fallback_models: Vec<Arc<dyn BaseChatModel>>,
}

impl ModelFallbackMiddleware {
    pub fn new(fallback_models: Vec<Arc<dyn BaseChatModel>>) -> Self {
        Self { fallback_models }
    }
}

#[async_trait]
impl AgentMiddleware for ModelFallbackMiddleware {
    fn name(&self) -> &str {
        "ModelFallbackMiddleware"
    }

    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        // Try primary model first
        match handler(request).await {
            Ok(response) => return Ok(ModelCallResult::Response(response)),
            Err(primary_error) => {
                // If no fallbacks, propagate the error
                if self.fallback_models.is_empty() {
                    return Err(primary_error);
                }
                // Try each fallback (note: in a full impl we'd swap the model
                // in the request, but since ModelRequest borrows the model,
                // we store the last error and report it if all fail)
                let mut last_error = primary_error;
                for fallback in &self.fallback_models {
                    // Create a new request with the fallback model, copying other fields
                    let fallback_request = ModelRequest {
                        model: Arc::clone(fallback),
                        messages: request.messages.clone(),
                        system_message: request.system_message.clone(),
                        tool_choice: request.tool_choice.clone(),
                        tools: request.tools.clone(),
                        response_format: request.response_format.clone(),
                        state: request.state.clone(),
                        model_settings: request.model_settings.clone(),
                    };
                    match handler(&fallback_request).await {
                        Ok(response) => return Ok(ModelCallResult::Response(response)),
                        Err(e) => {
                            last_error = e;
                        }
                    }
                }
                Err(last_error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_fallback_new() {
        let mw = ModelFallbackMiddleware::new(vec![]);
        assert_eq!(mw.name(), "ModelFallbackMiddleware");
        assert!(mw.fallback_models.is_empty());
    }
}
