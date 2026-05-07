//! Routing — dispatch each call to one of N providers based on a
//! user-supplied predicate.
//!
//! Use cases: route fast queries to a cheap model, route long-context
//! queries to a large-context model, route by user-tier metadata, etc.
//!
//! Customization:
//! - Implement [`RoutingStrategy`] for custom dispatch logic.
//! - Or use [`ProviderRoute`]'s closure-based predicate for inline policies.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::tools::ToolDefinition;
use crate::Message;

/// Pluggable routing decision. Returns the index of the chosen route in
/// the [`RoutingProvider`]'s table, or `None` to fall through to the
/// default route.
pub trait RoutingStrategy: Send + Sync {
    /// Pick a route index for the given chat call.
    fn pick(&self, messages: &[Message], opts: &ChatOptions) -> Option<usize>;
}

/// Closure-based strategy.
impl<F> RoutingStrategy for F
where
    F: Fn(&[Message], &ChatOptions) -> Option<usize> + Send + Sync,
{
    fn pick(&self, messages: &[Message], opts: &ChatOptions) -> Option<usize> {
        (self)(messages, opts)
    }
}

/// Boxed predicate: `(messages, opts) → bool`.
pub type RoutePredicate = Arc<dyn Fn(&[Message], &ChatOptions) -> bool + Send + Sync>;

/// One named route in a [`RoutingProvider`]. The `predicate` decides
/// whether the route claims the call; the first claiming predicate wins.
pub struct ProviderRoute {
    /// Friendly route label (e.g. `"long-context"`).
    pub name: String,
    /// Provider used when this route matches.
    pub provider: Arc<dyn LLMProvider>,
    /// Predicate. Receives `(messages, options)`; returns `true` if the
    /// route should handle this call.
    pub predicate: RoutePredicate,
}

impl ProviderRoute {
    /// Build a route with an inline closure predicate.
    pub fn new<F>(name: impl Into<String>, provider: Arc<dyn LLMProvider>, predicate: F) -> Self
    where
        F: Fn(&[Message], &ChatOptions) -> bool + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            provider,
            predicate: Arc::new(predicate),
        }
    }
}

/// Routing wrapper. Holds an ordered list of [`ProviderRoute`]s and a
/// default provider for the no-match case.
pub struct RoutingProvider {
    routes: Vec<ProviderRoute>,
    default_route: Arc<dyn LLMProvider>,
    strategy: Option<Box<dyn RoutingStrategy>>,
    name: String,
}

impl RoutingProvider {
    /// Build with a default provider used when no route claims the call.
    pub fn new(name: impl Into<String>, default_route: Arc<dyn LLMProvider>) -> Self {
        Self {
            routes: Vec::new(),
            default_route,
            strategy: None,
            name: name.into(),
        }
    }

    /// Append a route. Routes are evaluated in registration order; the
    /// first matching predicate wins.
    pub fn route(mut self, route: ProviderRoute) -> Self {
        self.routes.push(route);
        self
    }

    /// Override route selection with a [`RoutingStrategy`]. When set,
    /// the strategy's `pick` is consulted first; per-route predicates
    /// only apply when the strategy returns `None`.
    pub fn with_strategy<S>(mut self, strategy: S) -> Self
    where
        S: RoutingStrategy + 'static,
    {
        self.strategy = Some(Box::new(strategy));
        self
    }

    /// Borrow the registered routes.
    pub fn routes(&self) -> &[ProviderRoute] {
        &self.routes
    }

    /// Resolve a call to the chosen provider.
    fn resolve(&self, messages: &[Message], opts: &ChatOptions) -> &Arc<dyn LLMProvider> {
        if let Some(s) = &self.strategy {
            if let Some(idx) = s.pick(messages, opts) {
                if let Some(r) = self.routes.get(idx) {
                    return &r.provider;
                }
            }
        }
        for r in &self.routes {
            if (r.predicate)(messages, opts) {
                return &r.provider;
            }
        }
        &self.default_route
    }
}

#[async_trait]
impl LLMProvider for RoutingProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn provider_type(&self) -> Provider {
        self.default_route.provider_type()
    }
    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let p = self.resolve(&messages, &opts).clone();
        p.chat_completion(messages, opts).await
    }
    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        let p = self.resolve(&messages, &opts).clone();
        p.chat_completion_stream(messages, opts).await
    }
    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let p = self.resolve(&messages, &opts).clone();
        p.chat_completion_with_tools(messages, tools, opts).await
    }
    async fn health_check(&self) -> Result<HealthStatus> {
        // Only check the default route — the others may be intentionally
        // unreachable when their predicates don't match.
        self.default_route.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Tagged(Arc<Mutex<Vec<&'static str>>>, &'static str);
    #[async_trait]
    impl LLMProvider for Tagged {
        fn name(&self) -> &str {
            self.1
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            self.0.lock().unwrap().push(self.1);
            Ok(ChatResponse {
                message: Message::ai(self.1),
                usage: None,
                finish_reason: "stop".into(),
                model: self.1.into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn ep(seen: Arc<Mutex<Vec<&'static str>>>, tag: &'static str) -> Arc<dyn LLMProvider> {
        Arc::new(Tagged(seen, tag))
    }

    #[tokio::test]
    async fn predicate_routes_call() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let r = RoutingProvider::new("router", ep(seen.clone(), "default"))
            .route(ProviderRoute::new(
                "long-context",
                ep(seen.clone(), "big-model"),
                |msgs, _| msgs.iter().map(|m| m.content().len()).sum::<usize>() > 100,
            ))
            .route(ProviderRoute::new(
                "tiny",
                ep(seen.clone(), "small-model"),
                |msgs, _| msgs.iter().map(|m| m.content().len()).sum::<usize>() < 5,
            ));

        let _ = r
            .chat_completion(
                vec![Message::human("a".repeat(200))],
                ChatOptions::default(),
            )
            .await;
        let _ = r
            .chat_completion(vec![Message::human("hi")], ChatOptions::default())
            .await;
        let _ = r
            .chat_completion(
                vec![Message::human("medium length text")],
                ChatOptions::default(),
            )
            .await;

        let s = seen.lock().unwrap().clone();
        assert_eq!(s, vec!["big-model", "small-model", "default"]);
    }

    #[tokio::test]
    async fn strategy_overrides_predicates() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let r = RoutingProvider::new("router", ep(seen.clone(), "default"))
            .route(ProviderRoute::new(
                "a",
                ep(seen.clone(), "first"),
                |_, _| false, // predicate would never match
            ))
            .with_strategy(|_msgs: &[Message], _opts: &ChatOptions| Some(0));

        let _ = r.chat_completion(vec![], ChatOptions::default()).await;
        assert_eq!(seen.lock().unwrap()[0], "first");
    }

    #[tokio::test]
    async fn falls_through_to_default_when_no_match() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let r = RoutingProvider::new("router", ep(seen.clone(), "default"));
        let _ = r.chat_completion(vec![], ChatOptions::default()).await;
        assert_eq!(seen.lock().unwrap()[0], "default");
    }
}
