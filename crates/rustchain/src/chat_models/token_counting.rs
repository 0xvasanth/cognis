//! Token counting and cost estimation wrapper for chat models.
//!
//! Provides [`TokenCountingModel`], a chat model wrapper that tracks token
//! usage across calls and optionally estimates costs using a [`PricingRegistry`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::Message;
use rustchain_core::outputs::ChatResult;
use rustchain_core::tools::ToolSchema;
use rustchain_core::utils::tokens::estimate_token_count;

/// Pricing information for a specific model.
#[derive(Debug, Clone)]
pub struct ModelPricing {
    /// Cost per 1,000 input tokens in USD.
    pub input_cost_per_1k: f64,
    /// Cost per 1,000 output tokens in USD.
    pub output_cost_per_1k: f64,
    /// The model name this pricing applies to.
    pub model_name: String,
}

impl ModelPricing {
    /// Create a new pricing entry.
    pub fn new(model_name: impl Into<String>, input_cost_per_1k: f64, output_cost_per_1k: f64) -> Self {
        Self {
            model_name: model_name.into(),
            input_cost_per_1k,
            output_cost_per_1k,
        }
    }

    /// Calculate cost for the given token counts.
    pub fn calculate_cost(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        (input_tokens as f64 / 1000.0) * self.input_cost_per_1k
            + (output_tokens as f64 / 1000.0) * self.output_cost_per_1k
    }
}

/// Registry of model pricing information.
///
/// Pre-populated with common model prices via [`Default`]. Custom models can
/// be registered with [`PricingRegistry::register`].
#[derive(Debug, Clone)]
pub struct PricingRegistry {
    prices: HashMap<String, ModelPricing>,
}

impl PricingRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            prices: HashMap::new(),
        }
    }

    /// Look up pricing for a model.
    pub fn get_pricing(&self, model: &str) -> Option<&ModelPricing> {
        self.prices.get(model)
    }

    /// Register or update pricing for a model.
    pub fn register(&mut self, pricing: ModelPricing) {
        self.prices.insert(pricing.model_name.clone(), pricing);
    }
}

impl Default for PricingRegistry {
    /// Returns a registry pre-populated with common model prices.
    fn default() -> Self {
        let mut registry = Self::new();

        // OpenAI models
        registry.register(ModelPricing::new("gpt-4o", 2.50, 10.00));
        registry.register(ModelPricing::new("gpt-4o-mini", 0.15, 0.60));
        registry.register(ModelPricing::new("gpt-4-turbo", 10.00, 30.00));

        // Anthropic models
        registry.register(ModelPricing::new("claude-3.5-sonnet", 3.00, 15.00));
        registry.register(ModelPricing::new("claude-3-opus", 15.00, 75.00));
        registry.register(ModelPricing::new("claude-3-haiku", 0.25, 1.25));

        // Google models
        registry.register(ModelPricing::new("gemini-1.5-pro", 3.50, 10.50));
        registry.register(ModelPricing::new("gemini-1.5-flash", 0.075, 0.30));

        registry
    }
}

/// Token usage statistics for a model call or cumulative session.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenUsage {
    /// Number of input (prompt) tokens.
    pub input_tokens: usize,
    /// Number of output (completion) tokens.
    pub output_tokens: usize,
    /// Total tokens (input + output).
    pub total_tokens: usize,
    /// Estimated cost in USD, if pricing is available.
    pub estimated_cost: Option<f64>,
}

impl TokenUsage {
    /// Create a new usage record.
    pub fn new(input_tokens: usize, output_tokens: usize, estimated_cost: Option<f64>) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            estimated_cost,
        }
    }

    /// Zero usage.
    pub fn zero() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            estimated_cost: None,
        }
    }
}

/// A chat model wrapper that tracks token usage and estimates costs.
///
/// Delegates all calls to the inner model while counting input and output
/// tokens. Maintains both cumulative and last-call usage statistics.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::chat_models::token_counting::{TokenCountingModel, ModelPricing};
///
/// let counting = TokenCountingModel::builder(Box::new(my_model))
///     .with_pricing(ModelPricing::new("gpt-4o", 2.50, 10.00))
///     .build();
///
/// // After calls...
/// let usage = counting.get_usage();
/// println!("Total tokens: {}", usage.total_tokens);
/// ```
pub struct TokenCountingModel {
    inner: Box<dyn BaseChatModel>,
    cumulative_input: AtomicUsize,
    cumulative_output: AtomicUsize,
    last_input: AtomicUsize,
    last_output: AtomicUsize,
    pricing: Option<ModelPricing>,
    /// Protects the update of last-call counters so both are set atomically
    /// with respect to concurrent readers.
    _lock: Mutex<()>,
}

impl TokenCountingModel {
    /// Create a new token counting wrapper.
    pub fn new(inner: Box<dyn BaseChatModel>, pricing: Option<ModelPricing>) -> Self {
        Self {
            inner,
            cumulative_input: AtomicUsize::new(0),
            cumulative_output: AtomicUsize::new(0),
            last_input: AtomicUsize::new(0),
            last_output: AtomicUsize::new(0),
            pricing,
            _lock: Mutex::new(()),
        }
    }

    /// Create a builder for configuring a token counting model.
    pub fn builder(inner: Box<dyn BaseChatModel>) -> TokenCountingModelBuilder {
        TokenCountingModelBuilder {
            inner,
            pricing: None,
        }
    }

    /// Get cumulative token usage across all calls.
    pub fn get_usage(&self) -> TokenUsage {
        let input = self.cumulative_input.load(Ordering::SeqCst);
        let output = self.cumulative_output.load(Ordering::SeqCst);
        let cost = self
            .pricing
            .as_ref()
            .map(|p| p.calculate_cost(input, output));
        TokenUsage::new(input, output, cost)
    }

    /// Get token usage from the last call only.
    pub fn get_last_usage(&self) -> TokenUsage {
        let input = self.last_input.load(Ordering::SeqCst);
        let output = self.last_output.load(Ordering::SeqCst);
        let cost = self
            .pricing
            .as_ref()
            .map(|p| p.calculate_cost(input, output));
        TokenUsage::new(input, output, cost)
    }

    /// Reset all usage counters to zero.
    pub fn reset_usage(&self) {
        self.cumulative_input.store(0, Ordering::SeqCst);
        self.cumulative_output.store(0, Ordering::SeqCst);
        self.last_input.store(0, Ordering::SeqCst);
        self.last_output.store(0, Ordering::SeqCst);
    }

    /// Record token usage for a single call.
    fn record_usage(&self, input_tokens: usize, output_tokens: usize) {
        let _guard = self._lock.lock().unwrap_or_else(|e| e.into_inner());
        self.last_input.store(input_tokens, Ordering::SeqCst);
        self.last_output.store(output_tokens, Ordering::SeqCst);
        self.cumulative_input.fetch_add(input_tokens, Ordering::SeqCst);
        self.cumulative_output.fetch_add(output_tokens, Ordering::SeqCst);
    }

    /// Estimate input tokens from messages.
    fn estimate_input_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| estimate_token_count(&m.content().text()))
            .sum()
    }

    /// Estimate output tokens from a chat result.
    fn estimate_output_tokens(&self, result: &ChatResult) -> usize {
        result
            .generations
            .iter()
            .map(|g| estimate_token_count(&g.text))
            .sum()
    }
}

/// Builder for [`TokenCountingModel`].
pub struct TokenCountingModelBuilder {
    inner: Box<dyn BaseChatModel>,
    pricing: Option<ModelPricing>,
}

impl TokenCountingModelBuilder {
    /// Set pricing for cost estimation.
    pub fn with_pricing(mut self, pricing: ModelPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Build the token counting model.
    pub fn build(self) -> TokenCountingModel {
        TokenCountingModel::new(self.inner, self.pricing)
    }
}

#[async_trait]
impl BaseChatModel for TokenCountingModel {
    async fn _generate(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let input_tokens = self.estimate_input_tokens(messages);
        let result = self.inner._generate(messages, stop).await?;
        let output_tokens = self.estimate_output_tokens(&result);
        self.record_usage(input_tokens, output_tokens);
        Ok(result)
    }

    fn llm_type(&self) -> &str {
        self.inner.llm_type()
    }

    async fn _stream(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        self.inner._stream(messages, stop).await
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
    use rustchain_core::language_models::fake::FakeListChatModel;
    use rustchain_core::messages::{HumanMessage, Message};

    fn human(text: &str) -> Message {
        Message::Human(HumanMessage::new(text))
    }

    fn make_fake(responses: Vec<&str>) -> FakeListChatModel {
        FakeListChatModel::new(responses.into_iter().map(String::from).collect())
    }

    #[tokio::test]
    async fn test_token_counting_single_call() {
        let model = TokenCountingModel::new(Box::new(make_fake(vec!["Hello there"])), None);
        let msgs = vec![human("Hi")];
        let result = model._generate(&msgs, None).await.unwrap();
        assert_eq!(result.generations[0].text, "Hello there");

        let usage = model.get_usage();
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
        assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
    }

    #[tokio::test]
    async fn test_cumulative_tracking() {
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Response one", "Response two"])),
            None,
        );

        let msgs = vec![human("First question")];
        model._generate(&msgs, None).await.unwrap();
        let usage1 = model.get_usage();

        let msgs2 = vec![human("Second question")];
        model._generate(&msgs2, None).await.unwrap();
        let usage2 = model.get_usage();

        assert!(usage2.input_tokens > usage1.input_tokens);
        assert!(usage2.output_tokens > usage1.output_tokens);
        assert!(usage2.total_tokens > usage1.total_tokens);
    }

    #[tokio::test]
    async fn test_cost_estimation_with_pricing() {
        let pricing = ModelPricing::new("test-model", 2.0, 4.0);
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Hello world response"])),
            Some(pricing),
        );

        let msgs = vec![human("Test input")];
        model._generate(&msgs, None).await.unwrap();

        let usage = model.get_usage();
        assert!(usage.estimated_cost.is_some());
        assert!(usage.estimated_cost.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_reset_usage() {
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Response"])),
            None,
        );

        let msgs = vec![human("Hello")];
        model._generate(&msgs, None).await.unwrap();
        assert!(model.get_usage().total_tokens > 0);

        model.reset_usage();
        let usage = model.get_usage();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn test_last_usage_tracking() {
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Short", "A much longer response than the first one"])),
            None,
        );

        let msgs = vec![human("Q1")];
        model._generate(&msgs, None).await.unwrap();
        let last1 = model.get_last_usage();

        let msgs2 = vec![human("Q2")];
        model._generate(&msgs2, None).await.unwrap();
        let last2 = model.get_last_usage();

        // The second response is longer, so output tokens should differ
        assert!(last2.output_tokens > last1.output_tokens);
        // last_usage should reflect only the second call
        assert_ne!(last1.total_tokens, last2.total_tokens);
    }

    #[tokio::test]
    async fn test_no_pricing_returns_none_cost() {
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Response"])),
            None,
        );

        let msgs = vec![human("Hello")];
        model._generate(&msgs, None).await.unwrap();

        let usage = model.get_usage();
        assert!(usage.estimated_cost.is_none());
    }

    #[test]
    fn test_pricing_registry_lookup() {
        let registry = PricingRegistry::default();
        let pricing = registry.get_pricing("gpt-4o");
        assert!(pricing.is_some());
        let p = pricing.unwrap();
        assert_eq!(p.model_name, "gpt-4o");
        assert!(p.input_cost_per_1k > 0.0);
        assert!(p.output_cost_per_1k > 0.0);
    }

    #[test]
    fn test_pricing_registry_custom_registration() {
        let mut registry = PricingRegistry::default();
        let custom = ModelPricing::new("my-custom-model", 1.0, 2.0);
        registry.register(custom);

        let pricing = registry.get_pricing("my-custom-model");
        assert!(pricing.is_some());
        assert_eq!(pricing.unwrap().input_cost_per_1k, 1.0);
        assert_eq!(pricing.unwrap().output_cost_per_1k, 2.0);
    }

    #[test]
    fn test_default_pricing_known_models() {
        let registry = PricingRegistry::default();

        let known = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "claude-3.5-sonnet",
            "claude-3-opus",
            "claude-3-haiku",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ];

        for model in &known {
            assert!(
                registry.get_pricing(model).is_some(),
                "Missing pricing for {}",
                model
            );
        }
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let pricing = ModelPricing::new("test-model", 1.0, 2.0);
        let model = TokenCountingModel::builder(Box::new(make_fake(vec!["Built response"])))
            .with_pricing(pricing)
            .build();

        let msgs = vec![human("Test")];
        model._generate(&msgs, None).await.unwrap();

        let usage = model.get_usage();
        assert!(usage.total_tokens > 0);
        assert!(usage.estimated_cost.is_some());
    }

    #[tokio::test]
    async fn test_delegates_llm_type() {
        let model = TokenCountingModel::new(
            Box::new(make_fake(vec!["Response"])),
            None,
        );
        assert_eq!(model.llm_type(), "fake_list_chat_model");
    }

    #[test]
    fn test_pricing_registry_unknown_model_returns_none() {
        let registry = PricingRegistry::default();
        assert!(registry.get_pricing("nonexistent-model").is_none());
    }

    #[test]
    fn test_model_pricing_calculate_cost() {
        let pricing = ModelPricing::new("test", 2.0, 4.0);
        // 1000 input tokens at $2/1k = $2, 500 output tokens at $4/1k = $2
        let cost = pricing.calculate_cost(1000, 500);
        assert!((cost - 4.0).abs() < f64::EPSILON);
    }
}
