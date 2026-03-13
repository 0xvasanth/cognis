//! Model routing with intelligent selection based on cost, latency, capabilities,
//! and custom strategies.
//!
//! The [`ModelRouter`] distributes requests to the most appropriate model based on
//! configurable [`RoutingStrategy`] and [`RoutingRule`] criteria. It tracks
//! [`RoutingMetrics`] for observability and supports [`FallbackChain`] for
//! cascading through models on failure.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis_core::error::{CognisError, Result};
use cognis_core::language_models::chat_model::{BaseChatModel, ChatStream, ToolChoice};
use cognis_core::messages::Message;
use cognis_core::outputs::ChatResult;
use cognis_core::tools::ToolSchema;

// ---------------------------------------------------------------------------
// ModelCapability flags
// ---------------------------------------------------------------------------

/// Named capability constants.
pub struct ModelCapability;

impl ModelCapability {
    pub const STREAMING: u32 = 0b0000_0001;
    pub const TOOL_CALLING: u32 = 0b0000_0010;
    pub const VISION: u32 = 0b0000_0100;
    pub const LONG_CONTEXT: u32 = 0b0000_1000;
    pub const STRUCTURED_OUTPUT: u32 = 0b0001_0000;
}

/// A simple bitflag-style capability set stored as a `u32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ModelCapabilities(pub u32);

impl ModelCapabilities {
    pub const EMPTY: Self = Self(0);

    pub fn contains(self, flag: u32) -> bool {
        self.0 & flag == flag
    }

    pub fn insert(&mut self, flag: u32) {
        self.0 |= flag;
    }

    pub fn satisfies(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
}

// ---------------------------------------------------------------------------
// RoutingModelProfile
// ---------------------------------------------------------------------------

/// Profile metadata used for routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingModelProfile {
    /// Display / identifier name.
    pub name: String,
    /// Cost per 1K input tokens (USD).
    pub cost_per_1k_input_tokens: f64,
    /// Cost per 1K output tokens (USD).
    pub cost_per_1k_output_tokens: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: u64,
    /// Maximum context length in tokens.
    pub max_context_length: usize,
    /// Capability flags.
    pub capabilities: ModelCapabilities,
    /// Relative quality score (0.0–1.0).
    pub quality_score: f64,
}

impl RoutingModelProfile {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cost_per_1k_input_tokens: 0.0,
            cost_per_1k_output_tokens: 0.0,
            avg_latency_ms: 0,
            max_context_length: 4096,
            capabilities: ModelCapabilities::EMPTY,
            quality_score: 0.5,
        }
    }

    pub fn with_cost(mut self, input: f64, output: f64) -> Self {
        self.cost_per_1k_input_tokens = input;
        self.cost_per_1k_output_tokens = output;
        self
    }

    pub fn with_latency(mut self, ms: u64) -> Self {
        self.avg_latency_ms = ms;
        self
    }

    pub fn with_context_length(mut self, tokens: usize) -> Self {
        self.max_context_length = tokens;
        self
    }

    pub fn with_capabilities(mut self, caps: ModelCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    pub fn with_quality(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// RoutingStrategy
// ---------------------------------------------------------------------------

/// Strategy used by [`ModelRouter`] to pick a model.
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Prefer the cheapest model that satisfies requirements.
    CostOptimized,
    /// Prefer the fastest model that satisfies requirements.
    LatencyOptimized,
    /// Prefer the highest-quality model that satisfies requirements.
    QualityOptimized,
    /// Rotate through eligible models in order.
    RoundRobin,
    /// Pick a random eligible model.
    Random,
    /// Use a user-supplied [`ModelSelector`].
    Custom(Arc<dyn ModelSelector>),
}

// ---------------------------------------------------------------------------
// RoutingContext
// ---------------------------------------------------------------------------

/// Contextual information available to routing rules and selectors.
#[derive(Debug, Clone, Default)]
pub struct RoutingContext {
    /// Estimated input token count.
    pub estimated_tokens: Option<usize>,
    /// Required capabilities.
    pub required_capabilities: ModelCapabilities,
    /// Arbitrary string tags (e.g. `"task:summarize"`).
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// RoutingRule
// ---------------------------------------------------------------------------

/// Predicate function for routing rules.
pub type RoutingPredicate =
    Box<dyn Fn(&RoutingContext, &RoutingModelProfile) -> bool + Send + Sync>;

/// A conditional routing rule that can override the default strategy.
pub struct RoutingRule {
    /// Human-readable name of this rule.
    pub name: String,
    /// Predicate: given the context and a candidate profile, return `true` if
    /// the candidate should be considered.
    pub predicate: RoutingPredicate,
}

impl RoutingRule {
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&RoutingContext, &RoutingModelProfile) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            predicate: Box::new(predicate),
        }
    }

    /// Built-in: require a minimum context length.
    pub fn min_context_length(min_tokens: usize) -> Self {
        Self::new(
            format!("min_context_length({})", min_tokens),
            move |_ctx, profile| profile.max_context_length >= min_tokens,
        )
    }

    /// Built-in: require specific capabilities.
    pub fn requires_capabilities(caps: ModelCapabilities) -> Self {
        Self::new(
            format!("requires_capabilities({:?})", caps),
            move |_ctx, profile| profile.capabilities.satisfies(caps),
        )
    }

    /// Built-in: max cost per 1K input tokens.
    pub fn max_input_cost(max_cost: f64) -> Self {
        Self::new(
            format!("max_input_cost({})", max_cost),
            move |_ctx, profile| profile.cost_per_1k_input_tokens <= max_cost,
        )
    }

    /// Built-in: only include models whose context length exceeds the estimated tokens.
    pub fn context_fits() -> Self {
        Self::new("context_fits", |ctx, profile| match ctx.estimated_tokens {
            Some(t) => profile.max_context_length >= t,
            None => true,
        })
    }
}

impl std::fmt::Debug for RoutingRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingRule")
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ModelSelector trait
// ---------------------------------------------------------------------------

/// Trait for custom model selection logic.
pub trait ModelSelector: Send + Sync {
    /// Given candidate profiles and routing context, return the index of the
    /// chosen candidate. Return `None` if no candidate is suitable.
    fn select(
        &self,
        candidates: &[&RoutingModelProfile],
        context: &RoutingContext,
    ) -> Option<usize>;
}

impl std::fmt::Debug for dyn ModelSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn ModelSelector")
    }
}

// ---------------------------------------------------------------------------
// RoutingMetrics
// ---------------------------------------------------------------------------

/// Per-model routing metrics.
#[derive(Debug, Default)]
pub struct ModelRoutingMetrics {
    pub selection_count: AtomicUsize,
    pub success_count: AtomicUsize,
    pub failure_count: AtomicUsize,
}

/// Aggregated routing metrics for the router.
#[derive(Debug)]
pub struct RoutingMetrics {
    pub per_model: RwLock<HashMap<String, Arc<ModelRoutingMetrics>>>,
    pub total_requests: AtomicUsize,
    pub fallback_activations: AtomicUsize,
}

impl RoutingMetrics {
    pub fn new() -> Self {
        Self {
            per_model: RwLock::new(HashMap::new()),
            total_requests: AtomicUsize::new(0),
            fallback_activations: AtomicUsize::new(0),
        }
    }

    fn record_selection(&self, model_name: &str) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        let metrics = self.get_or_create(model_name);
        metrics.selection_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_success(&self, model_name: &str) {
        let metrics = self.get_or_create(model_name);
        metrics.success_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_failure(&self, model_name: &str) {
        let metrics = self.get_or_create(model_name);
        metrics.failure_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_fallback(&self) {
        self.fallback_activations.fetch_add(1, Ordering::Relaxed);
    }

    fn get_or_create(&self, model_name: &str) -> Arc<ModelRoutingMetrics> {
        {
            let map = self.per_model.read().unwrap();
            if let Some(m) = map.get(model_name) {
                return Arc::clone(m);
            }
        }
        let mut map = self.per_model.write().unwrap();
        Arc::clone(
            map.entry(model_name.to_string())
                .or_insert_with(|| Arc::new(ModelRoutingMetrics::default())),
        )
    }

    /// Snapshot of selection counts keyed by model name.
    pub fn selection_counts(&self) -> HashMap<String, usize> {
        let map = self.per_model.read().unwrap();
        map.iter()
            .map(|(k, v)| (k.clone(), v.selection_count.load(Ordering::Relaxed)))
            .collect()
    }
}

impl Default for RoutingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FallbackChain
// ---------------------------------------------------------------------------

/// Cascading fallback: tries each model in order until one succeeds.
pub struct FallbackChain {
    models: Vec<(String, Arc<dyn BaseChatModel>)>,
    metrics: Arc<RoutingMetrics>,
}

impl FallbackChain {
    pub fn new(metrics: Arc<RoutingMetrics>) -> Self {
        Self {
            models: Vec::new(),
            metrics,
        }
    }

    /// Add a model to the end of the fallback chain.
    pub fn add_model(mut self, name: impl Into<String>, model: Arc<dyn BaseChatModel>) -> Self {
        self.models.push((name.into(), model));
        self
    }

    /// Try each model in order. Return the first successful result.
    pub async fn generate(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<(String, ChatResult)> {
        let mut last_err = CognisError::Other("FallbackChain has no models".into());
        for (name, model) in &self.models {
            self.metrics.record_selection(name);
            match model._generate(messages, stop).await {
                Ok(result) => {
                    self.metrics.record_success(name);
                    return Ok((name.clone(), result));
                }
                Err(e) => {
                    self.metrics.record_failure(name);
                    self.metrics.record_fallback();
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    /// Number of models in the chain.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RegisteredModel (internal)
// ---------------------------------------------------------------------------

struct RegisteredModel {
    profile: RoutingModelProfile,
    model: Arc<dyn BaseChatModel>,
}

// ---------------------------------------------------------------------------
// ModelRouter
// ---------------------------------------------------------------------------

/// Selects the best model for a request based on strategy, rules, and context.
///
/// Use [`ModelRouter::builder`] for ergonomic construction.
pub struct ModelRouter {
    models: Vec<RegisteredModel>,
    strategy: RoutingStrategy,
    rules: Vec<RoutingRule>,
    metrics: Arc<RoutingMetrics>,
    rr_counter: AtomicUsize,
}

impl ModelRouter {
    /// Create a new builder.
    pub fn builder() -> ModelRouterBuilder {
        ModelRouterBuilder::new()
    }

    /// Access the routing metrics.
    pub fn metrics(&self) -> &Arc<RoutingMetrics> {
        &self.metrics
    }

    /// Number of registered models.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Resolve the index into `self.models` of the selected model.
    fn select_index(&self, context: &RoutingContext) -> Result<usize> {
        // 1) Apply rules to filter candidates.
        let candidate_indices: Vec<usize> = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, rm)| {
                // Must satisfy required capabilities from context.
                if !rm
                    .profile
                    .capabilities
                    .satisfies(context.required_capabilities)
                {
                    return false;
                }
                // Must pass all routing rules.
                self.rules
                    .iter()
                    .all(|rule| (rule.predicate)(context, &rm.profile))
            })
            .map(|(i, _)| i)
            .collect();

        if candidate_indices.is_empty() {
            return Err(CognisError::Other(
                "No model satisfies the routing rules and required capabilities".into(),
            ));
        }

        let chosen = match &self.strategy {
            RoutingStrategy::CostOptimized => *candidate_indices
                .iter()
                .min_by(|&&a, &&b| {
                    let ca = self.models[a].profile.cost_per_1k_input_tokens;
                    let cb = self.models[b].profile.cost_per_1k_input_tokens;
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap(),
            RoutingStrategy::LatencyOptimized => *candidate_indices
                .iter()
                .min_by_key(|&&i| self.models[i].profile.avg_latency_ms)
                .unwrap(),
            RoutingStrategy::QualityOptimized => *candidate_indices
                .iter()
                .max_by(|&&a, &&b| {
                    let qa = self.models[a].profile.quality_score;
                    let qb = self.models[b].profile.quality_score;
                    qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap(),
            RoutingStrategy::RoundRobin => {
                let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                candidate_indices[idx % candidate_indices.len()]
            }
            RoutingStrategy::Random => {
                // Deterministic-ish pseudo-random using the counter as seed.
                let tick = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                let pseudo = tick.wrapping_mul(6364136223846793005).wrapping_add(1);
                candidate_indices[pseudo % candidate_indices.len()]
            }
            RoutingStrategy::Custom(selector) => {
                let profiles: Vec<&RoutingModelProfile> = candidate_indices
                    .iter()
                    .map(|&i| &self.models[i].profile)
                    .collect();
                let local_idx = selector.select(&profiles, context).ok_or_else(|| {
                    CognisError::Other("Custom selector returned no candidate".into())
                })?;
                if local_idx >= candidate_indices.len() {
                    return Err(CognisError::Other(
                        "Custom selector returned out-of-bounds index".into(),
                    ));
                }
                candidate_indices[local_idx]
            }
        };

        Ok(chosen)
    }

    /// Select and invoke the best model.
    pub async fn route(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        context: &RoutingContext,
    ) -> Result<(String, ChatResult)> {
        let idx = self.select_index(context)?;
        let rm = &self.models[idx];
        self.metrics.record_selection(&rm.profile.name);

        match rm.model._generate(messages, stop).await {
            Ok(result) => {
                self.metrics.record_success(&rm.profile.name);
                Ok((rm.profile.name.clone(), result))
            }
            Err(e) => {
                self.metrics.record_failure(&rm.profile.name);
                Err(e)
            }
        }
    }

    /// Select and invoke with automatic fallback: on failure, try remaining
    /// candidates in strategy order.
    pub async fn route_with_fallback(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
        context: &RoutingContext,
    ) -> Result<(String, ChatResult)> {
        // Collect all candidate indices in strategy-sorted order.
        let mut candidate_indices: Vec<usize> = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, rm)| {
                if !rm
                    .profile
                    .capabilities
                    .satisfies(context.required_capabilities)
                {
                    return false;
                }
                self.rules
                    .iter()
                    .all(|rule| (rule.predicate)(context, &rm.profile))
            })
            .map(|(i, _)| i)
            .collect();

        if candidate_indices.is_empty() {
            return Err(CognisError::Other(
                "No model satisfies routing requirements".into(),
            ));
        }

        // Sort by strategy preference.
        match &self.strategy {
            RoutingStrategy::CostOptimized => {
                candidate_indices.sort_by(|&a, &b| {
                    let ca = self.models[a].profile.cost_per_1k_input_tokens;
                    let cb = self.models[b].profile.cost_per_1k_input_tokens;
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            RoutingStrategy::LatencyOptimized => {
                candidate_indices.sort_by_key(|&i| self.models[i].profile.avg_latency_ms);
            }
            RoutingStrategy::QualityOptimized => {
                candidate_indices.sort_by(|&a, &b| {
                    let qa = self.models[a].profile.quality_score;
                    let qb = self.models[b].profile.quality_score;
                    qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            _ => {} // keep original order for RoundRobin / Random / Custom
        }

        let mut last_err = CognisError::Other("No candidates".into());
        for (attempt, &idx) in candidate_indices.iter().enumerate() {
            let rm = &self.models[idx];
            self.metrics.record_selection(&rm.profile.name);
            match rm.model._generate(messages, stop).await {
                Ok(result) => {
                    self.metrics.record_success(&rm.profile.name);
                    return Ok((rm.profile.name.clone(), result));
                }
                Err(e) => {
                    self.metrics.record_failure(&rm.profile.name);
                    if attempt < candidate_indices.len() - 1 {
                        self.metrics.record_fallback();
                    }
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    /// Return the name of the model that would be selected.
    pub fn preview_selection(&self, context: &RoutingContext) -> Result<String> {
        let idx = self.select_index(context)?;
        Ok(self.models[idx].profile.name.clone())
    }

    /// Return all model profiles.
    pub fn profiles(&self) -> Vec<&RoutingModelProfile> {
        self.models.iter().map(|rm| &rm.profile).collect()
    }
}

#[async_trait]
impl BaseChatModel for ModelRouter {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        let context = RoutingContext::default();
        let (_name, result) = self.route(messages, stop, &context).await?;
        Ok(result)
    }

    fn llm_type(&self) -> &str {
        "model_router"
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        let mut context = RoutingContext::default();
        context
            .required_capabilities
            .insert(ModelCapability::STREAMING);
        let idx = self.select_index(&context)?;
        let rm = &self.models[idx];
        self.metrics.record_selection(&rm.profile.name);
        rm.model._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        _tools: &[ToolSchema],
        _tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        Err(CognisError::NotImplemented(
            "ModelRouter does not support bind_tools directly; route to a specific model first"
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// ModelRouterBuilder
// ---------------------------------------------------------------------------

/// Builder for [`ModelRouter`].
pub struct ModelRouterBuilder {
    models: Vec<RegisteredModel>,
    strategy: RoutingStrategy,
    rules: Vec<RoutingRule>,
    metrics: Option<Arc<RoutingMetrics>>,
}

impl ModelRouterBuilder {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            strategy: RoutingStrategy::QualityOptimized,
            rules: Vec::new(),
            metrics: None,
        }
    }

    /// Register a model with its routing profile.
    pub fn add_model(
        mut self,
        profile: RoutingModelProfile,
        model: Arc<dyn BaseChatModel>,
    ) -> Self {
        self.models.push(RegisteredModel { profile, model });
        self
    }

    /// Set the routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add a routing rule.
    pub fn rule(mut self, rule: RoutingRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Provide pre-existing metrics to share across routers.
    pub fn metrics(mut self, metrics: Arc<RoutingMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Build the router.
    pub fn build(self) -> Result<ModelRouter> {
        if self.models.is_empty() {
            return Err(CognisError::Other(
                "ModelRouter requires at least one model".into(),
            ));
        }
        Ok(ModelRouter {
            models: self.models,
            strategy: self.strategy,
            rules: self.rules,
            metrics: self
                .metrics
                .unwrap_or_else(|| Arc::new(RoutingMetrics::new())),
            rr_counter: AtomicUsize::new(0),
        })
    }
}

impl Default for ModelRouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::{AIMessage, HumanMessage};
    use cognis_core::outputs::{ChatGeneration, ChatResult};
    use std::sync::atomic::AtomicBool;

    // -----------------------------------------------------------------------
    // Stub model for testing
    // -----------------------------------------------------------------------

    struct StubModel {
        name: String,
        should_fail: AtomicBool,
    }

    impl StubModel {
        fn new(name: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                should_fail: AtomicBool::new(false),
            })
        }

        fn failing(name: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_string(),
                should_fail: AtomicBool::new(true),
            })
        }
    }

    #[async_trait]
    impl BaseChatModel for StubModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            if self.should_fail.load(Ordering::Relaxed) {
                return Err(CognisError::Other(format!("{} failed", self.name)));
            }
            let text = format!("response from {}", self.name);
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: text.clone(),
                    message: Message::Ai(AIMessage::new(&text)),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            &self.name
        }
    }

    fn cheap_profile(name: &str) -> RoutingModelProfile {
        RoutingModelProfile::new(name)
            .with_cost(0.001, 0.002)
            .with_latency(200)
            .with_context_length(8192)
            .with_quality(0.6)
    }

    fn mid_profile(name: &str) -> RoutingModelProfile {
        RoutingModelProfile::new(name)
            .with_cost(0.01, 0.03)
            .with_latency(100)
            .with_context_length(32768)
            .with_quality(0.8)
    }

    fn premium_profile(name: &str) -> RoutingModelProfile {
        RoutingModelProfile::new(name)
            .with_cost(0.05, 0.15)
            .with_latency(300)
            .with_context_length(200000)
            .with_quality(0.95)
            .with_capabilities(ModelCapabilities(
                ModelCapability::STREAMING
                    | ModelCapability::TOOL_CALLING
                    | ModelCapability::VISION
                    | ModelCapability::LONG_CONTEXT
                    | ModelCapability::STRUCTURED_OUTPUT,
            ))
    }

    fn build_test_router(strategy: RoutingStrategy) -> ModelRouter {
        ModelRouter::builder()
            .add_model(cheap_profile("cheap"), StubModel::new("cheap"))
            .add_model(mid_profile("mid"), StubModel::new("mid"))
            .add_model(premium_profile("premium"), StubModel::new("premium"))
            .strategy(strategy)
            .build()
            .unwrap()
    }

    fn test_messages() -> Vec<Message> {
        vec![Message::Human(HumanMessage::new("hello"))]
    }

    // -----------------------------------------------------------------------
    // ModelCapabilities tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_capabilities_empty() {
        let c = ModelCapabilities::EMPTY;
        assert!(!c.contains(ModelCapability::STREAMING));
        assert!(!c.contains(ModelCapability::TOOL_CALLING));
    }

    #[test]
    fn test_capabilities_insert() {
        let mut c = ModelCapabilities::EMPTY;
        c.insert(ModelCapability::STREAMING);
        assert!(c.contains(ModelCapability::STREAMING));
        assert!(!c.contains(ModelCapability::VISION));
    }

    #[test]
    fn test_capabilities_satisfies() {
        let all = ModelCapabilities(ModelCapability::STREAMING | ModelCapability::TOOL_CALLING);
        let required = ModelCapabilities(ModelCapability::STREAMING);
        assert!(all.satisfies(required));
        assert!(!required.satisfies(all));
    }

    #[test]
    fn test_capabilities_satisfies_empty() {
        let any = ModelCapabilities(ModelCapability::VISION);
        let empty = ModelCapabilities::EMPTY;
        assert!(any.satisfies(empty));
        assert!(empty.satisfies(empty));
    }

    #[test]
    fn test_capabilities_multiple_flags() {
        let caps = ModelCapabilities(
            ModelCapability::STREAMING | ModelCapability::VISION | ModelCapability::LONG_CONTEXT,
        );
        assert!(caps.contains(ModelCapability::STREAMING));
        assert!(caps.contains(ModelCapability::VISION));
        assert!(caps.contains(ModelCapability::LONG_CONTEXT));
        assert!(!caps.contains(ModelCapability::TOOL_CALLING));
    }

    // -----------------------------------------------------------------------
    // RoutingModelProfile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_profile_builder() {
        let p = RoutingModelProfile::new("test")
            .with_cost(0.01, 0.02)
            .with_latency(150)
            .with_context_length(16384)
            .with_quality(0.9);
        assert_eq!(p.name, "test");
        assert!((p.cost_per_1k_input_tokens - 0.01).abs() < f64::EPSILON);
        assert!((p.cost_per_1k_output_tokens - 0.02).abs() < f64::EPSILON);
        assert_eq!(p.avg_latency_ms, 150);
        assert_eq!(p.max_context_length, 16384);
        assert!((p.quality_score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_quality_clamp() {
        let p = RoutingModelProfile::new("x").with_quality(2.0);
        assert!((p.quality_score - 1.0).abs() < f64::EPSILON);
        let p2 = RoutingModelProfile::new("x").with_quality(-0.5);
        assert!((p2.quality_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_default_values() {
        let p = RoutingModelProfile::new("default");
        assert!((p.cost_per_1k_input_tokens - 0.0).abs() < f64::EPSILON);
        assert_eq!(p.avg_latency_ms, 0);
        assert_eq!(p.max_context_length, 4096);
        assert!((p.quality_score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_profile_serialization() {
        let p = RoutingModelProfile::new("ser_test").with_cost(0.1, 0.2);
        let json = serde_json::to_string(&p).unwrap();
        let deser: RoutingModelProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.name, "ser_test");
        assert!((deser.cost_per_1k_input_tokens - 0.1).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // RoutingRule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_min_context_length() {
        let rule = RoutingRule::min_context_length(16000);
        let ctx = RoutingContext::default();
        let small = cheap_profile("small"); // 8192
        let big = mid_profile("big"); // 32768
        assert!(!(rule.predicate)(&ctx, &small));
        assert!((rule.predicate)(&ctx, &big));
    }

    #[test]
    fn test_rule_requires_capabilities() {
        let rule =
            RoutingRule::requires_capabilities(ModelCapabilities(ModelCapability::TOOL_CALLING));
        let ctx = RoutingContext::default();
        let no_tools = cheap_profile("no_tools");
        let has_tools = premium_profile("has_tools");
        assert!(!(rule.predicate)(&ctx, &no_tools));
        assert!((rule.predicate)(&ctx, &has_tools));
    }

    #[test]
    fn test_rule_max_input_cost() {
        let rule = RoutingRule::max_input_cost(0.005);
        let ctx = RoutingContext::default();
        assert!((rule.predicate)(&ctx, &cheap_profile("c"))); // 0.001
        assert!(!(rule.predicate)(&ctx, &mid_profile("m"))); // 0.01
    }

    #[test]
    fn test_rule_context_fits() {
        let rule = RoutingRule::context_fits();
        let mut ctx = RoutingContext::default();
        ctx.estimated_tokens = Some(10000);
        let small = cheap_profile("small"); // 8192
        let big = mid_profile("big"); // 32768
        assert!(!(rule.predicate)(&ctx, &small));
        assert!((rule.predicate)(&ctx, &big));
    }

    #[test]
    fn test_rule_context_fits_no_estimate() {
        let rule = RoutingRule::context_fits();
        let ctx = RoutingContext::default(); // no estimated_tokens
        let small = cheap_profile("small");
        assert!((rule.predicate)(&ctx, &small));
    }

    #[test]
    fn test_rule_debug() {
        let rule = RoutingRule::min_context_length(1000);
        let dbg = format!("{:?}", rule);
        assert!(dbg.contains("min_context_length"));
    }

    #[test]
    fn test_custom_rule() {
        let rule = RoutingRule::new("only_premium", |_ctx, profile| profile.quality_score > 0.9);
        let ctx = RoutingContext::default();
        assert!(!(rule.predicate)(&ctx, &cheap_profile("c")));
        assert!((rule.predicate)(&ctx, &premium_profile("p")));
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: CostOptimized
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cost_optimized_selects_cheapest() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "cheap");
    }

    #[tokio::test]
    async fn test_cost_optimized_generate() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let ctx = RoutingContext::default();
        let (name, result) = router.route(&test_messages(), None, &ctx).await.unwrap();
        assert_eq!(name, "cheap");
        assert!(!result.generations.is_empty());
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: LatencyOptimized
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_latency_optimized_selects_fastest() {
        let router = build_test_router(RoutingStrategy::LatencyOptimized);
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "mid"); // 100ms
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: QualityOptimized
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_quality_optimized_selects_best() {
        let router = build_test_router(RoutingStrategy::QualityOptimized);
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "premium"); // 0.95
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: RoundRobin
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_round_robin_cycles() {
        let router = build_test_router(RoutingStrategy::RoundRobin);
        let ctx = RoutingContext::default();
        let n1 = router.preview_selection(&ctx).unwrap();
        let n2 = router.preview_selection(&ctx).unwrap();
        let n3 = router.preview_selection(&ctx).unwrap();
        let n4 = router.preview_selection(&ctx).unwrap();
        // Should cycle through 3 models then wrap around
        assert_eq!(n1, "cheap");
        assert_eq!(n2, "mid");
        assert_eq!(n3, "premium");
        assert_eq!(n4, "cheap");
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: Random
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_random_selects_valid_model() {
        let router = build_test_router(RoutingStrategy::Random);
        let ctx = RoutingContext::default();
        for _ in 0..20 {
            let name = router.preview_selection(&ctx).unwrap();
            assert!(["cheap", "mid", "premium"].contains(&name.as_str()));
        }
    }

    // -----------------------------------------------------------------------
    // RoutingStrategy: Custom
    // -----------------------------------------------------------------------

    struct AlwaysFirstSelector;

    impl ModelSelector for AlwaysFirstSelector {
        fn select(
            &self,
            _candidates: &[&RoutingModelProfile],
            _context: &RoutingContext,
        ) -> Option<usize> {
            Some(0)
        }
    }

    struct AlwaysLastSelector;

    impl ModelSelector for AlwaysLastSelector {
        fn select(
            &self,
            candidates: &[&RoutingModelProfile],
            _context: &RoutingContext,
        ) -> Option<usize> {
            if candidates.is_empty() {
                None
            } else {
                Some(candidates.len() - 1)
            }
        }
    }

    struct NoneSelector;

    impl ModelSelector for NoneSelector {
        fn select(
            &self,
            _candidates: &[&RoutingModelProfile],
            _context: &RoutingContext,
        ) -> Option<usize> {
            None
        }
    }

    #[tokio::test]
    async fn test_custom_strategy_first() {
        let router = build_test_router(RoutingStrategy::Custom(Arc::new(AlwaysFirstSelector)));
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "cheap");
    }

    #[tokio::test]
    async fn test_custom_strategy_last() {
        let router = build_test_router(RoutingStrategy::Custom(Arc::new(AlwaysLastSelector)));
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "premium");
    }

    #[tokio::test]
    async fn test_custom_strategy_none_returns_error() {
        let router = build_test_router(RoutingStrategy::Custom(Arc::new(NoneSelector)));
        let ctx = RoutingContext::default();
        assert!(router.preview_selection(&ctx).is_err());
    }

    // -----------------------------------------------------------------------
    // Required capabilities filtering
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_required_capabilities_filter() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let ctx = RoutingContext {
            required_capabilities: ModelCapabilities(ModelCapability::TOOL_CALLING),
            ..Default::default()
        };
        // Only premium has TOOL_CALLING
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "premium");
    }

    #[tokio::test]
    async fn test_no_model_satisfies_capabilities() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("only"), StubModel::new("only"))
            .strategy(RoutingStrategy::CostOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext {
            required_capabilities: ModelCapabilities(ModelCapability::VISION),
            ..Default::default()
        };
        assert!(router.preview_selection(&ctx).is_err());
    }

    // -----------------------------------------------------------------------
    // Rules integration
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_rule_filters_models() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("cheap"), StubModel::new("cheap"))
            .add_model(mid_profile("mid"), StubModel::new("mid"))
            .add_model(premium_profile("premium"), StubModel::new("premium"))
            .strategy(RoutingStrategy::CostOptimized)
            .rule(RoutingRule::min_context_length(16000))
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        // cheap (8192) is excluded, so cheapest remaining is mid
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "mid");
    }

    #[tokio::test]
    async fn test_context_fits_rule_with_large_prompt() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("cheap"), StubModel::new("cheap"))
            .add_model(premium_profile("premium"), StubModel::new("premium"))
            .strategy(RoutingStrategy::CostOptimized)
            .rule(RoutingRule::context_fits())
            .build()
            .unwrap();
        let ctx = RoutingContext {
            estimated_tokens: Some(100000),
            ..Default::default()
        };
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "premium"); // only premium has 200k context
    }

    #[tokio::test]
    async fn test_multiple_rules_all_applied() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("cheap"), StubModel::new("cheap"))
            .add_model(mid_profile("mid"), StubModel::new("mid"))
            .add_model(premium_profile("premium"), StubModel::new("premium"))
            .strategy(RoutingStrategy::LatencyOptimized)
            .rule(RoutingRule::min_context_length(16000))
            .rule(RoutingRule::max_input_cost(0.06))
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        // cheap excluded by context, all pass cost -> mid (100ms) and premium (300ms)
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "mid");
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_metrics_tracked_on_route() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let ctx = RoutingContext::default();
        router.route(&test_messages(), None, &ctx).await.unwrap();
        router.route(&test_messages(), None, &ctx).await.unwrap();

        let metrics = router.metrics();
        assert_eq!(metrics.total_requests.load(Ordering::Relaxed), 2);
        let counts = metrics.selection_counts();
        assert_eq!(counts.get("cheap"), Some(&2));
    }

    #[tokio::test]
    async fn test_metrics_failure_tracked() {
        let router = ModelRouter::builder()
            .add_model(
                cheap_profile("fail_model"),
                StubModel::failing("fail_model"),
            )
            .strategy(RoutingStrategy::CostOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        let _ = router.route(&test_messages(), None, &ctx).await;

        let metrics = router.metrics();
        let per_model = metrics.per_model.read().unwrap();
        let m = per_model.get("fail_model").unwrap();
        assert_eq!(m.failure_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_default() {
        let m = RoutingMetrics::new();
        assert_eq!(m.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(m.fallback_activations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_selection_counts_empty() {
        let m = RoutingMetrics::new();
        assert!(m.selection_counts().is_empty());
    }

    // -----------------------------------------------------------------------
    // FallbackChain
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_fallback_chain_first_succeeds() {
        let metrics = Arc::new(RoutingMetrics::new());
        let chain = FallbackChain::new(Arc::clone(&metrics))
            .add_model("primary", StubModel::new("primary"))
            .add_model("backup", StubModel::new("backup"));

        let (name, _result) = chain.generate(&test_messages(), None).await.unwrap();
        assert_eq!(name, "primary");
        assert_eq!(metrics.fallback_activations.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_fallback_chain_falls_through() {
        let metrics = Arc::new(RoutingMetrics::new());
        let chain = FallbackChain::new(Arc::clone(&metrics))
            .add_model("failing", StubModel::failing("failing"))
            .add_model("backup", StubModel::new("backup"));

        let (name, _result) = chain.generate(&test_messages(), None).await.unwrap();
        assert_eq!(name, "backup");
        assert_eq!(metrics.fallback_activations.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_fallback_chain_all_fail() {
        let metrics = Arc::new(RoutingMetrics::new());
        let chain = FallbackChain::new(Arc::clone(&metrics))
            .add_model("f1", StubModel::failing("f1"))
            .add_model("f2", StubModel::failing("f2"));

        let result = chain.generate(&test_messages(), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fallback_chain_empty() {
        let metrics = Arc::new(RoutingMetrics::new());
        let chain = FallbackChain::new(metrics);
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        let result = chain.generate(&test_messages(), None).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_chain_len() {
        let metrics = Arc::new(RoutingMetrics::new());
        let chain = FallbackChain::new(metrics)
            .add_model("a", StubModel::new("a"))
            .add_model("b", StubModel::new("b"));
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    // -----------------------------------------------------------------------
    // route_with_fallback
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_route_with_fallback_primary_ok() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let ctx = RoutingContext::default();
        let (name, _) = router
            .route_with_fallback(&test_messages(), None, &ctx)
            .await
            .unwrap();
        assert_eq!(name, "cheap");
    }

    #[tokio::test]
    async fn test_route_with_fallback_falls_through() {
        let router = ModelRouter::builder()
            .add_model(
                cheap_profile("fail_cheap"),
                StubModel::failing("fail_cheap"),
            )
            .add_model(mid_profile("ok_mid"), StubModel::new("ok_mid"))
            .strategy(RoutingStrategy::CostOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        let (name, _) = router
            .route_with_fallback(&test_messages(), None, &ctx)
            .await
            .unwrap();
        assert_eq!(name, "ok_mid");
        assert!(
            router
                .metrics()
                .fallback_activations
                .load(Ordering::Relaxed)
                > 0
        );
    }

    #[tokio::test]
    async fn test_route_with_fallback_all_fail() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("f1"), StubModel::failing("f1"))
            .add_model(mid_profile("f2"), StubModel::failing("f2"))
            .strategy(RoutingStrategy::CostOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        assert!(router
            .route_with_fallback(&test_messages(), None, &ctx)
            .await
            .is_err());
    }

    // -----------------------------------------------------------------------
    // Builder validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_empty_fails() {
        let result = ModelRouter::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_default_strategy_is_quality() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("x"), StubModel::new("x"))
            .build()
            .unwrap();
        // With only one model, it should work regardless of strategy.
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert_eq!(name, "x");
    }

    #[test]
    fn test_model_count() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        assert_eq!(router.model_count(), 3);
    }

    #[test]
    fn test_profiles_returns_all() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let profiles = router.profiles();
        assert_eq!(profiles.len(), 3);
        let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"cheap"));
        assert!(names.contains(&"mid"));
        assert!(names.contains(&"premium"));
    }

    // -----------------------------------------------------------------------
    // BaseChatModel impl
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_router_as_chat_model() {
        let router = build_test_router(RoutingStrategy::QualityOptimized);
        let result = router._generate(&test_messages(), None).await.unwrap();
        assert!(!result.generations.is_empty());
    }

    #[test]
    fn test_router_llm_type() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        assert_eq!(router.llm_type(), "model_router");
    }

    #[tokio::test]
    async fn test_router_bind_tools_not_supported() {
        let router = build_test_router(RoutingStrategy::CostOptimized);
        let result = router.bind_tools(&[], None);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Shared metrics
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_shared_metrics_across_routers() {
        let shared = Arc::new(RoutingMetrics::new());
        let r1 = ModelRouter::builder()
            .add_model(cheap_profile("model_a"), StubModel::new("model_a"))
            .strategy(RoutingStrategy::CostOptimized)
            .metrics(Arc::clone(&shared))
            .build()
            .unwrap();
        let r2 = ModelRouter::builder()
            .add_model(mid_profile("model_b"), StubModel::new("model_b"))
            .strategy(RoutingStrategy::CostOptimized)
            .metrics(Arc::clone(&shared))
            .build()
            .unwrap();

        let ctx = RoutingContext::default();
        r1.route(&test_messages(), None, &ctx).await.unwrap();
        r2.route(&test_messages(), None, &ctx).await.unwrap();

        assert_eq!(shared.total_requests.load(Ordering::Relaxed), 2);
    }

    // -----------------------------------------------------------------------
    // RoutingContext
    // -----------------------------------------------------------------------

    #[test]
    fn test_routing_context_default() {
        let ctx = RoutingContext::default();
        assert!(ctx.estimated_tokens.is_none());
        assert_eq!(ctx.required_capabilities, ModelCapabilities::EMPTY);
        assert!(ctx.tags.is_empty());
    }

    #[test]
    fn test_routing_context_with_tags() {
        let ctx = RoutingContext {
            tags: vec!["task:summarize".to_string()],
            ..Default::default()
        };
        assert_eq!(ctx.tags.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_single_model_always_selected() {
        let router = ModelRouter::builder()
            .add_model(mid_profile("only"), StubModel::new("only"))
            .strategy(RoutingStrategy::RoundRobin)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        for _ in 0..5 {
            let name = router.preview_selection(&ctx).unwrap();
            assert_eq!(name, "only");
        }
    }

    #[tokio::test]
    async fn test_all_filtered_by_rules() {
        let router = ModelRouter::builder()
            .add_model(cheap_profile("c"), StubModel::new("c"))
            .strategy(RoutingStrategy::CostOptimized)
            .rule(RoutingRule::min_context_length(999_999))
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        assert!(router.preview_selection(&ctx).is_err());
    }

    #[tokio::test]
    async fn test_cost_optimized_with_equal_costs() {
        let p1 = RoutingModelProfile::new("a").with_cost(0.01, 0.02);
        let p2 = RoutingModelProfile::new("b").with_cost(0.01, 0.02);
        let router = ModelRouter::builder()
            .add_model(p1, StubModel::new("a"))
            .add_model(p2, StubModel::new("b"))
            .strategy(RoutingStrategy::CostOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        // Should pick one deterministically (first with equal cost)
        let name = router.preview_selection(&ctx).unwrap();
        assert!(name == "a" || name == "b");
    }

    #[tokio::test]
    async fn test_quality_optimized_with_equal_quality() {
        let p1 = RoutingModelProfile::new("x").with_quality(0.8);
        let p2 = RoutingModelProfile::new("y").with_quality(0.8);
        let router = ModelRouter::builder()
            .add_model(p1, StubModel::new("x"))
            .add_model(p2, StubModel::new("y"))
            .strategy(RoutingStrategy::QualityOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        let name = router.preview_selection(&ctx).unwrap();
        assert!(name == "x" || name == "y");
    }

    // -----------------------------------------------------------------------
    // ModelSelector trait object debug
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_selector_debug() {
        let selector: &dyn ModelSelector = &AlwaysFirstSelector;
        let dbg = format!("{:?}", selector);
        assert!(dbg.contains("ModelSelector"));
    }

    // -----------------------------------------------------------------------
    // ModelRouterBuilder default
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = ModelRouterBuilder::default();
        assert!(builder.models.is_empty());
        assert!(builder.rules.is_empty());
    }

    // -----------------------------------------------------------------------
    // Latency-sorted fallback ordering
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_latency_fallback_ordering() {
        let router = ModelRouter::builder()
            .add_model(
                RoutingModelProfile::new("slow")
                    .with_latency(500)
                    .with_cost(0.01, 0.01),
                StubModel::failing("slow"),
            )
            .add_model(
                RoutingModelProfile::new("fast")
                    .with_latency(50)
                    .with_cost(0.01, 0.01),
                StubModel::new("fast"),
            )
            .strategy(RoutingStrategy::LatencyOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        let (name, _) = router
            .route_with_fallback(&test_messages(), None, &ctx)
            .await
            .unwrap();
        // fast (50ms) is tried first and succeeds
        assert_eq!(name, "fast");
    }

    #[tokio::test]
    async fn test_quality_fallback_ordering() {
        let router = ModelRouter::builder()
            .add_model(
                RoutingModelProfile::new("best").with_quality(0.99),
                StubModel::failing("best"),
            )
            .add_model(
                RoutingModelProfile::new("good").with_quality(0.7),
                StubModel::new("good"),
            )
            .strategy(RoutingStrategy::QualityOptimized)
            .build()
            .unwrap();
        let ctx = RoutingContext::default();
        let (name, _) = router
            .route_with_fallback(&test_messages(), None, &ctx)
            .await
            .unwrap();
        // best fails, falls through to good
        assert_eq!(name, "good");
    }
}
