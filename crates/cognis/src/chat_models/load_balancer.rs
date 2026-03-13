//! Load-balanced chat model wrapper.
//!
//! Distributes requests across multiple chat model instances using configurable
//! strategies (round-robin, random, least-latency, weighted round-robin) with
//! automatic health tracking and failover.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use async_trait::async_trait;

use cognis_core::error::{Result, CognisError};
use cognis_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use cognis_core::messages::Message;
use cognis_core::outputs::ChatResult;
use cognis_core::tools::ToolSchema;

/// Strategy for selecting which model to route a request to.
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Rotate through models sequentially.
    RoundRobin,
    /// Pick a model using a pseudo-random selection.
    Random,
    /// Pick the model with the lowest average latency.
    LeastLatency,
    /// Weighted round-robin distribution. Each weight determines the
    /// relative share of requests a model receives.
    WeightedRoundRobin(Vec<u32>),
}

/// Health metrics for a single model instance.
pub struct ModelHealth {
    /// Whether the model is considered healthy.
    pub is_healthy: AtomicBool,
    /// Total number of requests sent to this model.
    pub total_requests: AtomicUsize,
    /// Number of failed requests.
    pub failed_requests: AtomicUsize,
    /// Average latency in milliseconds (exponential moving average).
    pub avg_latency_ms: AtomicU64,
    /// Last error message, if any.
    pub last_error: RwLock<Option<String>>,
    /// Timestamp of the last health check or request.
    pub last_check: RwLock<Instant>,
    /// Consecutive failure count (used for unhealthy threshold).
    consecutive_failures: AtomicUsize,
}

impl ModelHealth {
    fn new() -> Self {
        Self {
            is_healthy: AtomicBool::new(true),
            total_requests: AtomicUsize::new(0),
            failed_requests: AtomicUsize::new(0),
            avg_latency_ms: AtomicU64::new(0),
            last_error: RwLock::new(None),
            last_check: RwLock::new(Instant::now()),
            consecutive_failures: AtomicUsize::new(0),
        }
    }

    fn record_success(&self, latency_ms: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.is_healthy.store(true, Ordering::Relaxed);

        // Exponential moving average: new_avg = old_avg * 0.8 + sample * 0.2
        let old = self.avg_latency_ms.load(Ordering::Relaxed);
        let new_avg = if old == 0 {
            latency_ms
        } else {
            (old * 4 + latency_ms) / 5
        };
        self.avg_latency_ms.store(new_avg, Ordering::Relaxed);

        if let Ok(mut last) = self.last_check.write() {
            *last = Instant::now();
        }
    }

    fn record_failure(&self, error: &CognisError, unhealthy_threshold: usize) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        let consecutive = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;

        if consecutive >= unhealthy_threshold {
            self.is_healthy.store(false, Ordering::Relaxed);
        }

        if let Ok(mut last_err) = self.last_error.write() {
            *last_err = Some(error.to_string());
        }
        if let Ok(mut last) = self.last_check.write() {
            *last = Instant::now();
        }
    }
}

/// A snapshot of a model's health status.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Index of the model in the load balancer's model list.
    pub index: usize,
    /// Whether the model is currently healthy.
    pub is_healthy: bool,
    /// Total requests sent to this model.
    pub total_requests: usize,
    /// Total failed requests.
    pub failed_requests: usize,
    /// Average latency in milliseconds.
    pub avg_latency_ms: u64,
    /// Error rate as a fraction (0.0 to 1.0).
    pub error_rate: f64,
}

/// A chat model that distributes requests across multiple backing models.
///
/// Supports round-robin, random, least-latency, and weighted round-robin
/// strategies. Automatically tracks model health and retries on the next
/// available model when a request fails.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::chat_models::load_balancer::{LoadBalancedChatModel, LoadBalancingStrategy};
///
/// let lb = LoadBalancerBuilder::new()
///     .add_model(Arc::new(model_a))
///     .add_model(Arc::new(model_b))
///     .strategy(LoadBalancingStrategy::RoundRobin)
///     .build()
///     .unwrap();
/// ```
pub struct LoadBalancedChatModel {
    models: Vec<Arc<dyn BaseChatModel>>,
    strategy: LoadBalancingStrategy,
    health: Vec<ModelHealth>,
    /// Counter for round-robin strategies.
    counter: AtomicUsize,
    /// Number of consecutive failures before marking unhealthy.
    unhealthy_threshold: usize,
}

impl LoadBalancedChatModel {
    /// Create a new load-balanced chat model.
    ///
    /// # Arguments
    /// * `models` - The backing chat model instances.
    /// * `strategy` - The load balancing strategy to use.
    /// * `unhealthy_threshold` - Number of consecutive failures before a model
    ///   is marked unhealthy. Defaults to 3.
    pub fn new(
        models: Vec<Arc<dyn BaseChatModel>>,
        strategy: LoadBalancingStrategy,
        unhealthy_threshold: usize,
    ) -> std::result::Result<Self, String> {
        if models.is_empty() {
            return Err("At least one model is required".into());
        }
        if let LoadBalancingStrategy::WeightedRoundRobin(ref weights) = strategy {
            if weights.len() != models.len() {
                return Err(format!(
                    "Weight count ({}) must match model count ({})",
                    weights.len(),
                    models.len()
                ));
            }
            if weights.iter().all(|w| *w == 0) {
                return Err("At least one weight must be non-zero".into());
            }
        }

        let health: Vec<ModelHealth> = (0..models.len()).map(|_| ModelHealth::new()).collect();

        Ok(Self {
            models,
            strategy,
            health,
            counter: AtomicUsize::new(0),
            unhealthy_threshold,
        })
    }

    /// Returns a health report for all models.
    pub fn get_health_report(&self) -> Vec<HealthReport> {
        self.health
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let total = h.total_requests.load(Ordering::Relaxed);
                let failed = h.failed_requests.load(Ordering::Relaxed);
                let error_rate = if total > 0 {
                    failed as f64 / total as f64
                } else {
                    0.0
                };
                HealthReport {
                    index: i,
                    is_healthy: h.is_healthy.load(Ordering::Relaxed),
                    total_requests: total,
                    failed_requests: failed,
                    avg_latency_ms: h.avg_latency_ms.load(Ordering::Relaxed),
                    error_rate,
                }
            })
            .collect()
    }

    /// Select the next model index based on the current strategy, skipping
    /// unhealthy models when possible.
    fn select_model_index(&self) -> usize {
        let n = self.models.len();
        let healthy_indices: Vec<usize> = (0..n)
            .filter(|i| self.health[*i].is_healthy.load(Ordering::Relaxed))
            .collect();

        // If no healthy models, fall back to all models.
        let candidates = if healthy_indices.is_empty() {
            (0..n).collect::<Vec<_>>()
        } else {
            healthy_indices
        };

        match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed);
                candidates[idx % candidates.len()]
            }
            LoadBalancingStrategy::Random => {
                // Simple pseudo-random using current time nanos and counter.
                let seed = {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as usize;
                    let cnt = self.counter.fetch_add(1, Ordering::Relaxed);
                    now.wrapping_add(cnt).wrapping_mul(6364136223846793005)
                };
                candidates[seed % candidates.len()]
            }
            LoadBalancingStrategy::LeastLatency => {
                let mut best_idx = candidates[0];
                let mut best_latency = self.health[best_idx].avg_latency_ms.load(Ordering::Relaxed);
                for &i in &candidates[1..] {
                    let lat = self.health[i].avg_latency_ms.load(Ordering::Relaxed);
                    if lat < best_latency {
                        best_latency = lat;
                        best_idx = i;
                    }
                }
                best_idx
            }
            LoadBalancingStrategy::WeightedRoundRobin(weights) => {
                // Build cumulative weights for healthy candidates only.
                let candidate_weights: Vec<u32> = candidates.iter().map(|&i| weights[i]).collect();
                let total_weight: u32 = candidate_weights.iter().sum();
                if total_weight == 0 {
                    return candidates[0];
                }
                let tick = self.counter.fetch_add(1, Ordering::Relaxed) as u32 % total_weight;
                let mut cumulative = 0u32;
                for (j, &w) in candidate_weights.iter().enumerate() {
                    cumulative += w;
                    if tick < cumulative {
                        return candidates[j];
                    }
                }
                *candidates.last().unwrap()
            }
        }
    }
}

#[async_trait]
impl BaseChatModel for LoadBalancedChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        let n = self.models.len();
        let start_idx = self.select_model_index();
        let mut last_error = None;

        for attempt in 0..n {
            let idx = (start_idx + attempt) % n;
            let start_time = Instant::now();

            match self.models[idx]._generate(messages, stop).await {
                Ok(result) => {
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    self.health[idx].record_success(elapsed);
                    return Ok(result);
                }
                Err(e) => {
                    self.health[idx].record_failure(&e, self.unhealthy_threshold);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| CognisError::Other("All models failed".into())))
    }

    fn llm_type(&self) -> &str {
        "load_balanced"
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        let n = self.models.len();
        let start_idx = self.select_model_index();
        let mut last_error = None;

        for attempt in 0..n {
            let idx = (start_idx + attempt) % n;
            let start_time = Instant::now();

            match self.models[idx]._stream(messages, stop).await {
                Ok(stream) => {
                    let elapsed = start_time.elapsed().as_millis() as u64;
                    self.health[idx].record_success(elapsed);
                    return Ok(stream);
                }
                Err(e) => {
                    self.health[idx].record_failure(&e, self.unhealthy_threshold);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| CognisError::Other("All models failed".into())))
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        // Try to bind tools on the first available model.
        for model in &self.models {
            if let Ok(bound) = model.bind_tools(tools, tool_choice.clone()) {
                return Ok(bound);
            }
        }
        Err(CognisError::NotImplemented(
            "No model in the load balancer supports tool binding".into(),
        ))
    }

    fn profile(&self) -> ModelProfile {
        // Return profile from first model.
        if let Some(m) = self.models.first() {
            m.profile()
        } else {
            ModelProfile::default()
        }
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        if let Some(m) = self.models.first() {
            m.get_num_tokens_from_messages(messages)
        } else {
            0
        }
    }
}

/// Builder for constructing a [`LoadBalancedChatModel`].
///
/// # Example
///
/// ```rust,ignore
/// let lb = LoadBalancerBuilder::new()
///     .add_model(Arc::new(model_a))
///     .add_model(Arc::new(model_b))
///     .strategy(LoadBalancingStrategy::LeastLatency)
///     .unhealthy_threshold(5)
///     .build()
///     .unwrap();
/// ```
pub struct LoadBalancerBuilder {
    models: Vec<Arc<dyn BaseChatModel>>,
    strategy: LoadBalancingStrategy,
    unhealthy_threshold: usize,
}

impl LoadBalancerBuilder {
    /// Create a new builder with default settings (round-robin, threshold 3).
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            strategy: LoadBalancingStrategy::RoundRobin,
            unhealthy_threshold: 3,
        }
    }

    /// Add a model to the load balancer.
    pub fn add_model(mut self, model: Arc<dyn BaseChatModel>) -> Self {
        self.models.push(model);
        self
    }

    /// Set the load balancing strategy.
    pub fn strategy(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the number of consecutive failures before a model is marked unhealthy.
    pub fn unhealthy_threshold(mut self, threshold: usize) -> Self {
        self.unhealthy_threshold = threshold;
        self
    }

    /// Build the [`LoadBalancedChatModel`].
    ///
    /// Returns an error if no models were added or if weighted round-robin
    /// weights do not match the number of models.
    pub fn build(self) -> std::result::Result<LoadBalancedChatModel, String> {
        LoadBalancedChatModel::new(self.models, self.strategy, self.unhealthy_threshold)
    }
}

impl Default for LoadBalancerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::{AIMessage, AIMessageChunk, HumanMessage};
    use cognis_core::outputs::{ChatGeneration, ChatGenerationChunk};
    use std::sync::atomic::AtomicUsize;

    /// A mock chat model that always succeeds with a configurable response.
    struct SuccessModel {
        id: String,
    }

    impl SuccessModel {
        fn new(id: &str) -> Self {
            Self { id: id.into() }
        }
    }

    #[async_trait]
    impl BaseChatModel for SuccessModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: format!("Response from {}", self.id),
                    message: Message::Ai(AIMessage::new(&format!("Response from {}", self.id))),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "success_mock"
        }

        async fn _stream(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatStream> {
            let id = self.id.clone();
            let chunk = ChatGenerationChunk {
                text: format!("Stream from {}", id),
                message: AIMessageChunk::new(&format!("Stream from {}", id)),
                generation_info: None,
            };
            Ok(Box::pin(futures::stream::once(async move { Ok(chunk) })))
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

    /// A mock model that tracks how many times it was called.
    struct CountingModel {
        id: String,
        count: AtomicUsize,
    }

    impl CountingModel {
        fn new(id: &str) -> Self {
            Self {
                id: id.into(),
                count: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl BaseChatModel for CountingModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: format!("Response from {}", self.id),
                    message: Message::Ai(AIMessage::new(&format!("Response from {}", self.id))),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "counting_mock"
        }
    }

    /// A model with configurable latency simulation for testing least-latency.
    struct SlowModel {
        id: String,
        delay_ms: u64,
    }

    impl SlowModel {
        fn new(id: &str, delay_ms: u64) -> Self {
            Self {
                id: id.into(),
                delay_ms,
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for SlowModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: format!("Response from {}", self.id),
                    message: Message::Ai(AIMessage::new(&format!("Response from {}", self.id))),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "slow_mock"
        }
    }

    fn test_messages() -> Vec<Message> {
        vec![Message::Human(HumanMessage::new("hello"))]
    }

    // --- Tests ---

    #[tokio::test]
    async fn test_round_robin_distributes_requests() {
        let m1 = Arc::new(CountingModel::new("A"));
        let m2 = Arc::new(CountingModel::new("B"));
        let m1c = m1.clone();
        let m2c = m2.clone();

        let lb = LoadBalancedChatModel::new(
            vec![m1 as Arc<dyn BaseChatModel>, m2 as Arc<dyn BaseChatModel>],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        for _ in 0..4 {
            lb._generate(&msgs, None).await.unwrap();
        }

        assert_eq!(m1c.call_count(), 2);
        assert_eq!(m2c.call_count(), 2);
    }

    #[tokio::test]
    async fn test_single_model_always_selected() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("only"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        let result = lb._generate(&msgs, None).await.unwrap();
        assert_eq!(result.generations[0].text, "Response from only");
    }

    #[tokio::test]
    async fn test_failover_to_next_model() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(SuccessModel::new("backup"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        let result = lb._generate(&msgs, None).await.unwrap();
        assert_eq!(result.generations[0].text, "Response from backup");
    }

    #[tokio::test]
    async fn test_all_models_fail() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(FailModel)],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        let result = lb._generate(&msgs, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_tracking_success() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("A"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        lb._generate(&msgs, None).await.unwrap();

        let report = lb.get_health_report();
        assert_eq!(report.len(), 1);
        assert!(report[0].is_healthy);
        assert_eq!(report[0].total_requests, 1);
        assert_eq!(report[0].failed_requests, 0);
        assert!((report[0].error_rate - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_health_tracking_failure() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(SuccessModel::new("B"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        // First request: model 0 fails, model 1 succeeds.
        lb._generate(&msgs, None).await.unwrap();

        let report = lb.get_health_report();
        assert_eq!(report[0].failed_requests, 1);
        assert_eq!(report[1].total_requests, 1);
        assert_eq!(report[1].failed_requests, 0);
    }

    #[tokio::test]
    async fn test_model_marked_unhealthy_after_threshold() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(SuccessModel::new("B"))],
            LoadBalancingStrategy::RoundRobin,
            2,
        )
        .unwrap();

        let msgs = test_messages();
        // Make multiple requests so model 0 fails enough times.
        for _ in 0..3 {
            lb._generate(&msgs, None).await.unwrap();
        }

        let report = lb.get_health_report();
        assert!(!report[0].is_healthy);
        assert!(report[1].is_healthy);
    }

    #[tokio::test]
    async fn test_weighted_round_robin() {
        let m1 = Arc::new(CountingModel::new("A"));
        let m2 = Arc::new(CountingModel::new("B"));
        let m1c = m1.clone();
        let m2c = m2.clone();

        let lb = LoadBalancedChatModel::new(
            vec![m1 as Arc<dyn BaseChatModel>, m2 as Arc<dyn BaseChatModel>],
            LoadBalancingStrategy::WeightedRoundRobin(vec![3, 1]),
            3,
        )
        .unwrap();

        let msgs = test_messages();
        // 4 requests with weights [3, 1] should give ~3 to A and ~1 to B.
        for _ in 0..4 {
            lb._generate(&msgs, None).await.unwrap();
        }

        assert_eq!(m1c.call_count(), 3);
        assert_eq!(m2c.call_count(), 1);
    }

    #[tokio::test]
    async fn test_weighted_round_robin_weight_mismatch() {
        let result = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("A"))],
            LoadBalancingStrategy::WeightedRoundRobin(vec![1, 2]),
            3,
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_empty_models_rejected() {
        let result = LoadBalancedChatModel::new(vec![], LoadBalancingStrategy::RoundRobin, 3);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_random_strategy_works() {
        let lb = LoadBalancedChatModel::new(
            vec![
                Arc::new(SuccessModel::new("A")),
                Arc::new(SuccessModel::new("B")),
            ],
            LoadBalancingStrategy::Random,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        // Just verify it doesn't panic and returns a valid result.
        for _ in 0..10 {
            let result = lb._generate(&msgs, None).await.unwrap();
            assert!(result.generations[0].text.starts_with("Response from "));
        }
    }

    #[tokio::test]
    async fn test_least_latency_strategy() {
        let lb = LoadBalancedChatModel::new(
            vec![
                Arc::new(SlowModel::new("slow", 50)),
                Arc::new(SlowModel::new("fast", 1)),
            ],
            LoadBalancingStrategy::LeastLatency,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        // First round: both have 0 latency, picks first.
        lb._generate(&msgs, None).await.unwrap();
        // Second round: the first call establishes latency. Now fast should win.
        lb._generate(&msgs, None).await.unwrap();
        // Third round: fast should have lower avg latency.
        let result = lb._generate(&msgs, None).await.unwrap();
        assert_eq!(result.generations[0].text, "Response from fast");
    }

    #[tokio::test]
    async fn test_llm_type() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("A"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        assert_eq!(lb.llm_type(), "load_balanced");
    }

    #[tokio::test]
    async fn test_stream_works() {
        use futures::StreamExt;

        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("A"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        let mut stream = lb._stream(&msgs, None).await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.text, "Stream from A");
    }

    #[tokio::test]
    async fn test_stream_failover() {
        use futures::StreamExt;

        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(SuccessModel::new("B"))],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let msgs = test_messages();
        let mut stream = lb._stream(&msgs, None).await.unwrap();
        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.text, "Stream from B");
    }

    #[tokio::test]
    async fn test_builder_default() {
        let lb = LoadBalancerBuilder::new()
            .add_model(Arc::new(SuccessModel::new("A")))
            .build()
            .unwrap();

        let msgs = test_messages();
        let result = lb._generate(&msgs, None).await.unwrap();
        assert_eq!(result.generations[0].text, "Response from A");
    }

    #[tokio::test]
    async fn test_builder_with_strategy() {
        let lb = LoadBalancerBuilder::new()
            .add_model(Arc::new(SuccessModel::new("A")))
            .add_model(Arc::new(SuccessModel::new("B")))
            .strategy(LoadBalancingStrategy::LeastLatency)
            .unhealthy_threshold(5)
            .build()
            .unwrap();

        let msgs = test_messages();
        let result = lb._generate(&msgs, None).await.unwrap();
        assert!(result.generations[0].text.starts_with("Response from "));
    }

    #[tokio::test]
    async fn test_builder_empty_fails() {
        let result = LoadBalancerBuilder::new().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_error_rate_calculation() {
        let lb = LoadBalancedChatModel::new(
            vec![Arc::new(FailModel), Arc::new(SuccessModel::new("B"))],
            LoadBalancingStrategy::RoundRobin,
            100, // High threshold so model stays "healthy"
        )
        .unwrap();

        let msgs = test_messages();
        for _ in 0..5 {
            lb._generate(&msgs, None).await.unwrap();
        }

        let report = lb.get_health_report();
        // Model 0 fails every time, model 1 succeeds every time.
        assert_eq!(report[0].failed_requests, report[0].total_requests);
        assert!((report[0].error_rate - 1.0).abs() < f64::EPSILON);
        assert!((report[1].error_rate - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_health_report_initial_state() {
        let lb = LoadBalancedChatModel::new(
            vec![
                Arc::new(SuccessModel::new("A")),
                Arc::new(SuccessModel::new("B")),
            ],
            LoadBalancingStrategy::RoundRobin,
            3,
        )
        .unwrap();

        let report = lb.get_health_report();
        assert_eq!(report.len(), 2);
        for r in &report {
            assert!(r.is_healthy);
            assert_eq!(r.total_requests, 0);
            assert_eq!(r.failed_requests, 0);
            assert_eq!(r.avg_latency_ms, 0);
            assert!((r.error_rate - 0.0).abs() < f64::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_zero_weights_rejected() {
        let result = LoadBalancedChatModel::new(
            vec![Arc::new(SuccessModel::new("A"))],
            LoadBalancingStrategy::WeightedRoundRobin(vec![0]),
            3,
        );
        assert!(result.is_err());
    }
}
