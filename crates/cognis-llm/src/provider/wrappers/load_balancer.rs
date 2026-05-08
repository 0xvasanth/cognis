//! Load balancer — distribute calls across a fleet of inner providers.
//!
//! Three strategies ship in the box; users plug in their own by
//! implementing [`LoadBalancingStrategy`]:
//!
//! - [`RoundRobinStrategy`] — cycle through endpoints in order.
//! - [`WeightedRoundRobinStrategy`] — RR with per-endpoint weights.
//! - [`RandomStrategy`] — uniform random pick.
//!
//! Failover: when the chosen endpoint errors, the load balancer can
//! optionally retry on the next endpoint. Toggle via
//! [`LoadBalancerProvider::with_failover`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::tools::ToolDefinition;
use crate::Message;

/// Pluggable load-balancing policy.
pub trait LoadBalancingStrategy: Send + Sync {
    /// Pick an index into the endpoints array. `attempt` is 0 for the
    /// first try, 1 for the failover retry, etc. Implementations must
    /// return a value in `[0, n_endpoints)`.
    fn pick(&self, n_endpoints: usize, attempt: usize) -> usize;
}

/// Round-robin: every call increments a shared counter.
#[derive(Default)]
pub struct RoundRobinStrategy {
    cursor: AtomicUsize,
}

impl RoundRobinStrategy {
    /// New strategy.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LoadBalancingStrategy for RoundRobinStrategy {
    fn pick(&self, n: usize, attempt: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let base = self.cursor.fetch_add(1, Ordering::Relaxed);
        (base + attempt) % n
    }
}

/// Weighted round-robin: endpoints with higher weights get proportionally
/// more traffic.
pub struct WeightedRoundRobinStrategy {
    weights: Vec<u32>,
    /// Pre-expanded schedule (e.g. weights [3, 1] → schedule [0,0,0,1]).
    schedule: Vec<usize>,
    cursor: AtomicUsize,
}

impl WeightedRoundRobinStrategy {
    /// Build with `weights[i]` = weight of endpoint `i`. Empty weights
    /// degrade to round-robin.
    pub fn new(weights: Vec<u32>) -> Self {
        let mut schedule = Vec::new();
        for (idx, &w) in weights.iter().enumerate() {
            for _ in 0..w {
                schedule.push(idx);
            }
        }
        Self {
            weights,
            schedule,
            cursor: AtomicUsize::new(0),
        }
    }

    /// Borrow the configured weights.
    pub fn weights(&self) -> &[u32] {
        &self.weights
    }
}

impl LoadBalancingStrategy for WeightedRoundRobinStrategy {
    fn pick(&self, n: usize, attempt: usize) -> usize {
        if n == 0 {
            return 0;
        }
        if self.schedule.is_empty() {
            // No weights configured — fall back to plain RR.
            let base = self.cursor.fetch_add(1, Ordering::Relaxed);
            return (base + attempt) % n;
        }
        let base = self.cursor.fetch_add(1, Ordering::Relaxed);
        let idx = (base + attempt) % self.schedule.len();
        // schedule entries are guaranteed < weights.len(); the caller's
        // n_endpoints should match; if it's smaller, modulo-clamp.
        self.schedule[idx] % n
    }
}

/// Uniform-random pick.
pub struct RandomStrategy {
    counter: AtomicUsize,
}

impl Default for RandomStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomStrategy {
    /// New random strategy with a deterministic-but-shuffled progression
    /// (uses a multiplicative hash; no `rand` dep needed).
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingStrategy for RandomStrategy {
    fn pick(&self, n: usize, attempt: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let c = self.counter.fetch_add(1, Ordering::Relaxed);
        // Cheap LCG-style mix to avoid clustering.
        let mixed = (c.wrapping_mul(6364136223846793005).wrapping_add(1)) ^ attempt;
        mixed % n
    }
}

/// Closure-based custom strategy.
impl<F> LoadBalancingStrategy for F
where
    F: Fn(usize, usize) -> usize + Send + Sync,
{
    fn pick(&self, n: usize, attempt: usize) -> usize {
        (self)(n, attempt)
    }
}

/// Load-balancing wrapper across multiple provider instances.
pub struct LoadBalancerProvider {
    endpoints: Vec<Arc<dyn LLMProvider>>,
    strategy: Box<dyn LoadBalancingStrategy>,
    failover_attempts: usize,
    name: String,
}

impl LoadBalancerProvider {
    /// Build with a fleet of endpoints and a strategy. `name` is the
    /// reported provider name (e.g. `"openai-pool"`).
    pub fn new(
        name: impl Into<String>,
        endpoints: Vec<Arc<dyn LLMProvider>>,
        strategy: Box<dyn LoadBalancingStrategy>,
    ) -> Result<Self> {
        if endpoints.is_empty() {
            return Err(CognisError::Configuration(
                "LoadBalancerProvider requires at least one endpoint".into(),
            ));
        }
        Ok(Self {
            endpoints,
            strategy,
            failover_attempts: 0,
            name: name.into(),
        })
    }

    /// Failover behavior on error: if the chosen endpoint errors, retry
    /// up to `n` other endpoints before propagating the failure.
    /// Default: 0 (no failover).
    pub fn with_failover(mut self, n: usize) -> Self {
        self.failover_attempts = n;
        self
    }

    /// Borrow the wrapped endpoints.
    pub fn endpoints(&self) -> &[Arc<dyn LLMProvider>] {
        &self.endpoints
    }

    fn pick(&self, attempt: usize) -> &Arc<dyn LLMProvider> {
        let idx = self.strategy.pick(self.endpoints.len(), attempt);
        // Defensive: clamp, in case a custom strategy returns out-of-range.
        let idx = idx.min(self.endpoints.len() - 1);
        &self.endpoints[idx]
    }
}

#[async_trait]
impl LLMProvider for LoadBalancerProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn provider_type(&self) -> Provider {
        // Load balancer's "type" is whatever its first endpoint reports.
        // (All endpoints are expected to be the same provider; this is
        // a soft assumption.)
        self.endpoints[0].provider_type()
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut last_err: Option<CognisError> = None;
        let n = self.endpoints.len();
        let first_idx = self.strategy.pick(n, 0).min(n - 1);
        for attempt in 0..=self.failover_attempts {
            let idx = (first_idx + attempt) % n;
            let ep = &self.endpoints[idx];
            match ep.chat_completion(messages.clone(), opts.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| CognisError::Internal("load balancer reached no endpoints".into())))
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        // Streaming doesn't have a clean failover path (chunks already
        // emitted couldn't be undone), so we only attempt the first pick.
        let ep = self.pick(0);
        ep.chat_completion_stream(messages, opts).await
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut last_err: Option<CognisError> = None;
        let n = self.endpoints.len();
        let first_idx = self.strategy.pick(n, 0).min(n - 1);
        for attempt in 0..=self.failover_attempts {
            let idx = (first_idx + attempt) % n;
            let ep = &self.endpoints[idx];
            match ep
                .chat_completion_with_tools(messages.clone(), tools.clone(), opts.clone())
                .await
            {
                Ok(r) => return Ok(r),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err
            .unwrap_or_else(|| CognisError::Internal("load balancer reached no endpoints".into())))
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        // Healthy if any endpoint is healthy. Returns the first Healthy
        // result verbatim (with its latency).
        let mut last: Result<HealthStatus> = Err(CognisError::Internal("no endpoints".into()));
        for ep in &self.endpoints {
            match ep.health_check().await {
                Ok(s @ HealthStatus::Healthy { .. }) => return Ok(s),
                Ok(s) => last = Ok(s),
                Err(e) => last = Err(e),
            }
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Tagged {
        tag: &'static str,
        ok: bool,
        seen: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl LLMProvider for Tagged {
        fn name(&self) -> &str {
            self.tag
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(self.tag);
            if self.ok {
                Ok(ChatResponse {
                    message: Message::ai(self.tag),
                    usage: None,
                    finish_reason: "stop".into(),
                    model: self.tag.into(),
                })
            } else {
                Err(CognisError::Internal("nope".into()))
            }
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(if self.ok {
                HealthStatus::Healthy { latency_ms: 0 }
            } else {
                HealthStatus::Unhealthy {
                    reason: "scripted".into(),
                }
            })
        }
    }

    fn ep(
        tag: &'static str,
        ok: bool,
        seen: Arc<Mutex<Vec<&'static str>>>,
    ) -> Arc<dyn LLMProvider> {
        Arc::new(Tagged { tag, ok, seen })
    }

    #[tokio::test]
    async fn round_robin_cycles_endpoints() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let lb = LoadBalancerProvider::new(
            "pool",
            vec![ep("a", true, seen.clone()), ep("b", true, seen.clone())],
            Box::new(RoundRobinStrategy::new()),
        )
        .unwrap();
        for _ in 0..4 {
            let _ = lb.chat_completion(vec![], ChatOptions::default()).await;
        }
        let s = seen.lock().unwrap().clone();
        assert_eq!(s.iter().filter(|t| **t == "a").count(), 2);
        assert_eq!(s.iter().filter(|t| **t == "b").count(), 2);
    }

    #[tokio::test]
    async fn weighted_rr_respects_weights() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let lb = LoadBalancerProvider::new(
            "pool",
            vec![ep("a", true, seen.clone()), ep("b", true, seen.clone())],
            Box::new(WeightedRoundRobinStrategy::new(vec![3, 1])),
        )
        .unwrap();
        for _ in 0..8 {
            let _ = lb.chat_completion(vec![], ChatOptions::default()).await;
        }
        let s = seen.lock().unwrap().clone();
        assert_eq!(s.iter().filter(|t| **t == "a").count(), 6);
        assert_eq!(s.iter().filter(|t| **t == "b").count(), 2);
    }

    #[tokio::test]
    async fn failover_retries_on_error() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let lb = LoadBalancerProvider::new(
            "pool",
            vec![
                ep("bad", false, seen.clone()),
                ep("good", true, seen.clone()),
            ],
            Box::new(RoundRobinStrategy::new()),
        )
        .unwrap()
        .with_failover(1);
        let res = lb.chat_completion(vec![], ChatOptions::default()).await;
        assert!(res.is_ok());
        // Both endpoints saw the call (first failed, second succeeded).
        let s = seen.lock().unwrap().clone();
        assert!(s.contains(&"bad"));
        assert!(s.contains(&"good"));
    }

    #[tokio::test]
    async fn rejects_empty_endpoints() {
        let res = LoadBalancerProvider::new("x", Vec::new(), Box::new(RoundRobinStrategy::new()));
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn closure_strategy_works() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        // Always pick endpoint 1.
        let lb = LoadBalancerProvider::new(
            "pool",
            vec![ep("a", true, seen.clone()), ep("b", true, seen.clone())],
            Box::new(|_n: usize, _a: usize| 1usize),
        )
        .unwrap();
        for _ in 0..3 {
            let _ = lb.chat_completion(vec![], ChatOptions::default()).await;
        }
        let s = seen.lock().unwrap().clone();
        assert!(s.iter().all(|t| *t == "b"));
    }
}
