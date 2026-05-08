//! Embedding model router with configurable routing rules.
//!
//! Routes embedding requests to different models based on text length,
//! content type, prefix matching, regex patterns, or custom predicates.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use regex::Regex;

use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;

/// Pattern for detecting content types.
#[derive(Debug, Clone)]
pub enum ContentPattern {
    /// Code-like content (contains braces, semicolons, keywords, etc.).
    Code,
    /// Natural language text.
    Natural,
    /// Short query or question.
    Query,
    /// Long document text.
    Document,
}

impl ContentPattern {
    /// Test whether the given text matches this content pattern.
    pub fn matches(&self, text: &str) -> bool {
        match self {
            ContentPattern::Code => Self::is_code(text),
            ContentPattern::Natural => !Self::is_code(text) && text.len() <= 1000,
            ContentPattern::Query => {
                text.len() <= 200 && (text.contains('?') || text.split_whitespace().count() <= 15)
            }
            ContentPattern::Document => text.len() > 1000,
        }
    }

    fn is_code(text: &str) -> bool {
        let code_indicators = [
            "{",
            "}",
            ";",
            "fn ",
            "def ",
            "class ",
            "import ",
            "use ",
            "#include",
            "//",
            "/*",
            "pub ",
            "let ",
            "const ",
            "var ",
            "function ",
        ];
        let matches = code_indicators
            .iter()
            .filter(|ind| text.contains(*ind))
            .count();
        matches >= 2
    }
}

/// Condition that determines when a route is selected.
pub enum RouteCondition {
    /// Route based on text length bounds.
    TextLength {
        /// Minimum text length (inclusive). `None` means no lower bound.
        min: Option<usize>,
        /// Maximum text length (inclusive). `None` means no upper bound.
        max: Option<usize>,
    },
    /// Route based on content type detection.
    ContentType(ContentPattern),
    /// Route if text starts with the given prefix.
    Prefix(String),
    /// Route if text matches the given regex pattern.
    Regex(String),
    /// Route based on a custom predicate function.
    Custom(Arc<dyn Fn(&str) -> bool + Send + Sync>),
    /// Always matches (used for default/fallback routes).
    Always,
}

impl RouteCondition {
    /// Evaluate whether this condition matches the given text.
    pub fn matches(&self, text: &str) -> bool {
        match self {
            RouteCondition::TextLength { min, max } => {
                let len = text.len();
                if let Some(min_val) = min {
                    if len < *min_val {
                        return false;
                    }
                }
                if let Some(max_val) = max {
                    if len > *max_val {
                        return false;
                    }
                }
                true
            }
            RouteCondition::ContentType(pattern) => pattern.matches(text),
            RouteCondition::Prefix(prefix) => text.starts_with(prefix.as_str()),
            RouteCondition::Regex(pattern) => {
                if let Ok(re) = Regex::new(pattern) {
                    re.is_match(text)
                } else {
                    false
                }
            }
            RouteCondition::Custom(f) => f(text),
            RouteCondition::Always => true,
        }
    }
}

/// A single route in the embedding router.
pub struct EmbeddingRoute {
    /// Human-readable name for this route.
    pub name: String,
    /// The embedding model to use when this route is selected.
    pub model: Arc<dyn Embeddings>,
    /// Condition that determines when this route applies.
    pub condition: RouteCondition,
    /// Priority for route ordering (lower = higher priority).
    pub priority: u32,
}

/// Strategy for selecting among multiple matching routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Use the first matching route (ordered by priority).
    FirstMatch,
    /// Distribute requests across matching routes in round-robin fashion.
    RoundRobin,
    /// Randomly select from matching routes.
    Random,
    /// Route to the model with the fewest pending requests.
    LeastLoaded,
}

/// Statistics about routing decisions.
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Number of calls routed to each named route.
    pub route_counts: HashMap<String, usize>,
    /// Number of calls routed to the default model.
    pub default_count: usize,
    /// Total number of routing calls.
    pub total_calls: usize,
}

/// Routes embedding requests to different models based on configurable rules.
///
/// Supports multiple routing strategies including first-match, round-robin,
/// random, and least-loaded selection among matching routes.
pub struct EmbeddingRouter {
    routes: Vec<EmbeddingRoute>,
    default_model: Arc<dyn Embeddings>,
    strategy: RoutingStrategy,
    /// Per-route call counts. Indexed by route name.
    route_counts: Mutex<HashMap<String, usize>>,
    default_count: AtomicUsize,
    total_calls: AtomicUsize,
    /// Round-robin counter.
    rr_counter: AtomicUsize,
}

impl EmbeddingRouter {
    /// Create a new embedding router.
    pub fn new(
        routes: Vec<EmbeddingRoute>,
        default_model: Arc<dyn Embeddings>,
        strategy: RoutingStrategy,
    ) -> Self {
        let mut sorted_routes = routes;
        sorted_routes.sort_by_key(|r| r.priority);

        Self {
            routes: sorted_routes,
            default_model,
            strategy,
            route_counts: Mutex::new(HashMap::new()),
            default_count: AtomicUsize::new(0),
            total_calls: AtomicUsize::new(0),
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Create a builder for constructing an `EmbeddingRouter`.
    pub fn builder() -> EmbeddingRouterBuilder {
        EmbeddingRouterBuilder::new()
    }

    /// Return current routing statistics.
    pub fn get_stats(&self) -> RouterStats {
        let counts = self.route_counts.lock().unwrap();
        RouterStats {
            route_counts: counts.clone(),
            default_count: self.default_count.load(Ordering::Relaxed),
            total_calls: self.total_calls.load(Ordering::Relaxed),
        }
    }

    /// Reset all routing statistics to zero.
    pub fn reset_stats(&self) {
        let mut counts = self.route_counts.lock().unwrap();
        counts.clear();
        self.default_count.store(0, Ordering::Relaxed);
        self.total_calls.store(0, Ordering::Relaxed);
    }

    /// Select a route for the given text, returning the model and route name (if any).
    fn select_route(&self, text: &str) -> (Arc<dyn Embeddings>, Option<String>) {
        let matching: Vec<usize> = self
            .routes
            .iter()
            .enumerate()
            .filter(|(_, r)| r.condition.matches(text))
            .map(|(i, _)| i)
            .collect();

        if matching.is_empty() {
            return (Arc::clone(&self.default_model), None);
        }

        let idx = match self.strategy {
            RoutingStrategy::FirstMatch => matching[0],
            RoutingStrategy::RoundRobin => {
                let counter = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                matching[counter % matching.len()]
            }
            RoutingStrategy::Random => {
                // Simple pseudo-random based on text hash and counter
                let seed = {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    text.hash(&mut hasher);
                    self.total_calls.load(Ordering::Relaxed).hash(&mut hasher);
                    hasher.finish() as usize
                };
                matching[seed % matching.len()]
            }
            RoutingStrategy::LeastLoaded => {
                // Pick the matching route with the fewest historical calls
                let counts = self.route_counts.lock().unwrap();
                let mut best = matching[0];
                let mut best_count = counts.get(&self.routes[best].name).copied().unwrap_or(0);
                for &i in &matching[1..] {
                    let count = counts.get(&self.routes[i].name).copied().unwrap_or(0);
                    if count < best_count {
                        best = i;
                        best_count = count;
                    }
                }
                best
            }
        };

        let route = &self.routes[idx];
        (Arc::clone(&route.model), Some(route.name.clone()))
    }

    /// Record a routing decision in the stats.
    fn record_route(&self, route_name: Option<&str>) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        match route_name {
            Some(name) => {
                let mut counts = self.route_counts.lock().unwrap();
                *counts.entry(name.to_string()).or_insert(0) += 1;
            }
            None => {
                self.default_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[async_trait]
impl Embeddings for EmbeddingRouter {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Group consecutive texts going to the same route into batches for efficiency.
        struct Batch {
            model: Arc<dyn Embeddings>,
            route_name: Option<String>,
            texts: Vec<String>,
            indices: Vec<usize>,
        }

        let mut batches: Vec<Batch> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let (model, route_name) = self.select_route(text);
            self.record_route(route_name.as_deref());

            let same_batch = if let Some(last) = batches.last() {
                last.route_name == route_name
            } else {
                false
            };

            if same_batch {
                let last = batches.last_mut().unwrap();
                last.texts.push(text.clone());
                last.indices.push(i);
            } else {
                batches.push(Batch {
                    model,
                    route_name,
                    texts: vec![text.clone()],
                    indices: vec![i],
                });
            }
        }

        // Execute each batch and place results in order.
        let mut all_results = vec![Vec::new(); texts.len()];
        for batch in batches {
            let embeddings = batch.model.embed_documents(batch.texts).await?;
            for (idx, emb) in batch.indices.into_iter().zip(embeddings) {
                all_results[idx] = emb;
            }
        }

        Ok(all_results)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let (model, route_name) = self.select_route(text);
        self.record_route(route_name.as_deref());
        model.embed_query(text).await
    }
}

/// Builder for constructing an [`EmbeddingRouter`].
pub struct EmbeddingRouterBuilder {
    routes: Vec<EmbeddingRoute>,
    default_model: Option<Arc<dyn Embeddings>>,
    strategy: RoutingStrategy,
}

impl EmbeddingRouterBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            default_model: None,
            strategy: RoutingStrategy::FirstMatch,
        }
    }

    /// Set the default model used when no route matches.
    pub fn default_model(mut self, model: Arc<dyn Embeddings>) -> Self {
        self.default_model = Some(model);
        self
    }

    /// Add a route with the given name, model, and condition.
    pub fn add_route(
        mut self,
        name: impl Into<String>,
        model: Arc<dyn Embeddings>,
        condition: RouteCondition,
    ) -> Self {
        self.routes.push(EmbeddingRoute {
            name: name.into(),
            model,
            condition,
            priority: self.routes.len() as u32,
        });
        self
    }

    /// Add a route with explicit priority.
    pub fn add_route_with_priority(
        mut self,
        name: impl Into<String>,
        model: Arc<dyn Embeddings>,
        condition: RouteCondition,
        priority: u32,
    ) -> Self {
        self.routes.push(EmbeddingRoute {
            name: name.into(),
            model,
            condition,
            priority,
        });
        self
    }

    /// Set the routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Convenience: route short texts (up to `max_length` chars) to the given model.
    pub fn route_short_texts(self, model: Arc<dyn Embeddings>, max_length: usize) -> Self {
        self.add_route(
            "short_texts",
            model,
            RouteCondition::TextLength {
                min: None,
                max: Some(max_length),
            },
        )
    }

    /// Convenience: route long texts (at least `min_length` chars) to the given model.
    pub fn route_long_texts(self, model: Arc<dyn Embeddings>, min_length: usize) -> Self {
        self.add_route(
            "long_texts",
            model,
            RouteCondition::TextLength {
                min: Some(min_length),
                max: None,
            },
        )
    }

    /// Convenience: route code-like content to the given model.
    pub fn route_code(self, model: Arc<dyn Embeddings>) -> Self {
        self.add_route(
            "code",
            model,
            RouteCondition::ContentType(ContentPattern::Code),
        )
    }

    /// Build the [`EmbeddingRouter`].
    ///
    /// # Panics
    /// Panics if no default model has been set.
    pub fn build(self) -> EmbeddingRouter {
        let default_model = self
            .default_model
            .expect("EmbeddingRouterBuilder requires a default_model");
        EmbeddingRouter::new(self.routes, default_model, self.strategy)
    }
}

impl Default for EmbeddingRouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock embedding model that returns vectors filled with a fixed value.
    /// The value identifies which model produced the embedding.
    struct MockEmbeddings {
        fill_value: f32,
        dim: usize,
    }

    impl MockEmbeddings {
        fn new(fill_value: f32, dim: usize) -> Self {
            Self { fill_value, dim }
        }
    }

    #[async_trait]
    impl Embeddings for MockEmbeddings {
        async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|_| vec![self.fill_value; self.dim])
                .collect())
        }

        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![self.fill_value; self.dim])
        }
    }

    fn mock(fill: f32) -> Arc<dyn Embeddings> {
        Arc::new(MockEmbeddings::new(fill, 4))
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_route_by_text_length_short() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .route_short_texts(mock(1.0), 10)
            .build();

        let result = router.embed_query("hi").await.unwrap();
        assert_eq!(result, vec![1.0; 4]);
    }

    #[tokio::test]
    async fn test_route_by_text_length_long() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .route_long_texts(mock(2.0), 50)
            .build();

        let long_text = "a".repeat(100);
        let result = router.embed_query(&long_text).await.unwrap();
        assert_eq!(result, vec![2.0; 4]);
    }

    #[tokio::test]
    async fn test_content_type_detection_code() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .route_code(mock(3.0))
            .build();

        let code = "fn main() { let x = 42; }";
        let result = router.embed_query(code).await.unwrap();
        assert_eq!(result, vec![3.0; 4]);
    }

    #[tokio::test]
    async fn test_content_type_detection_natural() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "natural",
                mock(4.0),
                RouteCondition::ContentType(ContentPattern::Natural),
            )
            .build();

        let text = "The quick brown fox jumps over the lazy dog";
        let result = router.embed_query(text).await.unwrap();
        assert_eq!(result, vec![4.0; 4]);
    }

    #[tokio::test]
    async fn test_prefix_based_routing() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "search",
                mock(5.0),
                RouteCondition::Prefix("search:".to_string()),
            )
            .build();

        let result = router
            .embed_query("search: find me something")
            .await
            .unwrap();
        assert_eq!(result, vec![5.0; 4]);

        // Non-matching prefix falls to default
        let result2 = router.embed_query("just a normal text").await.unwrap();
        assert_eq!(result2, vec![0.0; 4]);
    }

    #[tokio::test]
    async fn test_regex_based_routing() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "email",
                mock(6.0),
                RouteCondition::Regex(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b".to_string(),
                ),
            )
            .build();

        let result = router
            .embed_query("Contact us at info@example.com")
            .await
            .unwrap();
        assert_eq!(result, vec![6.0; 4]);

        let result2 = router.embed_query("No email here").await.unwrap();
        assert_eq!(result2, vec![0.0; 4]);
    }

    #[tokio::test]
    async fn test_custom_predicate_routing() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "uppercase",
                mock(7.0),
                RouteCondition::Custom(Arc::new(|text: &str| {
                    text.chars()
                        .filter(|c| c.is_alphabetic())
                        .all(|c| c.is_uppercase())
                })),
            )
            .build();

        let result = router.embed_query("ALL CAPS TEXT").await.unwrap();
        assert_eq!(result, vec![7.0; 4]);

        let result2 = router.embed_query("not all caps").await.unwrap();
        assert_eq!(result2, vec![0.0; 4]);
    }

    #[tokio::test]
    async fn test_first_match_strategy_priority() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route_with_priority("low_priority", mock(8.0), RouteCondition::Always, 10)
            .add_route_with_priority("high_priority", mock(9.0), RouteCondition::Always, 1)
            .strategy(RoutingStrategy::FirstMatch)
            .build();

        let result = router.embed_query("anything").await.unwrap();
        // high_priority (priority=1) should be sorted first
        assert_eq!(result, vec![9.0; 4]);
    }

    #[tokio::test]
    async fn test_round_robin_distribution() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route("route_a", mock(1.0), RouteCondition::Always)
            .add_route("route_b", mock(2.0), RouteCondition::Always)
            .strategy(RoutingStrategy::RoundRobin)
            .build();

        let r1 = router.embed_query("text1").await.unwrap();
        let r2 = router.embed_query("text2").await.unwrap();
        let r3 = router.embed_query("text3").await.unwrap();
        let r4 = router.embed_query("text4").await.unwrap();

        // Should alternate between route_a (1.0) and route_b (2.0)
        assert_eq!(r1, vec![1.0; 4]);
        assert_eq!(r2, vec![2.0; 4]);
        assert_eq!(r3, vec![1.0; 4]);
        assert_eq!(r4, vec![2.0; 4]);
    }

    #[tokio::test]
    async fn test_default_model_fallback() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(99.0))
            .add_route(
                "short_only",
                mock(1.0),
                RouteCondition::TextLength {
                    min: None,
                    max: Some(5),
                },
            )
            .build();

        // This text is too long for the short_only route
        let result = router
            .embed_query("this is a longer piece of text")
            .await
            .unwrap();
        assert_eq!(result, vec![99.0; 4]);
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .route_short_texts(mock(1.0), 20)
            .route_long_texts(mock(2.0), 100)
            .route_code(mock(3.0))
            .strategy(RoutingStrategy::FirstMatch)
            .build();

        // Short text should go to short_texts route
        let result = router.embed_query("hello").await.unwrap();
        assert_eq!(result, vec![1.0; 4]);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "prefix_route",
                mock(1.0),
                RouteCondition::Prefix("go:".to_string()),
            )
            .build();

        router.embed_query("go: somewhere").await.unwrap();
        router.embed_query("go: elsewhere").await.unwrap();
        router.embed_query("no match").await.unwrap();

        let stats = router.get_stats();
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.route_counts.get("prefix_route"), Some(&2));
        assert_eq!(stats.default_count, 1);
    }

    #[tokio::test]
    async fn test_multiple_matching_routes_first_wins() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route_with_priority("always_a", mock(10.0), RouteCondition::Always, 1)
            .add_route_with_priority("always_b", mock(20.0), RouteCondition::Always, 2)
            .strategy(RoutingStrategy::FirstMatch)
            .build();

        let result = router.embed_query("text").await.unwrap();
        assert_eq!(result, vec![10.0; 4]);
    }

    #[tokio::test]
    async fn test_embed_documents_routing() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "short",
                mock(1.0),
                RouteCondition::TextLength {
                    min: None,
                    max: Some(10),
                },
            )
            .build();

        let results = router
            .embed_documents(vec![
                "hi".to_string(),
                "a very long piece of text that exceeds ten characters".to_string(),
                "hey".to_string(),
            ])
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], vec![1.0; 4]);
        assert_eq!(results[1], vec![0.0; 4]);
        assert_eq!(results[2], vec![1.0; 4]);
    }

    #[tokio::test]
    async fn test_embed_query_routing() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route(
                "question",
                mock(5.0),
                RouteCondition::ContentType(ContentPattern::Query),
            )
            .build();

        let result = router.embed_query("What is Rust?").await.unwrap();
        assert_eq!(result, vec![5.0; 4]);
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let router = EmbeddingRouter::builder()
            .default_model(mock(0.0))
            .add_route("always", mock(1.0), RouteCondition::Always)
            .build();

        router.embed_query("a").await.unwrap();
        router.embed_query("b").await.unwrap();

        let stats = router.get_stats();
        assert_eq!(stats.total_calls, 2);

        router.reset_stats();

        let stats = router.get_stats();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.default_count, 0);
        assert!(stats.route_counts.is_empty());
    }
}
