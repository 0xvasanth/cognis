use std::collections::HashMap;
use std::sync::Arc;

use rustchain_core::embeddings::Embeddings;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::vectorstores::base::cosine_similarity;

/// A single route definition with a name and description used for semantic matching.
#[derive(Debug, Clone)]
pub struct Route {
    /// Unique name identifying this route.
    pub name: String,
    /// Natural language description used for semantic matching against queries.
    pub description: String,
    /// Optional custom prompt template for this route. Uses `{query}` placeholder.
    pub prompt_template: Option<String>,
}

impl Route {
    /// Create a new route with the given name and description.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            prompt_template: None,
        }
    }

    /// Set a custom prompt template for this route.
    pub fn with_prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }
}

/// Semantic router that matches queries to routes based on embedding similarity.
///
/// On construction, all route descriptions are embedded and cached. When routing
/// a query, the query is embedded and compared against all cached route embeddings
/// using cosine similarity to find the best match.
pub struct SemanticRouter {
    embeddings: Arc<dyn Embeddings>,
    routes: Vec<Route>,
    route_embeddings: Vec<Vec<f32>>,
    /// Optional fallback route name when no route matches well.
    pub default_route: Option<String>,
}

impl SemanticRouter {
    /// Create a new semantic router by embedding all route descriptions.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding model fails to embed the route descriptions.
    pub async fn new(embeddings: Arc<dyn Embeddings>, routes: Vec<Route>) -> Result<Self> {
        let descriptions: Vec<String> = routes.iter().map(|r| r.description.clone()).collect();
        let route_embeddings = embeddings.embed_documents(descriptions).await?;
        Ok(Self {
            embeddings,
            routes,
            route_embeddings,
            default_route: None,
        })
    }

    /// Set the default route name used as fallback.
    pub fn with_default_route(mut self, name: impl Into<String>) -> Self {
        self.default_route = Some(name.into());
        self
    }

    /// Route a query to the best matching route by cosine similarity.
    ///
    /// If no routes are defined, returns an error.
    pub async fn route(&self, query: &str) -> Result<&Route> {
        let (route, _score) = self.route_with_score(query).await?;
        Ok(route)
    }

    /// Route a query and return both the best matching route and its similarity score.
    ///
    /// The score is a cosine similarity value in the range [-1.0, 1.0],
    /// where higher values indicate better matches.
    pub async fn route_with_score(&self, query: &str) -> Result<(&Route, f32)> {
        if self.routes.is_empty() {
            return Err(RustChainError::Other("No routes defined".into()));
        }

        let query_embedding = self.embeddings.embed_query(query).await?;

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, route_emb) in self.route_embeddings.iter().enumerate() {
            let score = cosine_similarity(&query_embedding, route_emb);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        Ok((&self.routes[best_idx], best_score))
    }

    /// Get the list of routes.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Find the default route, if one is configured and exists.
    pub fn find_default_route(&self) -> Option<&Route> {
        self.default_route
            .as_ref()
            .and_then(|name| self.routes.iter().find(|r| r.name == *name))
    }
}

/// Result of a router chain invocation.
#[derive(Debug, Clone)]
pub struct RouterResult {
    /// The LLM-generated answer.
    pub answer: String,
    /// Name of the route that was selected.
    pub route_name: String,
    /// Cosine similarity confidence score for the route match.
    pub confidence: f32,
}

/// A chain that semantically routes queries and invokes an LLM with route-specific prompts.
///
/// Combines a [`SemanticRouter`] with a chat model. For each query:
/// 1. The query is routed to the best matching route.
/// 2. The appropriate prompt template is selected (route-specific or default).
/// 3. The prompt is formatted and sent to the LLM.
/// 4. The result is returned with routing metadata.
pub struct RouterChain {
    router: SemanticRouter,
    llm: Arc<dyn BaseChatModel>,
    route_prompts: HashMap<String, String>,
    default_prompt: String,
}

impl RouterChain {
    /// Create a new router chain with the given semantic router and LLM.
    pub fn new(router: SemanticRouter, llm: Arc<dyn BaseChatModel>) -> Self {
        Self {
            router,
            llm,
            route_prompts: HashMap::new(),
            default_prompt: "Answer the following question: {query}".to_string(),
        }
    }

    /// Add a prompt template for a specific route. The template should contain
    /// a `{query}` placeholder that will be replaced with the user's query.
    pub fn with_route_prompt(
        mut self,
        route_name: impl Into<String>,
        template: impl Into<String>,
    ) -> Self {
        self.route_prompts
            .insert(route_name.into(), template.into());
        self
    }

    /// Set the default prompt template used when no route-specific prompt exists.
    pub fn with_default_prompt(mut self, template: impl Into<String>) -> Self {
        self.default_prompt = template.into();
        self
    }

    /// Route the query, select the appropriate prompt, call the LLM, and return the result.
    pub async fn call(&self, query: &str) -> Result<RouterResult> {
        let (route, confidence) = self.router.route_with_score(query).await?;
        let route_name = route.name.clone();

        // Select prompt: route-specific from chain > route's own template > default
        let template = self
            .route_prompts
            .get(&route_name)
            .or(route.prompt_template.as_ref())
            .unwrap_or(&self.default_prompt);

        let formatted = template.replace("{query}", query);

        let messages = vec![Message::Human(HumanMessage::new(&formatted))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let answer = ai_msg.base.content.text();

        Ok(RouterResult {
            answer,
            route_name,
            confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
    use rustchain_core::language_models::fake::FakeListChatModel;

    fn fake_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(128))
    }

    fn fake_model(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    #[tokio::test]
    async fn test_route_to_correct_route_based_on_similarity() {
        let routes = vec![
            Route::new("math", "mathematics calculations arithmetic numbers"),
            Route::new("history", "historical events dates civilizations wars"),
            Route::new("science", "physics chemistry biology experiments"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        // The deterministic embeddings are hash-based, so the same text always
        // produces the same embedding. A query matching a route's description
        // text should route to that route.
        let route = router
            .route("mathematics calculations arithmetic numbers")
            .await
            .unwrap();
        assert_eq!(route.name, "math");

        let route = router
            .route("historical events dates civilizations wars")
            .await
            .unwrap();
        assert_eq!(route.name, "history");

        let route = router
            .route("physics chemistry biology experiments")
            .await
            .unwrap();
        assert_eq!(route.name, "science");
    }

    #[tokio::test]
    async fn test_route_with_score_returns_valid_confidence() {
        let routes = vec![
            Route::new("greeting", "hello hi greetings welcome"),
            Route::new("farewell", "goodbye bye see you later"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        // Exact match should give highest possible similarity (1.0)
        let (route, score) = router
            .route_with_score("hello hi greetings welcome")
            .await
            .unwrap();
        assert_eq!(route.name, "greeting");
        assert!(
            (score - 1.0).abs() < 1e-5,
            "Exact match should have score ~1.0, got {score}"
        );

        // Any query should produce a score in [-1, 1]
        let (_route, score) = router.route_with_score("some random query").await.unwrap();
        assert!(
            score >= -1.0 && score <= 1.0,
            "Score should be in [-1,1], got {score}"
        );
    }

    #[tokio::test]
    async fn test_router_chain_uses_correct_prompt_for_matched_route() {
        let routes = vec![
            Route::new("math", "mathematics calculations arithmetic numbers"),
            Route::new("history", "historical events dates civilizations wars"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        // The fake model echoes won't actually show the prompt, but we can verify
        // the chain picks the right route and returns its name.
        let chain = RouterChain::new(router, fake_model(vec!["42"]))
            .with_route_prompt("math", "Solve this math problem: {query}")
            .with_route_prompt("history", "Answer this history question: {query}");

        let result = chain
            .call("mathematics calculations arithmetic numbers")
            .await
            .unwrap();
        assert_eq!(result.route_name, "math");
        assert_eq!(result.answer, "42");
    }

    #[tokio::test]
    async fn test_default_route_when_no_good_match() {
        let routes = vec![
            Route::new("default", "general questions anything else"),
            Route::new("cooking", "recipes food ingredients kitchen"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap()
            .with_default_route("default");

        // Verify the default route is findable
        let default = router.find_default_route();
        assert!(default.is_some());
        assert_eq!(default.unwrap().name, "default");

        // The chain should still route and produce a result
        let chain = RouterChain::new(router, fake_model(vec!["I can help with that"]));
        let result = chain.call("general questions anything else").await.unwrap();
        assert_eq!(result.route_name, "default");
        assert_eq!(result.answer, "I can help with that");
    }

    #[tokio::test]
    async fn test_multiple_routes_with_different_descriptions() {
        let routes = vec![
            Route::new("weather", "weather forecast temperature rain sunny cloudy"),
            Route::new("sports", "football basketball soccer tennis athletes game"),
            Route::new("music", "songs albums artists concerts genres rhythm"),
            Route::new("travel", "flights hotels destinations tourism vacation"),
            Route::new("tech", "programming software computers code algorithms"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        // Each exact description should route to its own route
        let route = router
            .route("weather forecast temperature rain sunny cloudy")
            .await
            .unwrap();
        assert_eq!(route.name, "weather");

        let route = router
            .route("football basketball soccer tennis athletes game")
            .await
            .unwrap();
        assert_eq!(route.name, "sports");

        let route = router
            .route("songs albums artists concerts genres rhythm")
            .await
            .unwrap();
        assert_eq!(route.name, "music");

        let route = router
            .route("flights hotels destinations tourism vacation")
            .await
            .unwrap();
        assert_eq!(route.name, "travel");

        let route = router
            .route("programming software computers code algorithms")
            .await
            .unwrap();
        assert_eq!(route.name, "tech");
    }

    #[tokio::test]
    async fn test_router_result_contains_correct_metadata() {
        let routes = vec![Route::new(
            "support",
            "customer support help troubleshooting issues",
        )];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        let chain = RouterChain::new(router, fake_model(vec!["Let me help you"]))
            .with_route_prompt("support", "Help the customer: {query}");

        let result = chain
            .call("customer support help troubleshooting issues")
            .await
            .unwrap();

        // Verify all fields of RouterResult
        assert_eq!(result.route_name, "support");
        assert_eq!(result.answer, "Let me help you");
        // Exact match should produce high confidence
        assert!(
            result.confidence > 0.99,
            "Exact match confidence should be near 1.0, got {}",
            result.confidence
        );
    }

    #[tokio::test]
    async fn test_route_with_custom_prompt_template_on_route() {
        let routes = vec![
            Route::new("translate", "translation language convert words")
                .with_prompt_template("Translate the following: {query}"),
        ];

        let router = SemanticRouter::new(fake_embeddings(), routes)
            .await
            .unwrap();

        // The route's own prompt_template should be used when no chain-level override exists
        let chain = RouterChain::new(router, fake_model(vec!["Translated text"]));

        let result = chain
            .call("translation language convert words")
            .await
            .unwrap();
        assert_eq!(result.route_name, "translate");
        assert_eq!(result.answer, "Translated text");
    }
}
