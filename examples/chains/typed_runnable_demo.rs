//! Typed Runnable Demo
//!
//! Demonstrates compile-time type-safe runnable composition using
//! `TypedRunnable<I, O>` alongside the dynamic `Runnable` (Value-based) system.
//!
//! Shows:
//! - Defining typed runnables with concrete I/O types
//! - Composing them with `TypedSequence`
//! - Bridging typed->dynamic with `DynRunnable` for use with `|` operator
//! - Bridging dynamic->typed with `FromDynRunnable`
//!
//! Run with: `cargo run -p cognis-examples --example typed_runnable_demo`

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use cognis_core::error::Result;
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;
use cognis_core::runnables::lambda::RunnableLambda;
use cognis_core::runnables::pipe::RunnableRef;
use cognis_core::runnables::typed::{DynRunnable, FromDynRunnable, TypedRunnable, TypedSequence};

// --- Typed Runnables ---------------------------------------------------------

/// Parses a raw text query into a structured request.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchRequest {
    query: String,
    max_results: usize,
}

struct ParseQuery;

#[async_trait]
impl TypedRunnable<String, SearchRequest> for ParseQuery {
    fn name(&self) -> &str {
        "parse_query"
    }

    async fn invoke(
        &self,
        input: String,
        _config: Option<&RunnableConfig>,
    ) -> Result<SearchRequest> {
        Ok(SearchRequest {
            query: input,
            max_results: 5,
        })
    }
}

/// Simulates a search, returning scored results.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    title: String,
    score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    total: usize,
}

struct ExecuteSearch;

#[async_trait]
impl TypedRunnable<SearchRequest, SearchResponse> for ExecuteSearch {
    fn name(&self) -> &str {
        "execute_search"
    }

    async fn invoke(
        &self,
        input: SearchRequest,
        _config: Option<&RunnableConfig>,
    ) -> Result<SearchResponse> {
        // Simulate search results
        let results = vec![
            SearchResult {
                title: format!("Best practices for {}", input.query),
                score: 0.95,
            },
            SearchResult {
                title: format!("{} tutorial", input.query),
                score: 0.87,
            },
            SearchResult {
                title: format!("Advanced {} techniques", input.query),
                score: 0.82,
            },
        ];
        let total = results.len();
        Ok(SearchResponse { results, total })
    }
}

/// Formats search results as a readable string.
struct FormatResults;

#[async_trait]
impl TypedRunnable<SearchResponse, String> for FormatResults {
    fn name(&self) -> &str {
        "format_results"
    }

    async fn invoke(
        &self,
        input: SearchResponse,
        _config: Option<&RunnableConfig>,
    ) -> Result<String> {
        let mut output = format!("Found {} results:\n", input.total);
        for (i, r) in input.results.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} (score: {:.2})\n",
                i + 1,
                r.title,
                r.score
            ));
        }
        Ok(output)
    }
}

// --- Numeric pipeline --------------------------------------------------------

struct AddOne;

#[async_trait]
impl TypedRunnable<i64, i64> for AddOne {
    fn name(&self) -> &str {
        "add_one"
    }
    async fn invoke(&self, input: i64, _config: Option<&RunnableConfig>) -> Result<i64> {
        Ok(input + 1)
    }
}

struct Double;

#[async_trait]
impl TypedRunnable<i64, i64> for Double {
    fn name(&self) -> &str {
        "double"
    }
    async fn invoke(&self, input: i64, _config: Option<&RunnableConfig>) -> Result<i64> {
        Ok(input * 2)
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Typed Runnable Demo ===\n");

    // -- 1. Type-safe composition with TypedSequence --------------------------

    println!("--- 1. TypedSequence: Compile-Time Type Safety ---\n");

    // String -> SearchRequest -> SearchResponse
    let search_pipeline = TypedSequence::new(
        Arc::new(ParseQuery) as Arc<dyn TypedRunnable<String, SearchRequest>>,
        Arc::new(ExecuteSearch) as Arc<dyn TypedRunnable<SearchRequest, SearchResponse>>,
    );

    let response = search_pipeline
        .invoke("Rust async".to_string(), None)
        .await?;
    println!("Search for 'Rust async': {} results found", response.total);

    // Chain further: String -> SearchRequest -> SearchResponse -> String
    let full_pipeline = TypedSequence::new(
        Arc::new(search_pipeline) as Arc<dyn TypedRunnable<String, SearchResponse>>,
        Arc::new(FormatResults) as Arc<dyn TypedRunnable<SearchResponse, String>>,
    );

    let formatted = full_pipeline
        .invoke("Rust ownership".to_string(), None)
        .await?;
    println!("{}", formatted);

    // -- 2. Numeric pipeline --------------------------------------------------

    println!("--- 2. Numeric Pipeline: (x + 1) * 2 ---\n");

    let numeric = TypedSequence::new(
        Arc::new(AddOne) as Arc<dyn TypedRunnable<i64, i64>>,
        Arc::new(Double) as Arc<dyn TypedRunnable<i64, i64>>,
    );

    let result = numeric.invoke(5, None).await?;
    println!("(5 + 1) * 2 = {}", result);

    let batch_results = numeric.batch(vec![1, 2, 3, 4, 5], None).await?;
    println!("Batch: {:?}\n", batch_results);

    // -- 3. Bridge: Typed -> Dynamic (for | operator) -------------------------

    println!("--- 3. DynRunnable Bridge: Typed -> Dynamic ---\n");

    let typed_add: Arc<dyn TypedRunnable<i64, i64>> = Arc::new(AddOne);
    let dynamic_add: Arc<dyn Runnable> = Arc::new(DynRunnable::new(typed_add));

    // Now usable with the | pipe operator
    let dynamic_double = RunnableRef::new(Arc::new(RunnableLambda::new(
        "double",
        |v: Value| async move {
            let n = v.as_i64().unwrap_or(0);
            Ok(json!(n * 2))
        },
    )));

    let chain = RunnableRef::new(dynamic_add) | dynamic_double;
    let result = chain.runnable().invoke(json!(10), None).await?;
    println!("Typed AddOne | Dynamic Double: 10 -> {}\n", result);

    // -- 4. Bridge: Dynamic -> Typed ------------------------------------------

    println!("--- 4. FromDynRunnable Bridge: Dynamic -> Typed ---\n");

    let dyn_runnable = Arc::new(RunnableLambda::new(
        "multiply_by_3",
        |v: Value| async move {
            let n = v.as_i64().unwrap_or(0);
            Ok(json!(n * 3))
        },
    )) as Arc<dyn Runnable>;

    let typed_mul3: FromDynRunnable<i64, i64> = FromDynRunnable::new(dyn_runnable);
    let result = typed_mul3.invoke(7, None).await?;
    println!("Dynamic multiply_by_3 as typed: 7 -> {}", result);

    // Compose the typed wrapper with another typed runnable
    let pipeline = TypedSequence::new(
        Arc::new(typed_mul3) as Arc<dyn TypedRunnable<i64, i64>>,
        Arc::new(AddOne) as Arc<dyn TypedRunnable<i64, i64>>,
    );
    let result = pipeline.invoke(10, None).await?;
    println!("(10 * 3) + 1 = {}\n", result);

    // -- 5. Type error demonstration ------------------------------------------

    println!("--- 5. Type Safety: Bad Input Detection ---\n");

    let typed_add: Arc<dyn TypedRunnable<i64, i64>> = Arc::new(AddOne);
    let dynamic = Arc::new(DynRunnable::new(typed_add)) as Arc<dyn Runnable>;

    // This will fail at runtime with a clear deserialization error
    let bad_result = dynamic.invoke(json!("not a number"), None).await;
    match bad_result {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Caught type error: {}", e),
    }

    println!("\n=== Done ===");
    Ok(())
}
