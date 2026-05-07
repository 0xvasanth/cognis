//! LLM-driven retrievers — live in `cognis` (not `cognis-rag`) because they
//! depend on `cognis-llm`'s `Client`.
//!
//! - [`MultiQueryRetriever`] — generate N rephrasings of the query and union
//!   their hits.
//! - [`ContextualCompressionRetriever`] — score each candidate doc with the
//!   LLM and drop low-relevance ones.

pub mod contextual_compression;
pub mod multi_query;
pub mod reranking;
pub mod self_query;
pub mod time_weighted;

pub use contextual_compression::ContextualCompressionRetriever;
pub use multi_query::MultiQueryRetriever;
pub use reranking::RerankingRetriever;
pub use self_query::{SearchSpec, SelfQueryRetriever};
pub use time_weighted::TimeWeightedRetriever;
