//! Chain abstractions for composing prompts, models, and sequential pipelines.
//!
//! This module provides a variety of chain types for building LLM workflows:
//!
//! - [`llm`] -- Basic LLM chain (prompt -> model -> output).
//! - [`sequential`] -- Sequential chain that pipes output from one chain to the next.
//! - [`conversation`] -- Conversation chain with memory integration.
//! - [`conversation_retrieval`] -- Conversational retrieval chain combining memory and document lookup.
//! - [`retrieval`] -- Retrieval QA chain for question answering over documents.
//! - [`map_reduce`] -- Map-reduce chain for processing document collections.
//! - [`refine`] -- Refine chain that iteratively improves answers over documents.
//! - [`router`] -- Router chain with semantic routing for dynamic dispatch.
//! - [`structured_output`] -- Structured output chain for schema-validated responses.
//! - [`summarize`] -- Summarization chains (stuff, map-reduce, refine strategies).
//! - [`sql`] -- Text-to-SQL chain with schema validation.
//! - [`api`] -- API chain for calling external HTTP endpoints.
//! - [`streaming`] -- Streaming chain execution with callback-driven token delivery.
//! - [`extraction`] -- Extraction chain for pulling structured data from text.
//! - [`transform`] -- Transform chains with sync/async transforms and pipeline composition.
//! - [`conditional`] -- Conditional, branch, and switch chains for routing based on predicates.
//! - [`documents`] -- Stuff documents chain for combining documents with formatting strategies.
//! - [`qa`] -- QA chain with retrieval integration and citation tracking.

pub mod conversation;
pub mod llm;
pub mod map_reduce;
pub mod refine;
pub mod retrieval;
pub mod sequential;

pub use conversation::ConversationChain;
pub use llm::LLMChain;
pub use map_reduce::MapReduceChain;
pub use refine::RefineChain;
pub use retrieval::{RetrievalQAChain, RetrievalResult};
pub use sequential::SequentialChain;
pub mod conversation_retrieval;
pub use conversation_retrieval::{ConversationalRetrievalChain, ConversationalRetrievalResult};
pub mod router;
pub use router::{Route, RouterChain, RouterResult, SemanticRouter};
pub mod structured_output;
pub use structured_output::StructuredOutputChain;
pub mod summarize;
pub use summarize::{
    MapReduceSummarizationChain, RefineSummarizationChain, StuffSummarizationChain,
};
pub mod sql;
pub use sql::{
    ColumnSchema, ColumnSchemaBuilder, DatabaseSchema, DatabaseSchemaBuilder, SQLQueryValidator,
    TableSchema, TableSchemaBuilder, TextToSQLChain, TextToSQLChainBuilder,
};
pub mod api;
pub use api::{
    APIChain, APIChainBuilder, APISpec, APISpecBuilder, EndpointSpec, EndpointSpecBuilder,
    ParameterSpec, ParameterSpecBuilder, RequestValidator,
};
pub mod transform;
pub use transform::{
    json_extract_transform, lowercase_transform, map_transform, regex_replace_transform,
    trim_transform, uppercase_transform, AsyncTransformChain, AsyncTransformFn, TransformChain,
    TransformChainBuilder, TransformFn, TransformPipeline,
};
pub mod streaming;
pub use streaming::{
    stream_chain, ConsoleStreamingCallback, StreamingCallback, StreamingCallbackAdapter,
    StreamingChainExecutor, StreamingChainResult, TokenCollector,
};
pub mod conditional;
pub use conditional::{
    BranchChain, BranchChainBuilder, ClosureCondition, Condition, ConditionalChain,
    ConditionalChainBuilder, KeyContainsCondition, KeyEqualsCondition, KeyExistsCondition,
    SwitchChain, SwitchChainBuilder,
};
pub mod extraction;
pub use extraction::{
    ExtractionChain, ExtractionChainBuilder, ExtractionExample, ExtractionResult, ExtractionSchema,
    ExtractionSchemaBuilder, FieldType, OutputFormat, SchemaField, SchemaFieldBuilder,
};
pub mod documents;
pub use documents::{
    create_stuff_documents_chain, CollapseStrategy, DocumentCombineStrategy, DocumentFormatter,
    StuffDocumentsChain, StuffDocumentsChainBuilder, StuffStrategy,
};
pub mod qa;
pub use qa::{
    create_qa_chain, Citation, CitedAnswer, QAChain, QAChainType, QAConfig, QAConfigBuilder,
    QAResult,
};
