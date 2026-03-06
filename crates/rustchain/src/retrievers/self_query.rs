//! Self-query retriever that uses an LLM to parse natural language queries
//! into structured queries with semantic and metadata filter components.
//!
//! Given a natural language query like "Find sci-fi movies made after 2020",
//! the self-query retriever uses an LLM to decompose it into:
//! - A semantic query: "sci-fi movies"
//! - A metadata filter: `Gt("year", 2020)`
//!
//! The semantic query is sent to the vector store for similarity search, and
//! the metadata filter is applied client-side to the results.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::vectorstores::base::VectorStore;

// ─── Filter Types ───

/// A value that can be used in metadata filter comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FilterValue {
    /// A string value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
}

impl FilterValue {
    /// Convert to a `serde_json::Value` for comparison with document metadata.
    pub fn to_json_value(&self) -> Value {
        match self {
            FilterValue::String(s) => Value::String(s.clone()),
            FilterValue::Integer(i) => Value::Number((*i).into()),
            FilterValue::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            FilterValue::Bool(b) => Value::Bool(*b),
        }
    }

    /// Extract a comparable f64 from this value, if numeric.
    fn as_f64(&self) -> Option<f64> {
        match self {
            FilterValue::Integer(i) => Some(*i as f64),
            FilterValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// Metadata filter expressions for structured queries.
///
/// These filters operate on document metadata fields and support comparison,
/// set membership, and logical combination operators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operator", content = "args")]
pub enum AttributeFilter {
    /// Equality: `field == value`
    Eq { field: String, value: FilterValue },
    /// Inequality: `field != value`
    Ne { field: String, value: FilterValue },
    /// Greater than: `field > value`
    Gt { field: String, value: FilterValue },
    /// Greater than or equal: `field >= value`
    Gte { field: String, value: FilterValue },
    /// Less than: `field < value`
    Lt { field: String, value: FilterValue },
    /// Less than or equal: `field <= value`
    Lte { field: String, value: FilterValue },
    /// Set membership: `field in [values]`
    In {
        field: String,
        values: Vec<FilterValue>,
    },
    /// Negated set membership: `field not in [values]`
    Nin {
        field: String,
        values: Vec<FilterValue>,
    },
    /// Logical AND of multiple filters.
    And(Vec<AttributeFilter>),
    /// Logical OR of multiple filters.
    Or(Vec<AttributeFilter>),
    /// Logical NOT of a filter.
    Not(Box<AttributeFilter>),
}

impl AttributeFilter {
    /// Evaluate this filter against a document's metadata.
    ///
    /// Returns `true` if the document matches the filter criteria.
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            AttributeFilter::Eq { field, value } => metadata
                .get(field)
                .is_some_and(|v| *v == value.to_json_value()),
            AttributeFilter::Ne { field, value } => metadata
                .get(field)
                .is_none_or(|v| *v != value.to_json_value()),
            AttributeFilter::Gt { field, value } => {
                compare_metadata_numeric(metadata, field, value, |a, b| a > b)
            }
            AttributeFilter::Gte { field, value } => {
                compare_metadata_numeric(metadata, field, value, |a, b| a >= b)
            }
            AttributeFilter::Lt { field, value } => {
                compare_metadata_numeric(metadata, field, value, |a, b| a < b)
            }
            AttributeFilter::Lte { field, value } => {
                compare_metadata_numeric(metadata, field, value, |a, b| a <= b)
            }
            AttributeFilter::In { field, values } => metadata
                .get(field)
                .is_some_and(|v| values.iter().any(|fv| *v == fv.to_json_value())),
            AttributeFilter::Nin { field, values } => metadata
                .get(field)
                .is_none_or(|v| !values.iter().any(|fv| *v == fv.to_json_value())),
            AttributeFilter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            AttributeFilter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            AttributeFilter::Not(filter) => !filter.matches(metadata),
        }
    }
}

/// Helper to compare a numeric metadata value against a filter value.
fn compare_metadata_numeric(
    metadata: &HashMap<String, Value>,
    field: &str,
    filter_value: &FilterValue,
    cmp: fn(f64, f64) -> bool,
) -> bool {
    let Some(meta_val) = metadata.get(field) else {
        return false;
    };
    let meta_f64 = match meta_val {
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    let filter_f64 = filter_value.as_f64();
    match (meta_f64, filter_f64) {
        (Some(a), Some(b)) => cmp(a, b),
        _ => false,
    }
}

// ─── Structured Query ───

/// A parsed query with a semantic component and an optional metadata filter.
///
/// Produced by [`QueryConstructor`] from a natural language query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredQuery {
    /// The semantic part of the query for similarity search.
    pub query: String,
    /// Optional metadata filter to apply to results.
    pub filter: Option<AttributeFilter>,
}

// ─── Attribute Info ───

/// Description of a metadata attribute for the query constructor prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeInfo {
    /// The name of the metadata field.
    pub name: String,
    /// The data type (e.g., "string", "integer", "float", "boolean").
    pub data_type: String,
    /// A human-readable description of this attribute.
    pub description: String,
}

impl AttributeInfo {
    /// Create a new attribute info entry.
    pub fn new(
        name: impl Into<String>,
        data_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            description: description.into(),
        }
    }
}

// ─── Query Constructor ───

/// Uses an LLM to parse natural language queries into [`StructuredQuery`] objects.
///
/// The constructor builds a prompt that describes the available metadata attributes
/// and asks the LLM to output a JSON object with `query` and `filter` fields.
pub struct QueryConstructor {
    /// The chat model used to parse queries.
    llm: Arc<dyn BaseChatModel>,
    /// Descriptions of the available metadata attributes.
    attribute_info: Vec<AttributeInfo>,
    /// A description of the document collection for prompt context.
    document_contents: String,
}

impl QueryConstructor {
    /// Create a new query constructor.
    pub fn new(
        llm: Arc<dyn BaseChatModel>,
        attribute_info: Vec<AttributeInfo>,
        document_contents: impl Into<String>,
    ) -> Self {
        Self {
            llm,
            attribute_info,
            document_contents: document_contents.into(),
        }
    }

    /// Build the prompt to send to the LLM.
    fn build_prompt(&self, query: &str) -> String {
        let mut attrs_desc = String::new();
        for attr in &self.attribute_info {
            attrs_desc.push_str(&format!(
                "- \"{}\": type={}, description=\"{}\"\n",
                attr.name, attr.data_type, attr.description
            ));
        }

        format!(
            r#"You are a query parser. Given a natural language query about a collection of documents, extract:
1. A semantic search query (the part about the content/meaning)
2. An optional metadata filter (conditions on document attributes)

The document collection contains: {document_contents}

Available metadata attributes:
{attrs_desc}
Supported filter operators: eq, ne, gt, gte, lt, lte, in, nin, and, or, not

Respond with ONLY a JSON object (no markdown, no explanation) in this exact format:
{{
  "query": "<semantic search query>",
  "filter": <filter object or null>
}}

Filter format examples:
- {{"operator": "eq", "field": "genre", "value": "sci-fi"}}
- {{"operator": "gt", "field": "year", "value": 2020}}
- {{"operator": "in", "field": "genre", "values": ["sci-fi", "action"]}}
- {{"operator": "and", "filters": [{{"operator": "eq", "field": "genre", "value": "sci-fi"}}, {{"operator": "gt", "field": "year", "value": 2020}}]}}
- {{"operator": "not", "filter": {{"operator": "eq", "field": "genre", "value": "horror"}}}}

Query: {query}"#,
            document_contents = self.document_contents,
            attrs_desc = attrs_desc,
            query = query,
        )
    }

    /// Parse a natural language query into a structured query using the LLM.
    pub async fn construct(&self, query: &str) -> Result<StructuredQuery> {
        let prompt = self.build_prompt(query);
        let messages = vec![Message::Human(HumanMessage::new(&prompt))];
        let ai_msg = self.llm.invoke_messages(&messages, None).await?;
        let response_text = ai_msg.base.content.text();
        parse_structured_query(&response_text)
    }
}

/// Parse the LLM response text into a [`StructuredQuery`].
///
/// Extracts JSON from the response, handling possible markdown code fences.
fn parse_structured_query(response: &str) -> Result<StructuredQuery> {
    // Strip markdown code fences if present.
    let trimmed = response.trim();
    let json_str = if trimmed.starts_with("```") {
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        inner
    } else {
        trimmed
    };

    let raw: Value =
        serde_json::from_str(json_str).map_err(|e| RustChainError::OutputParserError {
            message: format!("Failed to parse LLM response as JSON: {e}"),
            observation: Some(response.to_string()),
            llm_output: Some(response.to_string()),
        })?;

    let query = raw
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let filter = if let Some(filter_val) = raw.get("filter") {
        if filter_val.is_null() {
            None
        } else {
            Some(parse_filter(filter_val)?)
        }
    } else {
        None
    };

    Ok(StructuredQuery { query, filter })
}

/// Recursively parse a JSON value into an [`AttributeFilter`].
fn parse_filter(val: &Value) -> Result<AttributeFilter> {
    let obj = val
        .as_object()
        .ok_or_else(|| RustChainError::OutputParserError {
            message: "Filter must be a JSON object".into(),
            observation: Some(val.to_string()),
            llm_output: None,
        })?;

    let operator = obj
        .get("operator")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RustChainError::OutputParserError {
            message: "Filter must have an 'operator' field".into(),
            observation: Some(val.to_string()),
            llm_output: None,
        })?;

    match operator {
        "eq" | "ne" | "gt" | "gte" | "lt" | "lte" => {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: format!("Filter '{operator}' must have a 'field' string"),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?
                .to_string();
            let value = obj
                .get("value")
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: format!("Filter '{operator}' must have a 'value' field"),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?;
            let fv = json_to_filter_value(value)?;
            Ok(match operator {
                "eq" => AttributeFilter::Eq { field, value: fv },
                "ne" => AttributeFilter::Ne { field, value: fv },
                "gt" => AttributeFilter::Gt { field, value: fv },
                "gte" => AttributeFilter::Gte { field, value: fv },
                "lt" => AttributeFilter::Lt { field, value: fv },
                "lte" => AttributeFilter::Lte { field, value: fv },
                _ => unreachable!(),
            })
        }
        "in" | "nin" => {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: format!("Filter '{operator}' must have a 'field' string"),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?
                .to_string();
            let values_arr = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: format!("Filter '{operator}' must have a 'values' array"),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?;
            let values: Result<Vec<FilterValue>> =
                values_arr.iter().map(json_to_filter_value).collect();
            let values = values?;
            Ok(match operator {
                "in" => AttributeFilter::In { field, values },
                "nin" => AttributeFilter::Nin { field, values },
                _ => unreachable!(),
            })
        }
        "and" | "or" => {
            let filters_arr = obj
                .get("filters")
                .and_then(|v| v.as_array())
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: format!("Filter '{operator}' must have a 'filters' array"),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?;
            let filters: Result<Vec<AttributeFilter>> =
                filters_arr.iter().map(parse_filter).collect();
            let filters = filters?;
            Ok(match operator {
                "and" => AttributeFilter::And(filters),
                "or" => AttributeFilter::Or(filters),
                _ => unreachable!(),
            })
        }
        "not" => {
            let inner = obj
                .get("filter")
                .ok_or_else(|| RustChainError::OutputParserError {
                    message: "Filter 'not' must have a 'filter' field".into(),
                    observation: Some(val.to_string()),
                    llm_output: None,
                })?;
            let inner_filter = parse_filter(inner)?;
            Ok(AttributeFilter::Not(Box::new(inner_filter)))
        }
        other => Err(RustChainError::OutputParserError {
            message: format!("Unknown filter operator: '{other}'"),
            observation: Some(val.to_string()),
            llm_output: None,
        }),
    }
}

/// Convert a JSON value to a [`FilterValue`].
fn json_to_filter_value(val: &Value) -> Result<FilterValue> {
    match val {
        Value::String(s) => Ok(FilterValue::String(s.clone())),
        Value::Bool(b) => Ok(FilterValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(FilterValue::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(FilterValue::Float(f))
            } else {
                Err(RustChainError::OutputParserError {
                    message: format!("Cannot convert number to FilterValue: {n}"),
                    observation: None,
                    llm_output: None,
                })
            }
        }
        _ => Err(RustChainError::OutputParserError {
            message: format!("Unsupported filter value type: {val}"),
            observation: None,
            llm_output: None,
        }),
    }
}

// ─── Self Query Retriever ───

/// A retriever that uses an LLM to parse natural language queries into
/// structured queries with semantic and metadata filter components.
///
/// On each call to `get_relevant_documents`:
/// 1. The `QueryConstructor` sends the query to an LLM to extract the semantic
///    search query and optional metadata filter.
/// 2. The semantic query is sent to the wrapped `VectorStore` for similarity search.
/// 3. Metadata filters are applied client-side to the results.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use rustchain::retrievers::self_query::{SelfQueryRetriever, AttributeInfo};
///
/// let retriever = SelfQueryRetriever::builder()
///     .vectorstore(vectorstore)
///     .llm(llm)
///     .document_contents("A collection of movies")
///     .attribute_info(vec![
///         AttributeInfo::new("genre", "string", "The genre of the movie"),
///         AttributeInfo::new("year", "integer", "The release year"),
///     ])
///     .build();
///
/// let docs = retriever.get_relevant_documents("sci-fi movies after 2020").await?;
/// ```
pub struct SelfQueryRetriever {
    /// The vector store to search.
    vectorstore: Arc<dyn VectorStore>,
    /// The query constructor that parses natural language into structured queries.
    query_constructor: QueryConstructor,
    /// Number of documents to retrieve from the vector store before filtering.
    k: usize,
    /// Whether to apply metadata filters client-side.
    enable_filter: bool,
}

/// Builder for [`SelfQueryRetriever`].
pub struct SelfQueryRetrieverBuilder {
    vectorstore: Option<Arc<dyn VectorStore>>,
    llm: Option<Arc<dyn BaseChatModel>>,
    document_contents: Option<String>,
    attribute_info: Vec<AttributeInfo>,
    k: usize,
    enable_filter: bool,
}

impl SelfQueryRetrieverBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            vectorstore: None,
            llm: None,
            document_contents: None,
            attribute_info: Vec::new(),
            k: 4,
            enable_filter: true,
        }
    }

    /// Set the vector store (required).
    pub fn vectorstore(mut self, vectorstore: Arc<dyn VectorStore>) -> Self {
        self.vectorstore = Some(vectorstore);
        self
    }

    /// Set the LLM for query parsing (required).
    pub fn llm(mut self, llm: Arc<dyn BaseChatModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the document collection description (required).
    pub fn document_contents(mut self, description: impl Into<String>) -> Self {
        self.document_contents = Some(description.into());
        self
    }

    /// Set the metadata attribute descriptions.
    pub fn attribute_info(mut self, info: Vec<AttributeInfo>) -> Self {
        self.attribute_info = info;
        self
    }

    /// Set the number of documents to retrieve. Default: 4.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Enable or disable client-side metadata filtering. Default: true.
    pub fn enable_filter(mut self, enable: bool) -> Self {
        self.enable_filter = enable;
        self
    }

    /// Build the [`SelfQueryRetriever`].
    ///
    /// # Panics
    ///
    /// Panics if `vectorstore`, `llm`, or `document_contents` is not set.
    pub fn build(self) -> SelfQueryRetriever {
        let vectorstore = self
            .vectorstore
            .expect("vectorstore is required for SelfQueryRetriever");
        let llm = self.llm.expect("llm is required for SelfQueryRetriever");
        let document_contents = self
            .document_contents
            .expect("document_contents is required for SelfQueryRetriever");

        let query_constructor = QueryConstructor::new(llm, self.attribute_info, document_contents);

        SelfQueryRetriever {
            vectorstore,
            query_constructor,
            k: self.k,
            enable_filter: self.enable_filter,
        }
    }
}

impl Default for SelfQueryRetrieverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfQueryRetriever {
    /// Create a new builder.
    pub fn builder() -> SelfQueryRetrieverBuilder {
        SelfQueryRetrieverBuilder::new()
    }
}

#[async_trait]
impl BaseRetriever for SelfQueryRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Step 1: Parse the query into a structured query.
        let structured = self.query_constructor.construct(query).await?;

        // Step 2: Perform similarity search with the semantic query.
        let search_query = if structured.query.is_empty() {
            query.to_string()
        } else {
            structured.query
        };
        let docs = self
            .vectorstore
            .similarity_search(&search_query, self.k)
            .await?;

        // Step 3: Apply metadata filter client-side if enabled.
        if self.enable_filter {
            if let Some(filter) = &structured.filter {
                return Ok(docs
                    .into_iter()
                    .filter(|doc| filter.matches(&doc.metadata))
                    .collect());
            }
        }

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorstores::in_memory::InMemoryVectorStore;
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
    use rustchain_core::language_models::fake::FakeListChatModel;
    use serde_json::json;

    fn make_embeddings() -> Arc<dyn rustchain_core::embeddings::Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    fn fake_llm(responses: Vec<&str>) -> Arc<dyn BaseChatModel> {
        Arc::new(FakeListChatModel::new(
            responses.into_iter().map(String::from).collect(),
        ))
    }

    fn movie_docs() -> Vec<Document> {
        vec![
            Document::new("A mind-bending sci-fi thriller").with_metadata(HashMap::from([
                ("genre".into(), json!("sci-fi")),
                ("year".into(), json!(2023)),
                ("rating".into(), json!(8.5)),
            ])),
            Document::new("A romantic comedy about love").with_metadata(HashMap::from([
                ("genre".into(), json!("comedy")),
                ("year".into(), json!(2019)),
                ("rating".into(), json!(6.2)),
            ])),
            Document::new("An action-packed adventure film").with_metadata(HashMap::from([
                ("genre".into(), json!("action")),
                ("year".into(), json!(2021)),
                ("rating".into(), json!(7.8)),
            ])),
            Document::new("A horror movie set in a haunted house").with_metadata(HashMap::from([
                ("genre".into(), json!("horror")),
                ("year".into(), json!(2022)),
                ("rating".into(), json!(5.9)),
            ])),
        ]
    }

    fn movie_attribute_info() -> Vec<AttributeInfo> {
        vec![
            AttributeInfo::new("genre", "string", "The genre of the movie"),
            AttributeInfo::new("year", "integer", "The release year"),
            AttributeInfo::new("rating", "float", "The movie rating (0-10)"),
        ]
    }

    #[tokio::test]
    async fn test_self_query_with_eq_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "movie", "filter": {"operator": "eq", "field": "genre", "value": "sci-fi"}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("sci-fi movies")
            .await
            .unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata.get("genre").unwrap(), "sci-fi");
    }

    #[tokio::test]
    async fn test_self_query_with_gt_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response =
            r#"{"query": "movie", "filter": {"operator": "gt", "field": "year", "value": 2020}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("movies after 2020")
            .await
            .unwrap();

        // Should return movies from 2021, 2022, 2023
        assert_eq!(docs.len(), 3);
        for doc in &docs {
            let year = doc.metadata.get("year").unwrap().as_i64().unwrap();
            assert!(year > 2020);
        }
    }

    #[tokio::test]
    async fn test_self_query_with_and_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "movie", "filter": {"operator": "and", "filters": [{"operator": "gt", "field": "year", "value": 2020}, {"operator": "gte", "field": "rating", "value": 7.0}]}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("good movies after 2020")
            .await
            .unwrap();

        // sci-fi (2023, 8.5) and action (2021, 7.8) should match
        assert_eq!(docs.len(), 2);
        for doc in &docs {
            let year = doc.metadata.get("year").unwrap().as_i64().unwrap();
            let rating = doc.metadata.get("rating").unwrap().as_f64().unwrap();
            assert!(year > 2020);
            assert!(rating >= 7.0);
        }
    }

    #[tokio::test]
    async fn test_self_query_no_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "thriller movie", "filter": null}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("thriller movies")
            .await
            .unwrap();

        // No filter means all documents returned
        assert_eq!(docs.len(), 4);
    }

    #[tokio::test]
    async fn test_self_query_with_in_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "movie", "filter": {"operator": "in", "field": "genre", "values": ["sci-fi", "action"]}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("sci-fi or action movies")
            .await
            .unwrap();

        assert_eq!(docs.len(), 2);
        for doc in &docs {
            let genre = doc.metadata.get("genre").unwrap().as_str().unwrap();
            assert!(genre == "sci-fi" || genre == "action");
        }
    }

    #[tokio::test]
    async fn test_self_query_with_not_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "movie", "filter": {"operator": "not", "filter": {"operator": "eq", "field": "genre", "value": "horror"}}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("non-horror movies")
            .await
            .unwrap();

        assert_eq!(docs.len(), 3);
        for doc in &docs {
            let genre = doc.metadata.get("genre").unwrap().as_str().unwrap();
            assert_ne!(genre, "horror");
        }
    }

    #[tokio::test]
    async fn test_self_query_with_or_filter() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        let llm_response = r#"{"query": "movie", "filter": {"operator": "or", "filters": [{"operator": "eq", "field": "genre", "value": "comedy"}, {"operator": "eq", "field": "genre", "value": "horror"}]}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .build();

        let docs = retriever
            .get_relevant_documents("comedy or horror movies")
            .await
            .unwrap();

        assert_eq!(docs.len(), 2);
        for doc in &docs {
            let genre = doc.metadata.get("genre").unwrap().as_str().unwrap();
            assert!(genre == "comedy" || genre == "horror");
        }
    }

    #[tokio::test]
    async fn test_self_query_filter_disabled() {
        let embeddings = make_embeddings();
        let store = Arc::new(InMemoryVectorStore::new(embeddings));
        store.add_documents(movie_docs(), None).await.unwrap();

        // Even though filter is provided, it should be ignored when disabled
        let llm_response = r#"{"query": "movie", "filter": {"operator": "eq", "field": "genre", "value": "sci-fi"}}"#;
        let llm = fake_llm(vec![llm_response]);

        let retriever = SelfQueryRetriever::builder()
            .vectorstore(store)
            .llm(llm)
            .document_contents("A collection of movies")
            .attribute_info(movie_attribute_info())
            .k(10)
            .enable_filter(false)
            .build();

        let docs = retriever
            .get_relevant_documents("sci-fi movies")
            .await
            .unwrap();

        // All 4 documents should be returned since filtering is disabled
        assert_eq!(docs.len(), 4);
    }

    #[test]
    fn test_attribute_filter_eq_matches() {
        let metadata = HashMap::from([("genre".into(), json!("sci-fi"))]);
        let filter = AttributeFilter::Eq {
            field: "genre".into(),
            value: FilterValue::String("sci-fi".into()),
        };
        assert!(filter.matches(&metadata));

        let filter_ne = AttributeFilter::Eq {
            field: "genre".into(),
            value: FilterValue::String("action".into()),
        };
        assert!(!filter_ne.matches(&metadata));
    }

    #[test]
    fn test_attribute_filter_ne_matches() {
        let metadata = HashMap::from([("genre".into(), json!("comedy"))]);
        let filter = AttributeFilter::Ne {
            field: "genre".into(),
            value: FilterValue::String("horror".into()),
        };
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_attribute_filter_numeric_comparisons() {
        let metadata = HashMap::from([("year".into(), json!(2022))]);

        assert!(AttributeFilter::Gt {
            field: "year".into(),
            value: FilterValue::Integer(2020)
        }
        .matches(&metadata));

        assert!(!AttributeFilter::Gt {
            field: "year".into(),
            value: FilterValue::Integer(2022)
        }
        .matches(&metadata));

        assert!(AttributeFilter::Gte {
            field: "year".into(),
            value: FilterValue::Integer(2022)
        }
        .matches(&metadata));

        assert!(AttributeFilter::Lt {
            field: "year".into(),
            value: FilterValue::Integer(2025)
        }
        .matches(&metadata));

        assert!(AttributeFilter::Lte {
            field: "year".into(),
            value: FilterValue::Integer(2022)
        }
        .matches(&metadata));
    }

    #[test]
    fn test_attribute_filter_missing_field() {
        let metadata = HashMap::new();

        // Eq on missing field returns false
        assert!(!AttributeFilter::Eq {
            field: "genre".into(),
            value: FilterValue::String("sci-fi".into())
        }
        .matches(&metadata));

        // Ne on missing field returns true (field doesn't equal the value)
        assert!(AttributeFilter::Ne {
            field: "genre".into(),
            value: FilterValue::String("sci-fi".into())
        }
        .matches(&metadata));

        // Nin on missing field returns true
        assert!(AttributeFilter::Nin {
            field: "genre".into(),
            values: vec![FilterValue::String("sci-fi".into())]
        }
        .matches(&metadata));
    }

    #[test]
    fn test_parse_structured_query_valid() {
        let response = r#"{"query": "sci-fi movies", "filter": {"operator": "eq", "field": "genre", "value": "sci-fi"}}"#;
        let result = parse_structured_query(response).unwrap();
        assert_eq!(result.query, "sci-fi movies");
        assert!(result.filter.is_some());
    }

    #[test]
    fn test_parse_structured_query_with_code_fence() {
        let response = "```json\n{\"query\": \"test\", \"filter\": null}\n```";
        let result = parse_structured_query(response).unwrap();
        assert_eq!(result.query, "test");
        assert!(result.filter.is_none());
    }

    #[test]
    fn test_parse_structured_query_invalid_json() {
        let response = "this is not json";
        let result = parse_structured_query(response);
        assert!(result.is_err());
    }
}
