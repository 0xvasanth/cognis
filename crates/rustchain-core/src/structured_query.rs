//! Structured query IR for translating natural language queries into
//! vectorstore-specific filter expressions.
//!
//! Mirrors Python `langchain_core.structured_query`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;

/// Logical operators for combining filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operator {
    And,
    Or,
    Not,
}

impl Operator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
        }
    }
}

/// Comparison operators for attribute-value comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Comparator {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    Contain,
    Like,
    In,
    Nin,
}

impl Comparator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::Contain => "contain",
            Self::Like => "like",
            Self::In => "in",
            Self::Nin => "nin",
        }
    }
}

/// A comparison expression: `attribute comparator value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub comparator: Comparator,
    pub attribute: String,
    pub value: Value,
}

impl Comparison {
    pub fn new(comparator: Comparator, attribute: impl Into<String>, value: Value) -> Self {
        Self {
            comparator,
            attribute: attribute.into(),
            value,
        }
    }
}

/// A compound filter expression: `operator(arguments...)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub operator: Operator,
    pub arguments: Vec<FilterDirective>,
}

impl Operation {
    pub fn new(operator: Operator, arguments: Vec<FilterDirective>) -> Self {
        Self {
            operator,
            arguments,
        }
    }
}

/// A filter directive is either a comparison or a compound operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FilterDirective {
    Comparison(Comparison),
    Operation(Operation),
}

impl FilterDirective {
    /// Accept a visitor for this directive (visitor pattern).
    pub fn accept(&self, visitor: &dyn Visitor) -> Result<Value> {
        match self {
            Self::Comparison(c) => visitor.visit_comparison(c),
            Self::Operation(o) => visitor.visit_operation(o),
        }
    }
}

/// A structured query: natural language query + optional filter + optional limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredQuery {
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<FilterDirective>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

impl StructuredQuery {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            filter: None,
            limit: None,
        }
    }

    pub fn with_filter(mut self, filter: FilterDirective) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Visitor interface for translating the structured query IR into
/// backend-specific filter syntax (e.g., Chroma, Pinecone, Weaviate).
///
/// Implementors translate comparisons and operations into `Value`
/// representations that the target vectorstore understands.
pub trait Visitor: Send + Sync {
    /// Which comparators this visitor supports.
    fn allowed_comparators(&self) -> &[Comparator];

    /// Which operators this visitor supports.
    fn allowed_operators(&self) -> &[Operator];

    /// Translate a comparison into backend-specific form.
    fn visit_comparison(&self, comparison: &Comparison) -> Result<Value>;

    /// Translate a compound operation into backend-specific form.
    fn visit_operation(&self, operation: &Operation) -> Result<Value>;

    /// Translate a full structured query into backend-specific form.
    fn visit_structured_query(&self, query: &StructuredQuery) -> Result<(String, Value)>;
}
