//! Structured query translator for translating natural language queries into
//! structured filter queries for vector store metadata filtering.
//!
//! Provides two strategies:
//! - [`RuleBasedTranslator`] -- pattern-matching translator that needs no LLM.
//! - [`TranslatorChain`] -- tries rules first, then falls back to an LLM.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};

// ─── Core Types ───

/// Comparison operators for metadata filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOperator {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    NotIn,
    Contains,
    StartsWith,
}

impl fmt::Display for FilterOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq => write!(f, "=="),
            Self::Ne => write!(f, "!="),
            Self::Gt => write!(f, ">"),
            Self::Gte => write!(f, ">="),
            Self::Lt => write!(f, "<"),
            Self::Lte => write!(f, "<="),
            Self::In => write!(f, "in"),
            Self::NotIn => write!(f, "not in"),
            Self::Contains => write!(f, "contains"),
            Self::StartsWith => write!(f, "starts with"),
        }
    }
}

/// A value used in metadata filter comparisons.
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
    Boolean(bool),
    /// A list of values (for `In` / `NotIn` operators).
    List(Vec<FilterValue>),
}

/// A single metadata filter extracted from a query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryFilter {
    /// The metadata field to filter on.
    pub field: String,
    /// The comparison operator.
    pub operator: FilterOperator,
    /// The comparison value.
    pub value: FilterValue,
}

impl QueryFilter {
    /// Create a new query filter.
    pub fn new(field: impl Into<String>, operator: FilterOperator, value: FilterValue) -> Self {
        Self {
            field: field.into(),
            operator,
            value,
        }
    }
}

/// Sort direction for result ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Specification for sorting results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SortSpec {
    /// The field to sort by.
    pub field: String,
    /// The sort direction.
    pub direction: SortDirection,
}

impl SortSpec {
    /// Create a new sort specification.
    pub fn new(field: impl Into<String>, direction: SortDirection) -> Self {
        Self {
            field: field.into(),
            direction,
        }
    }
}

/// A structured query with semantic search text, metadata filters, limit, and sort.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredQuery {
    /// The semantic search query (stripped of filter terms).
    pub query: String,
    /// Metadata filters extracted from the natural language query.
    pub filters: Vec<QueryFilter>,
    /// Optional limit on the number of results.
    pub limit: Option<usize>,
    /// Optional sort specification.
    pub sort_by: Option<SortSpec>,
}

impl StructuredQuery {
    /// Create a new structured query with only a semantic search string.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            filters: Vec::new(),
            limit: None,
            sort_by: None,
        }
    }

    /// Add a filter to this query.
    pub fn with_filter(mut self, filter: QueryFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the sort specification.
    pub fn with_sort(mut self, sort: SortSpec) -> Self {
        self.sort_by = Some(sort);
        self
    }
}

// ─── Query Schema ───

/// The data type of a metadata field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Integer => write!(f, "integer"),
            Self::Float => write!(f, "float"),
            Self::Boolean => write!(f, "boolean"),
            Self::Date => write!(f, "date"),
        }
    }
}

/// Definition of a metadata field available for filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// The name of the metadata field.
    pub name: String,
    /// The data type of the field.
    pub field_type: FieldType,
    /// A human-readable description of this field.
    pub description: String,
    /// Optional list of allowed values (for enum-like fields).
    pub allowed_values: Option<Vec<String>>,
}

impl FieldDefinition {
    /// Create a new field definition.
    pub fn new(
        name: impl Into<String>,
        field_type: FieldType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field_type,
            description: description.into(),
            allowed_values: None,
        }
    }

    /// Set allowed values for this field.
    pub fn with_allowed_values(mut self, values: Vec<String>) -> Self {
        self.allowed_values = Some(values);
        self
    }
}

/// Schema describing the metadata fields available for filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySchema {
    /// Available metadata fields.
    pub fields: Vec<FieldDefinition>,
    /// Optional description of the document collection.
    pub description: Option<String>,
}

impl QuerySchema {
    /// Create a new query schema.
    pub fn new(fields: Vec<FieldDefinition>) -> Self {
        Self {
            fields,
            description: None,
        }
    }

    /// Set the collection description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Check whether a field name is present in the schema.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    /// Validate that all filters reference fields in this schema.
    pub fn validate_filters(&self, filters: &[QueryFilter]) -> Result<()> {
        for filter in filters {
            if !self.has_field(&filter.field) {
                return Err(RustChainError::Other(format!(
                    "Filter references unknown field '{}'. Available fields: {}",
                    filter.field,
                    self.fields
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }
        Ok(())
    }
}

// ─── QueryTranslator trait ───

/// Trait for translating natural language queries into structured filter queries.
#[async_trait]
pub trait QueryTranslator: Send + Sync {
    /// Translate a natural language query into a structured query.
    async fn translate(&self, natural_query: &str) -> Result<StructuredQuery>;
}

// ─── Rule-Based Translator ───

/// A single translation rule: a regex pattern mapped to a field and operator.
#[derive(Clone)]
struct TranslationRule {
    /// Compiled regex pattern.
    pattern: Regex,
    /// The metadata field this rule maps to.
    field: String,
    /// The filter operator to use.
    operator: FilterOperator,
    /// Which capture group contains the value (0-indexed after full match).
    value_group: usize,
}

/// Rule-based query translator that uses regex pattern matching instead of an LLM.
///
/// Detects common patterns like "category is X", "year > 2020",
/// "price between 10 and 50", "contains keyword".
pub struct RuleBasedTranslator {
    /// The schema describing available metadata fields.
    schema: QuerySchema,
    /// Custom translation rules (checked first).
    custom_rules: Vec<TranslationRule>,
}

impl RuleBasedTranslator {
    /// Create a new rule-based translator with the given schema.
    pub fn new(schema: QuerySchema) -> Self {
        Self {
            schema,
            custom_rules: Vec::new(),
        }
    }

    /// Register a custom translation rule.
    ///
    /// The `pattern` is a regex string. The first capture group is used as the
    /// filter value unless `value_group` is specified.
    pub fn add_rule(
        &mut self,
        pattern: &str,
        field: impl Into<String>,
        operator: FilterOperator,
    ) -> &mut Self {
        let re = Regex::new(pattern).expect("Invalid regex pattern");
        self.custom_rules.push(TranslationRule {
            pattern: re,
            field: field.into(),
            operator,
            value_group: 1,
        });
        self
    }

    /// Extract filters by applying built-in and custom rules to the query.
    fn extract_filters(&self, query: &str) -> Vec<QueryFilter> {
        let mut filters = Vec::new();
        let lower = query.to_lowercase();

        // Apply custom rules first.
        for rule in &self.custom_rules {
            if let Some(caps) = rule.pattern.captures(&lower) {
                if let Some(val) = caps.get(rule.value_group) {
                    let value = Self::parse_value(val.as_str());
                    if self.schema.has_field(&rule.field) {
                        filters.push(QueryFilter::new(rule.field.clone(), rule.operator, value));
                    }
                }
            }
        }

        // Built-in rules for each schema field.
        for field_def in &self.schema.fields {
            let field_name = &field_def.name;
            let field_lower = field_name.to_lowercase();

            // Pattern: "<field> is <value>" or "<field> = <value>"
            let eq_pattern = format!(r"{}(?:\s+is|\s*=)\s+(\S+)", regex::escape(&field_lower));
            if let Ok(re) = Regex::new(&eq_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    if let Some(val) = caps.get(1) {
                        let parsed = Self::parse_value(val.as_str());
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Eq) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Eq,
                                parsed,
                            ));
                        }
                    }
                }
            }

            // Pattern: "<field> > <number>"
            let gt_pattern = format!(r"{}\s*>\s*([\d.]+)", regex::escape(&field_lower));
            if let Ok(re) = Regex::new(&gt_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    if let Some(val) = caps.get(1) {
                        let parsed = Self::parse_value(val.as_str());
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Gt) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Gt,
                                parsed,
                            ));
                        }
                    }
                }
            }

            // Pattern: "<field> >= <number>"
            let gte_pattern = format!(r"{}\s*>=\s*([\d.]+)", regex::escape(&field_lower));
            if let Ok(re) = Regex::new(&gte_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    if let Some(val) = caps.get(1) {
                        let parsed = Self::parse_value(val.as_str());
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Gte) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Gte,
                                parsed,
                            ));
                        }
                    }
                }
            }

            // Pattern: "<field> < <number>"
            let lt_pattern = format!(r"{}\s*<\s*([\d.]+)", regex::escape(&field_lower));
            if let Ok(re) = Regex::new(&lt_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    if let Some(val) = caps.get(1) {
                        let parsed = Self::parse_value(val.as_str());
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Lt) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Lt,
                                parsed,
                            ));
                        }
                    }
                }
            }

            // Pattern: "<field> <= <number>"
            let lte_pattern = format!(r"{}\s*<=\s*([\d.]+)", regex::escape(&field_lower));
            if let Ok(re) = Regex::new(&lte_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    if let Some(val) = caps.get(1) {
                        let parsed = Self::parse_value(val.as_str());
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Lte) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Lte,
                                parsed,
                            ));
                        }
                    }
                }
            }

            // Pattern: "<field> between <low> and <high>"
            let between_pattern = format!(
                r"{}\s+between\s+([\d.]+)\s+and\s+([\d.]+)",
                regex::escape(&field_lower)
            );
            if let Ok(re) = Regex::new(&between_pattern) {
                if let Some(caps) = re.captures(&lower) {
                    let low = caps.get(1).map(|m| Self::parse_value(m.as_str()));
                    let high = caps.get(2).map(|m| Self::parse_value(m.as_str()));
                    if let (Some(lo), Some(hi)) = (low, high) {
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Gte) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Gte,
                                lo,
                            ));
                        }
                        if !Self::filter_exists(&filters, field_name, FilterOperator::Lte) {
                            filters.push(QueryFilter::new(
                                field_name.clone(),
                                FilterOperator::Lte,
                                hi,
                            ));
                        }
                    }
                }
            }

            // Pattern: "contains <value>" (for string fields)
            let contains_pattern =
                format!(r"(?:{}\s+)?contains\s+(\S+)", regex::escape(&field_lower));
            if field_def.field_type == FieldType::String {
                if let Ok(re) = Regex::new(&contains_pattern) {
                    if let Some(caps) = re.captures(&lower) {
                        if let Some(val) = caps.get(1) {
                            let parsed = Self::parse_value(val.as_str());
                            if !Self::filter_exists(&filters, field_name, FilterOperator::Contains)
                            {
                                filters.push(QueryFilter::new(
                                    field_name.clone(),
                                    FilterOperator::Contains,
                                    parsed,
                                ));
                            }
                        }
                    }
                }
            }
        }

        filters
    }

    /// Extract a limit from the query (e.g., "top 5 results").
    fn extract_limit(query: &str) -> Option<usize> {
        let lower = query.to_lowercase();
        let re = Regex::new(r"top\s+(\d+)").unwrap();
        if let Some(caps) = re.captures(&lower) {
            if let Some(m) = caps.get(1) {
                return m.as_str().parse().ok();
            }
        }
        let re2 = Regex::new(r"(?:limit|first)\s+(\d+)").unwrap();
        if let Some(caps) = re2.captures(&lower) {
            if let Some(m) = caps.get(1) {
                return m.as_str().parse().ok();
            }
        }
        None
    }

    /// Strip recognized filter and limit patterns from the query, leaving the
    /// semantic search portion.
    fn extract_semantic_query(&self, query: &str) -> String {
        let mut result = query.to_string();
        // Remove "top N" / "limit N" / "first N"
        let limit_re = Regex::new(r"(?i)(?:top|limit|first)\s+\d+\s*(?:results?)?\s*").unwrap();
        result = limit_re.replace_all(&result, " ").to_string();

        // Remove field filter patterns.
        for field_def in &self.schema.fields {
            let field_lower = field_def.name.to_lowercase();
            let patterns = [
                format!(
                    r"(?i){}\s+between\s+[\d.]+\s+and\s+[\d.]+",
                    regex::escape(&field_lower)
                ),
                format!(
                    r"(?i){}\s*(?:>=|<=|>|<|=)\s*\S+",
                    regex::escape(&field_lower)
                ),
                format!(r"(?i){}\s+is\s+\S+", regex::escape(&field_lower)),
                format!(r"(?i)(?:{}\s+)?contains\s+\S+", regex::escape(&field_lower)),
            ];
            for pat in &patterns {
                if let Ok(re) = Regex::new(pat) {
                    result = re.replace_all(&result, " ").to_string();
                }
            }
        }

        // Collapse whitespace and trim.
        let ws_re = Regex::new(r"\s+").unwrap();
        ws_re.replace_all(result.trim(), " ").trim().to_string()
    }

    /// Parse a string token into a typed `FilterValue`.
    fn parse_value(s: &str) -> FilterValue {
        if let Ok(i) = s.parse::<i64>() {
            return FilterValue::Integer(i);
        }
        if let Ok(f) = s.parse::<f64>() {
            return FilterValue::Float(f);
        }
        match s {
            "true" => FilterValue::Boolean(true),
            "false" => FilterValue::Boolean(false),
            _ => FilterValue::String(s.to_string()),
        }
    }

    /// Check if a filter for a given field and operator already exists.
    fn filter_exists(filters: &[QueryFilter], field: &str, op: FilterOperator) -> bool {
        filters.iter().any(|f| f.field == field && f.operator == op)
    }
}

#[async_trait]
impl QueryTranslator for RuleBasedTranslator {
    async fn translate(&self, natural_query: &str) -> Result<StructuredQuery> {
        let filters = self.extract_filters(natural_query);
        let limit = Self::extract_limit(natural_query);
        let semantic = self.extract_semantic_query(natural_query);

        let query_str = if semantic.is_empty() {
            natural_query.to_string()
        } else {
            semantic
        };

        Ok(StructuredQuery {
            query: query_str,
            filters,
            limit,
            sort_by: None,
        })
    }
}

// ─── TranslatorChain ───

/// Combines a rule-based translator with an optional LLM-based fallback.
///
/// The chain first tries the rule-based translator. If no filters are extracted,
/// it falls back to the LLM (when provided).
pub struct TranslatorChain {
    /// Rule-based translator (always tried first).
    rules: RuleBasedTranslator,
    /// Optional LLM for fallback translation.
    llm: Option<Arc<dyn BaseChatModel>>,
    /// Schema used for LLM prompt construction and validation.
    schema: QuerySchema,
}

impl TranslatorChain {
    /// Create a new translator chain.
    pub fn new(rules: RuleBasedTranslator, llm: Option<Arc<dyn BaseChatModel>>) -> Self {
        let schema = rules.schema.clone();
        Self { rules, llm, schema }
    }

    /// Build a prompt for the LLM to parse the query.
    fn build_llm_prompt(&self, query: &str) -> String {
        let mut fields_desc = String::new();
        for field in &self.schema.fields {
            fields_desc.push_str(&format!(
                "- \"{}\": type={}, description=\"{}\"",
                field.name, field.field_type, field.description
            ));
            if let Some(ref allowed) = field.allowed_values {
                fields_desc.push_str(&format!(", allowed_values={:?}", allowed));
            }
            fields_desc.push('\n');
        }

        let collection_desc = self
            .schema
            .description
            .as_deref()
            .unwrap_or("a document collection");

        format!(
            r#"You are a query parser. Given a natural language query about {collection_desc}, extract:
1. A semantic search query (the part about content/meaning)
2. Metadata filters (conditions on document attributes)
3. Optional result limit
4. Optional sort specification

Available metadata fields:
{fields_desc}
Supported operators: eq, ne, gt, gte, lt, lte, in, not_in, contains, starts_with

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "query": "<semantic search query>",
  "filters": [
    {{"field": "<field_name>", "operator": "<operator>", "value": <value>}}
  ],
  "limit": <number or null>,
  "sort_by": {{"field": "<field_name>", "direction": "asc"|"desc"}} or null
}}

Query: {query}"#,
        )
    }

    /// Parse the LLM response into a structured query.
    fn parse_llm_response(response: &str) -> Result<StructuredQuery> {
        let trimmed = response.trim();
        let json_str = if trimmed.starts_with("```") {
            trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim()
        } else {
            trimmed
        };

        serde_json::from_str(json_str).map_err(|e| RustChainError::OutputParserError {
            message: format!("Failed to parse LLM response as StructuredQuery: {e}"),
            observation: Some(response.to_string()),
            llm_output: Some(response.to_string()),
        })
    }
}

#[async_trait]
impl QueryTranslator for TranslatorChain {
    async fn translate(&self, natural_query: &str) -> Result<StructuredQuery> {
        // Try rule-based first.
        let rule_result = self.rules.translate(natural_query).await?;

        // If rules found filters, use the rule-based result.
        if !rule_result.filters.is_empty() {
            return Ok(rule_result);
        }

        // Fall back to LLM if available.
        if let Some(ref llm) = self.llm {
            let prompt = self.build_llm_prompt(natural_query);
            let messages = vec![Message::Human(HumanMessage::new(&prompt))];
            let ai_msg = llm.invoke_messages(&messages, None).await?;
            let response_text = ai_msg.base.content.text();
            let result = Self::parse_llm_response(&response_text)?;

            // Validate filters against schema.
            self.schema.validate_filters(&result.filters)?;

            return Ok(result);
        }

        // No filters found and no LLM -- return the rule-based result as-is.
        Ok(rule_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn movie_schema() -> QuerySchema {
        QuerySchema::new(vec![
            FieldDefinition::new("category", FieldType::String, "The category of the item"),
            FieldDefinition::new("year", FieldType::Integer, "The release year"),
            FieldDefinition::new("price", FieldType::Float, "The price in dollars"),
            FieldDefinition::new("title", FieldType::String, "The title of the item"),
            FieldDefinition::new(
                "available",
                FieldType::Boolean,
                "Whether the item is available",
            ),
        ])
    }

    // 1. StructuredQuery construction
    #[test]
    fn test_structured_query_construction() {
        let q = StructuredQuery::new("find movies")
            .with_filter(QueryFilter::new(
                "genre",
                FilterOperator::Eq,
                FilterValue::String("action".into()),
            ))
            .with_limit(10)
            .with_sort(SortSpec::new("year", SortDirection::Desc));

        assert_eq!(q.query, "find movies");
        assert_eq!(q.filters.len(), 1);
        assert_eq!(q.limit, Some(10));
        assert!(q.sort_by.is_some());
        assert_eq!(q.sort_by.as_ref().unwrap().field, "year");
        assert_eq!(q.sort_by.as_ref().unwrap().direction, SortDirection::Desc);
    }

    // 2. QueryFilter with various operators
    #[test]
    fn test_query_filter_operators() {
        let ops = vec![
            FilterOperator::Eq,
            FilterOperator::Ne,
            FilterOperator::Gt,
            FilterOperator::Gte,
            FilterOperator::Lt,
            FilterOperator::Lte,
            FilterOperator::In,
            FilterOperator::NotIn,
            FilterOperator::Contains,
            FilterOperator::StartsWith,
        ];
        for op in ops {
            let filter = QueryFilter::new("field", op, FilterValue::String("val".into()));
            assert_eq!(filter.operator, op);
        }
    }

    // 3. FilterValue types
    #[test]
    fn test_filter_value_types() {
        assert_eq!(
            FilterValue::String("hello".into()),
            FilterValue::String("hello".into())
        );
        assert_eq!(FilterValue::Integer(42), FilterValue::Integer(42));
        assert_eq!(FilterValue::Float(3.14), FilterValue::Float(3.14));
        assert_eq!(FilterValue::Boolean(true), FilterValue::Boolean(true));

        let list = FilterValue::List(vec![
            FilterValue::String("a".into()),
            FilterValue::Integer(1),
        ]);
        if let FilterValue::List(items) = &list {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected List variant");
        }
    }

    // 4. QuerySchema with field definitions
    #[test]
    fn test_query_schema_field_definitions() {
        let schema = movie_schema().with_description("A collection of movies");
        assert_eq!(schema.fields.len(), 5);
        assert!(schema.has_field("category"));
        assert!(schema.has_field("year"));
        assert!(!schema.has_field("nonexistent"));
        assert_eq!(
            schema.description.as_deref(),
            Some("A collection of movies")
        );
    }

    // 5. RuleBasedTranslator: "category is X"
    #[tokio::test]
    async fn test_rule_based_category_is() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("find items where category is action")
            .await
            .unwrap();

        assert!(!result.filters.is_empty());
        let cat_filter = result
            .filters
            .iter()
            .find(|f| f.field == "category")
            .unwrap();
        assert_eq!(cat_filter.operator, FilterOperator::Eq);
        assert_eq!(cat_filter.value, FilterValue::String("action".into()));
    }

    // 6. RuleBasedTranslator: "year > 2020"
    #[tokio::test]
    async fn test_rule_based_year_gt() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("movies with year > 2020")
            .await
            .unwrap();

        let year_filter = result.filters.iter().find(|f| f.field == "year").unwrap();
        assert_eq!(year_filter.operator, FilterOperator::Gt);
        assert_eq!(year_filter.value, FilterValue::Integer(2020));
    }

    // 7. RuleBasedTranslator: "price between 10 and 50"
    #[tokio::test]
    async fn test_rule_based_price_between() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("items with price between 10 and 50")
            .await
            .unwrap();

        let gte_filter = result
            .filters
            .iter()
            .find(|f| f.field == "price" && f.operator == FilterOperator::Gte)
            .unwrap();
        let lte_filter = result
            .filters
            .iter()
            .find(|f| f.field == "price" && f.operator == FilterOperator::Lte)
            .unwrap();

        assert_eq!(gte_filter.value, FilterValue::Integer(10));
        assert_eq!(lte_filter.value, FilterValue::Integer(50));
    }

    // 8. RuleBasedTranslator: "contains keyword"
    #[tokio::test]
    async fn test_rule_based_contains_keyword() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("title contains adventure")
            .await
            .unwrap();

        let title_filter = result.filters.iter().find(|f| f.field == "title").unwrap();
        assert_eq!(title_filter.operator, FilterOperator::Contains);
        assert_eq!(title_filter.value, FilterValue::String("adventure".into()));
    }

    // 9. Custom rule registration
    #[tokio::test]
    async fn test_custom_rule_registration() {
        let schema = movie_schema();
        let mut translator = RuleBasedTranslator::new(schema);

        translator.add_rule(r"rated\s+(\S+)", "category", FilterOperator::Eq);

        let result = translator
            .translate("find items rated action")
            .await
            .unwrap();

        let cat_filter = result
            .filters
            .iter()
            .find(|f| f.field == "category")
            .unwrap();
        assert_eq!(cat_filter.operator, FilterOperator::Eq);
        assert_eq!(cat_filter.value, FilterValue::String("action".into()));
    }

    // 10. TranslatorChain rule-first (no LLM needed)
    #[tokio::test]
    async fn test_translator_chain_rule_first() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let chain = TranslatorChain::new(translator, None);

        let result = chain
            .translate("category is drama year > 2019")
            .await
            .unwrap();

        // Rules should have found filters, so no LLM fallback needed.
        assert!(!result.filters.is_empty());
        assert!(result.filters.iter().any(|f| f.field == "category"));
        assert!(result.filters.iter().any(|f| f.field == "year"));
    }

    // 11. Schema validation (filter field must be in schema)
    #[test]
    fn test_schema_validation_unknown_field() {
        let schema = movie_schema();
        let filters = vec![QueryFilter::new(
            "unknown_field",
            FilterOperator::Eq,
            FilterValue::String("value".into()),
        )];

        let result = schema.validate_filters(&filters);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("unknown_field"));
    }

    // 12. Multiple filters in one query
    #[tokio::test]
    async fn test_multiple_filters_in_one_query() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("category is comedy year > 2015 price < 20")
            .await
            .unwrap();

        assert!(result.filters.len() >= 3);
        assert!(result
            .filters
            .iter()
            .any(|f| f.field == "category" && f.operator == FilterOperator::Eq));
        assert!(result
            .filters
            .iter()
            .any(|f| f.field == "year" && f.operator == FilterOperator::Gt));
        assert!(result
            .filters
            .iter()
            .any(|f| f.field == "price" && f.operator == FilterOperator::Lt));
    }

    // 13. Sort specification
    #[test]
    fn test_sort_specification() {
        let sort_asc = SortSpec::new("year", SortDirection::Asc);
        assert_eq!(sort_asc.field, "year");
        assert_eq!(sort_asc.direction, SortDirection::Asc);

        let sort_desc = SortSpec::new("price", SortDirection::Desc);
        assert_eq!(sort_desc.field, "price");
        assert_eq!(sort_desc.direction, SortDirection::Desc);

        let q = StructuredQuery::new("test").with_sort(sort_asc);
        assert!(q.sort_by.is_some());
    }

    // 14. Limit extraction ("top 5 results")
    #[tokio::test]
    async fn test_limit_extraction() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator
            .translate("top 5 results category is action")
            .await
            .unwrap();

        assert_eq!(result.limit, Some(5));
        assert!(!result.filters.is_empty());
    }

    // 15. Empty query handling
    #[tokio::test]
    async fn test_empty_query_handling() {
        let schema = movie_schema();
        let translator = RuleBasedTranslator::new(schema);

        let result = translator.translate("").await.unwrap();

        assert!(result.filters.is_empty());
        assert!(result.limit.is_none());
        assert!(result.sort_by.is_none());
    }

    // 16. FilterOperator display
    #[test]
    fn test_filter_operator_display() {
        assert_eq!(format!("{}", FilterOperator::Eq), "==");
        assert_eq!(format!("{}", FilterOperator::Ne), "!=");
        assert_eq!(format!("{}", FilterOperator::Gt), ">");
        assert_eq!(format!("{}", FilterOperator::Gte), ">=");
        assert_eq!(format!("{}", FilterOperator::Lt), "<");
        assert_eq!(format!("{}", FilterOperator::Lte), "<=");
        assert_eq!(format!("{}", FilterOperator::In), "in");
        assert_eq!(format!("{}", FilterOperator::NotIn), "not in");
        assert_eq!(format!("{}", FilterOperator::Contains), "contains");
        assert_eq!(format!("{}", FilterOperator::StartsWith), "starts with");
    }

    // 17. Schema validation passes for valid filters
    #[test]
    fn test_schema_validation_valid_filters() {
        let schema = movie_schema();
        let filters = vec![
            QueryFilter::new(
                "category",
                FilterOperator::Eq,
                FilterValue::String("action".into()),
            ),
            QueryFilter::new("year", FilterOperator::Gt, FilterValue::Integer(2020)),
        ];

        assert!(schema.validate_filters(&filters).is_ok());
    }

    // 18. FieldDefinition with allowed values
    #[test]
    fn test_field_definition_with_allowed_values() {
        let field = FieldDefinition::new("genre", FieldType::String, "The genre")
            .with_allowed_values(vec!["action".into(), "comedy".into(), "drama".into()]);

        assert_eq!(field.name, "genre");
        assert_eq!(field.field_type, FieldType::String);
        assert!(field.allowed_values.is_some());
        assert_eq!(field.allowed_values.unwrap().len(), 3);
    }
}
