//! Unified filter utilities for vector store backends.
//!
//! Provides composable filter expressions that can be evaluated against document
//! metadata, serialized to JSON for cross-backend compatibility, and constructed
//! via a fluent builder API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::documents::Document;

/// Comparison operators supported by metadata filters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    EndsWith,
    Exists,
    NotExists,
}

/// A value used in filter comparisons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<FilterValue>),
    Null,
}

impl FilterValue {
    /// Convert this `FilterValue` into a `serde_json::Value`.
    fn to_json_value(&self) -> Value {
        match self {
            FilterValue::String(s) => Value::String(s.clone()),
            FilterValue::Integer(i) => Value::Number((*i).into()),
            FilterValue::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            FilterValue::Boolean(b) => Value::Bool(*b),
            FilterValue::List(items) => {
                Value::Array(items.iter().map(|v| v.to_json_value()).collect())
            }
            FilterValue::Null => Value::Null,
        }
    }
}

/// A single metadata filter comparing a document field against a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

impl MetadataFilter {
    /// Create a new metadata filter.
    pub fn new(field: impl Into<String>, operator: FilterOperator, value: FilterValue) -> Self {
        Self {
            field: field.into(),
            operator,
            value,
        }
    }

    /// Evaluate this filter against a metadata map.
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self.operator {
            FilterOperator::Exists => metadata.contains_key(&self.field),
            FilterOperator::NotExists => !metadata.contains_key(&self.field),
            _ => {
                let Some(actual) = metadata.get(&self.field) else {
                    return false;
                };
                self.compare(actual)
            }
        }
    }

    /// Compare a JSON value from metadata against the filter value using the operator.
    fn compare(&self, actual: &Value) -> bool {
        match &self.operator {
            FilterOperator::Eq => self.values_equal(actual),
            FilterOperator::Ne => !self.values_equal(actual),
            FilterOperator::Gt => self.compare_ord(actual) == Some(std::cmp::Ordering::Greater),
            FilterOperator::Gte => matches!(
                self.compare_ord(actual),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
            FilterOperator::Lt => self.compare_ord(actual) == Some(std::cmp::Ordering::Less),
            FilterOperator::Lte => matches!(
                self.compare_ord(actual),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            FilterOperator::In => self.value_in_list(actual),
            FilterOperator::NotIn => !self.value_in_list(actual),
            FilterOperator::Contains => {
                self.string_op(actual, |haystack, needle| haystack.contains(needle))
            }
            FilterOperator::StartsWith => {
                self.string_op(actual, |haystack, needle| haystack.starts_with(needle))
            }
            FilterOperator::EndsWith => {
                self.string_op(actual, |haystack, needle| haystack.ends_with(needle))
            }
            FilterOperator::Exists | FilterOperator::NotExists => unreachable!(),
        }
    }

    fn values_equal(&self, actual: &Value) -> bool {
        let expected = self.value.to_json_value();
        // Allow numeric cross-comparison (i64 vs f64)
        match (actual, &expected) {
            (Value::Number(a), Value::Number(b)) => {
                let a_f = a.as_f64().unwrap_or(f64::NAN);
                let b_f = b.as_f64().unwrap_or(f64::NAN);
                a_f == b_f
            }
            _ => *actual == expected,
        }
    }

    fn compare_ord(&self, actual: &Value) -> Option<std::cmp::Ordering> {
        let expected = self.value.to_json_value();
        match (actual, &expected) {
            (Value::Number(a), Value::Number(b)) => {
                let a_f = a.as_f64()?;
                let b_f = b.as_f64()?;
                a_f.partial_cmp(&b_f)
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    fn value_in_list(&self, actual: &Value) -> bool {
        if let FilterValue::List(items) = &self.value {
            items.iter().any(|item| {
                let item_json = item.to_json_value();
                match (actual, &item_json) {
                    (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
                    _ => *actual == item_json,
                }
            })
        } else {
            false
        }
    }

    fn string_op<F>(&self, actual: &Value, op: F) -> bool
    where
        F: Fn(&str, &str) -> bool,
    {
        match (actual, &self.value) {
            (Value::String(a), FilterValue::String(b)) => op(a.as_str(), b.as_str()),
            _ => false,
        }
    }
}

/// A composable filter expression tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum FilterExpression {
    /// A single metadata filter.
    Single(MetadataFilter),
    /// All sub-expressions must match.
    And(Vec<FilterExpression>),
    /// At least one sub-expression must match.
    Or(Vec<FilterExpression>),
    /// The sub-expression must not match.
    Not(Box<FilterExpression>),
}

impl FilterExpression {
    /// Evaluate this expression against a metadata map.
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            FilterExpression::Single(f) => f.matches(metadata),
            FilterExpression::And(exprs) => exprs.iter().all(|e| e.matches(metadata)),
            FilterExpression::Or(exprs) => exprs.iter().any(|e| e.matches(metadata)),
            FilterExpression::Not(expr) => !expr.matches(metadata),
        }
    }
}

/// Intermediate builder for constructing a filter on a specific field.
pub struct FieldFilter {
    field: String,
}

impl FieldFilter {
    /// Equal to the given value.
    pub fn eq(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Eq, value)
    }

    /// Not equal to the given value.
    pub fn ne(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Ne, value)
    }

    /// Greater than the given value.
    pub fn gt(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Gt, value)
    }

    /// Greater than or equal to the given value.
    pub fn gte(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Gte, value)
    }

    /// Less than the given value.
    pub fn lt(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Lt, value)
    }

    /// Less than or equal to the given value.
    pub fn lte(self, value: FilterValue) -> FilterExpression {
        self.build_single(FilterOperator::Lte, value)
    }

    /// Value is contained in the given list.
    pub fn is_in(self, values: Vec<FilterValue>) -> FilterExpression {
        self.build_single(FilterOperator::In, FilterValue::List(values))
    }

    /// Value is not contained in the given list.
    pub fn not_in(self, values: Vec<FilterValue>) -> FilterExpression {
        self.build_single(FilterOperator::NotIn, FilterValue::List(values))
    }

    /// String field contains the given substring.
    pub fn contains(self, value: impl Into<String>) -> FilterExpression {
        self.build_single(FilterOperator::Contains, FilterValue::String(value.into()))
    }

    /// String field starts with the given prefix.
    pub fn starts_with(self, value: impl Into<String>) -> FilterExpression {
        self.build_single(
            FilterOperator::StartsWith,
            FilterValue::String(value.into()),
        )
    }

    /// String field ends with the given suffix.
    pub fn ends_with(self, value: impl Into<String>) -> FilterExpression {
        self.build_single(FilterOperator::EndsWith, FilterValue::String(value.into()))
    }

    /// Field exists in metadata.
    pub fn exists(self) -> FilterExpression {
        self.build_single(FilterOperator::Exists, FilterValue::Null)
    }

    /// Field does not exist in metadata.
    pub fn not_exists(self) -> FilterExpression {
        self.build_single(FilterOperator::NotExists, FilterValue::Null)
    }

    fn build_single(self, operator: FilterOperator, value: FilterValue) -> FilterExpression {
        FilterExpression::Single(MetadataFilter {
            field: self.field,
            operator,
            value,
        })
    }
}

/// Fluent API for constructing filter expressions.
pub struct FilterBuilder;

impl FilterBuilder {
    /// Start building a filter on the given metadata field.
    pub fn field(name: &str) -> FieldFilter {
        FieldFilter {
            field: name.to_string(),
        }
    }

    /// Combine multiple expressions with AND.
    pub fn and(expressions: Vec<FilterExpression>) -> FilterExpression {
        FilterExpression::And(expressions)
    }

    /// Combine multiple expressions with OR.
    pub fn or(expressions: Vec<FilterExpression>) -> FilterExpression {
        FilterExpression::Or(expressions)
    }

    /// Negate an expression.
    pub fn not(expression: FilterExpression) -> FilterExpression {
        FilterExpression::Not(Box::new(expression))
    }
}

/// Filter a slice of documents by a filter expression.
///
/// Returns references to documents whose metadata matches the expression.
pub fn filter_documents<'a>(docs: &'a [Document], filter: &FilterExpression) -> Vec<&'a Document> {
    docs.iter()
        .filter(|doc| filter.matches(&doc.metadata))
        .collect()
}

/// Serialize and deserialize filter expressions to/from JSON.
pub struct FilterSerializer;

impl FilterSerializer {
    /// Serialize a filter expression to a JSON string.
    pub fn to_json(filter: &FilterExpression) -> serde_json::Result<String> {
        serde_json::to_string(filter)
    }

    /// Serialize a filter expression to a pretty-printed JSON string.
    pub fn to_json_pretty(filter: &FilterExpression) -> serde_json::Result<String> {
        serde_json::to_string_pretty(filter)
    }

    /// Deserialize a filter expression from a JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<FilterExpression> {
        serde_json::from_str(json)
    }

    /// Serialize a filter expression to a `serde_json::Value`.
    pub fn to_value(filter: &FilterExpression) -> serde_json::Result<Value> {
        serde_json::to_value(filter)
    }

    /// Deserialize a filter expression from a `serde_json::Value`.
    pub fn from_value(value: Value) -> serde_json::Result<FilterExpression> {
        serde_json::from_value(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn make_doc(content: &str, pairs: &[(&str, Value)]) -> Document {
        Document::new(content).with_metadata(make_metadata(pairs))
    }

    // --- FilterOperator: Eq ---
    #[test]
    fn test_eq_string() {
        let f = MetadataFilter::new(
            "color",
            FilterOperator::Eq,
            FilterValue::String("red".into()),
        );
        let meta = make_metadata(&[("color", Value::String("red".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("color", Value::String("blue".into()))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_eq_integer() {
        let f = MetadataFilter::new("count", FilterOperator::Eq, FilterValue::Integer(42));
        let meta = make_metadata(&[("count", serde_json::json!(42))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("count", serde_json::json!(99))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_eq_boolean() {
        let f = MetadataFilter::new("active", FilterOperator::Eq, FilterValue::Boolean(true));
        let meta = make_metadata(&[("active", Value::Bool(true))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("active", Value::Bool(false))]);
        assert!(!f.matches(&meta2));
    }

    // --- FilterOperator: Ne ---
    #[test]
    fn test_ne() {
        let f = MetadataFilter::new(
            "status",
            FilterOperator::Ne,
            FilterValue::String("draft".into()),
        );
        let meta = make_metadata(&[("status", Value::String("published".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("status", Value::String("draft".into()))]);
        assert!(!f.matches(&meta2));
    }

    // --- FilterOperator: Gt, Gte, Lt, Lte ---
    #[test]
    fn test_gt() {
        let f = MetadataFilter::new("score", FilterOperator::Gt, FilterValue::Float(0.5));
        let meta = make_metadata(&[("score", serde_json::json!(0.8))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("score", serde_json::json!(0.5))]);
        assert!(!f.matches(&meta2));
        let meta3 = make_metadata(&[("score", serde_json::json!(0.2))]);
        assert!(!f.matches(&meta3));
    }

    #[test]
    fn test_gte() {
        let f = MetadataFilter::new("score", FilterOperator::Gte, FilterValue::Integer(10));
        let meta_eq = make_metadata(&[("score", serde_json::json!(10))]);
        assert!(f.matches(&meta_eq));
        let meta_gt = make_metadata(&[("score", serde_json::json!(15))]);
        assert!(f.matches(&meta_gt));
        let meta_lt = make_metadata(&[("score", serde_json::json!(5))]);
        assert!(!f.matches(&meta_lt));
    }

    #[test]
    fn test_lt() {
        let f = MetadataFilter::new("priority", FilterOperator::Lt, FilterValue::Integer(5));
        let meta = make_metadata(&[("priority", serde_json::json!(3))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("priority", serde_json::json!(5))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_lte() {
        let f = MetadataFilter::new("priority", FilterOperator::Lte, FilterValue::Integer(5));
        let meta = make_metadata(&[("priority", serde_json::json!(5))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("priority", serde_json::json!(6))]);
        assert!(!f.matches(&meta2));
    }

    // --- FilterOperator: In, NotIn ---
    #[test]
    fn test_in() {
        let f = MetadataFilter::new(
            "lang",
            FilterOperator::In,
            FilterValue::List(vec![
                FilterValue::String("en".into()),
                FilterValue::String("fr".into()),
            ]),
        );
        let meta = make_metadata(&[("lang", Value::String("fr".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("lang", Value::String("de".into()))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_not_in() {
        let f = MetadataFilter::new(
            "status",
            FilterOperator::NotIn,
            FilterValue::List(vec![
                FilterValue::String("deleted".into()),
                FilterValue::String("archived".into()),
            ]),
        );
        let meta = make_metadata(&[("status", Value::String("active".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("status", Value::String("deleted".into()))]);
        assert!(!f.matches(&meta2));
    }

    // --- FilterOperator: Contains, StartsWith, EndsWith ---
    #[test]
    fn test_contains() {
        let f = MetadataFilter::new(
            "title",
            FilterOperator::Contains,
            FilterValue::String("rust".into()),
        );
        let meta = make_metadata(&[("title", Value::String("learning rust programming".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("title", Value::String("learning python".into()))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_starts_with() {
        let f = MetadataFilter::new(
            "path",
            FilterOperator::StartsWith,
            FilterValue::String("/docs".into()),
        );
        let meta = make_metadata(&[("path", Value::String("/docs/intro.md".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("path", Value::String("/src/main.rs".into()))]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_ends_with() {
        let f = MetadataFilter::new(
            "file",
            FilterOperator::EndsWith,
            FilterValue::String(".rs".into()),
        );
        let meta = make_metadata(&[("file", Value::String("main.rs".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("file", Value::String("main.py".into()))]);
        assert!(!f.matches(&meta2));
    }

    // --- FilterOperator: Exists, NotExists ---
    #[test]
    fn test_exists() {
        let f = MetadataFilter::new("author", FilterOperator::Exists, FilterValue::Null);
        let meta = make_metadata(&[("author", Value::String("alice".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[]);
        assert!(!f.matches(&meta2));
    }

    #[test]
    fn test_not_exists() {
        let f = MetadataFilter::new("deprecated", FilterOperator::NotExists, FilterValue::Null);
        let meta = make_metadata(&[]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("deprecated", Value::Bool(true))]);
        assert!(!f.matches(&meta2));
    }

    // --- Missing field returns false for comparison operators ---
    #[test]
    fn test_missing_field_returns_false() {
        let f = MetadataFilter::new("missing", FilterOperator::Eq, FilterValue::Integer(1));
        let meta = make_metadata(&[]);
        assert!(!f.matches(&meta));
    }

    // --- FilterExpression: And ---
    #[test]
    fn test_and_expression() {
        let expr = FilterExpression::And(vec![
            FilterExpression::Single(MetadataFilter::new(
                "a",
                FilterOperator::Eq,
                FilterValue::Integer(1),
            )),
            FilterExpression::Single(MetadataFilter::new(
                "b",
                FilterOperator::Eq,
                FilterValue::Integer(2),
            )),
        ]);
        let meta_both = make_metadata(&[("a", serde_json::json!(1)), ("b", serde_json::json!(2))]);
        assert!(expr.matches(&meta_both));
        let meta_one = make_metadata(&[("a", serde_json::json!(1)), ("b", serde_json::json!(9))]);
        assert!(!expr.matches(&meta_one));
    }

    // --- FilterExpression: Or ---
    #[test]
    fn test_or_expression() {
        let expr = FilterExpression::Or(vec![
            FilterExpression::Single(MetadataFilter::new(
                "x",
                FilterOperator::Eq,
                FilterValue::String("yes".into()),
            )),
            FilterExpression::Single(MetadataFilter::new(
                "y",
                FilterOperator::Eq,
                FilterValue::String("yes".into()),
            )),
        ]);
        let meta1 = make_metadata(&[("x", Value::String("yes".into()))]);
        assert!(expr.matches(&meta1));
        let meta2 = make_metadata(&[("y", Value::String("yes".into()))]);
        assert!(expr.matches(&meta2));
        let meta3 = make_metadata(&[
            ("x", Value::String("no".into())),
            ("y", Value::String("no".into())),
        ]);
        assert!(!expr.matches(&meta3));
    }

    // --- FilterExpression: Not ---
    #[test]
    fn test_not_expression() {
        let inner = FilterExpression::Single(MetadataFilter::new(
            "status",
            FilterOperator::Eq,
            FilterValue::String("deleted".into()),
        ));
        let expr = FilterExpression::Not(Box::new(inner));
        let meta = make_metadata(&[("status", Value::String("active".into()))]);
        assert!(expr.matches(&meta));
        let meta2 = make_metadata(&[("status", Value::String("deleted".into()))]);
        assert!(!expr.matches(&meta2));
    }

    // --- Nested compound expression ---
    #[test]
    fn test_nested_compound() {
        // (a == 1 AND b == 2) OR (c == 3)
        let expr = FilterExpression::Or(vec![
            FilterExpression::And(vec![
                FilterExpression::Single(MetadataFilter::new(
                    "a",
                    FilterOperator::Eq,
                    FilterValue::Integer(1),
                )),
                FilterExpression::Single(MetadataFilter::new(
                    "b",
                    FilterOperator::Eq,
                    FilterValue::Integer(2),
                )),
            ]),
            FilterExpression::Single(MetadataFilter::new(
                "c",
                FilterOperator::Eq,
                FilterValue::Integer(3),
            )),
        ]);
        let meta1 = make_metadata(&[("a", serde_json::json!(1)), ("b", serde_json::json!(2))]);
        assert!(expr.matches(&meta1));
        let meta2 = make_metadata(&[("c", serde_json::json!(3))]);
        assert!(expr.matches(&meta2));
        let meta3 = make_metadata(&[("a", serde_json::json!(1)), ("b", serde_json::json!(9))]);
        assert!(!expr.matches(&meta3));
    }

    // --- FilterBuilder ---
    #[test]
    fn test_builder_eq() {
        let expr = FilterBuilder::field("name").eq(FilterValue::String("alice".into()));
        let meta = make_metadata(&[("name", Value::String("alice".into()))]);
        assert!(expr.matches(&meta));
    }

    #[test]
    fn test_builder_compound() {
        let expr = FilterBuilder::and(vec![
            FilterBuilder::field("age").gte(FilterValue::Integer(18)),
            FilterBuilder::field("role").eq(FilterValue::String("admin".into())),
        ]);
        let meta = make_metadata(&[
            ("age", serde_json::json!(25)),
            ("role", Value::String("admin".into())),
        ]);
        assert!(expr.matches(&meta));
    }

    #[test]
    fn test_builder_not() {
        let expr = FilterBuilder::not(
            FilterBuilder::field("status").eq(FilterValue::String("banned".into())),
        );
        let meta = make_metadata(&[("status", Value::String("active".into()))]);
        assert!(expr.matches(&meta));
    }

    // --- filter_documents ---
    #[test]
    fn test_filter_documents() {
        let docs = vec![
            make_doc("doc1", &[("lang", Value::String("en".into()))]),
            make_doc("doc2", &[("lang", Value::String("fr".into()))]),
            make_doc("doc3", &[("lang", Value::String("en".into()))]),
        ];
        let filter = FilterBuilder::field("lang").eq(FilterValue::String("en".into()));
        let result = filter_documents(&docs, &filter);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "doc1");
        assert_eq!(result[1].page_content, "doc3");
    }

    #[test]
    fn test_filter_documents_empty_result() {
        let docs = vec![make_doc("doc1", &[("score", serde_json::json!(10))])];
        let filter = FilterBuilder::field("score").gt(FilterValue::Integer(100));
        let result = filter_documents(&docs, &filter);
        assert!(result.is_empty());
    }

    // --- FilterSerializer ---
    #[test]
    fn test_serialize_deserialize_single() {
        let expr = FilterBuilder::field("color").eq(FilterValue::String("red".into()));
        let json = FilterSerializer::to_json(&expr).unwrap();
        let restored = FilterSerializer::from_json(&json).unwrap();
        assert_eq!(expr, restored);
    }

    #[test]
    fn test_serialize_deserialize_compound() {
        let expr = FilterBuilder::and(vec![
            FilterBuilder::field("a").gt(FilterValue::Integer(5)),
            FilterBuilder::or(vec![
                FilterBuilder::field("b").eq(FilterValue::Boolean(true)),
                FilterBuilder::field("c").contains("hello"),
            ]),
        ]);
        let json = FilterSerializer::to_json(&expr).unwrap();
        let restored = FilterSerializer::from_json(&json).unwrap();
        assert_eq!(expr, restored);
    }

    #[test]
    fn test_serialize_to_value_and_back() {
        let expr = FilterBuilder::not(FilterBuilder::field("x").is_in(vec![
            FilterValue::Integer(1),
            FilterValue::Integer(2),
            FilterValue::Integer(3),
        ]));
        let value = FilterSerializer::to_value(&expr).unwrap();
        let restored = FilterSerializer::from_value(value).unwrap();
        assert_eq!(expr, restored);
    }

    // --- Numeric cross-type comparison ---
    #[test]
    fn test_eq_integer_float_cross() {
        let f = MetadataFilter::new("val", FilterOperator::Eq, FilterValue::Integer(10));
        let meta = make_metadata(&[("val", serde_json::json!(10.0))]);
        assert!(f.matches(&meta));
    }

    // --- String ordering ---
    #[test]
    fn test_gt_string() {
        let f = MetadataFilter::new(
            "name",
            FilterOperator::Gt,
            FilterValue::String("alice".into()),
        );
        let meta = make_metadata(&[("name", Value::String("bob".into()))]);
        assert!(f.matches(&meta));
        let meta2 = make_metadata(&[("name", Value::String("alice".into()))]);
        assert!(!f.matches(&meta2));
    }
}
