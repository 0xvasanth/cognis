use std::collections::HashMap;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use serde_json::Value;

/// Trait for combining multiple documents into a single string.
pub trait DocumentCombineStrategy: Send + Sync {
    /// Combine the given documents into a single string using the specified separator.
    fn combine(&self, documents: &[Document], separator: &str) -> Result<String>;
}

/// A strategy that concatenates ("stuffs") all documents together with a separator.
///
/// Each document is optionally formatted using a configurable template before joining.
pub struct StuffStrategy {
    /// Template for formatting each document.
    /// Supports `{page_content}` and `{metadata.KEY}` placeholders.
    /// If `None`, only the raw `page_content` is used.
    document_prompt: Option<String>,
}

impl StuffStrategy {
    /// Create a new `StuffStrategy` with no document prompt (raw page content).
    pub fn new() -> Self {
        Self {
            document_prompt: None,
        }
    }

    /// Create a new `StuffStrategy` with a document prompt template.
    ///
    /// The template may contain `{page_content}` and `{metadata.KEY}` placeholders.
    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            document_prompt: Some(prompt.into()),
        }
    }
}

impl Default for StuffStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentCombineStrategy for StuffStrategy {
    fn combine(&self, documents: &[Document], separator: &str) -> Result<String> {
        let formatted: Vec<String> = documents
            .iter()
            .map(|doc| match &self.document_prompt {
                Some(template) => DocumentFormatter::format_document(doc, template),
                None => doc.page_content.clone(),
            })
            .collect();
        Ok(formatted.join(separator))
    }
}

/// A summary function used by [`CollapseStrategy`] to summarize a group of documents.
pub type SummaryFn = Box<dyn Fn(&[Document]) -> Result<String> + Send + Sync>;

/// A strategy that hierarchically collapses documents by grouping and summarizing
/// until a single result remains.
///
/// Documents are split into groups of `max_docs_per_group`, each group is summarized
/// using the provided summary function, and the process repeats until only one group
/// (and thus one summary) remains.
pub struct CollapseStrategy {
    /// Maximum number of documents per group in each collapse iteration.
    max_docs_per_group: usize,
    /// Function used to summarize a group of documents into a single string.
    summary_fn: SummaryFn,
}

impl CollapseStrategy {
    /// Create a new `CollapseStrategy`.
    ///
    /// - `max_docs_per_group`: Maximum documents per group (must be >= 2).
    /// - `summary_fn`: A function that takes a slice of documents and returns a summary string.
    pub fn new(max_docs_per_group: usize, summary_fn: SummaryFn) -> Self {
        Self {
            max_docs_per_group: max_docs_per_group.max(2),
            summary_fn,
        }
    }
}

impl DocumentCombineStrategy for CollapseStrategy {
    fn combine(&self, documents: &[Document], _separator: &str) -> Result<String> {
        if documents.is_empty() {
            return Ok(String::new());
        }

        let mut current_docs: Vec<Document> = documents.to_vec();

        // Repeatedly collapse until we have one or fewer groups
        while current_docs.len() > self.max_docs_per_group {
            let mut next_docs = Vec::new();
            for chunk in current_docs.chunks(self.max_docs_per_group) {
                let summary = (self.summary_fn)(chunk)?;
                next_docs.push(Document::new(summary));
            }
            current_docs = next_docs;
        }

        // Final summarization of the remaining group
        if current_docs.len() == 1 {
            Ok(current_docs[0].page_content.clone())
        } else {
            let summary = (self.summary_fn)(&current_docs)?;
            Ok(summary)
        }
    }
}

/// Utilities for formatting documents using template strings.
pub struct DocumentFormatter;

impl DocumentFormatter {
    /// Format a single document using a template string.
    ///
    /// Supported placeholders:
    /// - `{page_content}` — replaced with the document's page content
    /// - `{metadata.KEY}` — replaced with the string value of the metadata key,
    ///   or an empty string if the key is missing
    pub fn format_document(doc: &Document, template: &str) -> String {
        let result = template.replace("{page_content}", &doc.page_content);

        // Replace {metadata.KEY} placeholders
        // Find all occurrences of {metadata.SOMETHING}
        let mut output = String::with_capacity(result.len());
        let mut remaining = result.as_str();

        while let Some(start) = remaining.find("{metadata.") {
            output.push_str(&remaining[..start]);
            let after_prefix = &remaining[start + "{metadata.".len()..];
            if let Some(end) = after_prefix.find('}') {
                let key = &after_prefix[..end];
                let value = doc
                    .metadata
                    .get(key)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                output.push_str(&value);
                remaining = &after_prefix[end + 1..];
            } else {
                // No closing brace found, just copy the rest
                output.push_str(&remaining[start..]);
                remaining = "";
                break;
            }
        }
        output.push_str(remaining);

        output
    }

    /// Format multiple documents using a template and join them with a separator.
    pub fn format_documents(docs: &[Document], template: &str, separator: &str) -> String {
        let formatted: Vec<String> = docs
            .iter()
            .map(|doc| Self::format_document(doc, template))
            .collect();
        formatted.join(separator)
    }
}

/// A chain that combines input documents into a single output text using a
/// configurable [`DocumentCombineStrategy`].
///
/// This is the Rust equivalent of LangChain's `StuffDocumentsChain`.
///
/// # Example
///
/// ```rust
/// use rustchain::chains::documents::{StuffDocumentsChain, StuffStrategy};
/// use rustchain_core::documents::Document;
///
/// let chain = StuffDocumentsChain::builder()
///     .strategy(Box::new(StuffStrategy::new()))
///     .build();
///
/// let docs = vec![
///     Document::new("First document"),
///     Document::new("Second document"),
/// ];
/// let result = chain.invoke(&docs).unwrap();
/// assert_eq!(result, "First document\n\nSecond document");
/// ```
pub struct StuffDocumentsChain {
    /// The strategy used to combine documents.
    combine_strategy: Box<dyn DocumentCombineStrategy>,
    /// The key used for input documents (for map-based invocation).
    input_key: String,
    /// The key used for the output text (for map-based invocation).
    output_key: String,
    /// Separator inserted between formatted documents.
    document_separator: String,
    /// Optional per-document template applied before combining.
    /// This overrides the strategy's own formatting if set.
    document_prompt: Option<String>,
}

impl StuffDocumentsChain {
    /// Create a new builder for `StuffDocumentsChain`.
    pub fn builder() -> StuffDocumentsChainBuilder {
        StuffDocumentsChainBuilder::default()
    }

    /// Create a new `StuffDocumentsChain` with a given strategy and default settings.
    pub fn new(strategy: Box<dyn DocumentCombineStrategy>) -> Self {
        Self {
            combine_strategy: strategy,
            input_key: "input_documents".to_string(),
            output_key: "output_text".to_string(),
            document_separator: "\n\n".to_string(),
            document_prompt: None,
        }
    }

    /// Invoke the chain on a slice of documents, returning the combined text.
    pub fn invoke(&self, documents: &[Document]) -> Result<String> {
        self.combine_strategy
            .combine(documents, &self.document_separator)
    }

    /// Invoke the chain and return a map with the output key.
    pub fn invoke_as_map(&self, documents: &[Document]) -> Result<HashMap<String, String>> {
        let result = self.invoke(documents)?;
        let mut map = HashMap::new();
        map.insert(self.output_key.clone(), result);
        Ok(map)
    }

    /// Get the input key name.
    pub fn input_key(&self) -> &str {
        &self.input_key
    }

    /// Get the output key name.
    pub fn output_key(&self) -> &str {
        &self.output_key
    }

    /// Get the document separator.
    pub fn document_separator(&self) -> &str {
        &self.document_separator
    }

    /// Get the optional document prompt.
    pub fn document_prompt(&self) -> Option<&str> {
        self.document_prompt.as_deref()
    }
}

/// Builder for [`StuffDocumentsChain`].
pub struct StuffDocumentsChainBuilder {
    combine_strategy: Option<Box<dyn DocumentCombineStrategy>>,
    input_key: String,
    output_key: String,
    document_separator: String,
    document_prompt: Option<String>,
}

impl Default for StuffDocumentsChainBuilder {
    fn default() -> Self {
        Self {
            combine_strategy: None,
            input_key: "input_documents".to_string(),
            output_key: "output_text".to_string(),
            document_separator: "\n\n".to_string(),
            document_prompt: None,
        }
    }
}

impl StuffDocumentsChainBuilder {
    /// Set the document combination strategy.
    pub fn strategy(mut self, strategy: Box<dyn DocumentCombineStrategy>) -> Self {
        self.combine_strategy = Some(strategy);
        self
    }

    /// Set the input key (default: `"input_documents"`).
    pub fn input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the output key (default: `"output_text"`).
    pub fn output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set the document separator (default: `"\n\n"`).
    pub fn document_separator(mut self, sep: impl Into<String>) -> Self {
        self.document_separator = sep.into();
        self
    }

    /// Set a per-document prompt template.
    pub fn document_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.document_prompt = Some(prompt.into());
        self
    }

    /// Build the `StuffDocumentsChain`.
    ///
    /// If no strategy is provided, a default [`StuffStrategy`] is used.
    /// If a `document_prompt` is set and no explicit strategy is provided,
    /// the strategy will use that prompt.
    pub fn build(self) -> StuffDocumentsChain {
        let strategy = self
            .combine_strategy
            .unwrap_or_else(|| match &self.document_prompt {
                Some(prompt) => Box::new(StuffStrategy::with_prompt(prompt.clone())),
                None => Box::new(StuffStrategy::new()),
            });

        StuffDocumentsChain {
            combine_strategy: strategy,
            input_key: self.input_key,
            output_key: self.output_key,
            document_separator: self.document_separator,
            document_prompt: self.document_prompt,
        }
    }
}

/// Convenience factory for creating a [`StuffDocumentsChain`] with a document prompt template.
///
/// # Arguments
///
/// * `prompt_template` - A template string with `{page_content}` and optionally
///   `{metadata.KEY}` placeholders.
///
/// # Example
///
/// ```rust
/// use rustchain::chains::documents::create_stuff_documents_chain;
/// use rustchain_core::documents::Document;
///
/// let chain = create_stuff_documents_chain("Content: {page_content}");
/// let docs = vec![Document::new("Hello"), Document::new("World")];
/// let result = chain.invoke(&docs).unwrap();
/// assert_eq!(result, "Content: Hello\n\nContent: World");
/// ```
pub fn create_stuff_documents_chain(prompt_template: &str) -> StuffDocumentsChain {
    StuffDocumentsChain::builder()
        .strategy(Box::new(StuffStrategy::with_prompt(prompt_template)))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_metadata(content: &str, metadata: HashMap<String, Value>) -> Document {
        Document::new(content).with_metadata(metadata)
    }

    // ── StuffStrategy tests ──────────────────────────────────────────────

    #[test]
    fn test_stuff_strategy_simple_concat() {
        let strategy = StuffStrategy::new();
        let docs = vec![make_doc("Hello"), make_doc("World")];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "Hello\n\nWorld");
    }

    #[test]
    fn test_stuff_strategy_empty_docs() {
        let strategy = StuffStrategy::new();
        let docs: Vec<Document> = vec![];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_stuff_strategy_single_doc() {
        let strategy = StuffStrategy::new();
        let docs = vec![make_doc("Only one")];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "Only one");
    }

    #[test]
    fn test_stuff_strategy_custom_separator() {
        let strategy = StuffStrategy::new();
        let docs = vec![make_doc("A"), make_doc("B"), make_doc("C")];
        let result = strategy.combine(&docs, " | ").unwrap();
        assert_eq!(result, "A | B | C");
    }

    #[test]
    fn test_stuff_strategy_with_prompt() {
        let strategy = StuffStrategy::with_prompt("Content: {page_content}");
        let docs = vec![make_doc("Hello"), make_doc("World")];
        let result = strategy.combine(&docs, "\n").unwrap();
        assert_eq!(result, "Content: Hello\nContent: World");
    }

    #[test]
    fn test_stuff_strategy_with_metadata_prompt() {
        let strategy =
            StuffStrategy::with_prompt("Content: {page_content}\nSource: {metadata.source}");
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), json!("file.txt"));
        let docs = vec![make_doc_with_metadata("Hello", meta)];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "Content: Hello\nSource: file.txt");
    }

    // ── CollapseStrategy tests ───────────────────────────────────────────

    #[test]
    fn test_collapse_strategy_empty() {
        let strategy =
            CollapseStrategy::new(2, Box::new(|_docs: &[Document]| Ok("summary".to_string())));
        let docs: Vec<Document> = vec![];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_collapse_strategy_single_doc() {
        let strategy =
            CollapseStrategy::new(2, Box::new(|_docs: &[Document]| Ok("summary".to_string())));
        let docs = vec![make_doc("Only one")];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "Only one");
    }

    #[test]
    fn test_collapse_strategy_within_group_size() {
        let strategy = CollapseStrategy::new(
            3,
            Box::new(|docs: &[Document]| {
                let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
                Ok(format!("SUMMARY({})", contents.join(", ")))
            }),
        );
        let docs = vec![make_doc("A"), make_doc("B"), make_doc("C")];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        assert_eq!(result, "SUMMARY(A, B, C)");
    }

    #[test]
    fn test_collapse_strategy_hierarchical() {
        // 4 docs with max_group=2: first collapse -> 2 summaries, then final collapse -> 1
        let strategy = CollapseStrategy::new(
            2,
            Box::new(|docs: &[Document]| {
                let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
                Ok(format!("[{}]", contents.join("+")))
            }),
        );
        let docs = vec![make_doc("A"), make_doc("B"), make_doc("C"), make_doc("D")];
        let result = strategy.combine(&docs, "\n\n").unwrap();
        // First round: [A+B], [C+D]
        // Second round: [[A+B]+[C+D]]
        assert_eq!(result, "[[A+B]+[C+D]]");
    }

    #[test]
    fn test_collapse_strategy_min_group_size() {
        // Even if 1 is passed, it should be clamped to 2
        let strategy = CollapseStrategy::new(
            1,
            Box::new(|docs: &[Document]| {
                let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
                Ok(contents.join("+"))
            }),
        );
        assert_eq!(strategy.max_docs_per_group, 2);
    }

    // ── DocumentFormatter tests ──────────────────────────────────────────

    #[test]
    fn test_format_document_page_content() {
        let doc = make_doc("Hello world");
        let result = DocumentFormatter::format_document(&doc, "Text: {page_content}");
        assert_eq!(result, "Text: Hello world");
    }

    #[test]
    fn test_format_document_metadata_key() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), json!("wiki.txt"));
        meta.insert("page".to_string(), json!(42));
        let doc = make_doc_with_metadata("Content here", meta);
        let result = DocumentFormatter::format_document(
            &doc,
            "{page_content} from {metadata.source} page {metadata.page}",
        );
        assert_eq!(result, "Content here from wiki.txt page 42");
    }

    #[test]
    fn test_format_document_missing_metadata_key() {
        let doc = make_doc("Content");
        let result =
            DocumentFormatter::format_document(&doc, "{page_content} source={metadata.source}");
        assert_eq!(result, "Content source=");
    }

    #[test]
    fn test_format_documents_multiple() {
        let docs = vec![make_doc("A"), make_doc("B")];
        let result = DocumentFormatter::format_documents(&docs, "Doc: {page_content}", " | ");
        assert_eq!(result, "Doc: A | Doc: B");
    }

    // ── StuffDocumentsChain tests ────────────────────────────────────────

    #[test]
    fn test_chain_default_builder() {
        let chain = StuffDocumentsChain::builder().build();
        assert_eq!(chain.input_key(), "input_documents");
        assert_eq!(chain.output_key(), "output_text");
        assert_eq!(chain.document_separator(), "\n\n");
    }

    #[test]
    fn test_chain_custom_keys() {
        let chain = StuffDocumentsChain::builder()
            .input_key("docs")
            .output_key("result")
            .build();
        assert_eq!(chain.input_key(), "docs");
        assert_eq!(chain.output_key(), "result");
    }

    #[test]
    fn test_chain_invoke() {
        let chain = StuffDocumentsChain::builder().build();
        let docs = vec![make_doc("First"), make_doc("Second")];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(result, "First\n\nSecond");
    }

    #[test]
    fn test_chain_invoke_as_map() {
        let chain = StuffDocumentsChain::builder().build();
        let docs = vec![make_doc("Alpha"), make_doc("Beta")];
        let map = chain.invoke_as_map(&docs).unwrap();
        assert_eq!(map.get("output_text").unwrap(), "Alpha\n\nBeta");
    }

    #[test]
    fn test_chain_custom_separator() {
        let chain = StuffDocumentsChain::builder()
            .document_separator("---")
            .build();
        let docs = vec![make_doc("X"), make_doc("Y")];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(result, "X---Y");
    }

    #[test]
    fn test_chain_with_document_prompt() {
        let chain = StuffDocumentsChain::builder()
            .document_prompt(">> {page_content}")
            .build();
        let docs = vec![make_doc("Hello"), make_doc("World")];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(result, ">> Hello\n\n>> World");
    }

    #[test]
    fn test_chain_new_constructor() {
        let chain = StuffDocumentsChain::new(Box::new(StuffStrategy::new()));
        let docs = vec![make_doc("Test")];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(result, "Test");
    }

    // ── create_stuff_documents_chain factory tests ───────────────────────

    #[test]
    fn test_factory_function() {
        let chain = create_stuff_documents_chain("Content: {page_content}");
        let docs = vec![make_doc("Hello"), make_doc("World")];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(result, "Content: Hello\n\nContent: World");
    }

    #[test]
    fn test_factory_with_metadata() {
        let chain = create_stuff_documents_chain("{page_content} (source: {metadata.source})");
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), json!("test.pdf"));
        let docs = vec![
            make_doc_with_metadata("Page 1 content", meta.clone()),
            make_doc_with_metadata("Page 2 content", {
                let mut m = HashMap::new();
                m.insert("source".to_string(), json!("test2.pdf"));
                m
            }),
        ];
        let result = chain.invoke(&docs).unwrap();
        assert_eq!(
            result,
            "Page 1 content (source: test.pdf)\n\nPage 2 content (source: test2.pdf)"
        );
    }
}
