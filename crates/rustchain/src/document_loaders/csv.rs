//! CSV file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

/// Loads a CSV file, creating one [`Document`] per row.
///
/// By default, all columns are joined to form `page_content`. You can
/// select specific columns for content and/or metadata.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::csv::CsvLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = CsvLoader::new("data/people.csv")
///     .with_content_columns(vec!["name", "bio"])
///     .with_metadata_columns(vec!["id", "email"]);
/// let docs = loader.load().await?;
/// # Ok(())
/// # }
/// ```
pub struct CsvLoader {
    path: PathBuf,
    /// Columns whose values are joined to form `page_content`.
    content_columns: Option<Vec<String>>,
    /// Columns to include as document metadata.
    metadata_columns: Option<Vec<String>>,
    /// When true, the entire CSV is loaded as a single document instead of one per row.
    single_document: bool,
    /// Custom separator between column key-value pairs (default: `"\n"`).
    separator: String,
}

impl CsvLoader {
    /// Create a new `CsvLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            content_columns: None,
            metadata_columns: None,
            single_document: false,
            separator: "\n".to_string(),
        }
    }

    /// Specify which columns to use as `page_content`.
    pub fn with_content_columns(mut self, cols: Vec<impl Into<String>>) -> Self {
        self.content_columns = Some(cols.into_iter().map(|c| c.into()).collect());
        self
    }

    /// Specify which columns to include as metadata.
    pub fn with_metadata_columns(mut self, cols: Vec<impl Into<String>>) -> Self {
        self.metadata_columns = Some(cols.into_iter().map(|c| c.into()).collect());
        self
    }

    /// Load the entire CSV as a single document instead of one document per row.
    pub fn as_single_document(mut self) -> Self {
        self.single_document = true;
        self
    }

    /// Set a custom separator between column key-value pairs (default: `"\n"`).
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
}

#[async_trait]
impl BaseLoader for CsvLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await?;
        let source = self.path.display().to_string();

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(raw.as_bytes());

        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| RustChainError::Other(format!("CSV header error: {}", e)))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut docs: Vec<Result<Document>> = Vec::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record =
                result.map_err(|e| RustChainError::Other(format!("CSV row error: {}", e)))?;

            // Build a map of column_name -> value for this row.
            let row_map: HashMap<&str, &str> = headers
                .iter()
                .zip(record.iter())
                .map(|(h, v)| (h.as_str(), v))
                .collect();

            // Build page_content.
            let content = match &self.content_columns {
                Some(cols) => cols
                    .iter()
                    .filter_map(|c| row_map.get(c.as_str()).map(|v| format!("{}: {}", c, v)))
                    .collect::<Vec<_>>()
                    .join(&self.separator),
                None => headers
                    .iter()
                    .filter_map(|h| row_map.get(h.as_str()).map(|v| format!("{}: {}", h, v)))
                    .collect::<Vec<_>>()
                    .join(&self.separator),
            };

            // Build metadata.
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), Value::String(source.clone()));
            metadata.insert("row".to_string(), Value::Number((row_idx as u64).into()));

            match &self.metadata_columns {
                Some(cols) => {
                    for col in cols {
                        if let Some(val) = row_map.get(col.as_str()) {
                            metadata.insert(col.clone(), Value::String(val.to_string()));
                        }
                    }
                }
                None => {
                    // When no metadata columns specified, include all columns
                    // that are NOT content columns (if content columns are specified).
                    if let Some(content_cols) = &self.content_columns {
                        for h in &headers {
                            if !content_cols.contains(h) {
                                if let Some(val) = row_map.get(h.as_str()) {
                                    metadata.insert(h.clone(), Value::String(val.to_string()));
                                }
                            }
                        }
                    }
                }
            }

            docs.push(Ok(Document::new(content).with_metadata(metadata)));
        }

        if self.single_document {
            // Merge all row documents into a single document.
            let mut combined_content = String::new();
            for (i, doc_result) in docs.iter().enumerate() {
                if let Ok(doc) = doc_result {
                    if i > 0 {
                        combined_content.push_str("\n\n");
                    }
                    combined_content.push_str(&doc.page_content);
                }
            }
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), Value::String(source));
            metadata.insert(
                "row_count".to_string(),
                Value::Number((docs.len() as u64).into()),
            );
            return Ok(Box::pin(stream::iter(vec![Ok(Document::new(
                combined_content,
            )
            .with_metadata(metadata))])));
        }

        Ok(Box::pin(stream::iter(docs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_csv_loader() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "name,age,city\nAlice,30,NYC\nBob,25,LA\n").unwrap();

        let loader = CsvLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 2);
        assert!(docs[0].page_content.contains("Alice"));
        assert!(docs[0].page_content.contains("30"));
        assert_eq!(
            docs[0].metadata.get("row").unwrap(),
            &Value::Number(0.into())
        );
    }

    #[tokio::test]
    async fn test_csv_loader_specific_columns() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "id,name,bio\n1,Alice,Engineer\n2,Bob,Designer\n").unwrap();

        let loader = CsvLoader::new(tmp.path())
            .with_content_columns(vec!["name", "bio"])
            .with_metadata_columns(vec!["id"]);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 2);
        assert!(docs[0].page_content.contains("name: Alice"));
        assert!(docs[0].page_content.contains("bio: Engineer"));
        // Should NOT contain id in content.
        assert!(!docs[0].page_content.contains("id"));
        // id should be in metadata.
        assert_eq!(
            docs[0].metadata.get("id").unwrap(),
            &Value::String("1".to_string())
        );
    }

    #[tokio::test]
    async fn test_csv_loader_single_document() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "name,age\nAlice,30\nBob,25\nCharlie,35\n").unwrap();

        let loader = CsvLoader::new(tmp.path()).as_single_document();
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
        assert!(docs[0].page_content.contains("Bob"));
        assert!(docs[0].page_content.contains("Charlie"));
        assert_eq!(
            docs[0].metadata.get("row_count").unwrap(),
            &Value::Number(3.into())
        );
    }

    #[tokio::test]
    async fn test_csv_loader_custom_separator() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "name,age\nAlice,30\n").unwrap();

        let loader = CsvLoader::new(tmp.path()).with_separator(" | ");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "name: Alice | age: 30");
    }

    #[tokio::test]
    async fn test_csv_loader_source_metadata() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "x\n1\n").unwrap();

        let loader = CsvLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String(tmp.path().display().to_string())
        );
    }
}
