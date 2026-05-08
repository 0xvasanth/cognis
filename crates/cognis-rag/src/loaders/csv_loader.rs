//! CSV loader — yields one [`Document`] per row.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a CSV file. Behaviour:
/// - Yields one [`Document`] per data row.
/// - If `content_column` is set, that column's value becomes the document
///   content; remaining columns become metadata.
/// - Otherwise, all columns are concatenated `key=value` pairs joined by
///   `\n` and used as content.
pub struct CsvLoader {
    path: PathBuf,
    content_column: Option<String>,
    has_headers: bool,
    delimiter: u8,
}

impl CsvLoader {
    /// Construct a loader for the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            content_column: None,
            has_headers: true,
            delimiter: b',',
        }
    }

    /// Set the column whose value becomes the document content.
    pub fn with_content_column(mut self, name: impl Into<String>) -> Self {
        self.content_column = Some(name.into());
        self
    }

    /// Whether the first row is a header row (default: true).
    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    /// Override the delimiter (default: `,`).
    pub fn with_delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }
}

#[async_trait]
impl DocumentLoader for CsvLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let bytes = tokio::fs::read(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("CsvLoader: read `{}`: {e}", self.path.display()))
        })?;

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(self.has_headers)
            .delimiter(self.delimiter)
            .from_reader(bytes.as_slice());

        let headers: Vec<String> = if self.has_headers {
            rdr.headers()
                .map_err(|e| CognisError::Serialization(format!("CsvLoader: headers: {e}")))?
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let source = self.path.display().to_string();
        let mut docs: Vec<Document> = Vec::new();
        for (i, rec) in rdr.records().enumerate() {
            let rec =
                rec.map_err(|e| CognisError::Serialization(format!("CsvLoader row {i}: {e}")))?;
            let mut content = String::new();
            let mut doc = Document::default_for_row(&source, i);
            for (col_idx, field) in rec.iter().enumerate() {
                let key = headers
                    .get(col_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("col_{col_idx}"));
                if Some(&key) == self.content_column.as_ref() {
                    content = field.to_string();
                } else {
                    doc.metadata
                        .insert(key, serde_json::Value::String(field.to_string()));
                }
            }
            if self.content_column.is_none() {
                content = headers
                    .iter()
                    .zip(rec.iter())
                    .map(|(h, v)| format!("{h}={v}"))
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            doc.content = content;
            docs.push(doc);
        }

        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }
}

impl Document {
    fn default_for_row(source: &str, row: usize) -> Self {
        Self::new(String::new())
            .with_metadata("source", source.to_string())
            .with_metadata("row", serde_json::Value::Number(row.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_rows_with_content_column() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "id,text,kind").unwrap();
        writeln!(f, "1,hello,intro").unwrap();
        writeln!(f, "2,world,outro").unwrap();
        let docs = CsvLoader::new(f.path())
            .with_content_column("text")
            .load_all()
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "hello");
        assert_eq!(docs[0].metadata["id"], "1");
        assert_eq!(docs[0].metadata["kind"], "intro");
    }

    #[tokio::test]
    async fn loads_rows_without_content_column() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "a,b").unwrap();
        writeln!(f, "1,2").unwrap();
        let docs = CsvLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("a=1"));
        assert!(docs[0].content.contains("b=2"));
    }
}
