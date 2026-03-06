//! Directory document loader.

use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};

use super::csv::CsvLoader;
use super::json::JsonLoader;
use super::text::TextLoader;

/// Loads all matching files in a directory, dispatching to the appropriate
/// loader based on file extension.
///
/// Extension mapping:
/// - `.txt`, `.md` -> [`TextLoader`]
/// - `.json`, `.jsonl` -> [`JsonLoader`]
/// - `.csv` -> [`CsvLoader`]
/// - All other extensions -> [`TextLoader`] (fallback)
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::directory::DirectoryLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = DirectoryLoader::new("data/")
///     .with_glob("*.txt")
///     .recursive(true);
/// let docs = loader.load().await?;
/// # Ok(())
/// # }
/// ```
pub struct DirectoryLoader {
    path: PathBuf,
    glob_pattern: Option<String>,
    recursive: bool,
}

impl DirectoryLoader {
    /// Create a new `DirectoryLoader` for the given directory path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            glob_pattern: None,
            recursive: false,
        }
    }

    /// Set a glob pattern to filter files (e.g. `"*.txt"`). Default is `"*"`.
    pub fn with_glob(mut self, pattern: impl Into<String>) -> Self {
        self.glob_pattern = Some(pattern.into());
        self
    }

    /// Enable or disable recursive directory traversal.
    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Collect matching file paths from the directory.
    fn collect_files(&self) -> Result<Vec<PathBuf>> {
        let pattern = self.glob_pattern.as_deref().unwrap_or("*");
        let glob_str = if self.recursive {
            format!("{}/**/{}", self.path.display(), pattern)
        } else {
            format!("{}/{}", self.path.display(), pattern)
        };

        let mut files: Vec<PathBuf> = Vec::new();
        let entries = glob::glob(&glob_str)
            .map_err(|e| RustChainError::Other(format!("Invalid glob pattern: {}", e)))?;

        for entry in entries {
            let path =
                entry.map_err(|e| RustChainError::Other(format!("Glob entry error: {}", e)))?;
            if path.is_file() {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// Load documents from a single file using the appropriate loader.
    async fn load_file(&self, path: &PathBuf) -> Result<Vec<Document>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "json" | "jsonl" => {
                let loader = JsonLoader::new(path);
                loader.load().await
            }
            "csv" => {
                let loader = CsvLoader::new(path);
                loader.load().await
            }
            // txt, md, and everything else -> TextLoader
            _ => {
                let loader = TextLoader::new(path);
                loader.load().await
            }
        }
    }
}

#[async_trait]
impl BaseLoader for DirectoryLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let files = self.collect_files()?;
        let mut all_docs: Vec<Result<Document>> = Vec::new();

        for file in &files {
            match self.load_file(file).await {
                Ok(docs) => {
                    for doc in docs {
                        all_docs.push(Ok(doc));
                    }
                }
                Err(e) => {
                    all_docs.push(Err(e));
                }
            }
        }

        Ok(Box::pin(stream::iter(all_docs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_directory_loader() {
        let dir = TempDir::new().unwrap();

        // Create a text file.
        fs::write(dir.path().join("notes.txt"), "Some notes here").unwrap();

        // Create a JSON file.
        fs::write(
            dir.path().join("data.json"),
            r#"[{"text":"item1"},{"text":"item2"}]"#,
        )
        .unwrap();

        // Create a CSV file.
        fs::write(dir.path().join("people.csv"), "name,age\nAlice,30\n").unwrap();

        let loader = DirectoryLoader::new(dir.path());
        let docs = loader.load().await.unwrap();

        // 1 (txt) + 2 (json array) + 1 (csv row) = 4
        assert_eq!(docs.len(), 4);
    }
}
