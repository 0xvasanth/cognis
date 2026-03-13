//! Filesystem-backed key-value store.
//!
//! Each key maps to a file inside a configurable directory. Values are
//! JSON-serialized and writes are atomic (write to a temporary file, then
//! rename).

use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use cognis_core::error::{Result, CognisError};

use super::Store;

/// A key-value store backed by the local filesystem.
///
/// Keys are sanitized to safe filenames. Values are stored as JSON files.
/// Writes are atomic: data is first written to a temporary file and then
/// renamed into place.
#[derive(Debug)]
pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    /// Create a new `FileStore` rooted at `dir`.
    ///
    /// The directory is created if it does not already exist.
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let root = dir.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Sanitize a key into a safe filename component.
    fn key_to_filename(key: &str) -> String {
        key.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
            + ".json"
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(Self::key_to_filename(key))
    }
}

impl Store for FileStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(None);
        }
        let contents = fs::read_to_string(&path)?;
        let value: serde_json::Value = serde_json::from_str(&contents)?;
        match value {
            serde_json::Value::Array(arr) => {
                let bytes: Vec<u8> = arr
                    .into_iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u8)
                    .collect();
                Ok(Some(bytes))
            }
            _ => Err(CognisError::Other(
                "unexpected JSON format in file store".into(),
            )),
        }
    }

    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let path = self.path_for(key);

        // Serialize value as a JSON array of bytes.
        let json_value: Vec<serde_json::Value> =
            value.iter().map(|&b| serde_json::Value::from(b)).collect();
        let serialized = serde_json::to_string(&json_value)?;

        // Atomic write: write to temp file then rename.
        let tmp_path = path.with_extension("tmp");
        {
            let mut file = fs::File::create(&tmp_path)?;
            file.write_all(serialized.as_bytes())?;
            file.sync_all()?;
        }
        fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<bool> {
        let path = self.path_for(key);
        if path.exists() {
            fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn exists(&self, key: &str) -> bool {
        self.path_for(key).exists()
    }

    fn keys(&self) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(key) = name.strip_suffix(".json") {
                result.push(key.to_string());
            }
        }
        Ok(result)
    }

    fn clear(&self) -> Result<()> {
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}
