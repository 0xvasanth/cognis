//! In-memory backend — `HashMap<String, String>` rooted at logical paths.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;

use cognis2_core::{CognisError, Result};

use super::{Backend, GrepHit};

/// Holds files in memory keyed by their normalized path string. Useful as
/// a default for tests and tightly-sandboxed agents.
pub struct MemoryBackend {
    files: Mutex<BTreeMap<String, String>>,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend {
    /// Empty backend.
    pub fn new() -> Self {
        Self {
            files: Mutex::new(BTreeMap::new()),
        }
    }

    /// Pre-populate with `(path, contents)` pairs.
    pub fn with_files<I, P, C>(self, files: I) -> Self
    where
        I: IntoIterator<Item = (P, C)>,
        P: Into<String>,
        C: Into<String>,
    {
        {
            let mut map = self.files.lock().unwrap();
            for (p, c) in files {
                map.insert(p.into(), c.into());
            }
        }
        self
    }
}

fn normalize(p: &str) -> String {
    // Strip a leading `./` and collapse `//` runs. We don't resolve `..` in
    // memory paths — refuse any path containing `..` segments.
    p.trim_start_matches("./").replace("//", "/")
}

fn refuse_traversal(p: &str) -> Result<()> {
    if p.split('/').any(|seg| seg == "..") {
        return Err(CognisError::Configuration(format!(
            "MemoryBackend: path traversal not allowed: `{p}`"
        )));
    }
    Ok(())
}

#[async_trait]
impl Backend for MemoryBackend {
    async fn read(&self, path: &str) -> Result<String> {
        refuse_traversal(path)?;
        let p = normalize(path);
        self.files
            .lock()
            .unwrap()
            .get(&p)
            .cloned()
            .ok_or_else(|| CognisError::Configuration(format!("MemoryBackend: not found: `{p}`")))
    }

    async fn write(&self, path: &str, contents: &str) -> Result<()> {
        refuse_traversal(path)?;
        let p = normalize(path);
        self.files.lock().unwrap().insert(p, contents.to_string());
        Ok(())
    }

    async fn edit(
        &self,
        path: &str,
        find: &str,
        replace: &str,
        max_occurrences: usize,
    ) -> Result<usize> {
        refuse_traversal(path)?;
        let p = normalize(path);
        let mut files = self.files.lock().unwrap();
        let body = files
            .get(&p)
            .cloned()
            .ok_or_else(|| CognisError::Configuration(format!("edit: not found: `{p}`")))?;
        let count = body.matches(find).count();
        if count == 0 {
            return Err(CognisError::Configuration(format!(
                "edit: `find` not present in `{p}`"
            )));
        }
        if count > max_occurrences {
            return Err(CognisError::Configuration(format!(
                "edit: `find` occurs {count} times in `{p}`, exceeds max_occurrences={max_occurrences}"
            )));
        }
        let new_body = body.replacen(find, replace, max_occurrences);
        files.insert(p, new_body);
        Ok(count)
    }

    async fn ls(&self, dir: &str) -> Result<Vec<String>> {
        refuse_traversal(dir)?;
        let prefix = if dir.is_empty() || dir == "." {
            String::new()
        } else {
            let mut p = normalize(dir);
            if !p.ends_with('/') {
                p.push('/');
            }
            p
        };
        let map = self.files.lock().unwrap();
        let mut out: Vec<String> = map
            .keys()
            .filter_map(|k| {
                k.strip_prefix(&prefix).and_then(|rest| {
                    if rest.is_empty() || rest.contains('/') {
                        None
                    } else {
                        Some(k.clone())
                    }
                })
            })
            .collect();
        out.sort();
        Ok(out)
    }

    async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        refuse_traversal(pattern)?;
        let map = self.files.lock().unwrap();
        let mut out: Vec<String> = map
            .keys()
            .filter(|k| glob_match(pattern, k))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    async fn grep(&self, pattern: &str) -> Result<Vec<GrepHit>> {
        let map = self.files.lock().unwrap();
        let mut out = Vec::new();
        for (path, body) in map.iter() {
            for (i, line) in body.lines().enumerate() {
                if line.contains(pattern) {
                    out.push(GrepHit {
                        path: path.clone(),
                        line: (i + 1) as u64,
                        text: line.to_string(),
                    });
                }
            }
        }
        Ok(out)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        refuse_traversal(path)?;
        let p = normalize(path);
        Ok(self.files.lock().unwrap().contains_key(&p))
    }
}

/// Match `pattern` against `text` shell-style:
/// - `*` matches any number of non-`/` characters.
/// - `**` matches across path separators.
/// - `**/` at the start matches zero or more leading path segments.
/// - `?` matches exactly one character.
/// - All other characters match literally.
pub(crate) fn glob_match(pattern: &str, text: &str) -> bool {
    // Recursive matcher — handles `**/` (zero or more path segments) cleanly.
    glob_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_inner(pat: &[u8], text: &[u8]) -> bool {
    let mut p = 0;
    let mut t = 0;
    while p < pat.len() {
        match pat[p] {
            b'*' => {
                let double = pat.get(p + 1) == Some(&b'*');
                if double {
                    // `**/...` — try matching zero or more path segments.
                    if pat.get(p + 2) == Some(&b'/') {
                        let rest = &pat[p + 3..];
                        // Try the empty match first (zero segments).
                        if glob_match_inner(rest, &text[t..]) {
                            return true;
                        }
                        // Then try after every `/` boundary.
                        let mut i = t;
                        while i < text.len() {
                            if text[i] == b'/' && glob_match_inner(rest, &text[i + 1..]) {
                                return true;
                            }
                            i += 1;
                        }
                        return false;
                    }
                    // Bare `**` — match anything including `/`.
                    let rest = &pat[p + 2..];
                    for i in t..=text.len() {
                        if glob_match_inner(rest, &text[i..]) {
                            return true;
                        }
                    }
                    return false;
                }
                // Single `*` — match any chars except `/`.
                let rest = &pat[p + 1..];
                for i in t..=text.len() {
                    if glob_match_inner(rest, &text[i..]) {
                        return true;
                    }
                    if text.get(i) == Some(&b'/') {
                        break;
                    }
                }
                return false;
            }
            b'?' => {
                if t >= text.len() || text[t] == b'/' {
                    return false;
                }
                p += 1;
                t += 1;
            }
            c => {
                if t >= text.len() || text[t] != c {
                    return false;
                }
                p += 1;
                t += 1;
            }
        }
    }
    t == text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_write_roundtrip() {
        let b = MemoryBackend::new();
        b.write("a.txt", "hello").await.unwrap();
        assert_eq!(b.read("a.txt").await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn ls_returns_sorted_top_level() {
        let b =
            MemoryBackend::new().with_files([("a.txt", "1"), ("b.txt", "2"), ("sub/c.txt", "3")]);
        let out = b.ls(".").await.unwrap();
        assert_eq!(out, vec!["a.txt", "b.txt"]);
    }

    #[tokio::test]
    async fn ls_under_subdir() {
        let b = MemoryBackend::new().with_files([("sub/x.txt", "1"), ("sub/y.txt", "2")]);
        let out = b.ls("sub").await.unwrap();
        assert_eq!(out, vec!["sub/x.txt", "sub/y.txt"]);
    }

    #[tokio::test]
    async fn glob_simple_and_recursive() {
        let b = MemoryBackend::new().with_files([
            ("a.txt", "1"),
            ("sub/b.txt", "2"),
            ("sub/deep/c.txt", "3"),
            ("z.md", "4"),
        ]);
        assert_eq!(b.glob("*.txt").await.unwrap(), vec!["a.txt"]);
        let mut all = b.glob("**/*.txt").await.unwrap();
        all.sort();
        assert_eq!(all, vec!["a.txt", "sub/b.txt", "sub/deep/c.txt"]);
    }

    #[tokio::test]
    async fn grep_finds_matches() {
        let b = MemoryBackend::new()
            .with_files([("a.txt", "alpha\nbeta\nalpha\n"), ("b.txt", "beta\n")]);
        let hits = b.grep("alpha").await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].path, "a.txt");
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[1].line, 3);
    }

    #[tokio::test]
    async fn edit_replaces_unique_match() {
        let b = MemoryBackend::new().with_files([("a.txt", "hello world")]);
        b.edit("a.txt", "world", "rust", 1).await.unwrap();
        assert_eq!(b.read("a.txt").await.unwrap(), "hello rust");
    }

    #[tokio::test]
    async fn edit_errors_on_missing_target() {
        let b = MemoryBackend::new().with_files([("a.txt", "hello")]);
        assert!(b.edit("a.txt", "world", "rust", 1).await.is_err());
    }

    #[tokio::test]
    async fn edit_errors_on_too_many_occurrences() {
        let b = MemoryBackend::new().with_files([("a.txt", "x x x")]);
        assert!(b.edit("a.txt", "x", "y", 1).await.is_err());
        assert_eq!(b.edit("a.txt", "x", "y", 5).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn refuses_traversal() {
        let b = MemoryBackend::new();
        assert!(b.read("../escape").await.is_err());
        assert!(b.write("../escape", "x").await.is_err());
    }
}
