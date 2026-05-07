//! Language-aware code splitter — uses per-language separator preferences.
//!
//! This is a thin specialization of the recursive splitter that picks
//! sensible separators per language (function/class boundaries, blank
//! lines, then chars). It's not a parser — for AST-aware splitting,
//! pull in `tree-sitter` in a downstream crate.

use crate::document::Document;

use super::{recursive::RecursiveCharSplitter, TextSplitter};

/// Common programming languages we ship default separator orderings for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    /// Rust.
    Rust,
    /// Python.
    Python,
    /// JavaScript / TypeScript.
    JavaScript,
    /// Go.
    Go,
    /// Java.
    Java,
    /// C / C++ / similar curly-brace languages.
    Cpp,
    /// Generic / fallback (paragraph → line → space → char).
    Generic,
}

impl CodeLanguage {
    /// Coarsest-first separator list for this language.
    pub fn separators(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec![
                "\nimpl ",
                "\nfn ",
                "\nstruct ",
                "\nenum ",
                "\ntrait ",
                "\nmod ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            Self::Python => vec!["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " ", ""],
            Self::JavaScript => vec![
                "\nfunction ",
                "\nclass ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            Self::Go => vec!["\nfunc ", "\ntype ", "\n\n", "\n", " ", ""],
            Self::Java => vec![
                "\npublic class ",
                "\nclass ",
                "\npublic ",
                "\nprivate ",
                "\nprotected ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            Self::Cpp => vec!["\nclass ", "\nstruct ", "\nvoid ", "\n\n", "\n", " ", ""],
            Self::Generic => vec!["\n\n", "\n", " ", ""],
        }
    }
}

/// Code-aware splitter. Wraps [`RecursiveCharSplitter`] with language-tuned
/// separators.
pub struct CodeSplitter {
    inner: RecursiveCharSplitter,
}

impl CodeSplitter {
    /// Build a splitter for the given language.
    pub fn new(language: CodeLanguage) -> Self {
        Self {
            inner: RecursiveCharSplitter::new().with_separators(language.separators()),
        }
    }

    /// Cap chunk size.
    pub fn with_chunk_size(mut self, n: usize) -> Self {
        self.inner = self.inner.with_chunk_size(n);
        self
    }

    /// Set chunk overlap.
    pub fn with_overlap(mut self, n: usize) -> Self {
        self.inner = self.inner.with_overlap(n);
        self
    }
}

impl TextSplitter for CodeSplitter {
    fn split(&self, doc: &Document) -> Vec<Document> {
        self.inner.split(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_splits_at_fn_boundary() {
        let code = "fn a() { 1 }\n\nfn b() { 2 }\n\nfn c() { 3 }\n";
        let s = CodeSplitter::new(CodeLanguage::Rust)
            .with_chunk_size(15)
            .with_overlap(0);
        let chunks = s.split(&Document::new(code));
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.content.contains("fn a")));
    }

    #[test]
    fn python_splits_at_def_boundary() {
        let code = "def a():\n    return 1\n\ndef b():\n    return 2\n";
        let s = CodeSplitter::new(CodeLanguage::Python).with_chunk_size(20);
        let chunks = s.split(&Document::new(code));
        assert!(!chunks.is_empty());
    }
}
