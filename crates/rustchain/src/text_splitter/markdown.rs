use super::{TextSplitter};
use std::collections::HashMap;
use rustchain_core::documents::Document;
use serde_json::Value;

/// Splits markdown by header hierarchy, preserving header context in metadata.
pub struct MarkdownHeaderTextSplitter {
    pub headers_to_split_on: Vec<(String, String)>,
    pub strip_headers: bool,
}

impl MarkdownHeaderTextSplitter {
    pub fn new(headers_to_split_on: Vec<(&str, &str)>) -> Self {
        Self {
            headers_to_split_on: headers_to_split_on
                .into_iter()
                .map(|(h, name)| (h.to_string(), name.to_string()))
                .collect(),
            strip_headers: true,
        }
    }

    pub fn with_strip_headers(mut self, strip: bool) -> Self {
        self.strip_headers = strip;
        self
    }

    /// Split markdown text into documents based on headers.
    pub fn split_text(&self, text: &str) -> Vec<Document> {
        let mut result = Vec::new();
        let mut current_headers: HashMap<String, Value> = HashMap::new();
        let mut current_content = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            let mut matched_header = None;

            for (marker, name) in &self.headers_to_split_on {
                if trimmed.starts_with(marker.as_str())
                    && trimmed.len() > marker.len()
                    && trimmed.as_bytes()[marker.len()] == b' '
                {
                    matched_header = Some((marker.clone(), name.clone(), trimmed[marker.len()..].trim().to_string()));
                    break;
                }
            }

            if let Some((_marker, name, header_text)) = matched_header {
                // Flush current content
                let content = current_content.trim().to_string();
                if !content.is_empty() {
                    result.push(
                        Document::new(content).with_metadata(
                            current_headers.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                        ),
                    );
                }
                current_content.clear();

                // Update headers (reset lower-level headers)
                let current_level = self.headers_to_split_on
                    .iter()
                    .position(|(_, n)| n == &name)
                    .unwrap_or(0);
                // Remove headers at this level and below
                let names_to_remove: Vec<String> = self.headers_to_split_on[current_level..]
                    .iter()
                    .map(|(_, n)| n.clone())
                    .collect();
                for n in &names_to_remove {
                    current_headers.remove(n);
                }
                current_headers.insert(name, Value::String(header_text));
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }
        }

        // Flush remaining
        let content = current_content.trim().to_string();
        if !content.is_empty() {
            result.push(
                Document::new(content).with_metadata(
                    current_headers.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                ),
            );
        }

        result
    }
}

/// Simple markdown-aware splitter using RecursiveCharacterTextSplitter with markdown separators.
pub struct MarkdownTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for MarkdownTextSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 4000,
            chunk_overlap: 200,
        }
    }
}

impl MarkdownTextSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }
}

impl TextSplitter for MarkdownTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let separators = vec![
            "\n## ", "\n### ", "\n#### ", "\n##### ",
            "\n\n", "\n", " ", "",
        ];
        let splitter = super::RecursiveCharacterTextSplitter::new()
            .with_chunk_size(self.chunk_size)
            .with_chunk_overlap(self.chunk_overlap)
            .with_separators(separators.into_iter().map(|s| s.to_string()).collect());
        splitter.split_text(text)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}
