use rustchain_core::documents::Document;
use serde_json::Value;
use std::collections::HashMap;

/// Splits HTML text based on header tags (h1-h6), preserving header context in metadata.
pub struct HTMLHeaderTextSplitter {
    pub headers_to_split_on: Vec<(String, String)>,
}

impl HTMLHeaderTextSplitter {
    pub fn new(headers_to_split_on: Vec<(&str, &str)>) -> Self {
        Self {
            headers_to_split_on: headers_to_split_on
                .into_iter()
                .map(|(h, name)| (h.to_string(), name.to_string()))
                .collect(),
        }
    }

    /// Split HTML text into documents based on header tags.
    /// This is a simplified parser -- for production use, consider an HTML parser crate.
    pub fn split_text(&self, text: &str) -> Vec<Document> {
        let mut result = Vec::new();
        let mut current_headers: HashMap<String, Value> = HashMap::new();
        let mut current_content = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            let mut matched_header = None;

            for (tag, name) in &self.headers_to_split_on {
                let open = format!("<{}", tag);
                let close = format!("</{}>", tag);
                if let Some(start_pos) = trimmed.to_lowercase().find(&open) {
                    // Extract text between > and </tag>
                    if let Some(gt_pos) = trimmed[start_pos..].find('>') {
                        let after_tag = &trimmed[start_pos + gt_pos + 1..];
                        if let Some(end_pos) = after_tag.to_lowercase().find(&close) {
                            let header_text = after_tag[..end_pos].trim().to_string();
                            matched_header = Some((name.clone(), header_text));
                            break;
                        }
                    }
                }
            }

            if let Some((name, header_text)) = matched_header {
                let content = current_content.trim().to_string();
                if !content.is_empty() {
                    result.push(
                        Document::new(content).with_metadata(
                            current_headers.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                        ),
                    );
                }
                current_content.clear();
                current_headers.insert(name, Value::String(header_text));
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(trimmed);
            }
        }

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
