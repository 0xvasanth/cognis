use serde_json::Value;

/// Recursively splits JSON while preserving structure.
pub struct RecursiveJsonSplitter {
    pub max_chunk_size: usize,
}

impl Default for RecursiveJsonSplitter {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000,
        }
    }
}

impl RecursiveJsonSplitter {
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Split a JSON value into smaller JSON chunks.
    pub fn split_json(&self, data: &Value) -> Vec<Value> {
        let serialized = serde_json::to_string(data).unwrap_or_default();
        if serialized.len() <= self.max_chunk_size {
            return vec![data.clone()];
        }

        match data {
            Value::Object(map) => {
                let mut chunks = Vec::new();
                for (key, value) in map {
                    let sub = serde_json::json!({ key: value });
                    let sub_str = serde_json::to_string(&sub).unwrap_or_default();
                    if sub_str.len() <= self.max_chunk_size {
                        chunks.push(sub);
                    } else {
                        // Recurse into the value
                        let sub_chunks = self.split_json(value);
                        for sc in sub_chunks {
                            chunks.push(serde_json::json!({ key: sc }));
                        }
                    }
                }
                chunks
            }
            Value::Array(arr) => {
                let mut chunks = Vec::new();
                let mut current_batch: Vec<Value> = Vec::new();
                let mut current_size = 2; // []

                for item in arr {
                    let item_str = serde_json::to_string(item).unwrap_or_default();
                    if current_size + item_str.len() + 1 > self.max_chunk_size && !current_batch.is_empty() {
                        chunks.push(Value::Array(std::mem::take(&mut current_batch)));
                        current_size = 2;
                    }
                    current_size += item_str.len() + 1;
                    current_batch.push(item.clone());
                }

                if !current_batch.is_empty() {
                    chunks.push(Value::Array(current_batch));
                }
                chunks
            }
            _ => vec![data.clone()],
        }
    }

    /// Split JSON and return as strings.
    pub fn split_text(&self, json_str: &str) -> Vec<String> {
        let Ok(data) = serde_json::from_str::<Value>(json_str) else {
            return vec![json_str.to_string()];
        };
        self.split_json(&data)
            .into_iter()
            .filter_map(|v| serde_json::to_string(&v).ok())
            .collect()
    }
}
