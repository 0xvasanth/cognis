//! Vector store memory for semantic long-term recall.
//!
//! Uses a vector store to save conversation exchanges as documents and
//! retrieve the most semantically relevant memories for a given query.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::messages::Message;
use rustchain_core::vectorstores::base::VectorStore;

use super::BaseMemory;

/// Memory backed by a vector store for semantic retrieval of past conversations.
///
/// Instead of returning all past messages, `VectorStoreMemory` retrieves only
/// the most relevant prior exchanges based on embedding similarity to the
/// current input. This is useful for long-running conversations where keeping
/// every message would exceed context limits.
///
/// Each conversation turn is stored as a `Document` with the format
/// `"Human: {input}\nAI: {output}"`. On retrieval, the most semantically
/// similar documents are returned as context.
pub struct VectorStoreMemory {
    vectorstore: Arc<dyn VectorStore>,
    memory_key: String,
    input_key: String,
    /// Number of relevant memories to retrieve.
    k: usize,
    /// Whether to return raw `Document` objects (as JSON) or formatted text.
    return_docs: bool,
    /// IDs of documents added by this memory instance, for potential deletion.
    stored_ids: Arc<Mutex<Vec<String>>>,
    /// The last input query, used by `load_memory_variables` for retrieval.
    last_query: Arc<Mutex<Option<String>>>,
}

impl VectorStoreMemory {
    /// Create a new vector store memory wrapping the given vector store.
    pub fn new(vectorstore: Arc<dyn VectorStore>) -> Self {
        Self {
            vectorstore,
            memory_key: "history".to_string(),
            input_key: "input".to_string(),
            k: 4,
            return_docs: false,
            stored_ids: Arc::new(Mutex::new(Vec::new())),
            last_query: Arc::new(Mutex::new(None)),
        }
    }

    /// Set the memory key used in chain context.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set the input key used to extract the query from chain inputs.
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set the number of relevant memories to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set whether to return raw documents or formatted text.
    pub fn with_return_docs(mut self, return_docs: bool) -> Self {
        self.return_docs = return_docs;
        self
    }

    /// Retrieve the most relevant memories for the given query.
    ///
    /// Returns up to `k` documents (or a custom count) from the vector store
    /// that are most semantically similar to the query.
    pub async fn retrieve_relevant(&self, query: &str, k: Option<usize>) -> Result<Vec<Document>> {
        let num = k.unwrap_or(self.k);
        self.vectorstore.similarity_search(query, num).await
    }

    /// Manually add a memory to the vector store.
    ///
    /// This allows injecting memories that did not come from a conversation
    /// turn, such as background knowledge or user preferences.
    pub async fn add_memory(
        &self,
        text: &str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Result<()> {
        let texts = vec![text.to_string()];
        let metadatas = metadata.map(|m| vec![m]);
        let metadatas_ref = metadatas.as_deref();
        let ids = self
            .vectorstore
            .add_texts(&texts, metadatas_ref, None)
            .await?;
        let mut stored = self.stored_ids.lock().await;
        stored.extend(ids);
        Ok(())
    }

    /// Set the query to use for the next `load_memory_variables` call.
    pub async fn set_query(&self, query: &str) {
        let mut last = self.last_query.lock().await;
        *last = Some(query.to_string());
    }
}

#[async_trait]
impl BaseMemory for VectorStoreMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let mut vars = HashMap::new();

        let query = {
            let last = self.last_query.lock().await;
            last.clone()
        };

        let Some(query) = query else {
            // No query set yet — return empty history.
            if self.return_docs {
                vars.insert(self.memory_key.clone(), Value::Array(vec![]));
            } else {
                vars.insert(self.memory_key.clone(), Value::String(String::new()));
            }
            return Ok(vars);
        };

        let docs = self.retrieve_relevant(&query, None).await?;

        if self.return_docs {
            let serialized: Vec<Value> = docs
                .iter()
                .map(|d| serde_json::to_value(d).unwrap_or(Value::Null))
                .collect();
            vars.insert(self.memory_key.clone(), Value::Array(serialized));
        } else {
            let text = docs
                .iter()
                .map(|d| d.page_content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            vars.insert(self.memory_key.clone(), Value::String(text));
        }

        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        let input_text = input.content().text();
        let output_text = output.content().text();
        let combined = format!("Human: {}\nAI: {}", input_text, output_text);

        // Store the input as the last query for subsequent retrieval.
        {
            let mut last = self.last_query.lock().await;
            *last = Some(input_text.clone());
        }

        let texts = vec![combined];
        let ids = self.vectorstore.add_texts(&texts, None, None).await?;
        let mut stored = self.stored_ids.lock().await;
        stored.extend(ids);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut stored = self.stored_ids.lock().await;
        if !stored.is_empty() {
            let id_list: Vec<String> = stored.drain(..).collect();
            self.vectorstore.delete(Some(&id_list)).await?;
        }
        let mut last = self.last_query.lock().await;
        *last = None;
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorstores::in_memory::InMemoryVectorStore;
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
    use rustchain_core::messages::Message;

    fn make_store() -> Arc<InMemoryVectorStore> {
        let embeddings = Arc::new(DeterministicFakeEmbedding::new(16));
        Arc::new(InMemoryVectorStore::new(embeddings))
    }

    #[tokio::test]
    async fn test_save_and_retrieve_memory() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone());

        mem.add_memory("The user's favorite color is blue.", None)
            .await
            .unwrap();

        let docs = mem
            .retrieve_relevant("What is the user's favorite color?", None)
            .await
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("blue"));
    }

    #[tokio::test]
    async fn test_save_multiple_and_retrieve_most_relevant() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone()).with_k(2);

        mem.add_memory("I love programming in Rust.", None)
            .await
            .unwrap();
        mem.add_memory("The weather in Paris is sunny today.", None)
            .await
            .unwrap();
        mem.add_memory("Rust has zero-cost abstractions.", None)
            .await
            .unwrap();

        let docs = mem
            .retrieve_relevant("Tell me about Rust programming", None)
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_retrieve_with_custom_k() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone());

        mem.add_memory("Fact one", None).await.unwrap();
        mem.add_memory("Fact two", None).await.unwrap();
        mem.add_memory("Fact three", None).await.unwrap();

        let docs = mem.retrieve_relevant("facts", Some(1)).await.unwrap();
        assert_eq!(docs.len(), 1);

        let docs = mem.retrieve_relevant("facts", Some(3)).await.unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn test_save_context_formats_correctly() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone());

        let human = Message::human("What is Rust?");
        let ai = Message::ai("Rust is a systems programming language.");
        mem.save_context(&human, &ai).await.unwrap();

        let docs = mem
            .retrieve_relevant("Rust programming language", None)
            .await
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Human: What is Rust?"));
        assert!(docs[0]
            .page_content
            .contains("AI: Rust is a systems programming language."));
    }

    #[tokio::test]
    async fn test_clear_tracks_ids_for_deletion() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone());

        mem.add_memory("Memory to be cleared", None).await.unwrap();
        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();

        // Verify we have stored IDs.
        {
            let stored = mem.stored_ids.lock().await;
            assert_eq!(stored.len(), 2);
        }

        mem.clear().await.unwrap();

        // IDs should be drained.
        {
            let stored = mem.stored_ids.lock().await;
            assert!(stored.is_empty());
        }

        // Vector store should be empty after deletion.
        let docs = mem.retrieve_relevant("anything", Some(10)).await.unwrap();
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn test_load_memory_variables_after_setting_query() {
        let store = make_store();
        let mem = VectorStoreMemory::new(store.clone());

        // Without a query set, should return empty.
        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());

        // Save a conversation turn (this also sets last_query).
        mem.save_context(
            &Message::human("Tell me about Rust"),
            &Message::ai("Rust is great!"),
        )
        .await
        .unwrap();

        // Now load_memory_variables should retrieve relevant memories.
        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Human: Tell me about Rust"));
        assert!(history.contains("AI: Rust is great!"));

        // Test with explicit set_query.
        mem.set_query("custom query").await;
        let vars = mem.load_memory_variables().await.unwrap();
        // Should still return something (the stored memory is the closest match).
        assert!(vars.contains_key("history"));
    }
}
