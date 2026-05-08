//! Structured output: turn a `Client` into a `Runnable<Vec<Message>, T>` for
//! any `T: JsonSchema + DeserializeOwned`.
//!
//! Produces a typed value by:
//! 1. Generating a JSON Schema from `T` via `schemars`.
//! 2. Appending instructions to the conversation telling the model to
//!    reply with JSON matching the schema.
//! 3. Parsing the assistant's reply with [`cognis_core::output_parsers::JsonParser`].

use std::marker::PhantomData;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use cognis_core::output_parsers::{JsonParser, OutputParser};
use cognis_core::{Message, Result, Runnable, RunnableConfig};

use crate::client::Client;

/// `Client` with the output coerced to a typed value `T`.
///
/// Construct via [`Client::with_structured_output`].
pub struct StructuredClient<T> {
    client: Client,
    schema_json: String,
    parser: JsonParser<T>,
    _t: PhantomData<fn() -> T>,
}

impl<T> StructuredClient<T>
where
    T: JsonSchema + DeserializeOwned + Send + 'static,
{
    /// Build a `StructuredClient<T>` over an existing `Client`.
    pub fn new(client: Client) -> Self {
        let schema = schemars::schema_for!(T);
        let schema_json =
            serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".to_string());
        Self {
            client,
            schema_json,
            parser: JsonParser::new(),
            _t: PhantomData,
        }
    }

    /// The JSON Schema generated from `T`.
    pub fn schema(&self) -> &str {
        &self.schema_json
    }

    fn instructions(&self) -> String {
        format!(
            "Reply with a single JSON object matching this JSON Schema. \
             Do not include any text before or after the JSON. Do not wrap \
             the JSON in markdown code fences.\n\nSchema:\n{}",
            self.schema_json
        )
    }
}

#[async_trait]
impl<T> Runnable<Vec<Message>, T> for StructuredClient<T>
where
    T: JsonSchema + DeserializeOwned + Send + 'static,
{
    async fn invoke(&self, mut input: Vec<Message>, _: RunnableConfig) -> Result<T> {
        // Inject schema instructions as a system message at the head.
        input.insert(0, Message::system(self.instructions()));
        let reply = self.client.invoke(input).await?;
        self.parser.parse(reply.content())
    }

    fn name(&self) -> &str {
        "StructuredClient"
    }
}

impl Client {
    /// Coerce this client's output to a typed value `T`.
    ///
    /// `T` must derive `JsonSchema` (for prompt construction) and
    /// `Deserialize` (for parsing). The returned value is itself a
    /// `Runnable<Vec<Message>, T>`, so it composes with `.pipe()`.
    pub fn with_structured_output<T>(self) -> StructuredClient<T>
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
    {
        StructuredClient::new(self)
    }
}
