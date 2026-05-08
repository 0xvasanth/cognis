use async_trait::async_trait;

use crate::error::Result;
use crate::outputs::Generation;
use crate::runnables::RunnableStream;

/// Core trait for parsing LLM output into structured types.
///
/// Output parsers convert raw text from language models into typed outputs.
/// They implement `get_format_instructions()` to guide the LLM and `parse()`
/// to extract structured data from the response.
pub trait OutputParser: Send + Sync {
    /// Parse the raw text output from a language model.
    fn parse(&self, text: &str) -> Result<serde_json::Value>;

    /// Parse from a list of generations. Default uses the first generation's text.
    fn parse_result(&self, result: &[Generation], _partial: bool) -> Result<serde_json::Value> {
        if result.is_empty() {
            return Err(crate::error::CognisError::OutputParserError {
                message: "No generations to parse".into(),
                observation: None,
                llm_output: None,
            });
        }
        self.parse(&result[0].text)
    }

    /// Return format instructions to include in the prompt.
    fn get_format_instructions(&self) -> Option<String> {
        None
    }

    /// The parser type name.
    fn parser_type(&self) -> &str;
}

/// Extended output parser that supports streaming transformation.
///
/// Accumulates streaming chunks, then parses and yields the result.
#[async_trait]
pub trait TransformOutputParser: OutputParser {
    /// Transform a stream of chunks by accumulating and parsing.
    async fn transform(&self, input: RunnableStream) -> Result<RunnableStream>;
}

/// Default implementation that accumulates all chunks into a string, parses, and yields.
pub async fn default_transform_stream(
    parser: &dyn OutputParser,
    input: RunnableStream,
) -> Result<RunnableStream> {
    use futures::StreamExt;

    let parser_type = parser.parser_type().to_string();
    let mut chunks = Vec::new();
    let mut stream = input;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        chunks.push(match &chunk {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        });
    }

    let accumulated = chunks.join("");
    let parsed =
        parser
            .parse(&accumulated)
            .map_err(|e| crate::error::CognisError::OutputParserError {
                message: format!("{} transform failed: {}", parser_type, e),
                observation: Some(accumulated),
                llm_output: None,
            })?;

    Ok(Box::pin(futures::stream::once(async { Ok(parsed) })))
}
