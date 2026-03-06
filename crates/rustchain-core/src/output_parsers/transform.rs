//! Streaming transform output parsers.
//!
//! Mirrors Python `langchain_core.output_parsers.transform`.

use futures::StreamExt;
use serde_json::Value;

use crate::error::Result;
use crate::runnables::RunnableStream;

use super::base::OutputParser;

/// A cumulative transform parser that accumulates streaming chunks
/// and optionally returns diffs between successive parse results.
///
/// Each time a new chunk arrives, the accumulated text is re-parsed.
/// If `diff` is true, only the difference from the previous parse
/// result is yielded.
pub struct CumulativeTransformParser<P: OutputParser> {
    /// The underlying parser.
    pub parser: P,
    /// If true, yield diffs between successive parse results.
    pub diff: bool,
}

impl<P: OutputParser> CumulativeTransformParser<P> {
    pub fn new(parser: P) -> Self {
        Self { parser, diff: false }
    }

    pub fn with_diff(mut self, diff: bool) -> Self {
        self.diff = diff;
        self
    }

    /// Transform a stream by accumulating and parsing.
    pub async fn transform(&self, input: RunnableStream) -> Result<RunnableStream> {
        let mut stream = input;
        let mut accumulated = String::new();
        let mut results = Vec::new();
        let mut _prev_parsed: Option<Value> = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            match &chunk {
                Value::String(s) => accumulated.push_str(s),
                other => accumulated.push_str(&other.to_string()),
            }

            // Try parsing the accumulated text
            if let Ok(parsed) = self.parser.parse(&accumulated) {
                if self.diff {
                    // Only yield if different from previous
                    if _prev_parsed.as_ref() != Some(&parsed) {
                        results.push(Ok(parsed.clone()));
                        _prev_parsed = Some(parsed);
                    }
                } else {
                    results.push(Ok(parsed.clone()));
                    _prev_parsed = Some(parsed);
                }
            }
        }

        // If no results from incremental parsing, do a final parse
        if results.is_empty() {
            let final_parsed = self.parser.parse(&accumulated)?;
            results.push(Ok(final_parsed));
        }

        Ok(Box::pin(futures::stream::iter(results)))
    }
}
