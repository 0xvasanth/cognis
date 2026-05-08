//! In-memory exporter for tests. Always built (independent of features).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::TraceError;
use crate::exporter::TraceExporter;
use crate::span::{ScoreRecord, Span};

/// In-memory exporter. Spans and scores are appended to internal `Vec`s
/// readable via `collected_spans()` and `collected_scores()`.
#[derive(Default)]
pub struct MockExporter {
    spans: Arc<Mutex<Vec<Span>>>,
    scores: Arc<Mutex<Vec<ScoreRecord>>>,
}

impl MockExporter {
    /// Empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cloneable handle to the collected spans.
    pub fn collected(&self) -> Arc<Mutex<Vec<Span>>> {
        self.spans.clone()
    }

    /// Cloneable handle to the collected scores.
    pub fn collected_scores(&self) -> Arc<Mutex<Vec<ScoreRecord>>> {
        self.scores.clone()
    }
}

#[async_trait]
impl TraceExporter for MockExporter {
    async fn export_spans(&self, spans: Vec<Span>) -> Result<(), TraceError> {
        self.spans.lock().unwrap().extend(spans);
        Ok(())
    }

    async fn export_scores(&self, scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        self.scores.lock().unwrap().extend(scores);
        Ok(())
    }

    fn name(&self) -> &str {
        "mock"
    }
}
