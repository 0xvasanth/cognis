//! Pretty-prints spans to stdout for local development. Always-on under
//! the default feature set.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use crate::error::TraceError;
use crate::exporter::TraceExporter;
use crate::span::{ScoreRecord, Span};

/// stdout pretty-printer.
pub struct StdoutExporter {
    pretty: AtomicBool,
    out: Mutex<()>, // serialize prints
}

impl Default for StdoutExporter {
    fn default() -> Self {
        Self {
            pretty: AtomicBool::new(true),
            out: Mutex::new(()),
        }
    }
}

impl StdoutExporter {
    /// Pretty-formatted (multi-line) output.
    pub fn pretty() -> Self {
        Self {
            pretty: AtomicBool::new(true),
            out: Mutex::new(()),
        }
    }

    /// Compact single-line JSON output.
    pub fn compact() -> Self {
        Self {
            pretty: AtomicBool::new(false),
            out: Mutex::new(()),
        }
    }
}

#[async_trait]
impl TraceExporter for StdoutExporter {
    async fn export_spans(&self, spans: Vec<Span>) -> Result<(), TraceError> {
        let _g = self.out.lock().unwrap();
        for s in spans {
            let line = if self.pretty.load(Ordering::Relaxed) {
                serde_json::to_string_pretty(&s)?
            } else {
                serde_json::to_string(&s)?
            };
            println!("[trace] {line}");
        }
        Ok(())
    }

    async fn export_scores(&self, scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        let _g = self.out.lock().unwrap();
        for sc in scores {
            println!("[score] {}", serde_json::to_string(&sc)?);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "stdout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::{SpanKind, SpanBuilder};
    use std::time::SystemTime;
    use uuid::Uuid;

    #[tokio::test]
    async fn export_spans_prints_without_error() {
        let id = Uuid::new_v4();
        let span = SpanBuilder::open(id, None, id, SpanKind::Chain, "x", None, SystemTime::now())
            .finish_ok(None, SystemTime::now());
        StdoutExporter::compact().export_spans(vec![span]).await.unwrap();
    }
}
