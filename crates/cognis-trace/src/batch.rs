//! `Batcher<T>` — bounded queue + background flush task. Each exporter
//! gets its own batcher so a slow backend doesn't block the bridge.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// Shared statistics counted across the batcher's lifetime.
#[derive(Debug, Default)]
pub struct BatcherStats {
    /// Items successfully sent to the flush callback.
    pub sent: AtomicUsize,
    /// Items dropped because the queue was full.
    pub dropped: AtomicUsize,
    /// Flush callbacks that returned an error.
    pub failed: AtomicUsize,
}

impl BatcherStats {
    /// Returns `(sent, dropped, failed)`.
    pub fn snapshot(&self) -> (usize, usize, usize) {
        (
            self.sent.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
        )
    }
}

/// Configuration for a `Batcher`.
#[derive(Debug, Clone, Copy)]
pub struct BatcherConfig {
    /// Max items per flush.
    pub max_batch: usize,
    /// Max wait before flushing a partial batch.
    pub flush_interval: Duration,
    /// Max items the queue holds before drops.
    pub queue_capacity: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch: 100,
            flush_interval: Duration::from_secs(1),
            queue_capacity: 10_000,
        }
    }
}

/// Bounded queue + background flush task. Generic over the item type.
pub struct Batcher<T: Send + 'static> {
    tx: mpsc::Sender<T>,
    stats: Arc<BatcherStats>,
    handle: Option<JoinHandle<()>>,
}

impl<T: Send + 'static> Batcher<T> {
    /// Spawn a batcher. `flush` receives a non-empty `Vec<T>` whenever a
    /// batch is ready (size or time). Errors from `flush` increment
    /// `stats.failed`.
    pub fn spawn<F, Fut>(cfg: BatcherConfig, flush: F) -> Self
    where
        F: Fn(Vec<T>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), crate::TraceError>> + Send,
    {
        let (tx, mut rx) = mpsc::channel::<T>(cfg.queue_capacity);
        let stats = Arc::new(BatcherStats::default());
        let stats_for_task = stats.clone();
        let handle = tokio::spawn(async move {
            let mut buf: Vec<T> = Vec::with_capacity(cfg.max_batch);
            let mut deadline = Instant::now() + cfg.flush_interval;
            loop {
                tokio::select! {
                    biased;
                    item = rx.recv() => match item {
                        Some(x) => {
                            buf.push(x);
                            if buf.len() >= cfg.max_batch {
                                Self::do_flush(&mut buf, &flush, &stats_for_task).await;
                                deadline = Instant::now() + cfg.flush_interval;
                            }
                        }
                        None => {
                            if !buf.is_empty() {
                                Self::do_flush(&mut buf, &flush, &stats_for_task).await;
                            }
                            break;
                        }
                    },
                    _ = tokio::time::sleep_until(deadline) => {
                        if !buf.is_empty() {
                            Self::do_flush(&mut buf, &flush, &stats_for_task).await;
                        }
                        deadline = Instant::now() + cfg.flush_interval;
                    }
                }
            }
        });
        Self {
            tx,
            stats,
            handle: Some(handle),
        }
    }

    async fn do_flush<F, Fut>(buf: &mut Vec<T>, flush: &F, stats: &Arc<BatcherStats>)
    where
        F: Fn(Vec<T>) -> Fut,
        Fut: std::future::Future<Output = Result<(), crate::TraceError>>,
    {
        let count = buf.len();
        let batch = std::mem::take(buf);
        match flush(batch).await {
            Ok(()) => {
                stats.sent.fetch_add(count, Ordering::Relaxed);
            }
            Err(e) => {
                stats.failed.fetch_add(count, Ordering::Relaxed);
                tracing::warn!(error = %e, dropped = count, "trace batcher flush failed");
            }
        }
    }

    /// Non-blocking enqueue. Drops on overflow and increments the dropped
    /// counter.
    pub fn send(&self, item: T) {
        if let Err(_e) = self.tx.try_send(item) {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Stats handle (clone-cheap; counters are atomic).
    pub fn stats(&self) -> Arc<BatcherStats> {
        self.stats.clone()
    }

    /// Drop the sender and wait for the background task to drain remaining
    /// items. Use on graceful shutdown.
    pub async fn shutdown(mut self) {
        drop(self.tx);
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tokio::time::sleep;

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_when_batch_full() {
        let collected = Arc::new(Mutex::new(Vec::<u32>::new()));
        let c2 = collected.clone();
        let cfg = BatcherConfig {
            max_batch: 3,
            flush_interval: Duration::from_secs(60),
            queue_capacity: 100,
        };
        let b = Batcher::spawn(cfg, move |batch: Vec<u32>| {
            let c = c2.clone();
            async move {
                c.lock().unwrap().extend(batch);
                Ok(())
            }
        });
        b.send(1);
        b.send(2);
        b.send(3); // triggers flush
        // Yield once so the task can run.
        tokio::task::yield_now().await;
        sleep(Duration::from_millis(1)).await;
        assert_eq!(*collected.lock().unwrap(), vec![1, 2, 3]);
        b.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn flushes_on_interval_with_partial_batch() {
        let collected = Arc::new(Mutex::new(Vec::<u32>::new()));
        let c2 = collected.clone();
        let cfg = BatcherConfig {
            max_batch: 100,
            flush_interval: Duration::from_millis(100),
            queue_capacity: 100,
        };
        let b = Batcher::spawn(cfg, move |batch: Vec<u32>| {
            let c = c2.clone();
            async move {
                c.lock().unwrap().extend(batch);
                Ok(())
            }
        });
        b.send(7);
        sleep(Duration::from_millis(150)).await;
        assert_eq!(*collected.lock().unwrap(), vec![7]);
        b.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drops_count_when_queue_full() {
        let cfg = BatcherConfig {
            max_batch: 1,
            flush_interval: Duration::from_secs(60),
            queue_capacity: 1,
        };
        let b = Batcher::spawn(cfg, |_batch: Vec<u32>| async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        });
        // First send fills the channel; the batcher pulls and gets stuck
        // in the slow flush. Subsequent sends fill again then drop.
        for i in 0..50 {
            b.send(i);
        }
        let (_, dropped, _) = b.stats().snapshot();
        assert!(dropped > 0, "expected some drops, got {dropped}");
    }
}
