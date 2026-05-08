//! Task-local span stack used as the **fallback** parent-id source when
//! `RunnableConfig.parent_run_id` is `None` at a callback site. Composition
//! sites that propagate `parent_run_id` explicitly bypass this stack.
//!
//! See spec §3.3 — Phase 2 (post-v1) audits remove the fallback.

use std::cell::RefCell;

use uuid::Uuid;

tokio::task_local! {
    static SPAN_STACK: RefCell<Vec<Uuid>>;
}

/// Push a run_id onto the current task-local stack.
pub fn push(run_id: Uuid) {
    let _ = SPAN_STACK.try_with(|s| s.borrow_mut().push(run_id));
}

/// Pop the top run_id (if it equals `expected`).
pub fn pop(expected: Uuid) {
    let _ = SPAN_STACK.try_with(|s| {
        let mut v = s.borrow_mut();
        if v.last().copied() == Some(expected) {
            v.pop();
        }
    });
}

/// Peek the top of the stack (the current parent for a new child).
pub fn peek() -> Option<Uuid> {
    SPAN_STACK
        .try_with(|s| s.borrow().last().copied())
        .ok()
        .flatten()
}

/// Run `f` inside a fresh empty span stack scope. Use at the top-level
/// `Runnable::invoke` entry to ensure each chain invocation gets its own
/// stack rather than inheriting from a sibling.
pub async fn scope<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    SPAN_STACK.scope(RefCell::new(Vec::new()), f).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn peek_outside_scope_is_none() {
        assert_eq!(peek(), None);
    }

    #[tokio::test]
    async fn push_pop_within_scope() {
        scope(async {
            assert_eq!(peek(), None);
            let a = Uuid::new_v4();
            push(a);
            assert_eq!(peek(), Some(a));
            let b = Uuid::new_v4();
            push(b);
            assert_eq!(peek(), Some(b));
            pop(b);
            assert_eq!(peek(), Some(a));
            pop(a);
            assert_eq!(peek(), None);
        })
        .await;
    }

    #[tokio::test]
    async fn pop_only_when_top_matches() {
        scope(async {
            let a = Uuid::new_v4();
            let b = Uuid::new_v4();
            push(a);
            pop(b); // wrong id — no-op
            assert_eq!(peek(), Some(a));
            pop(a);
            assert_eq!(peek(), None);
        })
        .await;
    }
}
