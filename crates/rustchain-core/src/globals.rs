//! Global values and configuration that apply to all of RustChain.
//!
//! Mirrors Python `langchain_core.globals`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::RwLock;

use crate::caches::BaseCache;

static VERBOSE: AtomicBool = AtomicBool::new(false);
static DEBUG: AtomicBool = AtomicBool::new(false);
static LLM_CACHE: OnceLock<RwLock<Option<Box<dyn BaseCache>>>> = OnceLock::new();

/// Set the global `verbose` flag.
pub fn set_verbose(value: bool) {
    VERBOSE.store(value, Ordering::Relaxed);
}

/// Get the global `verbose` flag.
pub fn get_verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// Set the global `debug` flag.
pub fn set_debug(value: bool) {
    DEBUG.store(value, Ordering::Relaxed);
}

/// Get the global `debug` flag.
pub fn get_debug() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

fn cache_lock() -> &'static RwLock<Option<Box<dyn BaseCache>>> {
    LLM_CACHE.get_or_init(|| RwLock::new(None))
}

/// Set the global LLM cache. Pass `None` to disable caching.
pub fn set_llm_cache(value: Option<Box<dyn BaseCache>>) {
    let mut guard = cache_lock().write().expect("cache lock poisoned");
    *guard = value;
}

/// Returns `true` if a global LLM cache is currently set.
pub fn has_llm_cache() -> bool {
    let guard = cache_lock().read().expect("cache lock poisoned");
    guard.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verbose_default() {
        // Reset to known state
        set_verbose(false);
        assert!(!get_verbose());
    }

    #[test]
    fn test_set_verbose() {
        set_verbose(true);
        assert!(get_verbose());
        set_verbose(false);
        assert!(!get_verbose());
    }

    #[test]
    fn test_debug_default() {
        set_debug(false);
        assert!(!get_debug());
    }

    #[test]
    fn test_set_debug() {
        set_debug(true);
        assert!(get_debug());
        set_debug(false);
        assert!(!get_debug());
    }
}
