//! Environment variable utilities.

use std::collections::HashMap;

/// Check if an environment variable is set and not empty/false.
pub fn env_var_is_set(key: &str) -> bool {
    match std::env::var(key) {
        Ok(val) => !val.is_empty() && val.to_lowercase() != "false" && val != "0",
        Err(_) => false,
    }
}

/// Get information about the Cognis runtime environment.
pub fn get_runtime_environment() -> HashMap<&'static str, String> {
    let mut env = HashMap::new();
    env.insert("library_version", env!("CARGO_PKG_VERSION").to_string());
    env.insert("library", "cognis-core".to_string());
    env.insert("runtime", "rust".to_string());
    env.insert("runtime_version", rustc_version());
    env
}

/// Get the Rust compiler version used to build this crate.
fn rustc_version() -> String {
    option_env!("RUSTC_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_var_is_set() {
        std::env::set_var("RUSTCHAIN_TEST_TRUE", "1");
        assert!(env_var_is_set("RUSTCHAIN_TEST_TRUE"));

        std::env::set_var("RUSTCHAIN_TEST_FALSE", "false");
        assert!(!env_var_is_set("RUSTCHAIN_TEST_FALSE"));

        std::env::set_var("RUSTCHAIN_TEST_EMPTY", "");
        assert!(!env_var_is_set("RUSTCHAIN_TEST_EMPTY"));

        assert!(!env_var_is_set("RUSTCHAIN_NONEXISTENT_VAR"));
    }

    #[test]
    fn test_get_runtime_environment() {
        let env = get_runtime_environment();
        assert_eq!(env["library"], "cognis-core");
        assert_eq!(env["runtime"], "rust");
        assert!(!env["library_version"].is_empty());
    }
}
