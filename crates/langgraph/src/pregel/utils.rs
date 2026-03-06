//! Utility functions for the Pregel execution engine.

use std::collections::HashMap;

/// Channel version type — a monotonically increasing integer.
pub type ChannelVersion = u64;

/// Map of channel names to their versions.
pub type ChannelVersions = HashMap<String, ChannelVersion>;

/// Get the subset of `current_versions` that are newer than `previous_versions`.
///
/// If `previous_versions` is empty, returns all of `current_versions`.
pub fn get_new_channel_versions(
    previous_versions: &ChannelVersions,
    current_versions: &ChannelVersions,
) -> ChannelVersions {
    if previous_versions.is_empty() {
        return current_versions.clone();
    }
    current_versions
        .iter()
        .filter(|(k, v)| {
            let prev = previous_versions.get(*k).copied().unwrap_or(0);
            **v > prev
        })
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

/// Check if a string matches the format of an xxh3_128 hex digest (32 hex chars).
pub fn is_xxh3_128_hexdigest(value: &str) -> bool {
    value.len() == 32 && value.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_versions_empty_previous() {
        let previous = ChannelVersions::new();
        let current: ChannelVersions = [("a".to_string(), 1), ("b".to_string(), 2)]
            .into_iter()
            .collect();

        let result = get_new_channel_versions(&previous, &current);
        assert_eq!(result, current);
    }

    #[test]
    fn test_new_versions_filters_old() {
        let previous: ChannelVersions = [("a".to_string(), 3), ("b".to_string(), 2)]
            .into_iter()
            .collect();
        let current: ChannelVersions = [("a".to_string(), 3), ("b".to_string(), 5)]
            .into_iter()
            .collect();

        let result = get_new_channel_versions(&previous, &current);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("b"), Some(&5));
    }

    #[test]
    fn test_new_versions_includes_new_channels() {
        let previous: ChannelVersions = [("a".to_string(), 1)].into_iter().collect();
        let current: ChannelVersions = [("a".to_string(), 1), ("b".to_string(), 2)]
            .into_iter()
            .collect();

        let result = get_new_channel_versions(&previous, &current);
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("b"), Some(&2));
    }

    #[test]
    fn test_new_versions_all_same() {
        let versions: ChannelVersions = [("a".to_string(), 1), ("b".to_string(), 2)]
            .into_iter()
            .collect();

        let result = get_new_channel_versions(&versions, &versions);
        assert!(result.is_empty());
    }

    #[test]
    fn test_is_xxh3_128_hexdigest_valid() {
        assert!(is_xxh3_128_hexdigest("0123456789abcdef0123456789abcdef"));
        assert!(is_xxh3_128_hexdigest("ABCDEF0123456789ABCDEF0123456789"));
    }

    #[test]
    fn test_is_xxh3_128_hexdigest_invalid() {
        // Too short
        assert!(!is_xxh3_128_hexdigest("0123456789abcdef"));
        // Too long
        assert!(!is_xxh3_128_hexdigest("0123456789abcdef0123456789abcdef00"));
        // Non-hex character
        assert!(!is_xxh3_128_hexdigest("0123456789abcdef0123456789abcdeg"));
        // Empty
        assert!(!is_xxh3_128_hexdigest(""));
    }
}
