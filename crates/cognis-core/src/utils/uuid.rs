//! UUID utility functions.
//!
//! Mirrors Python `langchain_core.utils.uuid`.
//!
//! Provides UUID v4 and v7 generation for tracing and run identification.

use ::uuid::Uuid;

/// Generate a new random UUID v4.
pub fn uuid4() -> Uuid {
    Uuid::new_v4()
}

/// Generate a time-ordered UUID v7.
///
/// UUIDv7 objects feature monotonicity within a millisecond,
/// making them ideal for tracing and log ordering.
pub fn uuid7() -> Uuid {
    Uuid::now_v7()
}

/// Generate a UUID v7 from a specific Unix timestamp in nanoseconds.
pub fn uuid7_from_nanos(nanoseconds: u128) -> Uuid {
    let millis = nanoseconds / 1_000_000;
    let ts = uuid::Timestamp::from_unix(
        uuid::NoContext,
        (millis / 1000) as u64,
        ((millis % 1000) * 1_000_000) as u32,
    );
    Uuid::new_v7(ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid4_is_v4() {
        let id = uuid4();
        assert_eq!(id.get_version_num(), 4);
    }

    #[test]
    fn test_uuid7_is_v7() {
        let id = uuid7();
        assert_eq!(id.get_version_num(), 7);
    }

    #[test]
    fn test_uuid7_monotonic() {
        let a = uuid7();
        let b = uuid7();
        // v7 UUIDs should be monotonically increasing
        assert!(b >= a);
    }

    #[test]
    fn test_uuid7_from_nanos() {
        let nanos: u128 = 1_700_000_000_000_000_000; // ~2023-11-14
        let id = uuid7_from_nanos(nanos);
        assert_eq!(id.get_version_num(), 7);
    }

    #[test]
    fn test_uuid4_uniqueness() {
        let a = uuid4();
        let b = uuid4();
        assert_ne!(a, b);
    }
}
