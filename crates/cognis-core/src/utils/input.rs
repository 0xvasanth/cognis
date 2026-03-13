//! Input handling and text coloring utilities.
//!
//! Mirrors Python `langchain_core.utils.input`.

use std::collections::HashMap;
use std::io::Write;

/// ANSI color codes for terminal output.
const TEXT_COLOR_MAPPING: &[(&str, &str)] = &[
    ("blue", "36;1"),
    ("yellow", "33;1"),
    ("pink", "38;5;200"),
    ("green", "32;1"),
    ("red", "31;1"),
];

/// Get a mapping from items to colors, cycling through available colors.
///
/// # Errors
/// Returns an error if no colors remain after applying exclusions.
pub fn get_color_mapping(
    items: &[String],
    excluded_colors: Option<&[String]>,
) -> Result<HashMap<String, String>, String> {
    let colors: Vec<&str> = TEXT_COLOR_MAPPING
        .iter()
        .filter(|(name, _)| {
            excluded_colors
                .map(|exc| !exc.iter().any(|e| e == *name))
                .unwrap_or(true)
        })
        .map(|(name, _)| *name)
        .collect();

    if colors.is_empty() {
        return Err("No colors available after applying exclusions.".into());
    }

    let mapping = items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.clone(), colors[i % colors.len()].to_string()))
        .collect();

    Ok(mapping)
}

/// Wrap text in ANSI color escape codes.
pub fn get_colored_text(text: &str, color: &str) -> String {
    let color_str = TEXT_COLOR_MAPPING
        .iter()
        .find(|(name, _)| *name == color)
        .map(|(_, code)| *code)
        .unwrap_or("0");
    format!("\x1b[{}m\x1b[1;3m{}\x1b[0m", color_str, text)
}

/// Wrap text in ANSI bold escape codes.
pub fn get_bolded_text(text: &str) -> String {
    format!("\x1b[1m{}\x1b[0m", text)
}

/// Print text with optional color to a writer.
pub fn print_text(text: &str, color: Option<&str>, end: &str, writer: &mut dyn Write) {
    let text_to_print = match color {
        Some(c) => get_colored_text(text, c),
        None => text.to_string(),
    };
    let _ = write!(writer, "{}{}", text_to_print, end);
    let _ = writer.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_color_mapping() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mapping = get_color_mapping(&items, None).unwrap();
        assert_eq!(mapping.len(), 3);
        assert!(mapping.contains_key("a"));
        assert!(mapping.contains_key("b"));
        assert!(mapping.contains_key("c"));
    }

    #[test]
    fn test_get_color_mapping_cycles() {
        let items: Vec<String> = (0..10).map(|i| format!("item_{}", i)).collect();
        let mapping = get_color_mapping(&items, None).unwrap();
        assert_eq!(mapping.len(), 10);
        // Items 0 and 5 should get the same color (5 colors available)
        assert_eq!(mapping["item_0"], mapping["item_5"]);
    }

    #[test]
    fn test_get_color_mapping_with_exclusions() {
        let items = vec!["a".to_string()];
        let excluded = vec!["blue".to_string()];
        let mapping = get_color_mapping(&items, Some(&excluded)).unwrap();
        assert_ne!(mapping["a"], "blue");
    }

    #[test]
    fn test_get_color_mapping_all_excluded() {
        let items = vec!["a".to_string()];
        let all_colors: Vec<String> = TEXT_COLOR_MAPPING
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();
        let result = get_color_mapping(&items, Some(&all_colors));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_colored_text() {
        let result = get_colored_text("hello", "blue");
        assert!(result.contains("hello"));
        assert!(result.starts_with("\x1b["));
        assert!(result.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_get_bolded_text() {
        let result = get_bolded_text("hello");
        assert_eq!(result, "\x1b[1mhello\x1b[0m");
    }

    #[test]
    fn test_print_text_no_color() {
        let mut buf = Vec::new();
        print_text("hello", None, "", &mut buf);
        assert_eq!(String::from_utf8(buf).unwrap(), "hello");
    }

    #[test]
    fn test_print_text_with_color() {
        let mut buf = Vec::new();
        print_text("hello", Some("green"), "\n", &mut buf);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("hello"));
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn test_print_text_with_end() {
        let mut buf = Vec::new();
        print_text("hello", None, "!!", &mut buf);
        assert_eq!(String::from_utf8(buf).unwrap(), "hello!!");
    }
}
