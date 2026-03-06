//! HTML utilities for link extraction.

/// Extract all href links from HTML content.
pub fn find_all_links(html: &str) -> Vec<String> {
    let re = regex::Regex::new(r#"href=["']([^"']+)["']"#).unwrap();
    re.captures_iter(html)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

/// Extract sub-links from HTML, filtering to same base URL.
pub fn extract_sub_links(html: &str, base_url: &str) -> Vec<String> {
    find_all_links(html)
        .into_iter()
        .filter_map(|link| {
            if link.starts_with("http://") || link.starts_with("https://") {
                if link.starts_with(base_url) {
                    Some(link)
                } else {
                    None
                }
            } else if link.starts_with('/') {
                Some(format!("{}{}", base_url.trim_end_matches('/'), link))
            } else {
                None
            }
        })
        .collect()
}
