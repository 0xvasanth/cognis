//! Incremental JSON-array element extraction.
//!
//! Feeds streamed text into [`StreamingJsonArray`]; each time a top-level
//! array element closes, its raw JSON text is returned. Used to deliver
//! structured list output element-by-element before the whole array
//! finishes. Tracks brace/bracket depth and string-escape state.

/// Extracts top-level elements from a streamed JSON array.
#[derive(Debug, Default)]
pub struct StreamingJsonArray {
    started: bool,  // seen the opening '['
    finished: bool, // seen the matching ']'
    depth: i32,     // nesting depth inside the current element
    in_string: bool,
    escaped: bool,
    current: String,   // accumulating element text
    has_content: bool, // current holds a real (non-whitespace) element
}

impl StreamingJsonArray {
    /// New empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a text fragment; returns the JSON text of every element that
    /// completed within this fragment.
    pub fn push_str(&mut self, s: &str) -> Vec<String> {
        let mut done = Vec::new();
        for ch in s.chars() {
            if self.finished {
                break;
            }
            if !self.started {
                if ch == '[' {
                    self.started = true;
                }
                continue;
            }
            if self.in_string {
                self.current.push(ch);
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match ch {
                '"' => {
                    self.in_string = true;
                    self.current.push(ch);
                    self.has_content = true;
                }
                '{' | '[' => {
                    self.depth += 1;
                    self.current.push(ch);
                    self.has_content = true;
                }
                '}' | ']' if self.depth > 0 => {
                    self.depth -= 1;
                    self.current.push(ch);
                }
                ',' if self.depth == 0 => {
                    self.flush(&mut done);
                }
                ']' if self.depth == 0 => {
                    self.flush(&mut done);
                    self.finished = true;
                }
                c if c.is_whitespace() => {
                    if self.has_content {
                        self.current.push(c);
                    }
                }
                c => {
                    self.current.push(c);
                    self.has_content = true;
                }
            }
        }
        done
    }

    /// Flush any trailing element (e.g. stream ended without a closing `]`).
    pub fn finish(&mut self) -> Vec<String> {
        let mut done = Vec::new();
        if self.has_content {
            self.flush(&mut done);
        }
        done
    }

    fn flush(&mut self, done: &mut Vec<String>) {
        let elem = self.current.trim().to_string();
        self.current.clear();
        self.has_content = false;
        if !elem.is_empty() {
            done.push(elem);
        }
    }
}

/// Best-effort completion of a truncated JSON object prefix into parseable
/// text, for rendering progressively-filled snapshots of a streamed object.
///
/// Different from [`StreamingJsonArray`], which waits for whole elements:
/// this closes an open string, drops a dangling `key`, `key:` or trailing
/// comma, and appends the missing `}`/`]` closers. Text before the first
/// `{` is skipped; returns `None` if no `{` has arrived yet. A value cut
/// mid-literal (`tru`, `1.`) still yields text that fails to parse, so
/// callers should treat a parse failure as "no new snapshot".
pub fn close_partial_json(buf: &str) -> Option<String> {
    let start = buf.find('{')?;
    let mut out = String::with_capacity(buf.len() - start + 8);
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut last_significant = '\0';
    // Where the current `key: value` entry begins in `out`, so a dangling
    // entry is cut at a structural boundary rather than by scanning for commas
    // that may sit inside strings.
    let mut entry_start = 0;
    let mut string_is_key = false;
    let mut last_string_was_key = false;
    for ch in buf[start..].chars() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
                last_string_was_key = string_is_key;
                last_significant = '"';
            }
            continue;
        }
        match ch {
            '"' => {
                string_is_key = stack.last() == Some(&'}') && matches!(last_significant, '{' | ',');
                in_string = true;
                out.push(ch);
            }
            '{' | '[' => {
                stack.push(if ch == '{' { '}' } else { ']' });
                out.push(ch);
                entry_start = out.len();
                last_significant = ch;
            }
            ',' => {
                entry_start = out.len();
                out.push(ch);
                last_significant = ch;
            }
            '}' | ']' => {
                stack.pop();
                out.push(ch);
                last_significant = ch;
            }
            c if c.is_whitespace() => out.push(c),
            c => {
                out.push(c);
                last_significant = c;
            }
        }
    }
    if in_string {
        out.push('"');
        last_string_was_key = string_is_key;
        last_significant = '"';
    }
    let dangling = match last_significant {
        ':' | ',' => true,
        '"' => last_string_was_key,
        _ => false,
    };
    if dangling {
        out.truncate(entry_start);
    }
    out.extend(stack.iter().rev());
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_each_object_element() {
        let mut p = StreamingJsonArray::new();
        let mut out = Vec::new();
        out.extend(p.push_str("[{\"a\":1},"));
        out.extend(p.push_str("{\"a\":2}]"));
        assert_eq!(out, vec!["{\"a\":1}".to_string(), "{\"a\":2}".to_string()]);
    }

    #[test]
    fn element_split_across_chunks() {
        let mut p = StreamingJsonArray::new();
        let mut out = Vec::new();
        out.extend(p.push_str("[{\"a\":"));
        out.extend(p.push_str("1}"));
        out.extend(p.push_str("]"));
        assert_eq!(out, vec!["{\"a\":1}".to_string()]);
    }

    #[test]
    fn string_with_brackets_and_escapes_not_split() {
        let mut p = StreamingJsonArray::new();
        let out = p.push_str(r#"["a]b}\"c"]"#);
        assert_eq!(out, vec![r#""a]b}\"c""#.to_string()]);
    }

    #[test]
    fn ignores_prose_before_bracket() {
        let mut p = StreamingJsonArray::new();
        let out = p.push_str("Here you go:\n[1,2]");
        assert_eq!(out, vec!["1".to_string(), "2".to_string()]);
    }

    #[test]
    fn empty_array_emits_nothing() {
        let mut p = StreamingJsonArray::new();
        assert!(p.push_str("[]").is_empty());
    }

    #[test]
    fn multibyte_char_split_across_chunks_is_safe() {
        // push_str takes &str, so callers must not split a codepoint; this
        // test documents that whole-char input reassembles correctly.
        let mut p = StreamingJsonArray::new();
        let mut out = Vec::new();
        out.extend(p.push_str("[\"café"));
        out.extend(p.push_str(" ☕\"]"));
        assert_eq!(out, vec!["\"café ☕\"".to_string()]);
    }

    #[test]
    fn unterminated_array_flushes_on_finish() {
        let mut p = StreamingJsonArray::new();
        let mut out = p.push_str("[{\"a\":1}");
        out.extend(p.finish());
        assert_eq!(out, vec!["{\"a\":1}".to_string()]);
    }

    fn closed(buf: &str) -> serde_json::Value {
        let s = close_partial_json(buf).expect("object prefix");
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("{e}: {s}"))
    }

    #[test]
    fn closes_open_object_and_string() {
        let v = closed("{\"a\":1,\"b\":\"hel");
        assert_eq!(v["a"], serde_json::json!(1));
        assert_eq!(v["b"], serde_json::json!("hel"));
    }

    #[test]
    fn drops_dangling_key_without_value() {
        let v = closed("{\"a\":1,\"b\":");
        assert_eq!(v, serde_json::json!({"a": 1}));
    }

    #[test]
    fn drops_trailing_comma() {
        assert_eq!(closed("{\"a\":1,"), serde_json::json!({"a": 1}));
    }

    #[test]
    fn drops_key_whose_colon_has_not_arrived() {
        assert_eq!(closed("{\"a\":1,\"b"), serde_json::json!({"a": 1}));
    }

    #[test]
    fn drops_dangling_key_containing_comma_or_brace() {
        assert_eq!(closed("{\"a\":1,\"b,{c\":"), serde_json::json!({"a": 1}));
    }

    #[test]
    fn closes_nested_object_and_array() {
        let v = closed("{\"a\":{\"b\":[1,2,{\"c\":\"x");
        assert_eq!(v, serde_json::json!({"a": {"b": [1, 2, {"c": "x"}]}}));
    }

    #[test]
    fn ignores_braces_and_escaped_quotes_inside_strings() {
        let v = closed("{\"a\":\"x}\\\"[y");
        assert_eq!(v["a"], serde_json::json!("x}\"[y"));
    }

    #[test]
    fn skips_leading_prose_before_first_brace() {
        assert_eq!(closed("Sure: {\"a\":1"), serde_json::json!({"a": 1}));
    }

    #[test]
    fn leaves_complete_object_unchanged() {
        assert_eq!(closed("{\"a\":[1,2]}"), serde_json::json!({"a": [1, 2]}));
    }

    #[test]
    fn none_before_first_brace() {
        assert!(close_partial_json("  ").is_none());
        assert!(close_partial_json("").is_none());
    }
}
