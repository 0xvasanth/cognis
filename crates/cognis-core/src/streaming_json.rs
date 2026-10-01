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
}
