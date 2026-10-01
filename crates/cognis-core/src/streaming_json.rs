//! Incremental parsing of streamed JSON text.
//!
//! Reach for [`StreamingJsonArray`] to deliver a structured list element by
//! element before the whole array finishes, and for [`close_partial_json`]
//! to render progressively-filled snapshots of a single streamed object.
//! Both track brace/bracket depth and string-escape state.

/// Extracts top-level elements from a streamed JSON array.
///
/// The first `[` anywhere in the fed text starts parsing, so leading prose
/// or a code fence is skipped — and so is any `[` they happen to contain.
/// An element is emitted when the top-level `,` or `]` that follows it
/// arrives, not when its own closing brace does; the last element of an
/// array the stream never closed is only available from
/// [`StreamingJsonArray::finish`]. Everything after the array's closing `]`
/// is ignored.
#[derive(Debug, Default)]
pub struct StreamingJsonArray {
    started: bool,
    finished: bool,
    depth: i32,
    in_string: bool,
    escaped: bool,
    current: String,
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

    /// True once the opening `[` has been seen. Lets a caller tell "the
    /// array was empty" from "there never was an array" at end of stream.
    pub fn has_started(&self) -> bool {
        self.started
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
/// comma, and appends the missing `}`/`]` closers.
///
/// What a snapshot may contain:
/// - **Strings stream as prefixes.** A string field still being written is
///   closed where it stands, so its value may be a prefix of the final one
///   until the stream ends.
/// - **Numbers and literals appear only once complete.** A number, `true`,
///   `false` or `null` at the very end of the buffer cannot be told apart
///   from a longer one (`9` vs `92`), so its entry is withheld until a `,`,
///   `}`, `]` or whitespace terminates it.
/// - **The first `{` starts the object.** Text before it is skipped —
///   including prose or a code fence, and any `{` they happen to contain.
///   Returns `None` if no `{` has arrived yet.
/// - **Text after the root object is ignored.** Scanning stops at the `}`
///   that closes the root object, so a trailing code fence or sign-off does
///   not corrupt the result.
///
/// Malformed input can still yield text that fails to parse, so callers
/// should treat a parse failure as "no new snapshot".
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
    // True while the last thing scanned is a number/literal no delimiter has
    // terminated yet — it may still grow, so it is provisional.
    let mut in_scalar = false;
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
                in_scalar = false;
                out.push(ch);
            }
            '{' | '[' => {
                stack.push(if ch == '{' { '}' } else { ']' });
                out.push(ch);
                entry_start = out.len();
                last_significant = ch;
                in_scalar = false;
            }
            ',' => {
                entry_start = out.len();
                out.push(ch);
                last_significant = ch;
                in_scalar = false;
            }
            '}' | ']' => {
                stack.pop();
                out.push(ch);
                last_significant = ch;
                in_scalar = false;
                if stack.is_empty() {
                    // Root object closed: whatever follows is not part of it.
                    break;
                }
            }
            ':' => {
                out.push(ch);
                last_significant = ch;
                in_scalar = false;
            }
            c if c.is_whitespace() => {
                out.push(c);
                in_scalar = false;
            }
            c => {
                out.push(c);
                last_significant = c;
                in_scalar = true;
            }
        }
    }
    if in_string {
        out.push('"');
        last_string_was_key = string_is_key;
        last_significant = '"';
    }
    let dangling = in_scalar
        || match last_significant {
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
    fn has_started_distinguishes_empty_array_from_no_array() {
        let mut empty = StreamingJsonArray::new();
        assert!(!empty.has_started());
        empty.push_str("[]");
        assert!(empty.has_started());
        assert!(empty.finish().is_empty());

        let mut prose = StreamingJsonArray::new();
        prose.push_str("I cannot help with that.");
        assert!(!prose.has_started());
        assert!(prose.finish().is_empty());
    }

    #[test]
    fn reassembles_multibyte_text_when_split_at_char_boundary() {
        // push_str takes &str, so a chunk can never end inside a codepoint;
        // this covers a split right after a multibyte character.
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
        assert_eq!(
            closed("Sure: {\"a\":1,\"b\":\"x"),
            serde_json::json!({"a": 1, "b": "x"})
        );
    }

    #[test]
    fn leaves_complete_object_unchanged() {
        assert_eq!(closed("{\"a\":[1,2]}"), serde_json::json!({"a": [1, 2]}));
    }

    #[test]
    fn ignores_closing_code_fence_after_root_object() {
        let v = closed("{\"title\":\"Q3\",\"score\":92}\n```\n");
        assert_eq!(v, serde_json::json!({"title": "Q3", "score": 92}));
    }

    #[test]
    fn ignores_prose_after_root_object() {
        let v = closed("{\"title\":\"Q3\",\"score\":92} Hope this helps!");
        assert_eq!(v, serde_json::json!({"title": "Q3", "score": 92}));
    }

    #[test]
    fn extracts_fenced_object_arriving_as_single_chunk() {
        let v = closed("```json\n{\"title\":\"Q3\",\"score\":92}\n```");
        assert_eq!(v, serde_json::json!({"title": "Q3", "score": 92}));
    }

    #[test]
    fn ignores_second_object_and_stray_closers_after_root_object() {
        let v = closed("{\"a\":{\"b\":[1]}}}] {\"c\":2");
        assert_eq!(v, serde_json::json!({"a": {"b": [1]}}));
    }

    #[test]
    fn withholds_number_still_being_written_at_buffer_end() {
        assert_eq!(closed("{\"a\":1,\"score\":9"), serde_json::json!({"a": 1}));
    }

    #[test]
    fn keeps_number_once_closing_brace_terminates_it() {
        assert_eq!(
            closed("{\"a\":1,\"score\":92}"),
            serde_json::json!({"a": 1, "score": 92})
        );
    }

    #[test]
    fn keeps_number_once_comma_or_whitespace_terminates_it() {
        assert_eq!(
            closed("{\"a\":1,\"score\":92,"),
            serde_json::json!({"a": 1, "score": 92})
        );
        assert_eq!(
            closed("{\"a\":1,\"score\":92\n"),
            serde_json::json!({"a": 1, "score": 92})
        );
    }

    #[test]
    fn withholds_literal_still_being_written_at_buffer_end() {
        assert_eq!(closed("{\"a\":1,\"ok\":tru"), serde_json::json!({"a": 1}));
        assert_eq!(closed("{\"a\":1,\"ok\":true"), serde_json::json!({"a": 1}));
        assert_eq!(
            closed("{\"a\":1,\"ok\":true}"),
            serde_json::json!({"a": 1, "ok": true})
        );
    }

    #[test]
    fn withholds_unterminated_scalar_when_it_is_the_only_entry() {
        for buf in [
            "{\"score\":9",
            "{\"ok\":tru",
            "{\"x\":1.",
            "{\"n\":nul",
            "{\"v\":-",
        ] {
            assert_eq!(closed(buf), serde_json::json!({}), "input: {buf}");
        }
    }

    #[test]
    fn withholds_array_element_number_at_buffer_end() {
        assert_eq!(closed("{\"xs\":[1,2"), serde_json::json!({"xs": [1]}));
        assert_eq!(closed("{\"xs\":[1"), serde_json::json!({"xs": []}));
        assert_eq!(closed("{\"xs\":[1,2]"), serde_json::json!({"xs": [1, 2]}));
    }

    #[test]
    fn keeps_streaming_string_prefix_while_withholding_later_number() {
        assert_eq!(
            closed("{\"title\":\"Quar"),
            serde_json::json!({"title": "Quar"})
        );
        assert_eq!(
            closed("{\"title\":\"Quarterly\",\"score\":4"),
            serde_json::json!({"title": "Quarterly"})
        );
    }

    #[test]
    fn none_before_first_brace() {
        assert!(close_partial_json("  ").is_none());
        assert!(close_partial_json("").is_none());
    }
}
