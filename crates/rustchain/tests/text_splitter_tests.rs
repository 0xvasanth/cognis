use rustchain::text_splitter::*;

#[test]
fn test_character_splitter_basic() {
    let splitter = CharacterTextSplitter::new()
        .with_separator("\n\n")
        .with_chunk_size(15)
        .with_chunk_overlap(0);
    let text = "Hello world\n\nFoo bar\n\nBaz qux";
    let chunks = splitter.split_text(text);
    assert_eq!(chunks, vec!["Hello world", "Foo bar", "Baz qux"]);
}

#[test]
fn test_character_splitter_merges_small_chunks() {
    let splitter = CharacterTextSplitter::new()
        .with_separator(" ")
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let text = "a b c d e f";
    let chunks = splitter.split_text(text);
    for chunk in &chunks {
        assert!(chunk.len() <= 10, "chunk too long: {}", chunk);
    }
}

#[test]
fn test_character_splitter_overlap() {
    let splitter = CharacterTextSplitter::new()
        .with_separator(" ")
        .with_chunk_size(10)
        .with_chunk_overlap(3);
    let text = "abcd efgh ijkl mnop";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 2);
}

#[test]
fn test_character_splitter_create_documents() {
    let splitter = CharacterTextSplitter::new()
        .with_separator("\n")
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let docs = splitter.create_documents(&["Hello\nWorld"], None);
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].page_content, "Hello");
    assert_eq!(docs[1].page_content, "World");
}

#[test]
fn test_character_splitter_split_documents() {
    use rustchain_core::documents::Document;
    let splitter = CharacterTextSplitter::new()
        .with_separator("\n")
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let doc = Document::new("Line1\nLine2\nLine3");
    let result = splitter.split_documents(&[doc]);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_merge_splits_basic() {
    use rustchain::text_splitter::merge_splits;
    let splits = vec!["a", "b", "c"];
    let result = merge_splits(&splits, " ", 5, 0);
    assert_eq!(result, vec!["a b c"]);
}

#[test]
fn test_merge_splits_exceeds_chunk() {
    use rustchain::text_splitter::merge_splits;
    let splits = vec!["hello", "world", "foo"];
    let result = merge_splits(&splits, " ", 11, 0);
    assert_eq!(result, vec!["hello world", "foo"]);
}

#[test]
fn test_recursive_splitter_basic() {
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(20)
        .with_chunk_overlap(0);
    let text = "Hello world.\n\nThis is a test.\n\nAnother paragraph.";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 2);
    for chunk in &chunks {
        assert!(chunk.len() <= 20, "chunk too long: '{}'", chunk);
    }
}

#[test]
fn test_recursive_splitter_falls_through_separators() {
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(15)
        .with_chunk_overlap(0);
    // No double newlines, so falls through to \n, then space
    let text = "Hello world foo bar baz";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 2);
}

#[test]
fn test_recursive_splitter_from_language() {
    let splitter = RecursiveCharacterTextSplitter::from_language(Language::Rust)
        .with_chunk_size(50)
        .with_chunk_overlap(0);
    let code =
        "fn main() {\n    println!(\"hello\");\n}\n\nfn other() {\n    println!(\"world\");\n}";
    let chunks = splitter.split_text(code);
    assert!(chunks.len() >= 1);
}

#[test]
fn test_language_separators_not_empty() {
    let langs = vec![
        Language::Python,
        Language::JavaScript,
        Language::Rust,
        Language::Go,
        Language::Java,
        Language::Markdown,
    ];
    for lang in langs {
        let seps = lang.get_separators();
        assert!(!seps.is_empty(), "{:?} has no separators", lang);
    }
}

#[test]
fn test_recursive_splitter_split_documents() {
    use rustchain_core::documents::Document;
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(20)
        .with_chunk_overlap(0);
    let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird one.");
    let result = splitter.split_documents(&[doc]);
    assert!(result.len() >= 2);
}

// Token splitter tests
#[test]
fn test_token_splitter_basic() {
    let splitter = TokenTextSplitter::new()
        .with_chunk_size(5)
        .with_chunk_overlap(0)
        .with_chars_per_token(4);
    let text = "This is a test of the token text splitter functionality.";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 2);
}

#[test]
fn test_token_splitter_overlap() {
    let splitter = TokenTextSplitter::new()
        .with_chunk_size(5)
        .with_chunk_overlap(2)
        .with_chars_per_token(4);
    let text = "The quick brown fox jumps over the lazy dog and more text here.";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 2);
}

// Markdown header splitter tests
#[test]
fn test_markdown_header_splitter() {
    let splitter = MarkdownHeaderTextSplitter::new(vec![("#", "Header 1"), ("##", "Header 2")]);
    let text = "# Intro\nSome intro text.\n## Details\nDetail text here.\n## More\nMore text.";
    let docs = splitter.split_text(text);
    assert!(docs.len() >= 2);
    assert!(docs[0].metadata.contains_key("Header 1"));
}

#[test]
fn test_markdown_text_splitter() {
    let splitter = MarkdownTextSplitter::new()
        .with_chunk_size(30)
        .with_chunk_overlap(0);
    let text = "## Section 1\nShort text.\n## Section 2\nAnother section with more content here.";
    let chunks = splitter.split_text(text);
    assert!(chunks.len() >= 1);
}

// HTML header splitter tests
#[test]
fn test_html_header_splitter() {
    let splitter = HTMLHeaderTextSplitter::new(vec![("h1", "Header 1"), ("h2", "Header 2")]);
    let text = "<h1>Title</h1>\n<p>Intro text.</p>\n<h2>Section</h2>\n<p>Section text.</p>";
    let docs = splitter.split_text(text);
    assert!(docs.len() >= 1);
}

// JSON splitter tests
#[test]
fn test_json_splitter_small() {
    let splitter = RecursiveJsonSplitter::new(1000);
    let data = serde_json::json!({"key": "value"});
    let chunks = splitter.split_json(&data);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_json_splitter_large_object() {
    let splitter = RecursiveJsonSplitter::new(50);
    let data = serde_json::json!({
        "name": "Alice",
        "bio": "A very long biography that exceeds the chunk size limit for testing purposes"
    });
    let chunks = splitter.split_json(&data);
    assert!(chunks.len() >= 2);
}

#[test]
fn test_json_splitter_array() {
    let splitter = RecursiveJsonSplitter::new(15);
    let data = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let chunks = splitter.split_json(&data);
    assert!(chunks.len() >= 2);
}

#[test]
fn test_json_splitter_split_text() {
    let splitter = RecursiveJsonSplitter::new(50);
    let json_str = r#"{"a": "hello", "b": "world long text here for splitting"}"#;
    let chunks = splitter.split_text(json_str);
    assert!(chunks.len() >= 1);
}
