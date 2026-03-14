//! Multimodal Content Blocks Example
//!
//! Demonstrates the `cognis_core::messages::content_blocks` module for creating
//! and querying structured multimodal content: text, images, audio, video, PDFs,
//! citations, and annotations. Also shows serialization and LLM integration.

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::ChatModelRunnable;
use cognis_core::messages::content_blocks::*;
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multimodal Content Blocks Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating text blocks
    // -----------------------------------------------------------------------
    println!("--- 1. Text blocks ---\n");

    let greeting = text_block("Hello, world!");
    let explanation = text_block("Content blocks provide a unified representation for multimodal data.");
    println!("  {:?}", greeting);
    println!("  {:?}", explanation);

    // -----------------------------------------------------------------------
    // 2. Creating image blocks
    // -----------------------------------------------------------------------
    println!("\n--- 2. Image blocks ---\n");

    let img_url = image_block_url("https://example.com/photo.png", "image/png");
    let img_b64 = image_block_base64("iVBORw0KGgoAAAANSUhEUg==", "image/png");
    let img_alt = image_block_url_with_alt(
        "https://example.com/cat.jpg",
        "image/jpeg",
        "A fluffy orange cat",
    );
    println!("  URL image:    {:?}", img_url);
    println!("  Base64 image: {:?}", img_b64);
    println!("  With alt:     {:?}", img_alt);

    // -----------------------------------------------------------------------
    // 3. Creating audio blocks
    // -----------------------------------------------------------------------
    println!("\n--- 3. Audio blocks ---\n");

    let audio = audio_block_url("https://example.com/speech.mp3", "audio/mp3");
    let audio_transcript = audio_block_url_with_transcript(
        "https://example.com/interview.mp3",
        "audio/mp3",
        "Welcome to the show.",
    );
    println!("  Audio:           {:?}", audio);
    println!("  With transcript: {:?}", audio_transcript);

    // -----------------------------------------------------------------------
    // 4. Creating video blocks
    // -----------------------------------------------------------------------
    println!("\n--- 4. Video blocks ---\n");

    let video = video_block_url("https://example.com/demo.mp4", "video/mp4");
    println!("  Video: {:?}", video);

    // -----------------------------------------------------------------------
    // 5. Creating PDF blocks
    // -----------------------------------------------------------------------
    println!("\n--- 5. PDF blocks ---\n");

    let pdf = pdf_block_url("https://example.com/report.pdf");
    let pdf_pages = pdf_block_url_with_pages("https://example.com/book.pdf", "1-10");
    println!("  PDF:            {:?}", pdf);
    println!("  PDF with pages: {:?}", pdf_pages);

    // -----------------------------------------------------------------------
    // 6. Creating citation blocks
    // -----------------------------------------------------------------------
    println!("\n--- 6. Citation blocks ---\n");

    let cite = citation_block("doc-001", "Rust guarantees memory safety without a garbage collector.");
    let cite_idx = citation_block_with_indices(
        "doc-002",
        "Ownership is Rust's most unique feature.",
        0,
        42,
    );
    println!("  Citation:             {:?}", cite);
    println!("  Citation with index:  {:?}", cite_idx);

    // -----------------------------------------------------------------------
    // 7. Annotation and custom blocks
    // -----------------------------------------------------------------------
    println!("\n--- 7. Annotation and custom blocks ---\n");

    let annotation = annotation_block("highlight", json!({"color": "yellow", "note": "important"}));
    let custom = custom_block("code_snippet", json!({"language": "rust", "code": "fn main() {}"}));
    println!("  Annotation: {:?}", annotation);
    println!("  Custom:     {:?}", custom);

    // -----------------------------------------------------------------------
    // 8. ContentBlockList — building and querying
    // -----------------------------------------------------------------------
    println!("\n--- 8. ContentBlockList queries ---\n");

    let mut list = ContentBlockList::new();
    list.push(text_block("Introduction to Rust"));
    list.push(image_block_url("https://example.com/rust-logo.png", "image/png"));
    list.push(audio_block_url("https://example.com/podcast.mp3", "audio/mp3"));
    list.push(pdf_block_url("https://example.com/rust-book.pdf"));
    list.push(citation_block("ch1", "Rust is blazingly fast."));
    list.push(text_block("Conclusion"));

    println!("  Total blocks:    {}", list.len());
    println!("  Has images:      {}", list.has_images());
    println!("  Has audio:       {}", list.has_audio());
    println!("  Has video:       {}", list.has_video());
    println!("  Has PDFs:        {}", list.has_pdfs());
    println!("  Has citations:   {}", list.has_citations());
    println!("  Has annotations: {}", list.has_annotations());
    println!("  Has media:       {}", list.has_media());
    println!("  Text only:       {}", list.text_only());
    println!("  Text blocks:     {}", list.text_blocks().len());
    println!("  Extracted text:  {:?}", list.extract_text());

    // -----------------------------------------------------------------------
    // 9. Serialization to JSON
    // -----------------------------------------------------------------------
    println!("\n--- 9. JSON serialization ---\n");

    let sample = text_block("Serializable content");
    let json_str = serde_json::to_string_pretty(&sample)?;
    println!("  Serialized:\n{}", json_str);

    let deserialized: MultimodalContentBlock = serde_json::from_str(&json_str)?;
    println!("\n  Deserialized: {:?}", deserialized);

    // ContentBlockList serialization
    let text_only_list = ContentBlockList::from_blocks(vec![
        text_block("First paragraph."),
        text_block("Second paragraph."),
    ]);
    let list_json = serde_json::to_string_pretty(&text_only_list)?;
    println!("\n  ContentBlockList JSON:\n{}", list_json);

    // -----------------------------------------------------------------------
    // 10. LLM-generated text as a content block
    // -----------------------------------------------------------------------
    println!("\n--- 10. LLM-generated text as a content block ---\n");

    let model = shared::get_chat_model(vec![
        "Rust provides memory safety through its ownership system, eliminating data races at compile time.".to_string(),
    ]);

    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a concise technical writer."),
        ("human", "Describe Rust's memory safety in one sentence."),
    ])?;

    let parser = StrOutputParser;
    let model_runnable = ChatModelRunnable::new(model);
    let chain = cognis_core::chain!(prompt, model_runnable, parser)?;

    let result = chain.invoke(json!({}), None).await?;
    let llm_text = result.as_str().unwrap_or("").trim();
    println!("  LLM output: {}", llm_text);

    let llm_block = text_block(llm_text);
    println!("  As content block: {:?}", llm_block);

    // Add it to a list along with a citation
    let mut final_list = ContentBlockList::new();
    final_list.push(llm_block);
    final_list.push(citation_block("llm-response", llm_text));
    println!("  Final list has {} blocks, text_only={}", final_list.len(), final_list.text_only());

    println!("\n=== Multimodal Content Blocks Example Complete ===");
    Ok(())
}
