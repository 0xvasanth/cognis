//! Multimodal Content Blocks Example
//!
//! Demonstrates creating, querying, and serializing structured multimodal
//! content blocks: text, images, audio, video, PDFs, citations, and annotations.

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
    // --- Create various block types ---
    let _text = text_block("Hello, world!");
    let _img = image_block_url("https://example.com/photo.png", "image/png");
    let _img_alt = image_block_url_with_alt("https://example.com/cat.jpg", "image/jpeg", "A cat");
    let _audio = audio_block_url("https://example.com/speech.mp3", "audio/mp3");
    let _video = video_block_url("https://example.com/demo.mp4", "video/mp4");
    let _pdf = pdf_block_url("https://example.com/report.pdf");
    let _cite = citation_block("doc-001", "Rust guarantees memory safety.");
    let _annotation = annotation_block("highlight", json!({"color": "yellow"}));
    let _custom = custom_block(
        "code_snippet",
        json!({"language": "rust", "code": "fn main() {}"}),
    );

    // --- ContentBlockList: building and querying ---
    let mut list = ContentBlockList::new();
    list.push(text_block("Introduction to Rust"));
    list.push(image_block_url(
        "https://example.com/rust-logo.png",
        "image/png",
    ));
    list.push(audio_block_url(
        "https://example.com/podcast.mp3",
        "audio/mp3",
    ));
    list.push(pdf_block_url("https://example.com/rust-book.pdf"));
    list.push(citation_block("ch1", "Rust is blazingly fast."));
    list.push(text_block("Conclusion"));

    println!("Total blocks: {}", list.len());
    println!(
        "Has images: {}, audio: {}, pdfs: {}",
        list.has_images(),
        list.has_audio(),
        list.has_pdfs()
    );
    println!(
        "Text only: {}, text blocks: {}",
        list.text_only(),
        list.text_blocks().len()
    );
    println!("Extracted text: {:?}", list.extract_text());

    // --- JSON serialization round-trip ---
    let sample = text_block("Serializable content");
    let json_str = serde_json::to_string_pretty(&sample)?;
    let _deserialized: MultimodalContentBlock = serde_json::from_str(&json_str)?;
    println!("\nSerialized block:\n{json_str}");

    // --- LLM-generated text as a content block ---
    let model = shared::get_chat_model(vec![
        "Rust provides memory safety through its ownership system.".into(),
    ]);
    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a concise technical writer."),
        ("human", "Describe Rust's memory safety in one sentence."),
    ])?;
    let chain = cognis_core::chain!(prompt, ChatModelRunnable::new(model), StrOutputParser)?;
    let result = chain.invoke(json!({}), None).await?;
    let llm_text = result.as_str().unwrap_or("").trim();
    println!("\nLLM output: {llm_text}");

    let mut final_list = ContentBlockList::new();
    final_list.push(text_block(llm_text));
    final_list.push(citation_block("llm-response", llm_text));
    println!(
        "Final list: {} blocks, text_only={}",
        final_list.len(),
        final_list.text_only()
    );

    Ok(())
}
