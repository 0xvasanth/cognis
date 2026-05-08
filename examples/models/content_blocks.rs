//! What you'll learn:
//!   How to build a multi-part `Message` containing text plus an
//!   image reference and send it to a vision-capable model — the
//!   shape every modern multimodal request takes.
//!
//! Why this matters:
//!   Every modern provider takes mixed content — text, images, audio.
//!   `ContentPart` is the unified representation; whatever you
//!   construct here will be serialised correctly by OpenAI,
//!   Anthropic, Google, or any other multimodal provider. This is
//!   what your code looks like when the agent needs to "see" an
//!   uploaded screenshot, photo, or document scan.
//!
//! Scenario:
//!   Caption an image. We attach the image URL and ask the model
//!   what's in it. Note: llama3.1 is text-only — point the model at
//!   `llava` (or an OpenAI vision model) for the actual caption.
//!
//! Run with:
//!   # llama3.1 is NOT vision-capable; use a vision model:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llava \
//!     cargo run -p cognis-examples --example models_content_blocks
//!
//! Sample output (against ollama / llava):
//!   caption: There is no image provided, please share the image you would like me to caption and I will be happy to help!

use cognis::prelude::*;
use cognis_core::content::{ContentPart, ImageSource};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;

    // Build a single human message containing two parts: a text
    // instruction and an image. The provider serialiser turns this
    // into the right wire shape (OpenAI's `image_url`, Anthropic's
    // `image` block, Ollama's `images` array) automatically.
    let msg = Message::human_with_parts(
        "Caption this image in one sentence.",
        vec![
            ContentPart::Text { text: "Caption this image in one sentence.".into() },
            ContentPart::Image {
                source: ImageSource::url(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
                ),
                mime: "image/jpeg".into(),
            },
        ],
    );

    let resp = client.invoke(vec![msg]).await?;
    println!("caption: {}", resp.content().trim());
    Ok(())
}
