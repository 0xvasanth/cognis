//! Multimodal content via V2's ContentPart enum (text / image / audio).

use cognis::prelude::*;
use cognis_core::content::{ContentPart, ImageSource};

fn main() {
    let msg = Message::human_with_parts(
        "Describe this image briefly.",
        vec![
            ContentPart::Text { text: "(small caption attached)".into() },
            ContentPart::Image {
                source: ImageSource::url("https://example.com/cat.jpg"),
                mime: "image/jpeg".into(),
            },
        ],
    );
    println!("text: {}", msg.content());
    for (i, part) in msg.parts().iter().enumerate() {
        match part {
            ContentPart::Text { text } => println!("part {i} (text): {text}"),
            ContentPart::Image { source, mime } => println!("part {i} (image): {source:?} mime={mime}"),
            ContentPart::Audio { source, mime } => println!("part {i} (audio): {source:?} mime={mime}"),
        }
    }
}
