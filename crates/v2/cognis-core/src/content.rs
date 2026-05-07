//! Multimodal content parts.
//!
//! [`ContentPart`] / [`ImageSource`] / [`AudioSource`] are the canonical
//! building blocks for multimodal messages. They sit on the `parts` field
//! of [`HumanMessage`](crate::HumanMessage) / [`AiMessage`](crate::AiMessage);
//! providers that understand multimodal serialize them into their wire
//! formats (OpenAI's `image_url` parts, Anthropic's `image` blocks,
//! Gemini's `inline_data` / `file_data`).
//!
//! [`ContentPart::to_openai`] / [`ContentPart::to_anthropic`] /
//! [`ContentPart::to_gemini`] emit the provider-specific JSON; the
//! corresponding `from_*` functions parse it back.
//!
//! ```
//! use cognis2_core::{ContentPart, ImageSource};
//! let part = ContentPart::Image {
//!     source: ImageSource::url("https://example.com/cat.jpg"),
//!     mime: "image/jpeg".into(),
//! };
//! let openai = part.to_openai();
//! assert_eq!(openai["type"], "image_url");
//! ```

use serde::{Deserialize, Serialize};

/// One non-text part of a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text part. Most messages don't need this — the textual
    /// `content` field on the message itself is the standard way to carry
    /// text. Provided here for symmetry with multi-text-block providers.
    Text {
        /// The text.
        text: String,
    },
    /// An image, by URL or inline base64.
    Image {
        /// Where the image bytes come from.
        source: ImageSource,
        /// MIME type (e.g. `image/jpeg`).
        mime: String,
    },
    /// Audio, by URL or inline base64.
    Audio {
        /// Where the audio bytes come from.
        source: AudioSource,
        /// MIME type (e.g. `audio/wav`).
        mime: String,
    },
}

/// Source of an image part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageSource {
    /// Public URL the model can fetch.
    Url {
        /// The URL.
        url: String,
    },
    /// Inline base64 data — for providers that accept it.
    Base64 {
        /// Raw base64-encoded payload.
        data: String,
    },
}

impl ImageSource {
    /// Convenience: `ImageSource::url("https://...")`.
    pub fn url(u: impl Into<String>) -> Self {
        Self::Url { url: u.into() }
    }
    /// Convenience: `ImageSource::base64(data)`.
    pub fn base64(d: impl Into<String>) -> Self {
        Self::Base64 { data: d.into() }
    }
}

/// Source of an audio part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioSource {
    /// Public URL.
    Url {
        /// The URL.
        url: String,
    },
    /// Inline base64.
    Base64 {
        /// Raw base64-encoded payload.
        data: String,
    },
}

impl AudioSource {
    /// Convenience: `AudioSource::url("https://...")`.
    pub fn url(u: impl Into<String>) -> Self {
        Self::Url { url: u.into() }
    }
    /// Convenience: `AudioSource::base64(data)`.
    pub fn base64(d: impl Into<String>) -> Self {
        Self::Base64 { data: d.into() }
    }
}

// ---------------------------------------------------------------------------
// Image / media helper utilities.
// ---------------------------------------------------------------------------

/// Best-effort MIME type from a file extension. Returns `None` if the
/// extension is unknown or absent. Recognises common image, audio, and
/// document formats.
pub fn mime_from_path(path: &std::path::Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        "svg" => "image/svg+xml",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" => "audio/mp4",
        "flac" => "audio/flac",
        "ogg" => "audio/ogg",
        "pdf" => "application/pdf",
        _ => return None,
    })
}

/// Encode bytes as base64 (RFC 4648 standard, no padding rules changed).
/// Pure-Rust implementation — no `base64` crate dependency.
pub fn base64_encode(bytes: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
        out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
        out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
        out.push(CHARS[(n & 0x3f) as usize] as char);
    }
    let rem = chunks.remainder();
    match rem.len() {
        0 => {}
        1 => {
            let n = (rem[0] as u32) << 16;
            out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
            out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
            out.push(CHARS[((n >> 18) & 0x3f) as usize] as char);
            out.push(CHARS[((n >> 12) & 0x3f) as usize] as char);
            out.push(CHARS[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        _ => unreachable!(),
    }
    out
}

/// Decode a base64 string back into bytes. Tolerates whitespace; rejects
/// non-base64 characters with [`crate::CognisError::Serialization`].
pub fn base64_decode(s: &str) -> crate::Result<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let bytes: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut chunks = bytes.chunks_exact(4);
    for chunk in &mut chunks {
        let a = val(chunk[0]).ok_or_else(bad)?;
        let b = val(chunk[1]).ok_or_else(bad)?;
        let c = val(chunk[2]).ok_or_else(bad)?;
        let d = val(chunk[3]).ok_or_else(bad)?;
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6) | (d as u32);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    }
    let rem = chunks.remainder();
    match rem.len() {
        0 => {}
        2 => {
            let a = val(rem[0]).ok_or_else(bad)?;
            let b = val(rem[1]).ok_or_else(bad)?;
            let n = ((a as u32) << 18) | ((b as u32) << 12);
            out.push((n >> 16) as u8);
        }
        3 => {
            let a = val(rem[0]).ok_or_else(bad)?;
            let b = val(rem[1]).ok_or_else(bad)?;
            let c = val(rem[2]).ok_or_else(bad)?;
            let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6);
            out.push((n >> 16) as u8);
            out.push((n >> 8) as u8);
        }
        _ => {
            return Err(crate::CognisError::Serialization(
                "base64: malformed input length".into(),
            ))
        }
    }
    Ok(out)
}

fn bad() -> crate::CognisError {
    crate::CognisError::Serialization("base64: invalid character".into())
}

/// Read a file from disk and produce an [`ImageSource::Base64`] — the
/// common pattern for "attach this local image to a multimodal message."
pub async fn image_source_from_path(
    path: impl AsRef<std::path::Path>,
) -> crate::Result<(ImageSource, String)> {
    let path = path.as_ref();
    let bytes = tokio::fs::read(path).await.map_err(|e| {
        crate::CognisError::Configuration(format!(
            "image_source_from_path: read `{}`: {e}",
            path.display()
        ))
    })?;
    let mime = mime_from_path(path)
        .unwrap_or("application/octet-stream")
        .to_string();
    Ok((ImageSource::base64(base64_encode(&bytes)), mime))
}

// ---------------------------------------------------------------------------
// Provider wire-format translators.
// ---------------------------------------------------------------------------

impl ContentPart {
    /// Serialize as an OpenAI Chat Completions content part.
    ///
    /// - Text  → `{"type":"text","text":"..."}`
    /// - Image → `{"type":"image_url","image_url":{"url":"..."}}` (URL or
    ///   `data:` URI for base64).
    /// - Audio → `{"type":"input_audio","input_audio":{"data":"...","format":"..."}}`
    pub fn to_openai(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => {
                serde_json::json!({"type": "text", "text": text})
            }
            ContentPart::Image { source, mime } => {
                let url = match source {
                    ImageSource::Url { url } => url.clone(),
                    ImageSource::Base64 { data } => format!("data:{mime};base64,{data}"),
                };
                serde_json::json!({"type": "image_url", "image_url": {"url": url}})
            }
            ContentPart::Audio { source, mime } => {
                let data = match source {
                    AudioSource::Base64 { data } => data.clone(),
                    AudioSource::Url { url } => url.clone(),
                };
                let format = mime.split('/').nth(1).unwrap_or("wav").to_string();
                serde_json::json!({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": format},
                })
            }
        }
    }

    /// Parse an OpenAI content part back into [`ContentPart`].
    pub fn from_openai(v: &serde_json::Value) -> Option<Self> {
        let kind = v["type"].as_str()?;
        match kind {
            "text" => Some(ContentPart::Text {
                text: v["text"].as_str()?.to_string(),
            }),
            "image_url" => {
                let url = v["image_url"]["url"].as_str()?;
                if let Some(rest) = url.strip_prefix("data:") {
                    if let Some((mime_part, b64)) = rest.split_once(";base64,") {
                        return Some(ContentPart::Image {
                            source: ImageSource::base64(b64),
                            mime: mime_part.to_string(),
                        });
                    }
                }
                Some(ContentPart::Image {
                    source: ImageSource::url(url),
                    mime: String::new(),
                })
            }
            _ => None,
        }
    }

    /// Serialize as an Anthropic content block.
    ///
    /// - Text  → `{"type":"text","text":"..."}`
    /// - Image → `{"type":"image","source":{"type":"url","url":...}}`
    ///   or `{"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}`
    /// - Audio → not supported by Anthropic at time of writing; emitted
    ///   as an Anthropic `text` block describing the audio reference.
    pub fn to_anthropic(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => {
                serde_json::json!({"type": "text", "text": text})
            }
            ContentPart::Image { source, mime } => match source {
                ImageSource::Url { url } => serde_json::json!({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }),
                ImageSource::Base64 { data } => serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": data,
                    },
                }),
            },
            ContentPart::Audio { source, mime } => {
                let stub = match source {
                    AudioSource::Url { url } => format!("[audio: {url} ({mime})]"),
                    AudioSource::Base64 { .. } => format!("[audio: base64 ({mime})]"),
                };
                serde_json::json!({"type": "text", "text": stub})
            }
        }
    }

    /// Parse an Anthropic content block back into [`ContentPart`].
    pub fn from_anthropic(v: &serde_json::Value) -> Option<Self> {
        let kind = v["type"].as_str()?;
        match kind {
            "text" => Some(ContentPart::Text {
                text: v["text"].as_str()?.to_string(),
            }),
            "image" => {
                let source_kind = v["source"]["type"].as_str()?;
                match source_kind {
                    "url" => Some(ContentPart::Image {
                        source: ImageSource::url(v["source"]["url"].as_str()?),
                        mime: String::new(),
                    }),
                    "base64" => Some(ContentPart::Image {
                        source: ImageSource::base64(v["source"]["data"].as_str()?),
                        mime: v["source"]["media_type"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                    }),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Serialize as a Gemini `Part`.
    ///
    /// - Text  → `{"text":"..."}`
    /// - Image → `{"inline_data":{"mime_type":"...","data":"..."}}` for
    ///   base64; `{"file_data":{"mime_type":"...","file_uri":"..."}}` for URL.
    /// - Audio → same shape as image (Gemini accepts audio inline/file_data).
    pub fn to_gemini(&self) -> serde_json::Value {
        match self {
            ContentPart::Text { text } => serde_json::json!({"text": text}),
            ContentPart::Image { source, mime } => match source {
                ImageSource::Url { url } => serde_json::json!({
                    "file_data": {"mime_type": mime, "file_uri": url},
                }),
                ImageSource::Base64 { data } => serde_json::json!({
                    "inline_data": {"mime_type": mime, "data": data},
                }),
            },
            ContentPart::Audio { source, mime } => match source {
                AudioSource::Url { url } => serde_json::json!({
                    "file_data": {"mime_type": mime, "file_uri": url},
                }),
                AudioSource::Base64 { data } => serde_json::json!({
                    "inline_data": {"mime_type": mime, "data": data},
                }),
            },
        }
    }

    /// Parse a Gemini `Part` back into [`ContentPart`].
    pub fn from_gemini(v: &serde_json::Value) -> Option<Self> {
        if let Some(t) = v["text"].as_str() {
            return Some(ContentPart::Text {
                text: t.to_string(),
            });
        }
        if let Some(inline) = v["inline_data"].as_object() {
            let mime = inline["mime_type"].as_str()?.to_string();
            let data = inline["data"].as_str()?.to_string();
            return Some(if mime.starts_with("audio/") {
                ContentPart::Audio {
                    source: AudioSource::base64(data),
                    mime,
                }
            } else {
                ContentPart::Image {
                    source: ImageSource::base64(data),
                    mime,
                }
            });
        }
        if let Some(file) = v["file_data"].as_object() {
            let mime = file["mime_type"].as_str()?.to_string();
            let uri = file["file_uri"].as_str()?.to_string();
            return Some(if mime.starts_with("audio/") {
                ContentPart::Audio {
                    source: AudioSource::url(uri),
                    mime,
                }
            } else {
                ContentPart::Image {
                    source: ImageSource::url(uri),
                    mime,
                }
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_part_roundtrip() {
        let p = ContentPart::Image {
            source: ImageSource::url("https://x"),
            mime: "image/png".into(),
        };
        let s = serde_json::to_string(&p).unwrap();
        let back: ContentPart = serde_json::from_str(&s).unwrap();
        assert_eq!(p, back);
        assert!(s.contains("\"type\":\"image\""));
        assert!(s.contains("\"kind\":\"url\""));
    }

    #[test]
    fn openai_url_image_roundtrip() {
        let p = ContentPart::Image {
            source: ImageSource::url("https://x/cat.png"),
            mime: "image/png".into(),
        };
        let v = p.to_openai();
        assert_eq!(v["type"], "image_url");
        let back = ContentPart::from_openai(&v).unwrap();
        assert!(matches!(back, ContentPart::Image { .. }));
    }

    #[test]
    fn openai_base64_image_via_data_uri() {
        let p = ContentPart::Image {
            source: ImageSource::base64("AAAA"),
            mime: "image/png".into(),
        };
        let v = p.to_openai();
        let url = v["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,AAAA"));
        let back = ContentPart::from_openai(&v).unwrap();
        assert_eq!(
            back,
            ContentPart::Image {
                source: ImageSource::base64("AAAA"),
                mime: "image/png".into(),
            }
        );
    }

    #[test]
    fn anthropic_base64_image_roundtrip() {
        let p = ContentPart::Image {
            source: ImageSource::base64("BBBB"),
            mime: "image/jpeg".into(),
        };
        let v = p.to_anthropic();
        assert_eq!(v["type"], "image");
        assert_eq!(v["source"]["type"], "base64");
        let back = ContentPart::from_anthropic(&v).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn gemini_inline_data_roundtrip() {
        let p = ContentPart::Image {
            source: ImageSource::base64("CCCC"),
            mime: "image/jpeg".into(),
        };
        let v = p.to_gemini();
        assert!(v["inline_data"]["data"].is_string());
        let back = ContentPart::from_gemini(&v).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn gemini_file_data_roundtrip() {
        let p = ContentPart::Image {
            source: ImageSource::url("gs://bucket/x.png"),
            mime: "image/png".into(),
        };
        let v = p.to_gemini();
        assert!(v["file_data"]["file_uri"].is_string());
        let back = ContentPart::from_gemini(&v).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn mime_from_path_recognises_common_extensions() {
        use std::path::Path;
        assert_eq!(mime_from_path(Path::new("a.png")), Some("image/png"));
        assert_eq!(mime_from_path(Path::new("a.jpg")), Some("image/jpeg"));
        assert_eq!(mime_from_path(Path::new("A.JPEG")), Some("image/jpeg"));
        assert_eq!(mime_from_path(Path::new("a.unknown")), None);
        assert_eq!(mime_from_path(Path::new("noext")), None);
    }

    #[test]
    fn base64_roundtrip() {
        for v in [
            &[][..],
            b"a",
            b"ab",
            b"abc",
            b"hello, world!",
            &[0u8, 1, 2, 3, 254, 255][..],
        ] {
            let enc = base64_encode(v);
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(dec, v.to_vec(), "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn base64_known_vector() {
        // RFC 4648 §10
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_decode_rejects_garbage() {
        assert!(base64_decode("****").is_err());
    }
}
