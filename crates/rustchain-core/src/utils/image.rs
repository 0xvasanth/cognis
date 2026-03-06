//! Utilities for image processing in LLM contexts.
//!
//! Provides helpers for encoding image bytes to base64, detecting MIME types
//! from raw image data, and constructing data URIs suitable for inclusion in
//! multimodal LLM messages.

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};

/// Supported image MIME types that can be detected from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageMimeType {
    /// JPEG image (`image/jpeg`)
    Jpeg,
    /// PNG image (`image/png`)
    Png,
    /// GIF image (`image/gif`)
    Gif,
    /// WebP image (`image/webp`)
    Webp,
    /// BMP image (`image/bmp`)
    Bmp,
    /// SVG image (`image/svg+xml`)
    Svg,
    /// TIFF image (`image/tiff`)
    Tiff,
    /// ICO image (`image/x-icon`)
    Ico,
}

impl ImageMimeType {
    /// Return the MIME type string (e.g. `"image/png"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Bmp => "image/bmp",
            Self::Svg => "image/svg+xml",
            Self::Tiff => "image/tiff",
            Self::Ico => "image/x-icon",
        }
    }
}

impl std::fmt::Display for ImageMimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Detect the MIME type of an image from its raw bytes by inspecting magic
/// bytes (file signatures).
///
/// Returns `None` if the format is not recognized.
///
/// # Examples
///
/// ```
/// use rustchain_core::utils::image::detect_mime_type;
///
/// // PNG magic bytes
/// let png_header = b"\x89PNG\r\n\x1a\n";
/// assert_eq!(
///     detect_mime_type(png_header).map(|m| m.as_str()),
///     Some("image/png"),
/// );
/// ```
pub fn detect_mime_type(data: &[u8]) -> Option<ImageMimeType> {
    if data.len() < 4 {
        return None;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(ImageMimeType::Png);
    }

    // JPEG: FF D8 FF
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(ImageMimeType::Jpeg);
    }

    // GIF: "GIF87a" or "GIF89a"
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Some(ImageMimeType::Gif);
    }

    // WebP: "RIFF" ... "WEBP"
    if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        return Some(ImageMimeType::Webp);
    }

    // BMP: "BM"
    if data.starts_with(b"BM") {
        return Some(ImageMimeType::Bmp);
    }

    // TIFF: little-endian "II" 42 00  or big-endian "MM" 00 42
    if (data.starts_with(&[0x49, 0x49, 0x2A, 0x00]))
        || (data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]))
    {
        return Some(ImageMimeType::Tiff);
    }

    // ICO: 00 00 01 00
    if data.starts_with(&[0x00, 0x00, 0x01, 0x00]) {
        return Some(ImageMimeType::Ico);
    }

    // SVG: heuristic — look for "<?xml" or "<svg" near the start
    let prefix = if data.len() > 256 { &data[..256] } else { data };
    if let Ok(text) = std::str::from_utf8(prefix) {
        let trimmed = text.trim_start();
        if trimmed.starts_with("<?xml") || trimmed.starts_with("<svg") {
            return Some(ImageMimeType::Svg);
        }
    }

    None
}

/// Encode raw image bytes to a base64 string using standard (RFC 4648)
/// encoding with padding.
///
/// # Examples
///
/// ```
/// use rustchain_core::utils::image::encode_image;
///
/// let encoded = encode_image(b"hello");
/// assert_eq!(encoded, "aGVsbG8=");
/// ```
pub fn encode_image(data: &[u8]) -> String {
    BASE64_STANDARD.encode(data)
}

/// Convert raw image bytes into a
/// [data URI](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs)
/// string.
///
/// The MIME type is auto-detected from the image magic bytes. If the format
/// cannot be determined the function falls back to `"application/octet-stream"`.
///
/// The resulting string has the form:
/// ```text
/// data:<mime>;base64,<encoded_data>
/// ```
///
/// # Examples
///
/// ```
/// use rustchain_core::utils::image::image_to_data_uri;
///
/// let png_bytes: Vec<u8> = {
///     let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
///     v.extend_from_slice(&[0u8; 32]);
///     v
/// };
/// let uri = image_to_data_uri(&png_bytes);
/// assert!(uri.starts_with("data:image/png;base64,"));
/// ```
pub fn image_to_data_uri(data: &[u8]) -> String {
    let mime = detect_mime_type(data)
        .map(|m| m.as_str())
        .unwrap_or("application/octet-stream");
    let encoded = encode_image(data);
    format!("data:{mime};base64,{encoded}")
}

/// Build a data URI from pre-encoded base64 data and an explicit MIME type.
///
/// This is useful when you already have base64-encoded image data (e.g. from
/// an API response) and just need to wrap it as a data URI.
///
/// # Examples
///
/// ```
/// use rustchain_core::utils::image::data_uri_from_base64;
///
/// let uri = data_uri_from_base64("aGVsbG8=", "image/png");
/// assert_eq!(uri, "data:image/png;base64,aGVsbG8=");
/// ```
pub fn data_uri_from_base64(base64_data: &str, mime_type: &str) -> String {
    format!("data:{mime_type};base64,{base64_data}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_mime_type ──────────────────────────────────────────────

    #[test]
    fn test_detect_png() {
        let data = b"\x89PNG\r\n\x1a\nsome_image_data";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Png));
    }

    #[test]
    fn test_detect_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(detect_mime_type(&data), Some(ImageMimeType::Jpeg));
    }

    #[test]
    fn test_detect_gif87a() {
        let data = b"GIF87a\x00\x00\x00\x00";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Gif));
    }

    #[test]
    fn test_detect_gif89a() {
        let data = b"GIF89a\x01\x00\x01\x00";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Gif));
    }

    #[test]
    fn test_detect_webp() {
        let mut data = b"RIFF".to_vec();
        data.extend_from_slice(&[0x00; 4]); // file size placeholder
        data.extend_from_slice(b"WEBP");
        data.extend_from_slice(&[0x00; 20]);
        assert_eq!(detect_mime_type(&data), Some(ImageMimeType::Webp));
    }

    #[test]
    fn test_detect_bmp() {
        let data = b"BM\x00\x00\x00\x00";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Bmp));
    }

    #[test]
    fn test_detect_tiff_little_endian() {
        let data = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00];
        assert_eq!(detect_mime_type(&data), Some(ImageMimeType::Tiff));
    }

    #[test]
    fn test_detect_tiff_big_endian() {
        let data = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x08];
        assert_eq!(detect_mime_type(&data), Some(ImageMimeType::Tiff));
    }

    #[test]
    fn test_detect_ico() {
        let data = [0x00, 0x00, 0x01, 0x00, 0x01, 0x00];
        assert_eq!(detect_mime_type(&data), Some(ImageMimeType::Ico));
    }

    #[test]
    fn test_detect_svg_xml_header() {
        let data = b"<?xml version=\"1.0\"?><svg></svg>";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Svg));
    }

    #[test]
    fn test_detect_svg_direct() {
        let data = b"<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Svg));
    }

    #[test]
    fn test_detect_svg_with_leading_whitespace() {
        let data = b"  \n  <svg></svg>";
        assert_eq!(detect_mime_type(data), Some(ImageMimeType::Svg));
    }

    #[test]
    fn test_detect_unknown() {
        let data = b"\x00\x01\x02\x03\x04\x05";
        assert_eq!(detect_mime_type(data), None);
    }

    #[test]
    fn test_detect_too_short() {
        assert_eq!(detect_mime_type(b""), None);
        assert_eq!(detect_mime_type(b"AB"), None);
        assert_eq!(detect_mime_type(b"ABC"), None);
    }

    // ── encode_image ─────────────────────────────────────────────────

    #[test]
    fn test_encode_image_empty() {
        assert_eq!(encode_image(b""), "");
    }

    #[test]
    fn test_encode_image_hello() {
        assert_eq!(encode_image(b"hello"), "aGVsbG8=");
    }

    #[test]
    fn test_encode_image_binary() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = encode_image(&data);
        // Verify round-trip
        let decoded = BASE64_STANDARD.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    // ── image_to_data_uri ────────────────────────────────────────────

    #[test]
    fn test_data_uri_png() {
        let mut data = b"\x89PNG\r\n\x1a\n".to_vec();
        data.extend_from_slice(&[0u8; 8]);
        let uri = image_to_data_uri(&data);
        assert!(uri.starts_with("data:image/png;base64,"));
        // Verify the base64 portion decodes back
        let b64_part = uri.strip_prefix("data:image/png;base64,").unwrap();
        let decoded = BASE64_STANDARD.decode(b64_part).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_data_uri_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let uri = image_to_data_uri(&data);
        assert!(uri.starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn test_data_uri_unknown_format() {
        let data = b"not_an_image_really";
        let uri = image_to_data_uri(data);
        assert!(uri.starts_with("data:application/octet-stream;base64,"));
    }

    // ── data_uri_from_base64 ─────────────────────────────────────────

    #[test]
    fn test_data_uri_from_base64() {
        let uri = data_uri_from_base64("aGVsbG8=", "image/png");
        assert_eq!(uri, "data:image/png;base64,aGVsbG8=");
    }

    #[test]
    fn test_data_uri_from_base64_custom_mime() {
        let uri = data_uri_from_base64("AAAA", "image/webp");
        assert_eq!(uri, "data:image/webp;base64,AAAA");
    }

    // ── ImageMimeType ────────────────────────────────────────────────

    #[test]
    fn test_mime_type_display() {
        assert_eq!(ImageMimeType::Jpeg.to_string(), "image/jpeg");
        assert_eq!(ImageMimeType::Png.to_string(), "image/png");
        assert_eq!(ImageMimeType::Gif.to_string(), "image/gif");
        assert_eq!(ImageMimeType::Webp.to_string(), "image/webp");
        assert_eq!(ImageMimeType::Bmp.to_string(), "image/bmp");
        assert_eq!(ImageMimeType::Svg.to_string(), "image/svg+xml");
        assert_eq!(ImageMimeType::Tiff.to_string(), "image/tiff");
        assert_eq!(ImageMimeType::Ico.to_string(), "image/x-icon");
    }

    #[test]
    fn test_mime_type_as_str() {
        // Ensure as_str and Display agree
        for mime in [
            ImageMimeType::Jpeg,
            ImageMimeType::Png,
            ImageMimeType::Gif,
            ImageMimeType::Webp,
            ImageMimeType::Bmp,
            ImageMimeType::Svg,
            ImageMimeType::Tiff,
            ImageMimeType::Ico,
        ] {
            assert_eq!(mime.as_str(), mime.to_string());
        }
    }
}
