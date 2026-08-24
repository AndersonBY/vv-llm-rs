use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::io::Cursor;

use crate::{ChatRequest, MessageContent, VvLlmError};

/// Resize data-URL image inputs to the model's maximum single dimension.
///
/// Non-data URLs are preserved because their dimensions cannot be inspected
/// without introducing a provider/network request into request preparation.
pub fn normalize_image_inputs(
    request: &mut ChatRequest,
    max_image_dimension: Option<u32>,
) -> Result<(), VvLlmError> {
    let Some(max_image_dimension) = max_image_dimension else {
        return Ok(());
    };
    if max_image_dimension == 0 {
        return Err(VvLlmError::Configuration(
            "max_image_dimension must be at least 1".to_string(),
        ));
    }

    for message in &mut request.messages {
        for content in &mut message.content {
            if let MessageContent::ImageUrl { url, .. } = content {
                *url = normalize_image_url(url, max_image_dimension)?;
            }
        }
    }
    Ok(())
}

/// Async variant that also inspects HTTP(S) image URLs before sending them.
pub async fn normalize_image_inputs_async(
    request: &mut ChatRequest,
    max_image_dimension: Option<u32>,
) -> Result<(), VvLlmError> {
    let Some(max_image_dimension) = max_image_dimension else {
        return Ok(());
    };
    if max_image_dimension == 0 {
        return Err(VvLlmError::Configuration(
            "max_image_dimension must be at least 1".to_string(),
        ));
    }

    for message in &mut request.messages {
        for content in &mut message.content {
            let MessageContent::ImageUrl { url, .. } = content else {
                continue;
            };
            let source_url = url.clone();
            *url = normalize_image_url_async(&source_url, max_image_dimension).await?;
        }
    }
    Ok(())
}

fn normalize_image_url(url: &str, max_image_dimension: u32) -> Result<String, VvLlmError> {
    let Some((media_type, bytes)) = decode_data_url(url)? else {
        return Ok(url.to_string());
    };

    resize_image_data_url(url, &media_type, &bytes, max_image_dimension)
}

async fn normalize_image_url_async(
    url: &str,
    max_image_dimension: u32,
) -> Result<String, VvLlmError> {
    if url
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
    {
        return normalize_image_url(url, max_image_dimension);
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Ok(url.to_string());
    }

    let response = reqwest::get(url)
        .await
        .map_err(|error| VvLlmError::Http(format!("cannot fetch image input: {error}")))?;
    if !response.status().is_success() {
        return Err(VvLlmError::Http(format!(
            "image input request returned HTTP {}",
            response.status()
        )));
    }
    let media_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_ascii_lowercase();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| VvLlmError::Http(format!("cannot read image input: {error}")))?;

    resize_image_data_url(url, &media_type, &bytes, max_image_dimension)
}

fn resize_image_data_url(
    source_url: &str,
    media_type: &str,
    source_bytes: &[u8],
    max_image_dimension: u32,
) -> Result<String, VvLlmError> {
    let image = image::load_from_memory(source_bytes).map_err(|error| {
        VvLlmError::Configuration(format!("cannot decode image input: {error}"))
    })?;
    let (width, height) = image.dimensions();
    let longest_dimension = width.max(height);
    if longest_dimension <= max_image_dimension {
        return Ok(source_url.to_string());
    }

    let scale = max_image_dimension as f64 / longest_dimension as f64;
    let resized_width = ((width as f64 * scale).floor() as u32).max(1);
    let resized_height = ((height as f64 * scale).floor() as u32).max(1);
    let resized = image.resize_exact(
        resized_width,
        resized_height,
        image::imageops::FilterType::Lanczos3,
    );
    encode_image_data_url(resized, media_type, source_bytes)
}

fn decode_data_url(url: &str) -> Result<Option<(String, Vec<u8>)>, VvLlmError> {
    if !url
        .get(..5)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
    {
        return Ok(None);
    }

    let (metadata, encoded) = url.split_once(',').ok_or_else(|| {
        VvLlmError::Configuration("image data URL is missing comma separator".to_string())
    })?;
    let mut metadata_parts = metadata[5..].split(';');
    let media_type = metadata_parts
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            VvLlmError::Configuration("image data URL is missing media type".to_string())
        })?;
    if !metadata_parts.any(|part| part.eq_ignore_ascii_case("base64")) {
        return Err(VvLlmError::Configuration(
            "image URL must be base64 data URL".to_string(),
        ));
    }
    let bytes = STANDARD.decode(encoded).map_err(|error| {
        VvLlmError::Configuration(format!("invalid image data URL base64: {error}"))
    })?;
    Ok(Some((media_type, bytes)))
}

fn encode_image_data_url(
    image: DynamicImage,
    source_media_type: &str,
    source_bytes: &[u8],
) -> Result<String, VvLlmError> {
    let (format, output_media_type) = output_format(source_media_type, source_bytes)?;
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, format).map_err(|error| {
        VvLlmError::Configuration(format!("cannot encode resized image: {error}"))
    })?;
    Ok(format!(
        "data:{output_media_type};base64,{}",
        STANDARD.encode(encoded.into_inner())
    ))
}

fn output_format(
    source_media_type: &str,
    source_bytes: &[u8],
) -> Result<(ImageFormat, &'static str), VvLlmError> {
    match source_media_type {
        "image/png" => Ok((ImageFormat::Png, "image/png")),
        "image/jpeg" | "image/jpg" => Ok((ImageFormat::Jpeg, "image/jpeg")),
        "image/gif" => Ok((ImageFormat::Gif, "image/gif")),
        "image/webp" => Ok((ImageFormat::WebP, "image/webp")),
        "image/bmp" => Ok((ImageFormat::Bmp, "image/bmp")),
        "image/tiff" => Ok((ImageFormat::Tiff, "image/tiff")),
        "application/octet-stream" => {
            let format = image::guess_format(source_bytes).map_err(|error| {
                VvLlmError::Configuration(format!("cannot determine image format: {error}"))
            })?;
            let media_type = match format {
                ImageFormat::Png => "image/png",
                ImageFormat::Jpeg => "image/jpeg",
                ImageFormat::Gif => "image/gif",
                ImageFormat::WebP => "image/webp",
                ImageFormat::Bmp => "image/bmp",
                ImageFormat::Tiff => "image/tiff",
                _ => {
                    return Err(VvLlmError::Configuration(
                        "unsupported image format for resizing".to_string(),
                    ))
                }
            };
            Ok((format, media_type))
        }
        other => Err(VvLlmError::Configuration(format!(
            "unsupported image media type for resizing: {other}"
        ))),
    }
}
