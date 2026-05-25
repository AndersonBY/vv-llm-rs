pub fn count_tokens_fallback(text: &str) -> usize {
    text.split_whitespace().count()
}

pub fn count_tokens(text: &str, model: &str) -> Result<usize, crate::VvLlmError> {
    let bpe = if model == "gpt-3.5-turbo" {
        Some(tiktoken_rs::cl100k_base_singleton())
    } else if model.starts_with("gpt-4o") || model.starts_with("o1-") || model.starts_with("o3-") {
        Some(tiktoken_rs::o200k_base_singleton())
    } else {
        None
    };

    Ok(bpe
        .map(|bpe| bpe.encode_with_special_tokens(text).len())
        .unwrap_or_else(|| count_tokens_fallback(text)))
}
