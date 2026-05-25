pub fn count_tokens_fallback(text: &str) -> usize {
    text.split_whitespace().count()
}
