use crate::{ErrorDetails, ErrorKind, VvLlmError};
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter_basis_points: u16,
    total_timeout: Option<Duration>,
    retryable_kinds: u16,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            ..Self::default()
        }
    }

    pub fn max_attempts(self) -> u32 {
        self.max_attempts
    }

    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    pub fn with_jitter_ratio(mut self, ratio: f64) -> Self {
        self.jitter_basis_points = (ratio.clamp(0.0, 1.0) * 10_000.0).round() as u16;
        self
    }

    pub fn with_total_timeout(mut self, timeout: Duration) -> Self {
        self.total_timeout = Some(timeout);
        self
    }

    /// Replace the default transient error set used by this policy.
    pub fn with_retryable_kinds<I>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = ErrorKind>,
    {
        self.retryable_kinds = kinds
            .into_iter()
            .fold(0, |mask, kind| mask | error_kind_bit(kind));
        self
    }

    pub fn should_retry(self, error: &VvLlmError, attempt: u32) -> bool {
        attempt < self.max_attempts && self.retryable_kinds & error_kind_bit(error.kind()) != 0
    }

    pub fn delay_for(self, error: &VvLlmError, attempt: u32, random_unit: f64) -> Duration {
        if let Some(seconds) = error.retry_after_seconds() {
            return self.max_delay.min(Duration::from_secs_f64(seconds));
        }

        let exponent = attempt.saturating_sub(1).min(31);
        let delay = self
            .base_delay
            .checked_mul(1_u32 << exponent)
            .unwrap_or(self.max_delay)
            .min(self.max_delay);
        let jitter_ratio = f64::from(self.jitter_basis_points) / 10_000.0;
        let jitter_factor = 1.0 + jitter_ratio * (random_unit.clamp(0.0, 1.0) * 2.0 - 1.0);
        Duration::from_secs_f64(delay.as_secs_f64() * jitter_factor)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            jitter_basis_points: 2_000,
            total_timeout: None,
            retryable_kinds: default_retryable_kinds(),
        }
    }
}

fn default_retryable_kinds() -> u16 {
    [
        ErrorKind::RateLimited,
        ErrorKind::Network,
        ErrorKind::Timeout,
        ErrorKind::ProviderInternal,
    ]
    .into_iter()
    .fold(0, |mask, kind| mask | error_kind_bit(kind))
}

fn error_kind_bit(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::Authentication => 1 << 0,
        ErrorKind::RateLimited => 1 << 1,
        ErrorKind::Network => 1 << 2,
        ErrorKind::Timeout => 1 << 3,
        ErrorKind::InvalidRequest => 1 << 4,
        ErrorKind::ContextLength => 1 << 5,
        ErrorKind::ContentPolicy => 1 << 6,
        ErrorKind::ModelNotFound => 1 << 7,
        ErrorKind::ProviderInternal => 1 << 8,
        ErrorKind::Serialization => 1 << 9,
        ErrorKind::Configuration => 1 << 10,
        ErrorKind::Unknown => 1 << 11,
    }
}

pub fn parse_retry_after(
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    now: SystemTime,
) -> Option<Duration> {
    if let Some(duration) = retry_after_ms.and_then(|value| parse_seconds(value, 0.001)) {
        return Some(duration);
    }

    let value = retry_after?.trim();
    if let Some(duration) = parse_seconds(value, 1.0) {
        return Some(duration);
    }

    let retry_at = httpdate::parse_http_date(value).ok()?;
    Some(retry_at.duration_since(now).unwrap_or(Duration::ZERO))
}

pub(crate) fn parse_retry_after_headers(
    headers: &reqwest::header::HeaderMap,
    now: SystemTime,
) -> Option<Duration> {
    parse_retry_after(
        headers
            .get("retry-after-ms")
            .and_then(|value| value.to_str().ok()),
        headers
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        now,
    )
}

fn parse_seconds(value: &str, scale: f64) -> Option<Duration> {
    let seconds = value.trim().parse::<f64>().ok()? * scale;
    if !seconds.is_finite() {
        return None;
    }
    Some(Duration::from_secs_f64(seconds.max(0.0)))
}

pub async fn execute_with_retry<T, Operation, OperationFuture>(
    mut operation: Operation,
    policy: RetryPolicy,
) -> Result<T, VvLlmError>
where
    Operation: FnMut() -> OperationFuture,
    OperationFuture: Future<Output = Result<T, VvLlmError>>,
{
    let started = Instant::now();
    let mut attempt = 1;

    loop {
        let result = if let Some(total_timeout) = policy.total_timeout {
            let Some(remaining) = total_timeout.checked_sub(started.elapsed()) else {
                return Err(deadline_error());
            };
            match tokio::time::timeout(remaining, operation()).await {
                Ok(result) => result,
                Err(_) => return Err(deadline_error()),
            }
        } else {
            operation().await
        };

        match result {
            Ok(value) => return Ok(value),
            Err(error) if policy.should_retry(&error, attempt) => {
                let delay = policy.delay_for(&error, attempt, random_unit());
                if policy
                    .total_timeout
                    .is_some_and(|timeout| started.elapsed().saturating_add(delay) >= timeout)
                {
                    return Err(error);
                }
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

fn deadline_error() -> VvLlmError {
    VvLlmError::Classified(Box::new(ErrorDetails::new(
        ErrorKind::Timeout,
        "retry deadline exceeded",
    )))
}

fn random_unit() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    f64::from(nanos) / f64::from(u32::MAX)
}
