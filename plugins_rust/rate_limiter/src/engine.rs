// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// `RateLimiterEngine` — the single PyO3-exposed class (IFACE-02).
//
// Python calls `evaluate_many(checks)` once per hook invocation (ARCH-01).
// All dimension aggregation happens here (ARCH-02).
// The Python wrapper is policy-only and never does rate math (ARCH-03).

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::*;
use pyo3_async_runtimes::tokio::future_into_py;

use crate::clock::{Clock, SystemClock};
use crate::config::{ConfigError, EngineConfig};
use crate::memory::MemoryStore;
use crate::redis_backend::RedisRateLimiter;
use crate::types::{DimResult, EvalResult};

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

enum EngineBackend {
    Memory(Arc<MemoryStore>),
    Redis(Arc<RedisRateLimiter>),
}

// ---------------------------------------------------------------------------
// Check descriptor — one entry per active dimension
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// High-performance rate limiter engine.
///
/// Construct once per plugin instance (`__init__`), then call
/// `evaluate_many()` on every hook invocation.
///
/// Backend is selected at init time from the config dict:
/// - `backend: "memory"` (default) — in-process counting via `MemoryStore`
/// - `backend: "redis"` — Rust owns the Redis connection; same batch Lua
///   scripts as the Python `RedisBackend`, one EVAL per hook invocation
#[gen_stub_pyclass]
#[pyclass]
pub struct RateLimiterEngine {
    config: EngineConfig,
    backend: EngineBackend,
    clock: Arc<dyn Clock>,
}

enum AsyncBackend {
    Memory(Arc<MemoryStore>),
    Redis(Arc<RedisRateLimiter>),
}

impl RateLimiterEngine {
    /// Internal constructor — always uses the memory backend.
    /// Used by tests and benchmarks where clock injection is required.
    pub fn new_with_clock(config: EngineConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            backend: EngineBackend::Memory(Arc::new(MemoryStore::new())),
            config,
            clock,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl RateLimiterEngine {
    /// Construct from the Python config dict.
    ///
    /// Parses all rate strings and normalises `by_tool` keys at init time —
    /// never on the request path (IFACE-01, IFACE-05).
    ///
    /// Extra keys consumed here (not part of `EngineConfig`):
    /// - `backend`: `"memory"` (default) or `"redis"`
    /// - `redis_url`: required when `backend = "redis"`
    /// - `redis_key_prefix`: key namespace prefix (default `"rl"`)
    #[new]
    pub fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let by_user: Option<String> = config.get_item("by_user")?.and_then(|v| v.extract().ok());
        let by_tenant: Option<String> =
            config.get_item("by_tenant")?.and_then(|v| v.extract().ok());
        let algorithm: String = config
            .get_item("algorithm")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| "fixed_window".to_string());

        let by_tool: HashMap<String, String> = config
            .get_item("by_tool")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();

        let engine_config = EngineConfig::new(
            by_user.as_deref(),
            by_tenant.as_deref(),
            by_tool,
            &algorithm,
        )
        .map_err(|e: ConfigError| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let backend_str: String = config
            .get_item("backend")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| "memory".to_string());

        let backend = if backend_str == "redis" {
            let redis_url: String = config
                .get_item("redis_url")?
                .and_then(|v| v.extract().ok())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "redis_url is required when backend=redis",
                    )
                })?;
            let prefix: String = config
                .get_item("redis_key_prefix")?
                .and_then(|v| v.extract().ok())
                .unwrap_or_else(|| "rl".to_string());
            let redis_limiter = RedisRateLimiter::new(&redis_url, engine_config.algorithm, prefix)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            EngineBackend::Redis(Arc::new(redis_limiter))
        } else {
            EngineBackend::Memory(Arc::new(MemoryStore::new()))
        };

        Ok(Self {
            config: engine_config,
            backend,
            clock: Arc::new(SystemClock),
        })
    }

    /// Evaluate all active dimensions in a single call (ARCH-01, IFACE-02).
    ///
    /// `checks` is a list of `(key, limit_count, window_nanos)` tuples built
    /// by the Python wrapper from the request context.
    ///
    /// `now_unix` is `int(time.time())` from Python — passing it here means
    /// Python test mocks of `time.time()` propagate to header timestamps (CORR-02).
    ///
    /// Returns the most restrictive `EvalResult` across all dimensions (ARCH-02).
    pub fn evaluate_many(
        &self,
        checks: Vec<(String, u64, u64)>,
        now_unix: i64,
    ) -> PyResult<EvalResult> {
        Python::attach(|py| {
            let dim_results: Vec<DimResult> = py
                .detach(|| -> Result<Vec<DimResult>, String> {
                    match &self.backend {
                        EngineBackend::Memory(store) => {
                            let now_mono = self.clock.now_monotonic();
                            Ok(checks
                                .into_iter()
                                .map(|(key, limit_count, window_nanos)| {
                                    store.check_and_increment(
                                        &key,
                                        limit_count,
                                        window_nanos,
                                        self.config.algorithm,
                                        now_mono,
                                        now_unix,
                                    )
                                })
                                .collect())
                        }
                        EngineBackend::Redis(redis) => redis
                            .evaluate_many(&checks, now_unix)
                            .map_err(|e| e.to_string()),
                    }
                })
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            Ok(EvalResult::from_dims(&dim_results))
        })
    }

    /// Evaluate all active dimensions asynchronously.
    ///
    /// Intended for Redis-backed deployments so Python async hooks can await
    /// the Rust Redis path without blocking the event loop.
    pub fn evaluate_many_async<'py>(
        &self,
        py: Python<'py>,
        checks: Vec<(String, u64, u64)>,
        now_unix: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = match &self.backend {
            EngineBackend::Memory(store) => AsyncBackend::Memory(Arc::clone(store)),
            EngineBackend::Redis(redis) => AsyncBackend::Redis(Arc::clone(redis)),
        };
        let algorithm = self.config.algorithm;
        let clock = Arc::clone(&self.clock);

        future_into_py(py, async move {
            let dim_results: Vec<DimResult> = match backend {
                AsyncBackend::Memory(store) => {
                    let now_mono = clock.now_monotonic();
                    checks
                        .into_iter()
                        .map(|(key, limit_count, window_nanos)| {
                            store.check_and_increment(
                                &key,
                                limit_count,
                                window_nanos,
                                algorithm,
                                now_mono,
                                now_unix,
                            )
                        })
                        .collect()
                }
                AsyncBackend::Redis(redis) => redis
                    .evaluate_many_async(&checks, now_unix)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
            };

            Python::attach(|py| Py::new(py, EvalResult::from_dims(&dim_results)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FakeClock;
    use crate::config::Algorithm;

    fn init_python() {
        Python::initialize();
    }

    fn engine_with_fake_clock(
        by_user: Option<&str>,
        algorithm: Algorithm,
    ) -> (RateLimiterEngine, crate::clock::FakeClockHandle) {
        init_python();
        let (clock, handle) = FakeClock::new(1_000_000);
        let mut by_tool = HashMap::new();
        let cfg = EngineConfig {
            by_user: by_user.map(|s| crate::config::parse_rate(s).unwrap()),
            by_tenant: None,
            by_tool: {
                by_tool.insert(
                    "search".to_string(),
                    crate::config::parse_rate("5/m").unwrap(),
                );
                by_tool
            },
            algorithm,
        };
        let engine = RateLimiterEngine::new_with_clock(cfg, Arc::new(clock));
        (engine, handle)
    }

    // --- IFACE-01: config parsed at init ---

    #[test]
    fn config_parsed_at_init_by_tool_normalised() {
        let cfg = EngineConfig::new(
            Some("10/s"),
            None,
            {
                let mut m = HashMap::new();
                m.insert("Search".to_string(), "5/m".to_string());
                m
            },
            "fixed_window",
        )
        .unwrap();
        // Key must be lowercase
        assert!(cfg.by_tool.contains_key("search"));
        assert!(!cfg.by_tool.contains_key("Search"));
    }

    // --- IFACE-02: evaluate_many returns EvalResult ---

    #[test]
    fn evaluate_many_returns_eval_result_shape() {
        let (engine, handle) = engine_with_fake_clock(Some("10/s"), Algorithm::FixedWindow);
        let checks = vec![("user:alice".to_string(), 10, 1_000_000_000)];
        let result = engine.evaluate_many(checks, handle.unix_secs()).unwrap();
        // Shape: all fields present, first call always allowed
        assert!(result.allowed);
        assert_eq!(result.limit, 10);
        assert!(result.remaining > 0);
        assert!(result.retry_after.is_none());
    }

    // --- ARCH-01: evaluate_many is the only hot-path call ---
    // (Structural — enforced by the interface: Python has no other method to call)

    // --- CORR-03: reset_timestamp > now on allowed requests ---

    #[test]
    fn reset_timestamp_strictly_greater_than_now_on_allowed() {
        let (engine, handle) = engine_with_fake_clock(Some("10/s"), Algorithm::FixedWindow);
        let now_unix = handle.unix_secs();
        let checks = vec![("user:bob".to_string(), 10, 1_000_000_000)];
        let result = engine.evaluate_many(checks, now_unix).unwrap();
        assert!(result.allowed);
        assert!(
            result.reset_timestamp > now_unix,
            "reset_timestamp {} must be > now {}",
            result.reset_timestamp,
            now_unix
        );
    }

    // --- CORR-04: None tenant means no tenant check ---
    // (Structural — Python wrapper never adds a tenant check when tenant_id is None)

    // --- CORR-07: multi-dimension aggregation picks most restrictive ---

    #[test]
    fn evaluate_many_blocked_dimension_blocks_result() {
        let (engine, _handle) = engine_with_fake_clock(Some("2/s"), Algorithm::FixedWindow);
        // Exhaust the limit
        let checks = || vec![("user:carol".to_string(), 2, 1_000_000_000)];
        let _ = engine.evaluate_many(checks(), 1_000_000).unwrap(); // 1
        let _ = engine.evaluate_many(checks(), 1_000_000).unwrap(); // 2
        let result = engine.evaluate_many(checks(), 1_000_000).unwrap(); // 3 — must be blocked
        assert!(!result.allowed);
        assert_eq!(result.remaining, 0);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn evaluate_many_multiple_dims_picks_most_restrictive() {
        let (engine, _handle) = engine_with_fake_clock(None, Algorithm::FixedWindow);
        // user has 10/s, tenant has 2/s — after 2 requests tenant is exhausted
        let user_key = "user:dave".to_string();
        let tenant_key = "tenant:acme".to_string();
        let checks = || {
            vec![
                (user_key.clone(), 10, 1_000_000_000),
                (tenant_key.clone(), 2, 1_000_000_000),
            ]
        };
        let _ = engine.evaluate_many(checks(), 1_000_000).unwrap();
        let _ = engine.evaluate_many(checks(), 1_000_000).unwrap();
        let result = engine.evaluate_many(checks(), 1_000_000).unwrap();
        assert!(!result.allowed); // tenant exhausted → blocked
    }
}
