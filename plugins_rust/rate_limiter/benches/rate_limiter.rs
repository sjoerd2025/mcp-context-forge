// Copyright 2026
// SPDX-License-Identifier: Apache-2.0
//
// Criterion benchmarks for the rate limiter memory backend.
// PERF-01, MEM-02, MEM-03, MEM-04.
//
// Each iteration measures a single "under-limit" request — the realistic
// hot path in production.  The clock advances between iterations so the
// window resets and counters stay low, preventing benchmark drift into the
// "blocked" code path or unbounded VecDeque growth (sliding window).

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use rate_limiter_rust::{
    clock::{FakeClock, FakeClockHandle},
    config::{Algorithm, EngineConfig, parse_rate},
    engine::RateLimiterEngine,
};

const T0_UNIX: i64 = 1_000_000;
const LIMIT: u64 = 100;
const WINDOW: u64 = 60_000_000_000; // 60s in nanos

fn make_engine(algorithm: Algorithm) -> (RateLimiterEngine, FakeClockHandle) {
    let (clock, handle) = FakeClock::new(T0_UNIX);
    let cfg = EngineConfig {
        by_user: Some(parse_rate(&format!("{}/m", LIMIT)).unwrap()),
        by_tenant: None,
        by_tool: Default::default(),
        algorithm,
    };
    (RateLimiterEngine::new_with_clock(cfg, Arc::new(clock)), handle)
}

fn bench_fixed_window(c: &mut Criterion) {
    let (engine, handle) = make_engine(Algorithm::FixedWindow);
    c.bench_function("fixed_window/single_key", |b| {
        b.iter(|| {
            // Advance past the window so each iteration is a fresh "allowed" request.
            handle.advance_secs(61);
            engine
                .evaluate_many(
                    black_box(vec![("user:bench".to_string(), LIMIT, WINDOW)]),
                    handle.unix_secs(),
                )
                .unwrap()
        })
    });
}

fn bench_token_bucket(c: &mut Criterion) {
    let (engine, handle) = make_engine(Algorithm::TokenBucket);
    c.bench_function("token_bucket/single_key", |b| {
        b.iter(|| {
            handle.advance_secs(61);
            engine
                .evaluate_many(
                    black_box(vec![("user:bench".to_string(), LIMIT, WINDOW)]),
                    handle.unix_secs(),
                )
                .unwrap()
        })
    });
}

fn bench_sliding_window(c: &mut Criterion) {
    let (engine, handle) = make_engine(Algorithm::SlidingWindow);
    c.bench_function("sliding_window/single_key", |b| {
        b.iter(|| {
            handle.advance_secs(61);
            engine
                .evaluate_many(
                    black_box(vec![("user:bench".to_string(), LIMIT, WINDOW)]),
                    handle.unix_secs(),
                )
                .unwrap()
        })
    });
}

fn bench_multi_dim(c: &mut Criterion) {
    let (engine, handle) = make_engine(Algorithm::FixedWindow);
    c.bench_function("fixed_window/three_dims", |b| {
        b.iter(|| {
            handle.advance_secs(61);
            engine
                .evaluate_many(
                    black_box(vec![
                        ("user:alice".to_string(), LIMIT, WINDOW),
                        ("tenant:acme".to_string(), LIMIT * 100, WINDOW),
                        ("tool:search".to_string(), LIMIT / 10, WINDOW),
                    ]),
                    handle.unix_secs(),
                )
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_fixed_window,
    bench_token_bucket,
    bench_sliding_window,
    bench_multi_dim
);
criterion_main!(benches);
