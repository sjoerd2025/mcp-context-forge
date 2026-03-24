#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Python and Rust rate limiter hook performance.

This benchmark measures the real plugin hook path, not just the raw Rust engine.
It mirrors the comparison style used by other Rust plugins in this repository by
reporting Python-vs-Rust timings in ms/iteration for the same hook inputs.

Design choices for fairness:
- benchmark the allowed path only
- use fresh identities per iteration so counters do not accumulate differently
- compare the same hook (`prompt_pre_fetch` / `tool_pre_invoke`) with the same
  plugin config, only toggling whether the Rust engine is active
- use a dedicated Redis DB (default: /15) so the benchmark does not disturb the
  running local stack
"""

from __future__ import annotations

# Standard
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

# Third-Party
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# First-Party
from mcpgateway.plugins.framework import GlobalContext, PluginConfig, PluginContext, PromptPrehookPayload, ToolPreInvokePayload
from plugins.rate_limiter.rate_limiter import RateLimiterPlugin

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - dependency exists in repo venv
    aioredis = None


class BenchmarkResult(BaseModel):
    """One measured implementation result for a scenario."""

    implementation: str
    mean_ms: float
    median_ms: float
    p95_ms: float


@dataclass(frozen=True)
class Scenario:
    """A benchmark scenario."""

    algorithm: str
    backend: str
    hook: str


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a simple percentile from a sorted float sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _make_plugin_config(algorithm: str, backend: str, redis_url: str, redis_key_prefix: str) -> PluginConfig:
    """Create a minimal plugin config for the benchmark."""
    return PluginConfig(
        name=f"rate-limiter-bench-{algorithm}-{backend}",
        kind="plugins.rate_limiter.rate_limiter.RateLimiterPlugin",
        hooks=["prompt_pre_fetch", "tool_pre_invoke"],
        config={
            "algorithm": algorithm,
            "backend": backend,
            "by_user": "60/m",
            "redis_url": redis_url,
            "redis_key_prefix": redis_key_prefix,
            "redis_fallback": False,
        },
    )


def _build_plugin(algorithm: str, backend: str, use_rust: bool, redis_url: str, redis_key_prefix: str) -> RateLimiterPlugin:
    """Instantiate a plugin and force the requested implementation path."""
    plugin = RateLimiterPlugin(_make_plugin_config(algorithm, backend, redis_url, redis_key_prefix))
    if not use_rust:
        plugin._rust_engine = None
    elif plugin._rust_engine is None:
        raise RuntimeError("Rust rate limiter engine is not available. Run: make -C plugins_rust/rate_limiter install")
    return plugin


def _build_prompt_contexts(count: int) -> list[PluginContext]:
    """Build prompt benchmark contexts with fresh user identities."""
    return [
        PluginContext(global_context=GlobalContext(request_id=f"prompt-{i}", user=f"prompt-user-{i}@example.com"))
        for i in range(count)
    ]


def _build_tool_contexts(count: int) -> list[PluginContext]:
    """Build tool benchmark contexts with fresh user identities."""
    return [
        PluginContext(global_context=GlobalContext(request_id=f"tool-{i}", user=f"tool-user-{i}@example.com"))
        for i in range(count)
    ]


async def _invoke_hook(plugin: RateLimiterPlugin, hook: str, payload: Any, context: PluginContext) -> Any:
    """Invoke the selected plugin hook."""
    if hook == "prompt_pre_fetch":
        return await plugin.prompt_pre_fetch(payload, context)
    return await plugin.tool_pre_invoke(payload, context)


async def _cleanup_plugin(plugin: RateLimiterPlugin) -> None:
    """Cancel any sweep task left behind by the memory backend."""
    rate_backend = getattr(plugin, "_rate_backend", None)
    sweep_task = getattr(rate_backend, "_sweep_task", None)
    if sweep_task is not None:
        sweep_task.cancel()
        try:
            await sweep_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


async def _flush_redis(redis_url: str) -> None:
    """Flush the benchmark Redis DB for a clean run."""
    if aioredis is None:
        return
    client = aioredis.from_url(redis_url, decode_responses=False)
    try:
        await client.flushdb()
    finally:
        await client.aclose()


async def _redis_available(redis_url: str) -> bool:
    """Check whether the benchmark Redis target is reachable."""
    if aioredis is None:
        return False
    client = aioredis.from_url(redis_url, decode_responses=False)
    try:
        return bool(await client.ping())
    except Exception:
        return False
    finally:
        await client.aclose()


async def _parity_smoke_test(algorithm: str, backend: str, redis_url: str) -> None:
    """Quick sanity-check that Python and Rust agree on an allow/block sequence."""
    redis_key_prefix = f"rlbench-parity-{algorithm}-{backend}-{uuid4().hex}"
    if backend == "redis":
        await _flush_redis(redis_url)

    plugin_python = RateLimiterPlugin(
        PluginConfig(
            name="rate-limiter-parity-python",
            kind="plugins.rate_limiter.rate_limiter.RateLimiterPlugin",
            hooks=["tool_pre_invoke"],
            config={
                "algorithm": algorithm,
                "backend": backend,
                "by_user": "3/m",
                "redis_url": redis_url,
                "redis_key_prefix": redis_key_prefix,
                "redis_fallback": False,
            },
        )
    )
    plugin_python._rust_engine = None

    plugin_rust = RateLimiterPlugin(
        PluginConfig(
            name="rate-limiter-parity-rust",
            kind="plugins.rate_limiter.rate_limiter.RateLimiterPlugin",
            hooks=["tool_pre_invoke"],
            config={
                "algorithm": algorithm,
                "backend": backend,
                "by_user": "3/m",
                "redis_url": redis_url,
                "redis_key_prefix": f"{redis_key_prefix}-rust",
                "redis_fallback": False,
            },
        )
    )

    if plugin_rust._rust_engine is None:
        raise RuntimeError("Rust engine unavailable during parity check")

    payload = ToolPreInvokePayload(name="bench_tool", args={})
    python_sequence: list[bool] = []
    rust_sequence: list[bool] = []

    for idx in range(4):
        ctx_python = PluginContext(global_context=GlobalContext(request_id=f"parity-py-{idx}", user="same-user@example.com"))
        ctx_rust = PluginContext(global_context=GlobalContext(request_id=f"parity-rs-{idx}", user="same-user@example.com"))
        python_result = await plugin_python.tool_pre_invoke(payload, ctx_python)
        rust_result = await plugin_rust.tool_pre_invoke(payload, ctx_rust)
        python_sequence.append(python_result.continue_processing)
        rust_sequence.append(rust_result.continue_processing)

    await _cleanup_plugin(plugin_python)
    await _cleanup_plugin(plugin_rust)

    if python_sequence != rust_sequence:
        raise AssertionError(f"Parity failed for {algorithm}/{backend}: python={python_sequence}, rust={rust_sequence}")


async def _benchmark_scenario(
    scenario: Scenario,
    implementation: str,
    iterations: int,
    warmup: int,
    redis_url: str,
) -> BenchmarkResult:
    """Benchmark one scenario for either the Python or Rust path."""
    use_rust = implementation == "Rust"
    redis_key_prefix = f"rlbench-{scenario.algorithm}-{scenario.backend}-{scenario.hook}-{implementation.lower()}-{uuid4().hex}"

    if scenario.backend == "redis":
        await _flush_redis(redis_url)

    plugin = _build_plugin(
        algorithm=scenario.algorithm,
        backend=scenario.backend,
        use_rust=use_rust,
        redis_url=redis_url,
        redis_key_prefix=redis_key_prefix,
    )

    total_calls = iterations + warmup
    if scenario.hook == "prompt_pre_fetch":
        payload = PromptPrehookPayload(prompt_id="benchmark_prompt", args={})
        contexts = _build_prompt_contexts(total_calls)
    else:
        payload = ToolPreInvokePayload(name="benchmark_tool", args={})
        contexts = _build_tool_contexts(total_calls)

    # Warmup
    for idx in range(warmup):
        result = await _invoke_hook(plugin, scenario.hook, payload, contexts[idx])
        if not result.continue_processing:
            raise AssertionError(f"Unexpected rate-limit during warmup for {scenario.algorithm}/{scenario.backend}/{scenario.hook}")

    times_ms: list[float] = []
    for idx in range(warmup, total_calls):
        start = time.perf_counter()
        result = await _invoke_hook(plugin, scenario.hook, payload, contexts[idx])
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not result.continue_processing:
            raise AssertionError(f"Unexpected rate-limit during benchmark for {scenario.algorithm}/{scenario.backend}/{scenario.hook}")
        times_ms.append(elapsed_ms)

    await _cleanup_plugin(plugin)

    return BenchmarkResult(
        implementation=implementation,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        p95_ms=_percentile(times_ms, 0.95),
    )


async def _run(args: argparse.Namespace) -> int:
    """Run the benchmark suite."""
    scenarios = [
        Scenario(algorithm=algorithm, backend=backend, hook=hook)
        for algorithm in ("fixed_window", "sliding_window", "token_bucket")
        for backend in args.backends
        for hook in args.hooks
    ]

    redis_enabled = False
    if "redis" in args.backends:
        redis_enabled = await _redis_available(args.redis_url)
        if not redis_enabled:
            print(f"⚠️  Redis unavailable at {args.redis_url}; skipping Redis scenarios")

    print("🚦 Rate Limiter Performance Comparison (Plugin Hook Path)")
    print(f"Iterations: {args.iterations} (+ {args.warmup} warmup)")
    print(f"Hooks:      {', '.join(args.hooks)}")
    print(f"Backends:   {', '.join(args.backends)}")
    print(f"Redis URL:  {args.redis_url}")
    print()

    for algorithm in ("fixed_window", "sliding_window", "token_bucket"):
        for backend in args.backends:
            if backend == "redis" and not redis_enabled:
                continue
            await _parity_smoke_test(algorithm, backend, args.redis_url)

    print("Parity smoke checks: ✓")
    print()

    for scenario in scenarios:
        if scenario.backend == "redis" and not redis_enabled:
            continue
        print("=" * 88)
        print(f"Scenario: {scenario.algorithm} / {scenario.backend} / {scenario.hook}")
        print("=" * 88)
        python_result = await _benchmark_scenario(scenario, "Python", args.iterations, args.warmup, args.redis_url)
        rust_result = await _benchmark_scenario(scenario, "Rust", args.iterations, args.warmup, args.redis_url)
        speedup = python_result.mean_ms / rust_result.mean_ms if rust_result.mean_ms else 0.0
        print(f"  Python: mean {python_result.mean_ms:.3f} ms | median {python_result.median_ms:.3f} ms | p95 {python_result.p95_ms:.3f} ms")
        print(f"  Rust:   mean {rust_result.mean_ms:.3f} ms | median {rust_result.median_ms:.3f} ms | p95 {rust_result.p95_ms:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x faster")
        print()

    print("✅ Comparison complete")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Rate limiter Python vs Rust hook-path benchmark")
    parser.add_argument("--iterations", type=int, default=1000, help="Measured iterations per scenario")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations per scenario")
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/15",
        help="Dedicated Redis URL for benchmark scenarios (defaults to DB 15)",
    )
    parser.add_argument(
        "--hooks",
        nargs="+",
        default=["prompt_pre_fetch", "tool_pre_invoke"],
        choices=["prompt_pre_fetch", "tool_pre_invoke"],
        help="Hooks to benchmark",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["memory", "redis"],
        choices=["memory", "redis"],
        help="Backends to benchmark",
    )
    return parser.parse_args()


def main() -> int:
    """Run the async benchmark entrypoint."""
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
