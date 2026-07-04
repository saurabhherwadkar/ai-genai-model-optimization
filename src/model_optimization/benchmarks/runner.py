"""
Benchmark runner for comparing model optimization approaches.

Measures latency, throughput, memory usage, and output quality
across different quantization methods and serving configurations.
"""

import time
from model_optimization.config.settings import get_settings
from model_optimization.models.schemas import BenchmarkResult, CostComparison
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runs performance benchmarks comparing optimization methods."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._results: list[BenchmarkResult] = []

    def run_benchmark(
        self, model_name: str, method: str, inference_fn, prompts: list[str]
    ) -> BenchmarkResult:
        """
        Run a complete benchmark suite on a model.

        Args:
            model_name: Name/path of the model.
            method: Optimization method used.
            inference_fn: Callable that takes a prompt and returns output.
            prompts: List of test prompts.

        Returns:
            BenchmarkResult with performance metrics.
        """
        num_samples = min(len(prompts), self._settings.benchmarks.num_samples)
        test_prompts = prompts[:num_samples]

        # Warmup
        for prompt in test_prompts[:self._settings.benchmarks.warmup_iterations]:
            inference_fn(prompt)

        # Measure latency
        latencies = []
        total_tokens = 0
        for prompt in test_prompts:
            start = time.time()
            output = inference_fn(prompt)
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            total_tokens += len(output.split())  # Approximate token count

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_time_s = sum(latencies) / 1000
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0

        result = BenchmarkResult(
            model_name=model_name,
            method=method,
            latency_ms=round(avg_latency, 2),
            throughput_tps=round(throughput, 2),
            memory_mb=0.0,  # Would use torch.cuda.memory_allocated() in production
            quality_score=0.0,  # Would compare against reference outputs
            num_samples=num_samples,
        )

        self._results.append(result)
        logger.info("benchmark_complete", model=model_name, method=method, latency_ms=avg_latency)
        return result

    def compare_all(self) -> CostComparison:
        """
        Compare all benchmarked models and generate recommendation.

        Returns:
            CostComparison with ranked results and recommendation.
        """
        if not self._results:
            return CostComparison(recommendation="No benchmarks to compare.")

        # Find best by latency
        best_latency = min(self._results, key=lambda r: r.latency_ms)
        # Find best by throughput
        best_throughput = max(self._results, key=lambda r: r.throughput_tps)

        recommendation = (
            f"Best latency: {best_latency.model_name} ({best_latency.method}) at {best_latency.latency_ms}ms. "
            f"Best throughput: {best_throughput.model_name} ({best_throughput.method}) at {best_throughput.throughput_tps} tokens/s."
        )

        return CostComparison(
            models=self._results,
            recommendation=recommendation,
        )

    def get_results(self) -> list[BenchmarkResult]:
        """Return all benchmark results."""
        return self._results
