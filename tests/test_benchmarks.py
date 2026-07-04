"""Tests for the benchmark runner."""
import pytest
from model_optimization.benchmarks.runner import BenchmarkRunner


@pytest.fixture
def runner() -> BenchmarkRunner:
    return BenchmarkRunner()


class TestBenchmarkRunner:
    def test_run_benchmark(self, runner: BenchmarkRunner) -> None:
        def mock_inference(prompt: str) -> str:
            return "This is a generated response to " + prompt

        result = runner.run_benchmark("test-model", "gptq", mock_inference, ["Hello", "World", "Test"])
        assert result.model_name == "test-model"
        assert result.method == "gptq"
        assert result.latency_ms >= 0
        assert result.num_samples == 3

    def test_compare_multiple(self, runner: BenchmarkRunner) -> None:
        def fast_inference(prompt: str) -> str:
            return "fast " + prompt

        def slow_inference(prompt: str) -> str:
            import time
            time.sleep(0.01)
            return "slow " + prompt

        runner.run_benchmark("fast-model", "awq", fast_inference, ["test"] * 5)
        runner.run_benchmark("slow-model", "gptq", slow_inference, ["test"] * 5)

        comparison = runner.compare_all()
        assert len(comparison.models) == 2
        assert comparison.recommendation != ""

    def test_empty_comparison(self, runner: BenchmarkRunner) -> None:
        comparison = runner.compare_all()
        assert "No benchmarks" in comparison.recommendation
