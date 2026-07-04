"""Settings for model optimization."""
import os
from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class GPTQSettings(BaseSettings):
    bits: int = Field(default=4)
    group_size: int = Field(default=128)
    desc_act: bool = Field(default=True)
    dataset: str = Field(default="c4")
    num_samples: int = Field(default=128)


class AWQSettings(BaseSettings):
    bits: int = Field(default=4)
    group_size: int = Field(default=128)
    zero_point: bool = Field(default=True)


class GGUFSettings(BaseSettings):
    quantization_type: str = Field(default="q4_k_m")


class QuantizationSettings(BaseSettings):
    default_method: str = Field(default="gptq")
    gptq: GPTQSettings = Field(default_factory=GPTQSettings)
    awq: AWQSettings = Field(default_factory=AWQSettings)
    gguf: GGUFSettings = Field(default_factory=GGUFSettings)


class DistillationSettings(BaseSettings):
    teacher_model: str = Field(default="meta-llama/Llama-3.1-70B-Instruct")
    student_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    temperature: float = Field(default=2.0)
    alpha: float = Field(default=0.5)
    num_epochs: int = Field(default=3)
    batch_size: int = Field(default=4)


class ServingSettings(BaseSettings):
    engine: str = Field(default="vllm")
    max_model_len: int = Field(default=4096)
    gpu_memory_utilization: float = Field(default=0.9)
    tensor_parallel_size: int = Field(default=1)


class BenchmarkSettings(BaseSettings):
    metrics: list[str] = Field(default_factory=lambda: ["latency_ms", "throughput_tps", "memory_mb", "quality_score"])
    num_samples: int = Field(default=50)
    warmup_iterations: int = Field(default=5)


class APISettings(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)


class LoggingSettings(BaseSettings):
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    file: str = Field(default="logs/app.log")


class Settings(BaseSettings):
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    distillation: DistillationSettings = Field(default_factory=DistillationSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    benchmarks: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    huggingface_token: str = Field(default="")
    model_config = {"env_prefix": "", "env_nested_delimiter": "__"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = os.getenv("APP_ENV", "development")
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    env_map = {"development": "dev", "production": "prod"}
    suffix = env_map.get(env, "")
    config_file = config_dir / f"application-{suffix}.yaml" if suffix else config_dir / "application.yaml"
    if not config_file.exists():
        config_file = config_dir / "application.yaml"
    cfg = {}
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}
    quant_cfg = cfg.get("quantization", {})
    return Settings(
        quantization=QuantizationSettings(
            default_method=quant_cfg.get("default_method", "gptq"),
            gptq=GPTQSettings(**quant_cfg.get("gptq", {})) if quant_cfg.get("gptq") else GPTQSettings(),
            awq=AWQSettings(**quant_cfg.get("awq", {})) if quant_cfg.get("awq") else AWQSettings(),
            gguf=GGUFSettings(**quant_cfg.get("gguf", {})) if quant_cfg.get("gguf") else GGUFSettings(),
        ) if quant_cfg else QuantizationSettings(),
        distillation=DistillationSettings(**cfg.get("distillation", {})) if cfg.get("distillation") else DistillationSettings(),
        serving=ServingSettings(**cfg.get("serving", {})) if cfg.get("serving") else ServingSettings(),
        benchmarks=BenchmarkSettings(**cfg.get("benchmarks", {})) if cfg.get("benchmarks") else BenchmarkSettings(),
        api=APISettings(**cfg.get("api", {})) if cfg.get("api") else APISettings(),
        logging=LoggingSettings(**cfg.get("logging", {})) if cfg.get("logging") else LoggingSettings(),
    )
