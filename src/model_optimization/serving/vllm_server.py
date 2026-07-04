"""
vLLM production serving configuration and management.

Provides configuration generation and health monitoring for
vLLM-based model deployment with optimized inference.
"""

from model_optimization.config.settings import get_settings
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)


class VLLMServer:
    """
    vLLM serving configuration and management.

    Generates optimized vLLM configurations for quantized models
    and provides deployment helpers.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def generate_config(self, model_path: str) -> dict:
        """
        Generate vLLM server configuration for a model.

        Args:
            model_path: Path to the model to serve.

        Returns:
            Dictionary with vLLM configuration parameters.
        """
        config = {
            "model": model_path,
            "max_model_len": self._settings.serving.max_model_len,
            "gpu_memory_utilization": self._settings.serving.gpu_memory_utilization,
            "tensor_parallel_size": self._settings.serving.tensor_parallel_size,
            "dtype": "auto",
            "enforce_eager": False,
            "enable_prefix_caching": True,
        }
        logger.info("vllm_config_generated", model=model_path)
        return config

    def get_launch_command(self, model_path: str, port: int = 8080) -> str:
        """
        Generate the vLLM launch command for the model.

        Args:
            model_path: Path to the model.
            port: Port to serve on.

        Returns:
            Shell command string to launch vLLM.
        """
        return (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} "
            f"--max-model-len {self._settings.serving.max_model_len} "
            f"--gpu-memory-utilization {self._settings.serving.gpu_memory_utilization} "
            f"--tensor-parallel-size {self._settings.serving.tensor_parallel_size} "
            f"--port {port} "
            f"--enable-prefix-caching"
        )

    def get_docker_compose(self, model_path: str) -> str:
        """Generate docker-compose snippet for vLLM deployment."""
        return f"""version: "3.9"
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    ports:
      - "8080:8080"
    volumes:
      - {model_path}:/model
    environment:
      - HUGGING_FACE_HUB_TOKEN=${{HUGGINGFACE_TOKEN}}
    command: >
      --model /model
      --max-model-len {self._settings.serving.max_model_len}
      --gpu-memory-utilization {self._settings.serving.gpu_memory_utilization}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {self._settings.serving.tensor_parallel_size}
              capabilities: [gpu]
"""
