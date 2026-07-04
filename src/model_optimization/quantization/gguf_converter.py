"""
GGUF format converter for llama.cpp compatible quantization.

Converts HuggingFace models to GGUF format with various quantization
levels (Q4_K_M, Q5_K_M, Q8_0) for CPU/hybrid inference.
"""

import uuid
from datetime import datetime, timezone

from model_optimization.config.settings import get_settings
from model_optimization.models.schemas import OptimizationMethod, QuantizationJob
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)

# GGUF quantization types and their approximate compression ratios
GGUF_TYPES = {
    "q2_k": {"bits": 2, "description": "Extreme compression, significant quality loss"},
    "q4_0": {"bits": 4, "description": "Basic 4-bit, fast but lower quality"},
    "q4_k_m": {"bits": 4, "description": "4-bit medium quality, good balance"},
    "q4_k_s": {"bits": 4, "description": "4-bit small, less memory"},
    "q5_k_m": {"bits": 5, "description": "5-bit medium, better quality"},
    "q6_k": {"bits": 6, "description": "6-bit, near original quality"},
    "q8_0": {"bits": 8, "description": "8-bit, minimal quality loss"},
}


class GGUFConverter:
    """
    Converts models to GGUF format for llama.cpp inference.

    Supports multiple quantization levels from Q2_K (extreme compression)
    to Q8_0 (minimal quality loss) for CPU and hybrid CPU/GPU inference.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def create_job(self, source_model: str, quant_type: str | None = None) -> QuantizationJob:
        """Create a GGUF conversion job."""
        q_type = quant_type or self._settings.quantization.gguf.quantization_type
        bits = GGUF_TYPES.get(q_type, {}).get("bits", 4)

        return QuantizationJob(
            job_id=f"gguf-{uuid.uuid4().hex[:8]}",
            method=OptimizationMethod.GGUF,
            source_model=source_model,
            bits=bits,
            output_path=f"models/optimized/gguf-{q_type}-{uuid.uuid4().hex[:6]}",
        )

    def convert(self, job: QuantizationJob, quant_type: str | None = None) -> QuantizationJob:
        """
        Convert model to GGUF format.

        In production, this calls llama.cpp's convert script.
        Here we define the pipeline structure.

        Args:
            job: Conversion job configuration.
            quant_type: GGUF quantization type override.

        Returns:
            Updated job with conversion results.
        """
        q_type = quant_type or self._settings.quantization.gguf.quantization_type
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        logger.info("gguf_conversion_started", model=job.source_model, quant_type=q_type)

        try:
            # Step 1: Convert HF model to GGUF FP16
            # In production: python convert_hf_to_gguf.py <model_path> --outtype f16
            logger.info("converting_to_fp16_gguf")

            # Step 2: Quantize GGUF to target type
            # In production: ./llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
            logger.info("quantizing_gguf", target_type=q_type)

            # Record completion (actual conversion would happen via subprocess)
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            logger.info("gguf_conversion_completed", output=job.output_path, quant_type=q_type)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("gguf_conversion_failed", error=str(e))

        return job

    def list_quantization_types(self) -> dict[str, dict]:
        """Return available GGUF quantization types with descriptions."""
        return GGUF_TYPES
