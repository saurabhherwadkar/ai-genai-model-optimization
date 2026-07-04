"""
AWQ (Activation-Aware Weight Quantization) for efficient 4-bit inference.

Identifies salient weight channels based on activation magnitudes
and protects them during quantization for better quality retention.
"""

import uuid
from datetime import datetime, timezone

from model_optimization.config.settings import get_settings
from model_optimization.models.schemas import OptimizationMethod, QuantizationJob
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)


class AWQQuantizer:
    """
    AWQ quantizer preserving activation-aware salient channels.

    Achieves better quality than naive quantization by identifying
    and protecting important weight channels during compression.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def create_job(self, source_model: str) -> QuantizationJob:
        """Create an AWQ quantization job."""
        return QuantizationJob(
            job_id=f"awq-{uuid.uuid4().hex[:8]}",
            method=OptimizationMethod.AWQ,
            source_model=source_model,
            bits=self._settings.quantization.awq.bits,
            output_path=f"models/optimized/awq-{uuid.uuid4().hex[:6]}",
        )

    def quantize(self, job: QuantizationJob) -> QuantizationJob:
        """
        Execute AWQ quantization.

        Args:
            job: QuantizationJob configuration.

        Returns:
            Updated job with status and metrics.
        """
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        logger.info("awq_quantization_started", model=job.source_model)

        try:
            # AWQ quantization using autoawq library
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                job.source_model, token=self._settings.huggingface_token
            )

            model = AutoAWQForCausalLM.from_pretrained(
                job.source_model, token=self._settings.huggingface_token
            )

            # Quantization config
            quant_config = {
                "zero_point": self._settings.quantization.awq.zero_point,
                "q_group_size": self._settings.quantization.awq.group_size,
                "w_bit": self._settings.quantization.awq.bits,
            }

            # Run quantization
            model.quantize(tokenizer, quant_config=quant_config)

            # Save
            model.save_quantized(job.output_path)
            tokenizer.save_pretrained(job.output_path)

            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            logger.info("awq_quantization_completed", output=job.output_path)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("awq_quantization_failed", error=str(e))

        return job
