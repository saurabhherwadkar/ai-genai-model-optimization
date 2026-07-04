"""
GPTQ quantization for post-training weight compression.

Applies GPTQ algorithm to reduce model weights to 4-bit or 8-bit
precision with minimal quality degradation using calibration data.
"""

import uuid
from datetime import datetime, timezone

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

from model_optimization.config.settings import get_settings
from model_optimization.models.schemas import OptimizationMethod, QuantizationJob
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)


class GPTQQuantizer:
    """
    GPTQ quantizer for weight-only quantization.

    Uses calibration data to find optimal quantization parameters
    that minimize output degradation at 4-bit precision.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def create_job(self, source_model: str) -> QuantizationJob:
        """Create a GPTQ quantization job."""
        return QuantizationJob(
            job_id=f"gptq-{uuid.uuid4().hex[:8]}",
            method=OptimizationMethod.GPTQ,
            source_model=source_model,
            bits=self._settings.quantization.gptq.bits,
            output_path=f"models/optimized/gptq-{uuid.uuid4().hex[:6]}",
        )

    def quantize(self, job: QuantizationJob) -> QuantizationJob:
        """
        Execute GPTQ quantization on the source model.

        Args:
            job: QuantizationJob with source model and configuration.

        Returns:
            Updated job with results.
        """
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        logger.info("gptq_quantization_started", model=job.source_model, bits=job.bits)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                job.source_model, token=self._settings.huggingface_token
            )

            # Configure quantization parameters
            quantize_config = BaseQuantizeConfig(
                bits=self._settings.quantization.gptq.bits,
                group_size=self._settings.quantization.gptq.group_size,
                desc_act=self._settings.quantization.gptq.desc_act,
            )

            # Load model for quantization
            model = AutoGPTQForCausalLM.from_pretrained(
                job.source_model,
                quantize_config,
                token=self._settings.huggingface_token,
            )

            # Prepare calibration data
            calibration_data = self._get_calibration_data(tokenizer)

            # Run quantization
            model.quantize(calibration_data)

            # Save quantized model
            model.save_quantized(job.output_path)
            tokenizer.save_pretrained(job.output_path)

            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            logger.info("gptq_quantization_completed", output=job.output_path)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)
            logger.error("gptq_quantization_failed", error=str(e))

        return job

    def _get_calibration_data(self, tokenizer) -> list[str]:
        """Generate calibration dataset for quantization."""
        # Sample calibration prompts (production would use C4 or custom data)
        samples = [
            "The meaning of life is to find purpose and contribute to society.",
            "Machine learning algorithms learn patterns from data without explicit programming.",
            "Python is a versatile programming language used in web development, data science, and AI.",
            "Climate change is driven by greenhouse gas emissions from human activities.",
            "The Internet has transformed how people communicate, work, and access information.",
        ]
        # Repeat to reach desired sample count
        num_needed = self._settings.quantization.gptq.num_samples
        calibration = (samples * (num_needed // len(samples) + 1))[:num_needed]

        # Tokenize
        return [
            tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            for text in calibration
        ]
