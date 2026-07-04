"""FastAPI router for model optimization endpoints."""
from fastapi import APIRouter, HTTPException
from model_optimization.quantization import GPTQQuantizer, AWQQuantizer, GGUFConverter
from model_optimization.distillation import KnowledgeDistiller
from model_optimization.serving import VLLMServer
from model_optimization.benchmarks import BenchmarkRunner
from model_optimization.models.schemas import QuantizationJob, DistillationJob
from model_optimization.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/optimization", tags=["optimization"])

_gptq = GPTQQuantizer()
_awq = AWQQuantizer()
_gguf = GGUFConverter()
_distiller = KnowledgeDistiller()
_server = VLLMServer()
_benchmark = BenchmarkRunner()


@router.post("/quantize/gptq", response_model=QuantizationJob)
async def quantize_gptq(source_model: str) -> QuantizationJob:
    """Create GPTQ quantization job."""
    return _gptq.create_job(source_model)


@router.post("/quantize/awq", response_model=QuantizationJob)
async def quantize_awq(source_model: str) -> QuantizationJob:
    """Create AWQ quantization job."""
    return _awq.create_job(source_model)


@router.post("/quantize/gguf", response_model=QuantizationJob)
async def quantize_gguf(source_model: str, quant_type: str = "q4_k_m") -> QuantizationJob:
    """Create GGUF conversion job."""
    return _gguf.create_job(source_model, quant_type)


@router.get("/quantize/gguf/types")
async def list_gguf_types() -> dict:
    """List available GGUF quantization types."""
    return _gguf.list_quantization_types()


@router.post("/distill", response_model=DistillationJob)
async def create_distillation() -> DistillationJob:
    """Create knowledge distillation job."""
    return _distiller.create_job()


@router.post("/serve/config")
async def generate_serving_config(model_path: str) -> dict:
    """Generate vLLM serving configuration."""
    return _server.generate_config(model_path)


@router.get("/serve/command")
async def get_launch_command(model_path: str, port: int = 8080) -> dict:
    """Get vLLM launch command."""
    return {"command": _server.get_launch_command(model_path, port)}


@router.get("/benchmarks")
async def get_benchmarks() -> list:
    """Get all benchmark results."""
    return [r.model_dump() for r in _benchmark.get_results()]


@router.get("/benchmarks/compare")
async def compare_benchmarks() -> dict:
    """Compare all benchmarked models."""
    return _benchmark.compare_all().model_dump()


@router.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "model-optimization"}
