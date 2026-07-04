"""Main application entry point."""
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from model_optimization.api.router import router
from model_optimization.config.settings import get_settings

load_dotenv()

def create_app() -> FastAPI:
    app = FastAPI(title="Model Optimization API", description="Quantization, distillation, and production serving", version="1.0.0")
    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("model_optimization.main:app", host=settings.api.host, port=settings.api.port, reload=settings.api.reload)
