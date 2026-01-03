import os
# Suppress Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import get_settings
from app.api.v1.router import api_router
from app.services.tts import TTSEngine
from app.core.dependencies import set_tts_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    engine = TTSEngine(settings)
    engine.initialize()
    set_tts_engine(engine)
    
    yield
    
    # Shutdown
    engine.voice_manager.stop_cleanup_loop()

app = FastAPI(
    title=get_settings().service.name,
    version=get_settings().service.version,
    lifespan=lifespan
)

# Mount V1 API
app.include_router(api_router, prefix="/v1")

# Optional: Convenience root endpoint pointing to V1 info or Docs
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": get_settings().service.name,
        "version": get_settings().service.version,
        "docs": "/docs",
        "api_v1": "/v1/"
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.service.host, port=settings.service.port, reload=True)
