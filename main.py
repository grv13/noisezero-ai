import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi.errors import RateLimitExceeded

from routes import audio_denoise, video_denoise, video_caption, audio_enhance
from limiter import limiter, rate_limit_exceeded_handler
from settings import settings


# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Add the gRPC generated code path for studiovoice to allow for absolute imports
# studiovoice_interface_path = project_root / "ai_clients" / "studio-voice" / "interfaces" / "studiovoice"
# sys.path.insert(0, str(studiovoice_interface_path))

app = FastAPI(
    title="NoiseZero AI API",
    description="A FastAPI wrapper for the NVIDIA Background Noise Removal NIM.",
)

@app.on_event("startup")
async def startup_db_client():
    """Create the MongoDB client on application startup."""
    app.mongodb_client = AsyncIOMotorClient(settings.MONGO_URI)
    app.mongodb = app.mongodb_client.caption_generator

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close the MongoDB client on application shutdown."""
    app.mongodb_client.close()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=str(project_root / "templates"))

app.include_router(audio_denoise.router, prefix="/denoise", tags=["Audio Denoising"])
app.include_router(video_denoise.router, prefix="/denoise", tags=["Video Denoising"])
app.include_router(audio_enhance.router, prefix="/enhance", tags=["Audio Enhancement"])
app.include_router(video_caption.router, prefix="/caption", tags=["Video Captioning"])

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Simple health check endpoint to confirm the service is running."""
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serves the main UI page."""
    return templates.TemplateResponse(request, "index.html", {"request": request})
