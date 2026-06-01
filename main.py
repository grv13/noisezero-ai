import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi.errors import RateLimitExceeded

from routes import audio_denoise, video_denoise
from limiter import limiter, rate_limit_exceeded_handler

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="NVIDIA BNR API",
    description="A FastAPI wrapper for the NVIDIA Background Noise Removal NIM.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=str(project_root / "templates"))

app.include_router(audio_denoise.router, prefix="/denoise", tags=["Audio Denoising"])
app.include_router(video_denoise.router, prefix="/denoise", tags=["Video Denoising"])

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Simple health check endpoint to confirm the service is running."""
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serves the main UI page."""
    return templates.TemplateResponse(request, "index.html", {"request": request})
