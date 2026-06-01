import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from routes import audio_denoise, video_denoise

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="NVIDIA BNR API",
    description="A FastAPI wrapper for the NVIDIA Background Noise Removal NIM.",
)

templates = Jinja2Templates(directory=str(project_root / "templates"))

app.include_router(audio_denoise.router, prefix="/denoise", tags=["Audio Denoising"])
app.include_router(video_denoise.router, prefix="/denoise", tags=["Video Denoising"])

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    """Serves the main UI page."""
    return templates.TemplateResponse(request, "index.html", {"request": request})
