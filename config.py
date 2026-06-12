import shutil
from pathlib import Path

from settings import settings

# --- FFmpeg Dependency Check ---
# pydub and ffmpeg require FFmpeg for non-WAV audio format conversion.
# This check ensures it's available and provides a helpful error if not.
FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")

# Create a temporary directory for file processing
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# --- Caption Generation Directories ---
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
SUBTITLES_DIR = Path("subtitles")
SUBTITLES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)



def _cleanup_file(path: Path):
    """Background task to remove temporary files."""
    try:
        if path.exists():
            path.unlink()
    except OSError as e:
        # Log the error, but don't raise an exception in a background task
        print(f"Error cleaning up file {path}: {e}")