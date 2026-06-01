import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# -- Configuration --
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_FUNCTION_ID = os.getenv("NVIDIA_FUNCTION_ID")
NVIDIA_TARGET_URL = os.getenv("NVIDIA_TARGET_URL")

if not all([NVIDIA_API_KEY, NVIDIA_FUNCTION_ID, NVIDIA_TARGET_URL]):
    raise EnvironmentError("Missing required NVIDIA environment variables in your .env file.")

# --- FFmpeg Dependency Check ---
# pydub and ffmpeg require FFmpeg for non-WAV audio format conversion.
# This check ensures it's available and provides a helpful error if not.
FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")
if not FFMPEG_PATH or not FFPROBE_PATH:
    # Set the path for pydub if found, otherwise it will raise an exception on its own.
    # This is more for clarity and to provide a better error message.
    print("ERROR: FFmpeg not found. Please install FFmpeg on your system to support non-WAV audio formats.")
    print("On macOS, you can install it with: brew install ffmpeg")
    print("On Debian/Ubuntu, you can install it with: sudo apt update && sudo apt install ffmpeg")

# Create a temporary directory for file processing
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)


def _cleanup_file(path: Path):
    """Background task to remove temporary files."""
    try:
        if path.exists():
            path.unlink()
    except OSError as e:
        # Log the error, but don't raise an exception in a background task
        print(f"Error cleaning up file {path}: {e}")