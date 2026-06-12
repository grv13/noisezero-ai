import subprocess
import uuid
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import (APIRouter, BackgroundTasks, Depends, File, HTTPException,
                     Request, UploadFile)
from fastapi.responses import FileResponse
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from config import (FFMPEG_PATH, OUTPUTS_DIR, SUBTITLES_DIR, TEMP_DIR,
                    UPLOADS_DIR)
from settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_jobs_collection(request: Request) -> AsyncIOMotorCollection:
    return request.app.mongodb["video_jobs"]


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

if not settings.GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found in .env file. Captioning will not work."
    )
groq_client = Groq(api_key=settings.GROQ_API_KEY)


# ---------------------------------------------------------------------------
# FFmpeg capability detection  (runs once at import time)
# ---------------------------------------------------------------------------

def _probe_ffmpeg_filters() -> set[str]:
    """Return the set of filter names available in the current FFmpeg build."""
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-filters"],
            capture_output=True, text=True, check=True
        )
        filters: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            # Each line looks like:  "... ass  V->V  ..."
            if len(parts) >= 2:
                filters.add(parts[1])
        return filters
    except Exception as exc:
        logger.warning("Could not probe FFmpeg filters: %s", exc)
        return set()


def _probe_ffmpeg_encoders() -> set[str]:
    """Return the set of encoder names available in the current FFmpeg build."""
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-encoders"],
            capture_output=True, text=True, check=True
        )
        encoders: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                encoders.add(parts[1])
        return encoders
    except Exception as exc:
        logger.warning("Could not probe FFmpeg encoders: %s", exc)
        return set()


_FFMPEG_FILTERS  = _probe_ffmpeg_filters()
_FFMPEG_ENCODERS = _probe_ffmpeg_encoders()

HAS_LIBASS       = "ass"        in _FFMPEG_FILTERS
HAS_SUBTITLES    = "subtitles"  in _FFMPEG_FILTERS
HAS_DRAWTEXT     = "drawtext"   in _FFMPEG_FILTERS
HAS_LIBX264      = "libx264"    in _FFMPEG_ENCODERS

logger.info(
    "FFmpeg capabilities — libass=%s  subtitles=%s  drawtext=%s  libx264=%s",
    HAS_LIBASS, HAS_SUBTITLES, HAS_DRAWTEXT, HAS_LIBX264
)


# ---------------------------------------------------------------------------
# Subtitle / caption data structures
# ---------------------------------------------------------------------------

ASS_HEADER = """\
[Script Info]
Title: Auto-generated captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _to_ass_time(seconds: float) -> str:
    """Convert fractional seconds → ASS timestamp  H:MM:SS.cs"""
    h  = int(seconds / 3600)
    m  = int((seconds % 3600) / 60)
    s  = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _to_srt_time(seconds: float) -> str:
    """Convert fractional seconds → SRT timestamp  HH:MM:SS,mmm"""
    h   = int(seconds / 3600)
    m   = int((seconds % 3600) / 60)
    s   = int(seconds % 60)
    ms  = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_caption_chunks(words: list, max_words: int = 3) -> list:
    """Group word-level transcript tokens into display chunks."""
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Audio extraction & transcription
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, job_id: str) -> Path:
    audio_path = TEMP_DIR / f"{job_id}.wav"
    command = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-vn",
        str(audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return audio_path


def transcribe_audio(audio_path: Path) -> object:
    with open(audio_path, "rb") as fh:
        result = groq_client.audio.transcriptions.create(
            file=(audio_path.name, fh.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    return result


# ---------------------------------------------------------------------------
# Subtitle file writers
# ---------------------------------------------------------------------------

def create_ass_file(transcript, job_id: str) -> Path:
    subtitle_path = SUBTITLES_DIR / f"{job_id}.ass"
    chunks = _build_caption_chunks(transcript.words)
    with open(subtitle_path, "w", encoding="utf-8") as fh:
        fh.write(ASS_HEADER)
        for chunk in chunks:
            start = chunk[0]["start"]
            end   = chunk[-1]["end"]
            text  = " ".join(x["word"] for x in chunk)
            fh.write(
                f"Dialogue: 0,{_to_ass_time(start)},{_to_ass_time(end)},"
                f"Default,,0,0,0,,{text}\n"
            )
    return subtitle_path


def create_srt_file(transcript, job_id: str) -> Path:
    subtitle_path = SUBTITLES_DIR / f"{job_id}.srt"
    chunks = _build_caption_chunks(transcript.words)
    with open(subtitle_path, "w", encoding="utf-8") as fh:
        for idx, chunk in enumerate(chunks, start=1):
            start = chunk[0]["start"]
            end   = chunk[-1]["end"]
            text  = " ".join(x["word"] for x in chunk)
            fh.write(f"{idx}\n{_to_srt_time(start)} --> {_to_srt_time(end)}\n{text}\n\n")
    return subtitle_path


# ---------------------------------------------------------------------------
# FFmpeg filter-string helpers
# ---------------------------------------------------------------------------

def _escape_filter_path(path: str) -> str:
    """
    Escape a filesystem path for use inside an FFmpeg -vf filter value.

    FFmpeg's filtergraph parser treats  \  :  '  [  ]  ,  ;  as special.
    On POSIX paths only  \  and  :  can appear; on Windows  \  is common.
    We must escape them so the parser sees them as literal characters.
    """
    # Order matters: escape backslash first so later replacements
    # don't double-escape.
    return (
        path
        .replace("\\", "\\\\")
        .replace(":",  "\\:")
        .replace("'",  "\\'")
        .replace("[",  "\\[")
        .replace("]",  "\\]")
        .replace(",",  "\\,")
        .replace(";",  "\\;")
    )


def _video_encoder_flags() -> list[str]:
    """Return the best available video encoder flags for this FFmpeg build."""
    if HAS_LIBX264:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    # libx264 missing — fall back to the built-in mpeg4 encoder (always present)
    logger.warning("libx264 not available; falling back to mpeg4 encoder.")
    return ["-c:v", "mpeg4", "-q:v", "5"]


# ---------------------------------------------------------------------------
# Rendering strategies  (tried in order of quality)
# ---------------------------------------------------------------------------

def _render_with_libass(
    video_path: Path,
    subtitle_path: Path,   # must be .ass
    output_path: Path,
) -> None:
    """Burn ASS subtitles using the 'ass' filter (requires libass in FFmpeg)."""
    escaped = _escape_filter_path(str(subtitle_path.resolve()))
    vf = f"ass=filename={escaped}"
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path.resolve()),
        "-vf", vf,
        *_video_encoder_flags(),
        "-c:a", "copy",
        str(output_path.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _render_with_subtitles_filter(
    video_path: Path,
    subtitle_path: Path,   # .ass or .srt both work
    output_path: Path,
) -> None:
    """
    Burn subtitles using the 'subtitles' filter.
    Run FFmpeg with cwd = subtitle directory so the path never enters
    the filter string (sidesteps all escaping issues on every OS).
    """
    vf = f"subtitles={subtitle_path.name}"
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path.resolve()),
        "-vf", vf,
        *_video_encoder_flags(),
        "-c:a", "copy",
        str(output_path.resolve()),
    ]
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(subtitle_path.parent.resolve()),   # <-- the key trick
    )


def _render_with_soft_subtitles(
    video_path: Path,
    subtitle_path: Path,   # .srt
    output_path: Path,
) -> None:
    """
    Mux subtitles as a soft (non-burned) subtitle track inside an MKV.
    No libass required.  Output is always .mkv regardless of input extension.
    """
    mkv_output = output_path.with_suffix(".mkv")
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path.resolve()),
        "-i", str(subtitle_path.resolve()),
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "srt",
        "-map", "0:v", "-map", "0:a", "-map", "1:0",
        str(mkv_output.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    # Rename so callers always get the path they were given (as .mkv)
    if mkv_output != output_path:
        mkv_output.rename(output_path)


# ---------------------------------------------------------------------------
# Main render dispatcher
# ---------------------------------------------------------------------------

def render_video(
    video_path: Path,
    transcript,
    job_id: str,
    original_extension: str,
) -> Path:
    """
    Render a captioned video using the best strategy available.

    Strategy priority:
      1. libass  ('ass' filter)     – highest quality, requires libass
      2. subtitles filter           – good quality, requires libass or built-in
      3. soft subtitle mux (MKV)   – no quality loss, subtitles are selectable
         but not burned in; always works

    Both ASS and SRT files are written up-front so each strategy can
    pick the format it works best with.
    """
    # Determine output container
    keep_ext = original_extension.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    output_ext = original_extension if keep_ext else ".mp4"

    # Strategy 3 always produces MKV
    if not (HAS_LIBASS or HAS_SUBTITLES):
        output_ext = ".mkv"

    output_path = OUTPUTS_DIR / f"{job_id}{output_ext}"

    # Write subtitle files (we may need both formats)
    ass_path = create_ass_file(transcript, job_id)
    srt_path = create_srt_file(transcript, job_id)

    errors: list[str] = []

    # --- Strategy 1: libass 'ass' filter ---
    if HAS_LIBASS:
        try:
            logger.info("[%s] Rendering with libass 'ass' filter", job_id)
            _render_with_libass(video_path, ass_path, output_path)
            logger.info("[%s] Render complete (libass)", job_id)
            return output_path
        except subprocess.CalledProcessError as exc:
            msg = f"libass strategy failed: {exc.stderr[-400:]}"
            logger.warning("[%s] %s", job_id, msg)
            errors.append(msg)

    # --- Strategy 2: 'subtitles' filter ---
    if HAS_SUBTITLES:
        try:
            logger.info("[%s] Rendering with 'subtitles' filter", job_id)
            _render_with_subtitles_filter(video_path, ass_path, output_path)
            logger.info("[%s] Render complete (subtitles filter)", job_id)
            return output_path
        except subprocess.CalledProcessError as exc:
            msg = f"subtitles-filter strategy failed: {exc.stderr[-400:]}"
            logger.warning("[%s] %s", job_id, msg)
            errors.append(msg)

        # Retry strategy 2 with SRT (sometimes more compatible)
        try:
            logger.info("[%s] Retrying 'subtitles' filter with SRT", job_id)
            _render_with_subtitles_filter(video_path, srt_path, output_path)
            logger.info("[%s] Render complete (subtitles+SRT)", job_id)
            return output_path
        except subprocess.CalledProcessError as exc:
            msg = f"subtitles-filter+SRT strategy failed: {exc.stderr[-400:]}"
            logger.warning("[%s] %s", job_id, msg)
            errors.append(msg)

    # --- Strategy 3: soft subtitle mux ---
    try:
        logger.info("[%s] Falling back to soft subtitle mux (MKV)", job_id)
        soft_output = output_path.with_suffix(".mkv")
        _render_with_soft_subtitles(video_path, srt_path, soft_output)
        logger.info("[%s] Render complete (soft mux)", job_id)
        return soft_output
    except subprocess.CalledProcessError as exc:
        errors.append(f"soft-mux strategy failed: {exc.stderr[-400:]}")

    # All strategies exhausted
    raise RuntimeError(
        "All subtitle rendering strategies failed.\n\n"
        + "\n---\n".join(errors)
        + "\n\nFix: brew install ffmpeg (Homebrew default build includes libass)."
    )


# ---------------------------------------------------------------------------
# Background processing pipeline
# ---------------------------------------------------------------------------

@lru_cache()
def _get_bg_db():
    """One MongoDB client reused across all background tasks."""
    client = AsyncIOMotorClient(settings.MONGO_URI)
    return client.caption_generator


async def process_video(job_id: str, video_path: Path, original_extension: str):
    jobs_collection = _get_bg_db()["video_jobs"]
    try:
        await jobs_collection.update_one(
            {"_id": job_id},
            {"$set": {"status": "processing", "updated_at": datetime.utcnow()}},
        )

        audio_file = extract_audio(video_path, job_id)
        transcript = transcribe_audio(audio_file)
        output_video_path = render_video(
            video_path, transcript, job_id, original_extension
        )

        await jobs_collection.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "output_video_path": str(output_video_path),
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    except Exception as exc:
        error_message = (
            exc.stderr if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        logger.error("[%s] Processing failed: %s", job_id, error_message)
        await jobs_collection.update_one(
            {"_id": job_id},
            {"$set": {"status": "failed", "error": error_message, "updated_at": datetime.utcnow()}},
        )

    finally:
        for tmp in [TEMP_DIR / f"{job_id}.wav"]:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    jobs_collection: AsyncIOMotorCollection = Depends(get_jobs_collection),
):
    """Upload a video to generate captions. Returns a job_id immediately."""
    if not FFMPEG_PATH:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg is not configured. Set FFMPEG_PATH in your environment.",
        )

    job_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    video_path = UPLOADS_DIR / f"{job_id}{file_extension}"

    with open(video_path, "wb") as fh:
        fh.write(await file.read())

    await jobs_collection.insert_one({
        "_id": job_id,
        "status": "pending",
        "original_filename": file.filename,
        "original_video_path": str(video_path),
        "created_at": datetime.utcnow(),
    })

    background_tasks.add_task(process_video, job_id, video_path, file_extension)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    jobs_collection: AsyncIOMotorCollection = Depends(get_jobs_collection),
):
    """Poll the status of a captioning job."""
    job = await jobs_collection.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/download/{job_id}")
async def download_video(
    job_id: str,
    jobs_collection: AsyncIOMotorCollection = Depends(get_jobs_collection),
):
    """Download the final captioned video once the job is complete."""
    job = await jobs_collection.find_one({"_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job.get('status')}",
        )

    output_path = job.get("output_video_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output video file not found on disk.")

    _MEDIA_TYPES = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    suffix = Path(output_path).suffix.lower()
    media_type = _MEDIA_TYPES.get(suffix, "application/octet-stream")

    return FileResponse(
        output_path,
        media_type=media_type,
        filename=f"captioned_{Path(output_path).name}",
    )