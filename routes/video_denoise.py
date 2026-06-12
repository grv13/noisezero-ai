import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ai_clients.bnr.scripts.bnr import run_client as run_bnr_processing
from limiter import limiter
from config import FFMPEG_PATH, TEMP_DIR, _cleanup_file
from settings import settings

router = APIRouter()


@router.post("/denoise-video",
    responses={
        200: {
            "content": {"video/mp4": {}},
            "description": "Returns the video file with denoised audio.",
        },
        400: {"description": "Invalid input file or parameters."},
        500: {"description": "Internal server error during processing."},
    },
    summary="Remove background noise from a video file"
)
@limiter.limit("50/minute")
async def denoise_video(
    request: Request,
    file: UploadFile = File(..., description="Video file (e.g., MP4, MOV, etc.)."),
    intensity_ratio: float = Query(1.0, ge=0.0, le=1.0, description="Denoising intensity (0.0 to 1.0). Default is 1.0 (max)."),
):
    if not FFMPEG_PATH:
        raise HTTPException(status_code=500, detail="FFmpeg is not installed, which is required for video processing.")

    request_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()

    video_input_path = TEMP_DIR / f"{request_id}_video_input{file_extension}"
    audio_extracted_path = TEMP_DIR / f"{request_id}_extracted_audio.wav"
    audio_denoised_path = TEMP_DIR / f"{request_id}_denoised_audio.wav"
    video_output_path = TEMP_DIR / f"{request_id}_video_output{file_extension}"

    cleanup_paths = [video_input_path, audio_extracted_path, audio_denoised_path, video_output_path]

    try:
        with video_input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        target_sample_rate = 48000
        try:
            subprocess.run(
                [
                    FFMPEG_PATH, "-i", str(video_input_path), "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(target_sample_rate), "-ac", "1", str(audio_extracted_path)
                ],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract audio from video: {e.stderr}")

        if not audio_extracted_path.exists() or audio_extracted_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Audio extraction failed to produce a valid file.")

        try:
            run_bnr_processing(
                preview_mode=True,
                ssl_mode="TLS",
                target=settings.NVIDIA_TARGET_URL,
                function_id=settings.NVIDIA_FUNCTION_ID,
                api_key=settings.NVIDIA_API_KEY,
                input=str(audio_extracted_path),
                output=str(audio_denoised_path),
                sample_rate=target_sample_rate, streaming=False, intensity_ratio=intensity_ratio,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process audio with BNR model: {e}")

        if not audio_denoised_path.exists() or audio_denoised_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="BNR processing failed to produce a denoised audio file.")

        try:
            subprocess.run(
                [
                    FFMPEG_PATH, "-i", str(video_input_path), "-i", str(audio_denoised_path),
                    "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", str(video_output_path)
                ],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to merge denoised audio back into video: {e.stderr}")

        if not video_output_path.exists() or video_output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Final video creation failed.")

        return FileResponse(
            path=video_output_path, media_type=f"video/{file_extension.strip('.')}",
            filename=f"denoised_{file.filename}",
            # This background task will clean up ALL temp files after the response is sent.
            background=BackgroundTask(lambda: [_cleanup_file(p) for p in cleanup_paths])
        )
    finally:
        # This is a fallback cleanup for cases where an exception is raised before
        # the FileResponse is returned. We check if an exception occurred.
        # If no exception, the background task will handle cleanup.
        if sys.exc_info()[0]: # An exception is being handled
            for path in cleanup_paths:
                _cleanup_file(path)