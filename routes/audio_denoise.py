import shutil
import uuid
from pathlib import Path

import librosa
import soundfile as sf
import sys
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydub import AudioSegment
from starlette.background import BackgroundTask

from ai_clients.bnr.scripts.bnr import run_client as run_bnr_processing
from limiter import limiter
from config import TEMP_DIR, _cleanup_file
from settings import settings

router = APIRouter()


@router.post("/denoise-audio",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Returns the denoised audio file in WAV format.",
        },
        400: {"description": "Invalid input file or parameters."},
        500: {"description": "Internal server error during processing."},
    },
    summary="Remove background noise from an audio file"
)
@limiter.limit("50/minute")
async def denoise_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.). Will be converted to WAV for processing."),
    intensity_ratio: float = Query(1.0, ge=0.0, le=1.0, description="Denoising intensity (0.0 to 1.0). Default is 1.0 (max).")
):
    """
    Upload an audio file to remove background noise using NVIDIA's BNR model.

    - **file**: The input audio file. Non-WAV formats will be converted automatically.
    - **intensity_ratio**: A float between 0.0 and 1.0 to control the denoising strength.
    """
    request_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix.lower()
    
    original_temp_path = TEMP_DIR / f"{request_id}_original{file_extension}"
    input_path = TEMP_DIR / f"{request_id}_input.wav" 
    output_path = TEMP_DIR / f"{request_id}_output.wav"

    # List of files to clean up
    cleanup_paths = [original_temp_path, input_path, output_path]

    try:
        with original_temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file_extension != ".wav":
            try:
                audio = AudioSegment.from_file(original_temp_path)
                audio.export(input_path, format="wav")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read or convert audio file: {e}. Ensure it is a valid audio format.")
        else:
            shutil.move(original_temp_path, input_path)

        if not input_path.exists() or input_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Audio conversion to WAV failed.")

        try:
            info = sf.info(input_path)
            input_sample_rate = info.samplerate
            target_sample_rate = input_sample_rate

            if input_sample_rate not in [16000, 48000]:
                print(f"Unsupported sample rate: {input_sample_rate}. Resampling to 48000 Hz.")
                target_sample_rate = 48000
                y, sr = librosa.load(input_path, sr=None)
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
                sf.write(input_path, y_resampled, target_sample_rate)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not read or convert audio file: {e}")

        try:
            run_bnr_processing(
                preview_mode=True,
                ssl_mode="TLS",
                target=settings.NVIDIA_TARGET_URL,
                function_id=settings.NVIDIA_FUNCTION_ID,
                api_key=settings.NVIDIA_API_KEY,
                input=str(input_path),
                output=str(output_path),
                sample_rate=target_sample_rate, streaming=False, intensity_ratio=intensity_ratio,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process audio: {e}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processing failed to produce an output file.")

        return FileResponse(
            path=output_path, media_type="audio/wav", filename="denoised_output.wav",
            background=BackgroundTask(lambda: [_cleanup_file(p) for p in cleanup_paths])
        )
    finally:
        # This is a fallback cleanup for cases where an exception is raised before
        # the FileResponse is returned. We check if an exception occurred.
        # If no exception, the background task will handle cleanup.
        if sys.exc_info()[0]: # An exception is being handled
            for path in cleanup_paths:
                _cleanup_file(path)