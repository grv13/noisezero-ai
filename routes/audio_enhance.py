import shutil
import sys
import traceback
import uuid
import logging
from pathlib import Path

import librosa
import soundfile as sf
from fastapi import (APIRouter, File, HTTPException, Query, Request,
                     UploadFile)
from fastapi.responses import FileResponse
from pydub import AudioSegment
from starlette.background import BackgroundTask

from config import TEMP_DIR, _cleanup_file
from limiter import limiter
from settings import settings

# The project root is added to sys.path in main.py, so we can use absolute imports.
from ai_clients.studio_voice.scripts.studio_voice import run_client as run_studio_voice_processing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def _prepare_audio_file(file: UploadFile, request_id: str, target_sample_rate: int) -> Path:
    """
    Saves, converts, resamples, and converts the uploaded audio file to the target format.
    Ensures output is 16-bit mono WAV at the target sample rate (required by Studio Voice API).
    Returns the path to the prepared WAV file.
    Raises HTTPException on failure.
    """
    file_extension = Path(file.filename).suffix.lower()
    original_temp_path = TEMP_DIR / f"{request_id}_original{file_extension}"
    input_path = TEMP_DIR / f"{request_id}_input.wav"

    try:
        # Save the original uploaded file
        with original_temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use pydub to load and convert the audio file
        # This ensures proper format handling across different input types
        audio = AudioSegment.from_file(original_temp_path)
        
        # pydub can handle the conversion, resampling, and bit depth in one go.
        # This is more robust than manual processing with numpy/soundfile for this task.
        # Studio Voice requires: mono, 16-bit PCM.
        print(f"Input sample rate is {audio.frame_rate}Hz, but model requires {target_sample_rate}Hz. Resampling...")
        audio = (
            audio.set_frame_rate(target_sample_rate)
            .set_channels(1)
            .set_sample_width(2) # 2 bytes = 16 bits
        )

        # Export as a WAV file. pydub handles the WAV header correctly.
        audio.export(input_path, format="wav")
        
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise ValueError("Audio conversion to WAV failed, resulting in an empty file.")
        
        # Validate the output file by loading and checking it's not all zeros
        y, sr = librosa.load(input_path, sr=None, mono=True)
        if (y == 0).all():
            raise ValueError("Audio conversion resulted in silent/zero audio data.")
        
        print(f"Successfully prepared audio: {target_sample_rate}Hz, mono, 16-bit, duration: {len(y)/target_sample_rate:.2f}s")
        
        return input_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to prepare audio file: {e}")

@router.post("/enhance-audio",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Returns the enhanced audio file in WAV format.",
        },
        400: {"description": "Invalid input file or parameters."},
        500: {"description": "Internal server error during processing."},
        503: {"description": "External AI service unavailable. Retrying..."},
    },
    summary="Enhance speech in an audio file using Studio Voice"
)
@limiter.limit("50/minute")
async def enhance_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.). Will be converted to WAV for processing."),
    model_type: str = Query("48k-hq", enum=["48k-hq", "48k-ll", "16k-hq"], description="Studio Voice model type to use.")
):
    """
    Upload an audio file to enhance the speech quality using NVIDIA's Studio Voice model.

    - **file**: The input audio file. Non-WAV formats will be converted automatically.
    - **model_type**: The Studio Voice model to use. This determines the required sample rate (16kHz or 48kHz). Default is '48k-hq'.
    """
    request_id = str(uuid.uuid4())
    output_path = TEMP_DIR / f"{request_id}_output.wav"
    # We will add input_path to cleanup_paths after it's created
    cleanup_paths = [output_path]

    try:
        target_sample_rate = 16000 if model_type == "16k-hq" else 48000
        input_path = _prepare_audio_file(file, request_id, target_sample_rate)
        
        # Add all generated temp files for this request to the cleanup list
        # This includes the original upload, and the converted/resampled input
        for p in TEMP_DIR.glob(f"{request_id}_*"):
            cleanup_paths.append(p)

        logger.info(f"Request {request_id}: Processing audio with model {model_type}")
        
        try:
            run_studio_voice_processing(
                preview_mode=True,
                ssl_mode="TLS",
                target=settings.NVIDIA_TARGET_URL,
                function_id=settings.NVIDIA_FUNCTION_ID_STUDIO_VOICE,
                api_key=settings.NVIDIA_API_KEY_STUDIO_VOICE,
                input=str(input_path),
                output=str(output_path),
                # For HQ models, transactional mode (streaming=False) is more robust for file-based processing.
                # Streaming is better suited for the low-latency (ll) model.
                streaming=(model_type == "48k-ll"),
                model_type=model_type,
            ) # sample_rate is derived from model_type inside the client
            
            logger.info(f"Request {request_id}: Audio processing completed successfully")
            
        except Exception as e:
            # Log the full error for debugging
            error_msg = f"Studio Voice processing failed: {e}\n{traceback.format_exc()}"
            logger.error(f"Request {request_id}: {error_msg}")
            
            # Provide specific error messages based on exception type
            if "Server Error" in str(e) or "INTERNAL" in str(e):
                raise HTTPException(
                    status_code=503, 
                    detail="External AI service encountered an error. The system is configured with automatic retries. Please try again."
                )
            elif "UNAVAILABLE" in str(e) or "Connection" in str(e):
                raise HTTPException(
                    status_code=503,
                    detail="External AI service is temporarily unavailable. Please try again in a few moments."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to process audio with external AI service. Error: {str(e)}"
                )

        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.error(f"Request {request_id}: Output file not generated or is empty")
            raise HTTPException(status_code=500, detail="Processing failed to produce an output file.")

        logger.info(f"Request {request_id}: Returning enhanced audio file")
        
        return FileResponse(
            path=output_path, media_type="audio/wav", filename="enhanced_output.wav",
            background=BackgroundTask(lambda: [_cleanup_file(p) for p in cleanup_paths])
        )
    finally:
        if sys.exc_info()[0]:
            logger.info(f"Request {request_id}: Cleaning up temporary files due to error")
            for path in cleanup_paths:
                _cleanup_file(path)