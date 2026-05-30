import os
import sys
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
import soundfile as sf
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from nim_clients.bnr.scripts.bnr import run_client as run_bnr_processing

# Add the project root to the Python path to allow for absolute imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# -- Project Setup --

# -- Configuration --
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_FUNCTION_ID = os.getenv("NVIDIA_FUNCTION_ID")
NVIDIA_TARGET_URL = os.getenv("NVIDIA_TARGET_URL")

if not all([NVIDIA_API_KEY, NVIDIA_FUNCTION_ID, NVIDIA_TARGET_URL]):
    raise EnvironmentError("Missing required NVIDIA environment variables in your .env file.")

# Create a temporary directory for file processing
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="NVIDIA BNR API",
    description="A FastAPI wrapper for the NVIDIA Background Noise Removal NIM.",
)

def cleanup_files(paths: list[Path]):
    """Background task to remove temporary files."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            print(f"Error cleaning up file {path}: {e}")


@app.post("/denoise",
    response_class=FileResponse,
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
async def denoise_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="16-bit mono channel WAV audio file (< 35MB)."),
    intensity_ratio: float = Query(1.0, ge=0.0, le=1.0, description="Denoising intensity (0.0 to 1.0). Default is 1.0 (max).")
):
    """
    Upload a WAV audio file to remove background noise using NVIDIA's BNR model.

    - **file**: The input audio file. Must be a 16-bit mono channel WAV file.
    - **intensity_ratio**: A float between 0.0 and 1.0 to control the denoising strength.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")

    # Generate unique file paths to avoid conflicts
    request_id = str(uuid.uuid4())
    input_path = TEMP_DIR / f"{request_id}_input.wav"
    output_path = TEMP_DIR / f"{request_id}_output.wav"

    # Save the uploaded file temporarily
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # Add files to be cleaned up after the request is complete
    background_tasks.add_task(cleanup_files, [input_path, output_path])

    # Check the sample rate of the uploaded audio file
    try:
        info = sf.info(input_path)
        input_sample_rate = info.samplerate
        if input_sample_rate not in [16000, 48000]:
            raise HTTPException(status_code=400, detail=f"Unsupported sample rate: {input_sample_rate}. Please use 16000 or 48000 Hz.")
    except HTTPException:
        # Re-raise the specific HTTP exception for unsupported sample rates
        raise
    except Exception as e:
        print(f"Could not read audio info: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read audio file properties: {e}")

    try:
        print(f"Processing {input_path} -> {output_path}")
        # Call the NVIDIA BNR processing function
        run_bnr_processing(
            preview_mode=True,
            ssl_mode="TLS",
            target=NVIDIA_TARGET_URL,
            function_id=NVIDIA_FUNCTION_ID,
            api_key=NVIDIA_API_KEY,
            input=str(input_path),
            output=str(output_path),
            sample_rate=input_sample_rate,
            streaming=False,
            intensity_ratio=intensity_ratio,
        )
    except Exception as e:
        # Catch potential errors from the bnr script
        print(f"Error during BNR processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {e}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Processing failed to produce an output file.")

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="denoised_output.wav"
    )

@app.get("/", include_in_schema=False)
def root():
    return {"message": "NVIDIA BNR API is running. See /docs for API documentation."}
