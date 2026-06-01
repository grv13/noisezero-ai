# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - ffmpeg is required by pydub for audio format conversion.
# - git is included in case of any dependencies that might need it.
# - build-essential is for any packages that need to be compiled from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# First, copy only the requirements file to leverage Docker's cache.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use uvicorn to run the FastAPI app, binding to all interfaces on port 8000.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]