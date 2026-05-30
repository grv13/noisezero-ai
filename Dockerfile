# =========================================================================
# Stage 1: Build Stage
# =========================================================================
# Use a Python base image. The 'slim' variant is a good balance of size and functionality.
FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# =========================================================================
# Stage 2: Production Stage
# =========================================================================
# Use a minimal base image for a smaller and more secure final image.
FROM python:3.11-slim

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

WORKDIR /home/appuser/app

# Copy the virtual environment and source code from the builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app .

# Make the virtual environment's Python the default
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on (uvicorn's default is 8000)
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
# The host 0.0.0.0 is necessary to make it accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]