from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address, default_limits=["50/minute"])


def rate_limit_exceeded_handler(request: Request, exc: Exception):
    """Custom exception handler for rate limit exceeded errors."""
    return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})