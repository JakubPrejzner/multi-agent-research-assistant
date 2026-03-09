"""CORS, rate limiting, and error handling middleware."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import get_settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Register global error handlers."""

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "type": "about:blank",
                "title": "Rate Limit Exceeded",
                "status": 429,
                "detail": f"Rate limit exceeded: {exc.detail}",
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "type": "about:blank",
                "title": "Bad Request",
                "status": 400,
                "detail": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "type": "about:blank",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred.",
            },
        )
