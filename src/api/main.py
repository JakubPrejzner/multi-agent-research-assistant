"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from src.api.middleware import limiter, setup_cors, setup_error_handlers
from src.api.routes import health, research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Multi-Agent Research Assistant",
        description="Orchestrates specialized AI agents to research topics and produce structured reports.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.limiter = limiter
    setup_cors(app)
    setup_error_handlers(app)

    app.include_router(health.router, tags=["health"])
    app.include_router(research.router, tags=["research"])

    return app


app = create_app()
