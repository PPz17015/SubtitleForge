import argparse
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SubtitleForge Pro API",
    description="Professional subtitle generation and translation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": "SubtitleForge Pro API",
        "version": "1.0.0",
        "docs": "/docs"
    }


def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    logger.info(f"Starting SubtitleForge Pro API server on {host}:{port}")
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SubtitleForge Pro API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    run_server(args.host, args.port, args.reload)
