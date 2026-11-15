"""CircuitScope backend entry point. Binds to localhost by default."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import uvicorn

from .api.routes import router
from .storage import db


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    @asynccontextmanager
    async def lifespan(_app):
        # Respect a path chosen earlier (--db flag, tests); default otherwise.
        if not db.is_initialized():
            db.init()
        yield

    app = FastAPI(title="CircuitScope", version="0.1.0", lifespan=lifespan)
    # Local-first product: the dev frontend runs on a different localhost port.
    # Vite moves to 5174+ when 5173 is busy, so accept any local port while
    # still rejecting every non-local origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--db", default=None, help="SQLite path override")
    args = parser.parse_args()
    db.init(args.db)
    uvicorn.run(app, host=args.host, port=args.port)
