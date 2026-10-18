"""CircuitScope backend entry point. Binds to localhost by default."""

from __future__ import annotations

import argparse

import uvicorn

from .api.routes import router
from .storage import db


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="CircuitScope", version="0.1.0")
    # Local-first product: the dev frontend runs on a different localhost port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        db.init()

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
