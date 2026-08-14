"""FastAPI application for HandSign Conversion.

Run it with::

    uvicorn backend.app.main:app --reload

The heavy objects -- the MediaPipe landmarker and the trained random forest --
are created once in :func:`lifespan` and shared by every request and socket.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.deps import AppServices
from backend.app.api.routes import health, recognition, session, signs, suggestions
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.services.recognition_service import RecognitionService
from backend.app.services.session import SessionStore
from backend.app.services.sign_translation import SignTranslator
from backend.app.services.suggestion_service import SuggestionService
from ml.paths import PROJECT_ROOT, SIGN_ASSETS_DIR

logger = get_logger(__name__)

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("Loading recognition model ...")
    recognition_service = RecognitionService(settings)
    logger.info(
        "Model ready: %d classes (%s)",
        len(recognition_service.classes),
        "".join(recognition_service.classes),
    )

    suggestion_service = SuggestionService(settings)
    if settings.gemini_enabled:
        logger.info("Gemini configured (model=%s)", settings.gemini_model)
    else:
        logger.warning(
            "GEMINI_API_KEY is not set - running with local suggestions only."
        )

    app.state.services = AppServices(
        settings=settings,
        recognition=recognition_service,
        sessions=SessionStore(settings),
        suggestions=suggestion_service,
        signs=SignTranslator(recognition_service.classes),
    )
    try:
        yield
    finally:
        logger.info("Shutting down ...")
        await app.state.services.aclose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Real-time sign-language recognition with contextual text "
            "prediction, and the reverse: English text rendered as hand signs."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(recognition.router, prefix="/api")
    app.include_router(suggestions.router, prefix="/api")
    app.include_router(session.router, prefix="/api")
    app.include_router(signs.router, prefix="/api")

    # The live channel lives at /ws/recognition, outside the /api prefix.
    app.include_router(recognition.ws_router)

    # Sign illustrations are served by the backend in both dev and production,
    # so the letter -> asset mapping returned by /api/signs is always valid.
    if SIGN_ASSETS_DIR.exists():
        app.mount(
            "/signs",
            StaticFiles(directory=str(SIGN_ASSETS_DIR)),
            name="signs",
        )
    else:
        logger.warning(
            "No sign assets at %s - run `python -m ml.scripts.generate_sign_assets`",
            SIGN_ASSETS_DIR,
        )

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built UI, when there is one.

    During development the Vite dev server handles this; after
    ``npm run build`` the same uvicorn process serves the whole application.
    """
    if not FRONTEND_DIST.exists():
        @app.get("/", include_in_schema=False)
        async def root() -> JSONResponse:
            return JSONResponse(
                {
                    "app": "HandSign Conversion API",
                    "docs": "/docs",
                    "health": "/api/health",
                    "note": (
                        "The frontend is not built. Run `npm run dev` in "
                        "frontend/, or `npm run build` to serve it from here."
                    ),
                }
            )
        return

    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


app = create_app()
