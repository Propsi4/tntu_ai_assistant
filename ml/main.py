"""Main entry point for the FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml.config.settings import settings
from ml.config.logger import configure_root_logger
from ml.routes.agent_routes import router as agent_router
from ml.routes.chat_history_routes import router as chat_history_router
from ml.routes.rag_routes import router as rag_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured FastAPI application instance.
    """
    # Configure logging
    configure_root_logger(debug=settings.DEBUG)

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="API for Drug Assistant AI Agent",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(agent_router)
    app.include_router(chat_history_router)
    app.include_router(rag_router)

    @app.get("/health")
    async def health_check():
        """Global health check endpoint."""
        return {"status": "healthy", "app_name": settings.APP_NAME, "version": settings.APP_VERSION}

    return app


app = create_app()


def main():
    """Entry point for poetry script."""
    import uvicorn

    uvicorn.run("ml.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)


if __name__ == "__main__":
    main()
