from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logger import get_logger
from app.db import engine
from app.routes import chat, health, checkpoints

logger = get_logger()

app = FastAPI(title="Varsapradaya FAQ Chatbot Backend")

origins = [settings.FRONTEND_URL]
if settings.FRONTEND_URL == "http://localhost:5500":
    origins.append("http://127.0.0.1:5500")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(checkpoints.router, prefix="/api", tags=["Checkpoints"])
app.include_router(health.router, prefix="/api", tags=["Health"])

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    try:
        from app.rag.seed_embeddings import seed
        await seed()
    except Exception as e:
        logger.error(f"Failed to seed FAQ embeddings during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    try:
        await engine.dispose()
        logger.info("Database engine disposed.")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")
