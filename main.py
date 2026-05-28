import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Fix for psycopg on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.logger import get_logger
from app.db import engine
from app.routes import chat, health
from app.agents.graph import faq_graph_builder

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up — LangGraph FAQ Agent v2.0")
    
    # 1. Seed embeddings
    try:
        from app.rag.seed_embeddings import seed
        await seed()
    except Exception as e:
        logger.error(f"Failed to seed FAQ embeddings during startup: {e}")

    # Use MemorySaver for Windows dev environment to avoid psycopg3 async deadlocks
    checkpointer = MemorySaver()
    app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled with MemorySaver")
    
    yield  # Application is running

    # Shutdown
    logger.info("Application shutting down...")
    try:
        await engine.dispose()
        logger.info("Database engine disposed.")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")

app = FastAPI(
    title="Varsapradaya FAQ Chatbot — LangGraph Edition",
    description="Conversational FAQ agent powered by LangGraph. Single endpoint handles everything.",
    version="2.0.0",
    lifespan=lifespan,
)

origins = [settings.FRONTEND_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only 2 routes now — the agent handles everything else conversationally
app.include_router(chat.router,   prefix="/api", tags=["Chat"])
app.include_router(health.router, prefix="/api", tags=["Health"])