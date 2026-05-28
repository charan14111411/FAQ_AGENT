import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Windows asyncio fix for psycopg compatibility
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
    logger.info("Varsapradaya v3.0 starting — Master/Slave Agent Architecture")

    # 1. Seed FAQ embeddings into pgvector (idempotent)
    try:
        from app.rag.seed_embeddings import seed
        await seed()
    except Exception as e:
        logger.error(f"Embedding seed failed: {e}")

    # 2. Compile LangGraph with in-memory checkpointer (dev)
    #    For production: swap MemorySaver for a PostgreSQL-backed checkpointer
    checkpointer = MemorySaver()
    app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph v3 compiled with MemorySaver checkpointer")

    yield  # Application is running

    # Shutdown
    logger.info("Varsapradaya v3.0 shutting down...")
    try:
        await engine.dispose()
        logger.info("Database engine disposed.")
    except Exception as e:
        logger.error(f"Engine disposal error: {e}")


app = FastAPI(
    title="Varsapradaya FarmFuture — Master/Slave Agent v3",
    description=(
        "Master orchestrator routes free-text messages to specialist slave agents. "
        "Button clicks bypass master for instant slave access. "
        "4 agents: grower | investor | corporate | general."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,   prefix="/api", tags=["Chat"])
app.include_router(health.router, prefix="/api", tags=["Health"])
