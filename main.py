import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Required for psycopg/asyncpg compatibility on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import settings
from app.logger import get_logger
from app.db import engine
from app.routes import chat, health
from app.agents.graph import faq_graph_builder

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Varsapradaya FAQ Agent starting up...")

    # 1. Seed FAQ embeddings into pgvector on first run
    try:
        from app.rag.seed_embeddings import seed
        await seed()
        logger.info("FAQ embeddings seeded successfully.")
    except Exception as e:
        logger.error(f"Failed to seed FAQ embeddings: {e}")

    # 2. Compile LangGraph with PostgreSQL checkpointer
    # AsyncPostgresSaver persists full conversation state across server restarts.
    # It auto-creates the 'checkpoints' and 'checkpoint_blobs' tables in PostgreSQL.
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # Strip the SQLAlchemy driver prefix — psycopg3 uses plain postgresql:// or postgresql+psycopg://
        checkpoint_url = settings.CHECKPOINT_DB_URL

        async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as checkpointer:
            await checkpointer.setup()  # idempotent: safe to run every startup
            app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
            logger.info("LangGraph compiled with PostgreSQL checkpointer.")
            yield

    except Exception as e:
        logger.error(f"PostgreSQL checkpointer failed: {e}. Falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
        logger.warning("LangGraph compiled with MemorySaver (conversations will not persist across restarts).")
        yield

    # Shutdown cleanup
    logger.info("Varsapradaya FAQ Agent shutting down...")
    try:
        await engine.dispose()
        logger.info("Database engine disposed.")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")


app = FastAPI(
    title="Varsapradaya FAQ Agent",
    description=(
        "Production-ready multi-agent FAQ chatbot. "
        "4 agents (Grower, Investor, Corporate/Partnership, Just Exploring) — "
        "all powered by LangGraph with PostgreSQL state persistence."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (safe for local dev; restrict in production)
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,   prefix="/api", tags=["Chat"])
app.include_router(health.router, prefix="/api", tags=["Health"])