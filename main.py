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
from app.routes import chat, health, prospect
from app.agents.graph import faq_graph_builder

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Varsapradaya FAQ Agent starting up...")

    # 1. Start continuous idle session background monitoring loop
    from app.utils.scheduler import monitor_inactive_sessions_loop
    monitor_task = asyncio.create_task(monitor_inactive_sessions_loop())

    # 2. Seed FAQ embeddings into pgvector on first run
    try:
        from app.rag.seed_embeddings import seed
        await seed()
        logger.info("FAQ embeddings seeded successfully.")
    except Exception as e:
        logger.error(f"Failed to seed FAQ embeddings: {e}")

    # 3. Compile LangGraph with PostgreSQL checkpointer
    # AsyncPostgresSaver persists full conversation state across server restarts.
    # It auto-creates the 'checkpoints' and 'checkpoint_blobs' tables in PostgreSQL.
    try:
        from psycopg_pool import AsyncConnectionPool
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # Strip the SQLAlchemy driver prefix — psycopg3 uses plain postgresql:// or postgresql+psycopg://
        checkpoint_url = settings.CHECKPOINT_DB_URL

        # Create a robust connection pool for the checkpointer to handle timeouts & self-healing
        pool = AsyncConnectionPool(
            conninfo=checkpoint_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row}
        )
        await pool.open()

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # idempotent: safe to run every startup
        app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
        logger.info("LangGraph compiled with PostgreSQL checkpointer using AsyncConnectionPool.")
        yield
        
        # Shutdown: Close the connection pool gracefully
        await pool.close()

    except Exception as e:
        logger.error(f"PostgreSQL checkpointer failed: {e}. Falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        app.state.faq_graph = faq_graph_builder.compile(checkpointer=checkpointer)
        logger.warning("LangGraph compiled with MemorySaver (conversations will not persist across restarts).")
        yield

    # Shutdown cleanup
    logger.info("Varsapradaya FAQ Agent shutting down...")
    
    # Cancel the background monitoring daemon cleanly
    logger.info("Stopping idle session monitor daemon...")
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

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
app.include_router(prospect.router, prefix="/api", tags=["Prospect"])
# Also include prospect router without prefix to support direct access at root level (e.g. /fetch_prospect/...)
app.include_router(prospect.router, tags=["Prospect"])