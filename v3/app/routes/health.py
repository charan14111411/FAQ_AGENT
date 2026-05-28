from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db import get_db
from app.logger import get_logger

logger = get_logger()
router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "db_connected": True, "version": "3.0.0"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "degraded", "db_connected": False, "version": "3.0.0"}
