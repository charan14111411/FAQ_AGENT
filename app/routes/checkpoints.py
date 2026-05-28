from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db, get_checkpoints_by_session
from app.models import CheckpointListResponse
from app.logger import get_logger

logger = get_logger()
router = APIRouter()


@router.get("/checkpoints/{session_id}", response_model=CheckpointListResponse)
async def get_checkpoints(session_id: str, limit: int = Query(100, ge=1, le=500), db: AsyncSession = Depends(get_db)):
    try:
        items = await get_checkpoints_by_session(db, session_id=session_id, limit=limit)
        return CheckpointListResponse(session_id=session_id, count=len(items), items=items)
    except Exception as e:
        logger.error(f"Error fetching checkpoints: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching checkpoints.",
        )
