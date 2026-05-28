from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db, create_session, end_session, get_last_session_category, write_log
from app.models import SessionRequest, EndSessionRequest, CategorySwitchRequest
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

@router.post("/sessions")
async def handle_create_session(req: SessionRequest, db: AsyncSession = Depends(get_db)):
    try:
        last_cat = await get_last_session_category(db, req.user_id)
        is_returning = last_cat is not None
        
        session = await create_session(db, req.user_id, req.category, is_returning)
        session_dict = dict(session._mapping)
        session_dict["id"] = str(session_dict["id"])
        session_dict["user_id"] = str(session_dict["user_id"])
        
        await write_log(
            db,
            level="INFO",
            event="session_started",
            message=f"Session started for category {req.category}",
            user_id=req.user_id,
            session_id=session.id,
            meta={"category": req.category}
        )
        logger.info(
            f"Session started: {session_dict['id']}",
            extra={"event": "session_started", "user_id": req.user_id, "session_id": session_dict["id"], "meta": {"category": req.category}}
        )
        
        return session_dict
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start session."
        )

@router.patch("/sessions/end")
async def handle_end_session(req: EndSessionRequest, db: AsyncSession = Depends(get_db)):
    try:
        await end_session(db, req.session_id)
        
        await write_log(
            db,
            level="INFO",
            event="session_ended",
            message="Session ended",
            session_id=req.session_id
        )
        logger.info(
            f"Session ended: {req.session_id}",
            extra={"event": "session_ended", "session_id": req.session_id}
        )
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error ending session {req.session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session."
        )

@router.post("/sessions/switch-category")
async def handle_switch_category(req: CategorySwitchRequest, db: AsyncSession = Depends(get_db)):
    try:
        await end_session(db, req.session_id)
        
        new_session = await create_session(db, req.user_id, req.new_category, is_returning=True)
        new_session_dict = dict(new_session._mapping)
        new_session_dict["id"] = str(new_session_dict["id"])
        new_session_dict["user_id"] = str(new_session_dict["user_id"])
        
        await write_log(
            db,
            level="INFO",
            event="category_switched",
            message=f"Category switched to {req.new_category}",
            user_id=req.user_id,
            session_id=new_session.id,
            meta={"old_session_id": req.session_id, "new_category": req.new_category}
        )
        logger.info(
            f"Category switched: {req.session_id} -> {new_session_dict['id']} ({req.new_category})",
            extra={
                "event": "category_switched",
                "user_id": req.user_id,
                "session_id": new_session_dict["id"],
                "meta": {"old_session_id": req.session_id, "new_category": req.new_category}
            }
        )
        
        return new_session_dict
    except Exception as e:
        logger.error(f"Error switching category: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to switch category."
        )
