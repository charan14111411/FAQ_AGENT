from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db, get_onboarding_state
from app.models import ChatRequest, ChatResponse
from app.checkpointer import write_checkpoint
from app.graph.chat_graph import chat_graph
from app.graph.onboarding_graph import onboarding_graph
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    turn_id = str(uuid4())
    try:
        conversation_id = req.conversation_id or str(uuid4())
        session_id = req.session_id
        user_id = req.user_id
        category = (req.category or "").strip().lower() or None

        if not session_id or not user_id or not category:
            state = await get_onboarding_state(db, conversation_id)
            if state:
                session_id = session_id or state.get("session_id")
                user_id = user_id or state.get("user_id")

            if not session_id or not user_id or not category:
                onboarding_result = await onboarding_graph.ainvoke(
                    {
                        "db": db,
                        "conversation_id": conversation_id,
                        "message": req.message,
                    }
                )
                if not onboarding_result.get("onboarding_complete", False):
                    return ChatResponse(
                        reply=onboarding_result.get("reply", ""),
                        conversation_id=conversation_id,
                        step=onboarding_result.get("step", "name"),
                        onboarding_complete=False,
                        session_id=onboarding_result.get("session_id"),
                        user_id=onboarding_result.get("user_id"),
                        category=onboarding_result.get("category"),
                        agent="onboarding_agent",
                        switch_requested=False,
                    )
                session_id = onboarding_result.get("session_id")
                user_id = onboarding_result.get("user_id")
                category = onboarding_result.get("category")

        result = await chat_graph.ainvoke(
            {
                "db": db,
                "turn_id": turn_id,
                "session_id": session_id,
                "user_id": user_id,
                "category": category,
                "message": req.message,
            }
        )
        
        return ChatResponse(
            reply=result.get("reply", ""),
            conversation_id=conversation_id,
            step="done",
            onboarding_complete=True,
            session_id=session_id,
            user_id=user_id,
            category=result.get("resolved_category", category),
            agent=result.get("agent", "none"),
            switch_requested=result.get("switch_requested", False)
        )
        
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        error_text = str(e).lower()
        if "rate limit" in error_text or "rate_limit_exceeded" in error_text:
            return ChatResponse(
                reply="I am temporarily at capacity due to provider limits. Please retry in a few minutes.",
                conversation_id=req.conversation_id,
                step="done",
                onboarding_complete=bool(req.session_id and req.user_id),
                session_id=req.session_id,
                user_id=req.user_id,
                category=req.category,
                agent="system_fallback",
                switch_requested=False,
            )
        await write_checkpoint(
            db,
            turn_id=turn_id,
            checkpoint_type="response_failed",
            status="error",
            user_id=req.user_id,
            session_id=req.session_id,
            category=req.category,
            metadata={"error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message."
        )
