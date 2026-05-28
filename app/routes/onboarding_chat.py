from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.graph.onboarding_graph import onboarding_graph
from app.models import OnboardingChatRequest, OnboardingChatResponse
from app.logger import get_logger

logger = get_logger()
router = APIRouter()


@router.post("/chat/onboarding", response_model=OnboardingChatResponse)
async def onboarding_chat(req: OnboardingChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        conversation_id = req.conversation_id or str(uuid4())
        result = await onboarding_graph.ainvoke(
            {
                "db": db,
                "conversation_id": conversation_id,
                "message": req.message,
            }
        )
        return OnboardingChatResponse(
            conversation_id=conversation_id,
            reply=result.get("reply", ""),
            step=result.get("step", "name"),
            onboarding_complete=result.get("onboarding_complete", False),
            user_id=result.get("user_id"),
            session_id=result.get("session_id"),
            category=result.get("category"),
        )
    except Exception as e:
        logger.error(f"Error in onboarding chat route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing onboarding chat.",
        )
