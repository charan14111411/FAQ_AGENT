import json
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.logger import get_logger

logger = get_logger()


async def write_checkpoint(
    db: AsyncSession,
    *,
    turn_id: str,
    checkpoint_type: str,
    status: str = "ok",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    category: Optional[str] = None,
    agent: Optional[str] = None,
    user_message_id: Optional[str] = None,
    assistant_message_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    payload = metadata or {}
    query = """
        INSERT INTO checkpoints (
            turn_id, checkpoint_type, status, user_id, session_id, category, agent,
            user_message_id, assistant_message_id, metadata
        )
        VALUES (
            :turn_id, :checkpoint_type, :status, :user_id, :session_id, :category, :agent,
            :user_message_id, :assistant_message_id, :metadata
        )
    """
    try:
        await db.execute(
            text(query),
            {
                "turn_id": turn_id,
                "checkpoint_type": checkpoint_type,
                "status": status,
                "user_id": user_id,
                "session_id": session_id,
                "category": category,
                "agent": agent,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "metadata": json.dumps(payload),
            },
        )
        await db.commit()
    except Exception as exc:
        logger.error(
            f"Failed to write checkpoint: {exc}",
            extra={
                "event": "checkpoint_write_failure",
                "user_id": user_id,
                "session_id": session_id,
                "meta": {"turn_id": turn_id, "checkpoint_type": checkpoint_type},
            },
        )
