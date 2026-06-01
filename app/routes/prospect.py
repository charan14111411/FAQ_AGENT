"""
GET /api/prospect/{prospect_id}/conversations

Returns the full conversation history for a given CRM prospect ID.
Groups messages by session and shows them in order.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict
from app.db import AsyncSessionLocal
from sqlalchemy import text
from app.logger import get_logger

logger = get_logger()
router = APIRouter()


@router.get(
    "/prospect/{prospect_id}/conversations",
    response_model=Dict[str, Any],
    summary="Get full conversation history for a prospect",
    description=(
        "Given a CRM prospect_id, returns the user's profile and all their "
        "chat sessions with every message (user + AI) in chronological order."
    ),
)
async def get_prospect_conversations(prospect_id: str):
    """
    Fetch all sessions and messages for a given CRM prospect_id.
    """
    async with AsyncSessionLocal() as db:

        # 1. Fetch all sessions for this prospect_id (oldest first)
        sessions_rows = await db.execute(
            text("""
                SELECT s.id, s.category, s.is_returning, s.started_at, s.ended_at,
                       u.name, u.phone, u.email
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.prospect_id = :pid
                ORDER BY s.started_at ASC
            """),
            {"pid": prospect_id},
        )
        all_sessions = sessions_rows.fetchall()

        if not all_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No sessions found for prospect_id '{prospect_id}'.",
            )

        # Get user details from the first session row
        first_row = all_sessions[0]
        user_name = first_row[5]
        user_phone = first_row[6]
        user_email = first_row[7]

        # Build response matching the requested structure
        response_data: Dict[str, Any] = {
            "prospect id": prospect_id,
            "total_sessions": len(all_sessions),
        }

        # 2. For each session, fetch all messages in order
        for i, sess in enumerate(all_sessions, start=1):
            session_id = str(sess[0])
            started_at = sess[3].isoformat() if sess[3] else None
            ended_at = sess[4].isoformat() if sess[4] else None

            msgs_rows = await db.execute(
                text("""
                    SELECT role, content
                    FROM messages
                    WHERE session_id = :sid
                    ORDER BY created_at ASC
                """),
                {"sid": session_id},
            )
            
            conversation_lines = []
            for r in msgs_rows.fetchall():
                # Map role to user or ai as requested
                role_str = "user" if r[0] == "user" else "ai"
                conversation_lines.append(f"{role_str}: {r[1]}")

            session_key = f"session{i}"
            response_data[session_key] = {
                "username": user_name,
                "phone number": user_phone,
                "email": user_email,
                "category": sess[1],
                "started_at": started_at,
                "ended_at": ended_at,
                "conversation": conversation_lines,
            }

        logger.info(
            f"Prospect conversation fetched: prospect_id={prospect_id}, "
            f"user={user_name}, sessions={len(all_sessions)}",
            extra={"event": "prospect_conversation_fetched"},
        )

        return response_data
