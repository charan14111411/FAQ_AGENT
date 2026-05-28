from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import json
import asyncpg
from app.config import settings
from app.logger import get_logger

logger = get_logger()

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def find_user_by_email(db: AsyncSession, email: str):
    query = "SELECT id, name, phone, email, created_at, updated_at FROM users WHERE email = :email"
    result = await db.execute(text(query), {"email": email})
    return result.fetchone()

async def create_user(db: AsyncSession, name: str, phone: str, email: str):
    query = """
        INSERT INTO users (name, phone, email)
        VALUES (:name, :phone, :email)
        RETURNING id, name, phone, email, created_at, updated_at
    """
    result = await db.execute(text(query), {"name": name, "phone": phone, "email": email})
    await db.commit()
    return result.fetchone()

async def get_last_session_category(db: AsyncSession, user_id):
    query = "SELECT category FROM sessions WHERE user_id = :user_id ORDER BY started_at DESC LIMIT 1"
    result = await db.execute(text(query), {"user_id": user_id})
    row = result.fetchone()
    return row[0] if row else None

async def create_session(db: AsyncSession, user_id, category: str, is_returning: bool):
    query = """
        INSERT INTO sessions (user_id, category, is_returning)
        VALUES (:user_id, :category, :is_returning)
        RETURNING id, user_id, category, is_returning, started_at, ended_at
    """
    result = await db.execute(text(query), {
        "user_id": user_id,
        "category": category,
        "is_returning": is_returning
    })
    await db.commit()
    return result.fetchone()

async def end_session(db: AsyncSession, session_id):
    query = "UPDATE sessions SET ended_at = NOW() WHERE id = :session_id"
    await db.execute(text(query), {"session_id": session_id})
    await db.commit()

async def save_message(db: AsyncSession, session_id, role: str, content: str):
    query = """
        INSERT INTO messages (session_id, role, content)
        VALUES (:session_id, :role, :content)
        RETURNING id, session_id, role, content, created_at
    """
    result = await db.execute(text(query), {
        "session_id": session_id,
        "role": role,
        "content": content
    })
    await db.commit()
    return result.fetchone()

async def get_last_10_messages(db: AsyncSession, session_id):
    query = """
        SELECT role, content FROM (
            SELECT role, content, created_at FROM messages
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT 10
        ) sub
        ORDER BY created_at ASC
    """
    result = await db.execute(text(query), {"session_id": session_id})
    rows = result.fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

async def get_checkpoints_by_session(db: AsyncSession, session_id: str, limit: int = 100):
    safe_limit = min(max(limit, 1), 500)
    query = """
        SELECT
            id,
            turn_id::text,
            checkpoint_type,
            status,
            user_id::text,
            session_id::text,
            category,
            agent,
            user_message_id::text,
            assistant_message_id::text,
            metadata,
            created_at::text
        FROM checkpoints
        WHERE session_id = :session_id
        ORDER BY created_at DESC
        LIMIT :limit
    """
    result = await db.execute(text(query), {"session_id": session_id, "limit": safe_limit})
    rows = result.fetchall()
    items = []
    for r in rows:
        items.append(
            {
                "id": r[0],
                "turn_id": r[1],
                "checkpoint_type": r[2],
                "status": r[3],
                "user_id": r[4],
                "session_id": r[5],
                "category": r[6],
                "agent": r[7],
                "user_message_id": r[8],
                "assistant_message_id": r[9],
                "metadata": r[10] or {},
                "created_at": r[11],
            }
        )
    return items


async def get_onboarding_state(db: AsyncSession, conversation_id: str):
    query = """
        SELECT
            conversation_id::text,
            step,
            profile,
            user_id::text,
            session_id::text
        FROM onboarding_states
        WHERE conversation_id = :conversation_id
    """
    result = await db.execute(text(query), {"conversation_id": conversation_id})
    row = result.fetchone()
    if not row:
        return None
    return {
        "conversation_id": row[0],
        "step": row[1],
        "profile": row[2] or {},
        "user_id": row[3],
        "session_id": row[4],
    }


async def upsert_onboarding_state(
    db: AsyncSession,
    conversation_id: str,
    step: str,
    profile: dict,
    user_id: str | None = None,
    session_id: str | None = None,
):
    query = """
        INSERT INTO onboarding_states (conversation_id, step, profile, user_id, session_id, updated_at)
        VALUES (:conversation_id, :step, :profile, :user_id, :session_id, NOW())
        ON CONFLICT (conversation_id)
        DO UPDATE SET
            step = EXCLUDED.step,
            profile = EXCLUDED.profile,
            user_id = EXCLUDED.user_id,
            session_id = EXCLUDED.session_id,
            updated_at = NOW()
    """
    await db.execute(
        text(query),
        {
            "conversation_id": conversation_id,
            "step": step,
            "profile": json.dumps(profile or {}),
            "user_id": user_id,
            "session_id": session_id,
        },
    )
    await db.commit()

async def write_log(db: AsyncSession, level: str, event: str, message: str, user_id=None, session_id=None, meta={}):
    try:
        query = """
            INSERT INTO logs (level, event, message, user_id, session_id, meta)
            VALUES (:level, :event, :message, :user_id, :session_id, :meta)
        """
        meta_json = json.dumps(meta)
        await db.execute(text(query), {
            "level": level,
            "event": event,
            "message": message,
            "user_id": user_id,
            "session_id": session_id,
            "meta": meta_json
        })
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to write DB log: {e}", extra={"event": "db_log_failure"})

async def get_db_direct():
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    return conn
