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
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def find_user_by_email(db: AsyncSession, email: str):
    result = await db.execute(
        text("SELECT id, name, phone, email, created_at, updated_at FROM users WHERE email = :email"),
        {"email": email},
    )
    return result.fetchone()


async def create_user(db: AsyncSession, name: str, email: str = None, phone: str = None):
    result = await db.execute(
        text("""
            INSERT INTO users (name, phone, email)
            VALUES (:name, :phone, :email)
            RETURNING id, name, phone, email, created_at, updated_at
        """),
        {"name": name, "phone": phone, "email": email},
    )
    await db.commit()
    return result.fetchone()


async def get_last_session_category(db: AsyncSession, user_id):
    result = await db.execute(
        text("SELECT category FROM sessions WHERE user_id = :uid ORDER BY started_at DESC LIMIT 1"),
        {"uid": user_id},
    )
    row = result.fetchone()
    return row[0] if row else None


async def create_session(db: AsyncSession, user_id, category: str, is_returning: bool = False):
    result = await db.execute(
        text("""
            INSERT INTO sessions (user_id, category, is_returning)
            VALUES (:user_id, :category, :is_returning)
            RETURNING id, user_id, category, is_returning, started_at, ended_at
        """),
        {"user_id": user_id, "category": category, "is_returning": is_returning},
    )
    await db.commit()
    return result.fetchone()


async def end_session(db: AsyncSession, session_id):
    await db.execute(
        text("UPDATE sessions SET ended_at = NOW() WHERE id = :sid"),
        {"sid": session_id},
    )
    await db.commit()


async def save_message(db: AsyncSession, session_id, role: str, content: str):
    result = await db.execute(
        text("""
            INSERT INTO messages (session_id, role, content)
            VALUES (:session_id, :role, :content)
            RETURNING id, session_id, role, content, created_at
        """),
        {"session_id": session_id, "role": role, "content": content},
    )
    await db.commit()
    return result.fetchone()


async def get_last_10_messages(db: AsyncSession, session_id):
    result = await db.execute(
        text("""
            SELECT role, content FROM (
                SELECT role, content, created_at FROM messages
                WHERE session_id = :sid
                ORDER BY created_at DESC LIMIT 10
            ) sub ORDER BY created_at ASC
        """),
        {"sid": session_id},
    )
    rows = result.fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


async def write_log(db: AsyncSession, level: str, event: str, message: str,
                    user_id=None, session_id=None, meta={}):
    try:
        await db.execute(
            text("""
                INSERT INTO logs (level, event, message, user_id, session_id, meta)
                VALUES (:level, :event, :message, :user_id, :session_id, :meta)
            """),
            {
                "level": level, "event": event, "message": message,
                "user_id": user_id, "session_id": session_id,
                "meta": json.dumps(meta),
            },
        )
        await db.commit()
    except Exception as e:
        logger.error(f"DB log write failed: {e}")


async def get_db_direct():
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    return await asyncpg.connect(url)
