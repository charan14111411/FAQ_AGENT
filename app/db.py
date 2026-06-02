from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import json
import asyncpg
from app.config import settings
from app.logger import get_logger

logger = get_logger()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=15
)
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

async def find_user_by_phone(db: AsyncSession, phone: str):
    query = "SELECT id, name, phone, email, created_at, updated_at FROM users WHERE phone = :phone LIMIT 1"
    result = await db.execute(text(query), {"phone": phone})
    return result.fetchone()

async def update_user_name(db: AsyncSession, user_id, new_name: str):
    query = "UPDATE users SET name = :name, updated_at = NOW() WHERE id = :user_id"
    await db.execute(text(query), {"name": new_name, "user_id": user_id})
    await db.commit()

async def create_user(db: AsyncSession, name: str, email: str, phone: str = None):
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


async def update_session_prospect_id(db: AsyncSession, session_id, prospect_id: str):
    """Stores the CRM prospectID returned by BusinessCentral onto the session row."""
    query = "UPDATE sessions SET prospect_id = :prospect_id WHERE id = :session_id"
    await db.execute(text(query), {"session_id": session_id, "prospect_id": prospect_id})
    await db.commit()


async def get_prospect_id_for_user(db: AsyncSession, user_id) -> str | None:
    """
    Fetches the most recent prospect_id from any prior session of the given user.
    Used when CRM reports 'Mobile already exists' — so we reuse the existing prospectID
    rather than leaving the current session with a null.
    """
    query = """
        SELECT prospect_id FROM sessions
        WHERE user_id = :user_id
          AND prospect_id IS NOT NULL
        ORDER BY started_at DESC
        LIMIT 1
    """
    result = await db.execute(text(query), {"user_id": user_id})
    row = result.fetchone()
    return row[0] if row else None


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

async def get_all_session_messages(db: AsyncSession, session_id):
    query = """
        SELECT role, content FROM messages
        WHERE session_id = :session_id
        ORDER BY created_at ASC
    """
    result = await db.execute(text(query), {"session_id": session_id})
    rows = result.fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]

async def fetch_inactive_sessions(db: AsyncSession, threshold_minutes: int = 30):
    query = """
        SELECT 
            s.id AS session_id,
            s.category,
            u.email,
            u.name,
            u.phone
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        LEFT JOIN (
            SELECT session_id, MAX(created_at) AS last_msg_time
            FROM messages
            GROUP BY session_id
        ) m ON s.id = m.session_id
        WHERE s.ended_at IS NULL
          AND (
              (m.last_msg_time IS NOT NULL AND m.last_msg_time < NOW() - INTERVAL '1 minute' * :threshold)
              OR
              (m.last_msg_time IS NULL AND s.started_at < NOW() - INTERVAL '1 minute' * :threshold)
          );
    """
    result = await db.execute(text(query), {"threshold": threshold_minutes})
    return result.fetchall()




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
