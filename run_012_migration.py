import asyncio
import asyncpg
from app.config import settings

CHECK_SQL = (
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name='sessions' AND column_name='prospect_id'"
)

async def run():
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        sql = open("migrations/012_add_prospect_id_to_sessions.sql").read()
        await conn.execute(sql)
        print("Migration 012 applied successfully!")
        row = await conn.fetchrow(CHECK_SQL)
        print("Column check:", row)
    finally:
        await conn.close()

asyncio.run(run())
