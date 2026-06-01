import asyncio
import asyncpg
from app.config import settings

async def run():
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM faq_embeddings")
        rows = await conn.fetch("SELECT question FROM faq_embeddings ORDER BY question")
        print(f"Total FAQs in DB: {count}")
        print("\nAll FAQ questions:")
        for r in rows:
            print(f"  - {r[0]}")
    finally:
        await conn.close()

asyncio.run(run())
