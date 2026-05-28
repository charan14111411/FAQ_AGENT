from app.data.faq import FAQ
from app.rag.embedder import embed_text
from app.db import get_db_direct
from app.logger import get_logger

logger = get_logger()


async def seed():
    """Idempotent startup seeder — embeds question+answer text and inserts into faq_embeddings."""
    conn = None
    try:
        conn = await get_db_direct()
        seeded = 0
        for item in FAQ:
            combined = f"{item['question']} {item['answer']}"
            embedding = await embed_text(combined)
            emb_str = f"[{','.join(str(x) for x in embedding)}]"
            status = await conn.execute(
                """
                INSERT INTO faq_embeddings (category, question, answer, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (question) DO NOTHING
                """,
                item["category"], item["question"], item["answer"], emb_str,
            )
            if status == "INSERT 0 1":
                seeded += 1
        logger.info(f"Seeding complete. {seeded} new embeddings inserted.")
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        raise
    finally:
        if conn:
            await conn.close()
