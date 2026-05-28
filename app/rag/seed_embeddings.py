from app.data.faq import FAQ
from app.rag.embedder import embed_text
from app.db import get_db_direct
from app.logger import get_logger

logger = get_logger()

async def seed():
    conn = None
    try:
        conn = await get_db_direct()
        seeded_count = 0
        for item in FAQ:
            category = item["category"]
            question = item["question"]
            answer = item["answer"]
            
            text = f"{question} {answer}"
            embedding = await embed_text(text)
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            query = """
                INSERT INTO faq_embeddings (category, question, answer, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (question) DO NOTHING
            """
            status = await conn.execute(query, category, question, answer, embedding_str)
            if status == "INSERT 0 1":
                seeded_count += 1
                
        logger.info(f"Seed process finished. Seeded {seeded_count} new FAQ embeddings.")
    except Exception as e:
        logger.error(f"Failed to seed FAQ embeddings: {e}")
        raise e
    finally:
        if conn:
            await conn.close()
