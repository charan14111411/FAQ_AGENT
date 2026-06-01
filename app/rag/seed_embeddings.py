from app.data.faq import FAQ
from app.rag.embedder import embed_text
from app.db import get_db_direct
from app.logger import get_logger

logger = get_logger()

async def seed():
    conn = None
    try:
        conn = await get_db_direct()
        
        # 1. DELETE outdated FAQs (Pruning step)
        # Get all current questions from faq.py
        current_questions = [item["question"] for item in FAQ]
        
        # Delete any record in DB whose question is NOT in our current list
        delete_query = "DELETE FROM faq_embeddings WHERE question <> ALL($1)"
        delete_status = await conn.execute(delete_query, current_questions)
        logger.info(f"Pruned outdated FAQ records: {delete_status}")

        # 2. INSERT or UPDATE (Upsert step)
        inserted_count = 0
        updated_count = 0
        
        for item in FAQ:
            category = item["category"]
            question = item["question"]
            answer = item["answer"]
            
            # Check if this question already exists in the database
            existing = await conn.fetchrow(
                "SELECT answer, category FROM faq_embeddings WHERE question = $1", 
                question
            )
            
            if not existing:
                # New Question -> Embed and Insert
                text = f"{question} {answer}"
                embedding = await embed_text(text)
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                
                await conn.execute(
                    """
                    INSERT INTO faq_embeddings (category, question, answer, embedding)
                    VALUES ($1, $2, $3, $4::vector)
                    """,
                    category, question, answer, embedding_str
                )
                inserted_count += 1
                
            elif existing["answer"] != answer or existing["category"] != category:
                # Answer or Category changed -> Re-embed and Update
                text = f"{question} {answer}"
                embedding = await embed_text(text)
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                
                await conn.execute(
                    """
                    UPDATE faq_embeddings
                    SET answer = $1, category = $2, embedding = $3::vector
                    WHERE question = $4
                    """,
                    answer, category, embedding_str, question
                )
                updated_count += 1
                
        logger.info(f"FAQ Sync complete. Added: {inserted_count}, Updated: {updated_count}.")
        
    except Exception as e:
        logger.error(f"Failed to sync FAQ embeddings: {e}")
        raise e
    finally:
        if conn:
            await conn.close()
