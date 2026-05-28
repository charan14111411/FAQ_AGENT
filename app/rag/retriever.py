from sqlalchemy import text
from app.rag.embedder import embed_text
from app.logger import get_logger

logger = get_logger()

async def retrieve(db, user_message: str, top_k: int = 3) -> str:
    try:
        embedding = await embed_text(user_message)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        
        try:
            query = "SELECT question, answer FROM faq_embeddings ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :limit"
            result = await db.execute(text(query), {"embedding": embedding_str, "limit": top_k})
            rows = result.fetchall()
            
            context_parts = []
            for row in rows:
                context_parts.append(f"Q: {row[0]}\nA: {row[1]}")
                
            return "\n\n".join(context_parts) + "\n\n" if context_parts else ""
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error retrieving FAQ embeddings: {e}")
        return ""
