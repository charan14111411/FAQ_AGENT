from sqlalchemy import text
from app.rag.embedder import embed_text
from app.logger import get_logger

logger = get_logger()

async def retrieve(db, user_message: str, top_k: int = 3) -> str:
    try:
        embedding = await embed_text(user_message)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        
        try:
            # Only return FAQs with cosine distance <= 0.70 (threshold for semantic relevance).
            # pgvector <=> returns cosine distance: 0 = identical, 2 = opposite.
            # Without this filter, poor matches get injected and the LLM says "I don't have details".
            query = (
                "SELECT question, answer FROM faq_embeddings "
                "WHERE embedding <=> CAST(:embedding AS vector) <= 0.70 "
                "ORDER BY embedding <=> CAST(:embedding AS vector) "
                "LIMIT :limit"
            )
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
