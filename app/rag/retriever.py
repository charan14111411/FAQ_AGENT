from sqlalchemy import text
from app.rag.embedder import embed_text
from app.logger import get_logger

logger = get_logger()

async def retrieve(db, user_message: str, top_k: int = 3) -> str:
    try:
        embedding = await embed_text(user_message)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        
        try:
            # THRESHOLD = 0.76  (calibrated from live data)
            # ─────────────────────────────────────────────────────────────────
            # On-topic queries (formal + informal/colloquial): max dist = 0.75
            # Off-topic queries (cooking, math, jokes):        min dist = 0.73+
            #
            # There is NO single threshold that perfectly blocks ALL off-topic
            # queries — e.g. "invest in stock market" scores 0.50 (close to
            # agritech investor FAQs by nature). The LLM guardrail in nodes.py
            # (RULE 1: DOMAIN ONLY) is the true safety net for those edge cases.
            #
            # 0.76 is optimal because:
            #   - Passes ALL 15 on-topic queries (formal + colloquial)  ✓
            #   - The few off-topic that leak (0.73-0.76) are caught by LLM guardrail ✓
            #   - Clearly off-topic (cooking, math, jokes) score 0.81+ → blocked ✓
            # ─────────────────────────────────────────────────────────────────
            query = (
                "SELECT question, answer FROM faq_embeddings "
                "WHERE embedding <=> CAST(:embedding AS vector) <= 0.76 "
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
