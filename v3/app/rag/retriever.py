from sqlalchemy import text
from app.rag.embedder import embed_text
from app.logger import get_logger

logger = get_logger()


async def retrieve(db, user_message: str, category: str = None,
                   top_k: int = 3, include_agritech: bool = False) -> str:
    """
    Partitioned RAG retrieval.

    category=None        → cross-domain search (used by general_agent and master)
    category='corporate' + include_agritech=True → searches corporate AND agritech rows
    category='grower'    → searches only grower rows
    """
    try:
        embedding = await embed_text(user_message)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        if category is None:
            # Cross-domain — no partition filter
            query = """
                SELECT question, answer, category
                FROM faq_embeddings
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :lim
            """
            params = {"emb": embedding_str, "lim": top_k}

        elif category == "corporate" and include_agritech:
            # Corporate slave covers both corporate + agritech sub-domains
            query = """
                SELECT question, answer, category
                FROM faq_embeddings
                WHERE category IN ('corporate', 'agritech')
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :lim
            """
            params = {"emb": embedding_str, "lim": top_k}

        else:
            # Standard single-category partition
            query = """
                SELECT question, answer, category
                FROM faq_embeddings
                WHERE category = :cat
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :lim
            """
            params = {"emb": embedding_str, "lim": top_k, "cat": category}

        result = await db.execute(text(query), params)
        rows = result.fetchall()

        parts = [f"Q: {r[0]}\nA: {r[1]}" for r in rows]
        return "\n\n".join(parts) + "\n\n" if parts else ""

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return ""
