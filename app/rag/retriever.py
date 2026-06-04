from app.rag.embedder import embed_text
from app.config import settings
from app.logger import get_logger
from qdrant_client import AsyncQdrantClient

logger = get_logger()

def _get_client():
    if settings.QDRANT_URL.startswith("http://") or settings.QDRANT_URL.startswith("https://"):
        return AsyncQdrantClient(url=settings.QDRANT_URL)
    else:
        return AsyncQdrantClient(location=settings.QDRANT_URL)

async def retrieve(db, user_message: str, top_k: int = 3) -> str:
    """
    Retrieves matching FAQ entries using Qdrant vector search.
    - db: kept for backwards compatibility in function signature.
    - user_message: text query from the user.
    - top_k: maximum number of matches to return.
    """
    client = None
    try:
        embedding = await embed_text(user_message)
        client = _get_client()
        
        if not await client.collection_exists(settings.QDRANT_COLLECTION):
            return ""

        # Cosine Similarity threshold of 0.24 is equivalent to pgvector's Cosine Distance of 0.76.
        # (Cosine Distance = 1.0 - Cosine Similarity)
        search_results = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embedding,
            limit=top_k,
            score_threshold=0.24
        )
        
        context_parts = []
        for hit in search_results.points:
            payload = hit.payload
            context_parts.append(f"Q: {payload.get('question')}\nA: {payload.get('answer')}")
            
        return "\n\n".join(context_parts) + "\n\n" if context_parts else ""
        
    except Exception as e:
        logger.error(f"Error retrieving FAQ embeddings from Qdrant: {e}")
        return ""
    finally:
        if client:
            await client.close()

