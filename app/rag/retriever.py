from app.rag.embedder import embed_text
from app.config import settings
from app.logger import get_logger
from qdrant_client import AsyncQdrantClient
from app.agents.base_agent import _call_llm
import re

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
        # Preprocess/rewrite query using LLM if it contains pronouns or is in a non-English script
        query = user_message.strip()
        PRONOUN_PATTERN = re.compile(r"\b(you|u|your|ur|yourself|yours)\b", re.IGNORECASE)
        
        is_non_english = not query.isascii()
        contains_pronoun = bool(PRONOUN_PATTERN.search(query))
        
        if is_non_english or contains_pronoun:
            try:
                rewrite_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a search query optimizer for a precision agritech platform named Varsapradaya.\n"
                            "Translate and rewrite the user's conversational message into a clean, third-person, "
                            "grammatically correct English search query. Replace pronouns referring to the platform "
                            "('you', 'u', 'your', 'ur', 'yourself', 'yours') with 'Varsapradaya' or 'Varsapradaya's' "
                            "where grammatically appropriate. Keep the core intent identical.\n"
                            "Do NOT add any greetings, conversational filler, or explanations. Reply with ONLY the rewritten query text."
                        )
                    },
                    {"role": "user", "content": query}
                ]
                # Fast call with 30 max tokens
                res = await _call_llm(rewrite_messages, max_tokens=30, temperature=0.0)
                rewritten = res["reply"].strip().strip("\"'")
                if rewritten:
                    query = rewritten
            except Exception as le:
                logger.warning(f"Failed to rewrite query: {le}. Using original query.")

        embedding = await embed_text(query)
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

