import asyncio
import os

from app.config import settings
from app.logger import get_logger

logger = get_logger()

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local SentenceTransformer model 'all-MiniLM-L6-v2' (384 dimensions)...")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception as e:
            logger.error(
                "Local SentenceTransformer model load failed. "
                "Ensure 'all-MiniLM-L6-v2' is cached locally or allow internet access to download it."
            )
            raise
        logger.info("SentenceTransformer model loaded successfully.")
    return _model

def _get_local_embedding(text: str) -> list:
    """
    Generates a local semantic embedding using SentenceTransformer (all-MiniLM-L6-v2).
    """
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()

async def embed_text(text: str) -> list:
    """
    Returns the 384-dimensional semantic embedding for the given text.
    Uses SentenceTransformer locally inside a worker thread to prevent blocking.
    """
    try:
        return await asyncio.to_thread(_get_local_embedding, text)
    except Exception as e:
        logger.error(f"Failed to generate SentenceTransformer embedding: {e}")
        raise e
