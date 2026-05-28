import os
from app.logger import get_logger

logger = get_logger()
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2' (384-dim)...")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            logger.error("Local model load failed. Cache 'all-MiniLM-L6-v2' first.")
            raise
        logger.info("SentenceTransformer loaded.")
    return _model


def _embed_local(text: str) -> list:
    return get_model().encode(text).tolist()


async def embed_text(text: str) -> list:
    try:
        return _embed_local(text)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise
