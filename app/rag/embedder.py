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
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            logger.info("Configured PyTorch CPU thread limits (set to 1 thread) to optimize SentenceTransformer execution.")
        except ImportError:
            pass
        logger.info("Loading local SentenceTransformer model 'all-MiniLM-L6-v2' (384 dimensions)...")
        try:
            # 1. Try loading from local cache (fast, offline)
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            logger.warning("SentenceTransformer model 'all-MiniLM-L6-v2' not found in cache. Attempting online download...")
            try:
                # 2. Try online download (requires internet access on first run)
                _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=False)
                logger.info("SentenceTransformer model downloaded and cached successfully.")
            except Exception as download_err:
                logger.error(
                    f"Failed to load or download SentenceTransformer model: {download_err}\n\n"
                    "--- OFFLINE PRODUCTION DEPLOYMENT INSTRUCTIONS ---\n"
                    "If your production server has no internet access, you must transfer the model manually:\n"
                    "1. Copy this folder from your local development machine:\n"
                    "   C:\\Users\\Leela\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\n"
                    "2. Paste it on the production server under the same path in the user profile directory running the server:\n"
                    "   <user_home_dir>\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2\n"
                    "--------------------------------------------------\n"
                )
                raise download_err
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
