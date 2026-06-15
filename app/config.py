from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    CHECKPOINT_DB_URL: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "groq"
    FRONTEND_URL: str = "http://localhost:5500"
    BACKEND_URL: str = "https://aibackend.varsapradaya.com/"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SESSION_TIMEOUT_MINUTES: int = 30
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "faq_embeddings"
    CRM_PROSPECT_URL: str = "https://dev.businesscentral.in/rest/telecaller/backoffice/createProspect"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

