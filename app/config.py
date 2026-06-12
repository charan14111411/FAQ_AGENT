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

    # Dynamic UI Image Assets Config
    IMAGE_PATH_DEVICE: str = "./assets/images/microclime.png"
    IMAGE_PATH_MICROCLIME: str = "./assets/images/microclime.png"
    IMAGE_PATH_SOILSYNC: str = "./assets/images/soilsync.png"
    IMAGE_PATH_YIELDWHISPERER: str = "./assets/images/microclime.png"
    IMAGE_PATH_RAINSENSE: str = "./assets/images/soilsync.png"
    IMAGE_PATH_APP_DASHBOARD: str = "./assets/images/appscreenshots/dashboard.png"
    IMAGE_PATH_APP_CROP_HEALTH: str = "./assets/images/appscreenshots/crop_health.png"
    IMAGE_PATH_APP_ADVISORY: str = "./assets/images/appscreenshots/advisory.png"
    IMAGE_PATH_APP_AGRONOMY: str = "./assets/images/appscreenshots/agronomic_practices.png"
    IMAGE_PATH_APP_CALENDAR: str = "./assets/images/appscreenshots/calender.png"
    IMAGE_PATH_APP_CASHBOOK: str = "./assets/images/appscreenshots/cashbook.png"
    IMAGE_PATH_APP_LANGUAGE: str = "./assets/images/appscreenshots/language_selection.png"
    IMAGE_PATH_APP_LOGIN: str = "./assets/images/appscreenshots/login_screen.png"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

