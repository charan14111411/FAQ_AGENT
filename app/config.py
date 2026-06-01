from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    CHECKPOINT_DB_URL: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    LLM_PROVIDER: str = "groq"
    FRONTEND_URL: str = "http://localhost:5500"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
