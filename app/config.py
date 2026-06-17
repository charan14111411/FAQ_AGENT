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

    # SMTP settings for sending the thank-you / transcript email directly (replaces the
    # external api-mobile.farmfuture.io email endpoint). Mirrors the C# "InfoSMTP" section.
    INFO_SMTP_HOST: str = "smtp.outlook.com"
    INFO_SMTP_PORT: int = 587
    INFO_SMTP_USER: Optional[str] = None
    INFO_SMTP_PASSWORD: Optional[str] = None
    INFO_SMTP_FROM: Optional[str] = None

    # WhatsApp Cloud API settings for the post-chat followup (replaces the external
    # api-mobile.farmfuture.io/SendPostChatFollowup endpoint). Mirrors the C# "WhatsApp" section.
    WHATSAPP_ACCESS_TOKEN: Optional[str] = None
    WHATSAPP_PHONE_NUMBER_ID: Optional[str] = None
    WHATSAPP_API_VERSION: str = "v22.0"
    WHATSAPP_TEMPLATE_NAME: str = "post_chat_followup"
    WHATSAPP_TEMPLATE_LANG: str = "en"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

