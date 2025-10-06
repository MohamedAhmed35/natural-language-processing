import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================== APP Config ========================
    APP_NAME: str
    APP_VERSION: str
    PROJECT_ROOT: str = os.getenv("RAG_ROOT")

    # =========================== Log Config ========================
    LOG_LEVEL: str

    # =========================== Backend Config ====================
    RAG_API_URL: str

    # =========================== Groq Config =======================
    GROQ_API_KEY: str
    LLM_MODEL: str
    TRIM_MAX_TOKENS: int

    # =========================== HuggingFace Config ================
    EMBEDDING_MODEL: str


    # Load environment file from PROJECT_ROOT/.env when available
    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        extra="ignore")


settings = Settings()
