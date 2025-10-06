from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =========================== APP Config ========================
    APP_NAME: str
    APP_VERSION: str

    # =========================== Groq Config =======================
    GROQ_API_KEY: str
    LLM_MODEL: str


    # Load environment file from PROJECT_ROOT/.env when available
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore")


settings = Settings()
