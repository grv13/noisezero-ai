from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Manages application settings and environment variables.
    """
    # Load environment variables from a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # NVIDIA BNR settings
    NVIDIA_API_KEY: str
    NVIDIA_FUNCTION_ID: str
    NVIDIA_FUNCTION_ID_STUDIO_VOICE: str
    NVIDIA_API_KEY_STUDIO_VOICE: str
    NVIDIA_TARGET_URL: str

    # Captioning settings
    MONGO_URI: str = "mongodb://localhost:27017"
    GROQ_API_KEY: str

settings = Settings()