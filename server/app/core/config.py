from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Trajectory Detection Research Server"
    environment: str = "dev"
    database_url: str = "sqlite:///server/data/app.db"
    template_dir: str = "server/app/templates"
    app_password: str
    app_auth_secret: str
    session_max_age_seconds: int = 86400

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
