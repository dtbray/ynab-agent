"""Application configuration with Pydantic Settings."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """YNAB Agent configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # YNAB
    ynab_auth_mode: str = "oauth"
    ynab_access_token: SecretStr | None = None
    ynab_plan_id: str = "last-used"
    ynab_oauth_client_id: str | None = None
    ynab_oauth_client_secret: SecretStr | None = None
    ynab_oauth_redirect_uri: str = "http://localhost:8765/callback"
    ynab_oauth_scope: str = "read-only"
    ynab_oauth_token_path: str = ".ynab-oauth-token.json"
    
    # 1Password
    op_service_account_token: SecretStr | None = None
    op_vault_name: str = ""
    op_item_name: str = ""
    
    # Notifications
    discord_enabled: bool = False
    discord_target: str = ""
    
    # Polling
    poll_interval_minutes: int = 15
    
    # Logging
    log_level: str = "INFO"
    
    # Database
    database_url: str | None = None
    database_path: str = "ynab_agent.db"

    @property
    def effective_database_url(self) -> str:
        """Return the configured SQLAlchemy database URL."""
        return self.database_url or f"sqlite+aiosqlite:///{self.database_path}"


settings = Settings()
