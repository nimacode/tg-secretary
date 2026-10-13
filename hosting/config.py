from pathlib import Path
from typing import Literal

from pydantic import model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict


# Hosted tenants always have a USD cap: only models with known native pricing are safe.
# Keep aligned with secretary.gemini.PRICES without importing tenant settings.
GEMINI_METERED_MODELS = frozenset({
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
})

class HostingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file="hosting.env", env_file_encoding="utf-8", extra="ignore")

    platform_bot_token: str
    platform_bot_username: str
    # Empty = log in with the classic Login Widget (needs /setdomain on the platform bot).
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    # Shared native keys are metered locally; OpenRouter gets one capped key per tenant.
    llm_provider: Literal["auto", "openrouter", "gemini", "anthropic"] = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_extractor_model: str = "gemini-3.1-flash-lite"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    openrouter_mgmt_key: str = ""
    public_url: str  # https://example.com, no trailing slash
    admin_tg_id: int
    db_path: Path = Path("./hosting.db")
    tenants_root: Path = Path("./tenants")
    repo_root: Path = Path(__file__).resolve().parent.parent
    listen_port: int = 8700
    port_range_start: int = 9100
    port_range_end: int = 9199
    trial_credit_usd: float = 0.0
    consent_version: int = 1
    payment_instructions: str = ""  # shown on the awaiting-credit page (card number, wallet)

    @property
    def provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        return "gemini" if self.gemini_api_key else "openrouter"

    @model_validator(mode="after")
    def validate_provider(self) -> "HostingSettings":
        required = {
            "anthropic": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            "gemini": ("GEMINI_API_KEY", self.gemini_api_key),
            "openrouter": ("OPENROUTER_MGMT_KEY", self.openrouter_mgmt_key),
        }
        name, key = required[self.provider]
        if self.llm_provider != "auto" and not key.strip():
            raise ValueError(f"LLM_PROVIDER={self.provider} requires {name}")
        if self.provider == "gemini":
            for name, model in (("GEMINI_MODEL", self.gemini_model),
                                ("GEMINI_EXTRACTOR_MODEL", self.gemini_extractor_model)):
                if model not in GEMINI_METERED_MODELS:
                    raise ValueError(f"{name} must be a known-priced native Gemini model for hosted credit caps")
        return self


settings = HostingSettings()  # type: ignore[call-arg]
