from __future__ import annotations
from pathlib import Path
import os
from pydantic import Field
from pydantic_settings import BaseSettings
import subprocess

def _detect_project_root() -> Path:
    # 1) ENV override (most explicit)
    env = os.getenv("ALBCOVIS_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # 2) If running inside a Git clone, use repo top-level
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if root:
            return Path(root).resolve()
    except Exception:
        pass

    # 3) Fallback: assume this file lives at <repo>/src/albcovis/settings.py
    # so project root is 2 levels up from this file.
    return Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    """
    Typed environment configuration for ALBCOVIS.
    Loads from OS env and local `.env` file (for dev).
    """

    user_agent: str = Field(
        default="ALBCOVIS/1.0 (contact@example.com)",
        alias="USER_AGENT",
        description="Contact-style User-Agent string for API calls."
    )

    discogs_token: str = Field(
        default="",
        alias="DISCOGS_TOKEN",
        description="Discogs personal token (required for Discogs API)."
    )

    # Project paths
    project_root: Path = Field(default_factory=_detect_project_root)
    source_data_dir: Path = Field(default_factory=lambda: _detect_project_root() / "data" / "source")
    source_images_dir: Path = Field(default_factory=lambda: _detect_project_root() / "data" / "source" / "images")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()
