from __future__ import annotations

import tarfile
from functools import cached_property
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    timestamp: float = 0.0

    @computed_field
    @property
    def assets(self) -> Path:
        return Path(__file__).resolve().parents[2] / "assets"

    @computed_field
    @property
    def settings_path(self) -> Path:
        return self.assets / "TUM1.yaml"

    @computed_field
    @cached_property
    def vocabulary_path(self) -> Path:
        path = Path.home() / ".cache" / "astral-plane" / "ORBvoc.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.assets / "ORBvoc.txt.tar.gz", "r:gz") as archive:
                archive.extract("ORBvoc.txt", path.parent)
        return path


settings = Settings()
