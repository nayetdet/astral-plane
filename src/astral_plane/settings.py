from __future__ import annotations

import tarfile
from functools import cached_property
from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    @property
    def assets_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "assets"

    @property
    def slam_config_path(self) -> Path:
        return self.assets_dir / "KinectV1.yaml"

    @cached_property
    def vocabulary_path(self) -> Path:
        vocabulary = Path.home() / ".cache" / "astral-plane" / "ORBvoc.txt"
        if not vocabulary.exists():
            vocabulary.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.assets_dir / "ORBvoc.txt.tar.gz", "r:gz") as archive:
                archive.extract("ORBvoc.txt", vocabulary.parent)
        return vocabulary


settings: Final[Settings] = Settings()
