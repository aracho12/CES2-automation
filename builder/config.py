from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def workdir(self) -> Path:
        return Path(self.raw.get("project", {}).get("workdir", "./")).resolve()

def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(raw)
