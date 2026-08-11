from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_WATER_MODEL_FALLBACK = "TIP3P"


def load_forcefield_db(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate the shared MM force-field database."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Force-field database not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "sets" not in data or "water_models" not in data:
        raise ValueError(f"{p} missing required 'sets'/'water_models' keys")
    return data


def resolve_water_model_entry(
    data: Dict[str, Any],
    water_model: str,
    *,
    source: str | Path,
) -> Dict[str, Any]:
    """Return a water-model mapping, retaining the legacy TIP3P fallback."""
    wm = str(water_model).upper()
    entry = data["water_models"].get(wm)
    if entry is not None:
        return entry

    entry = data["water_models"].get(_WATER_MODEL_FALLBACK)
    if entry is None:
        raise KeyError(
            f"water_model '{wm}' not in {source} and no "
            f"'{_WATER_MODEL_FALLBACK}' fallback entry present"
        )
    print(
        f"[forcefield_db] water_model '{wm}' not in force-field DB; "
        f"using '{_WATER_MODEL_FALLBACK}' sets"
    )
    return entry


def resolve_hydroxide_set_name(
    data: Dict[str, Any],
    water_model: str,
    hydroxide_model: Optional[str] = None,
    *,
    source: str | Path,
) -> str:
    """Resolve ``auto`` or a config-friendly OH model name to a set name."""
    raw = "auto" if hydroxide_model is None else str(hydroxide_model).strip()
    if not raw or raw.lower() == "auto":
        entry = resolve_water_model_entry(data, water_model, source=source)
        set_name = entry.get("hydroxide_set")
        if not set_name:
            raise KeyError(
                f"water_model '{str(water_model).upper()}' has no hydroxide_set in {source}"
            )
    else:
        set_name = raw if raw.startswith("hydroxide_") else f"hydroxide_{raw}"

    if set_name not in data["sets"]:
        available = sorted(k for k in data["sets"] if k.startswith("hydroxide_"))
        raise KeyError(
            f"Unknown hydroxide model/set '{raw}' in {source}; available: {available}"
        )
    return str(set_name)
