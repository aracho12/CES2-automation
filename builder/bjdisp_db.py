"""
bjdisp_db.py — BJ-dispersion parameter database and lookup logic.

Lookup priority (for each type_label):
  1. species YAML   → atom.bjdisp field         (MM atoms: water, ions, ...)
  2. qm_params YAML → species_db/qm_params/*.yaml (QM surface atoms: Ir, O in IrO2, ...)
  3. config fallback → config.yaml [bjdisp] section
  4. KeyError with helpful message

Main public functions:
  load_qm_params_db(qm_params_dir)      → dict[type_label, BjdispParams]
  get_bjdisp(type_label, ...)            → BjdispParams
  compute_pair_coeff(qm_params, mm_params) → (alpha_rms, s, C6_combined)
  build_bjdisp_table(type_ids, ...)      → list of pair_coeff lines for LAMMPS
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import yaml


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class BjdispParams:
    """BJdisp parameters for one atom type."""
    type_label: str
    alpha_iso: float   # isotropic polarizability [a.u.]
    C6: float          # dispersion coefficient   [kcal/mol * Ang^6]
    s: float = 1.40    # BJ scaling parameter (0.53 for H-bond O, 1.40 otherwise)
    source: str = ""   # traceability


# ---------------------------------------------------------------------------
# Loader: qm_params/*.yaml
# ---------------------------------------------------------------------------

def load_qm_params_db(qm_params_dir: Path) -> Dict[str, BjdispParams]:
    """
    Load all *.yaml files under qm_params_dir.
    Returns dict: type_label -> BjdispParams.

    Each YAML has structure:
      system: IrO2
      types:
        Ir:
          alpha_iso: 31.4
          C6: 4209.986
          s: 1.40
          note: "..."
    """
    db: Dict[str, BjdispParams] = {}

    if not qm_params_dir.exists():
        return db

    for yaml_file in sorted(qm_params_dir.glob("*.yaml")):
        with yaml_file.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        system_name = data.get("system", yaml_file.stem)
        types_block = data.get("types", {})

        for type_label, fields in types_block.items():
            if type_label in db:
                # Later file wins; warn if values differ
                pass
            db[type_label] = BjdispParams(
                type_label=type_label,
                alpha_iso=float(fields["alpha_iso"]),
                C6=float(fields["C6"]),
                s=float(fields.get("s", 1.40)),
                source=f"{yaml_file.name} [{system_name}]: {fields.get('note', '')}",
            )

    return db


# ---------------------------------------------------------------------------
# Loader: species YAML bjdisp field (MM atoms)
# ---------------------------------------------------------------------------

def extract_mm_bjdisp_from_species(species_db: Dict[str, Any]) -> Dict[str, BjdispParams]:
    """
    Walk all loaded Species objects in species_db and pull out bjdisp fields.
    Returns dict: type_label -> BjdispParams.
    Atoms with bjdisp: null are silently skipped.
    """
    db: Dict[str, BjdispParams] = {}

    for sp_id, sp in species_db.items():
        for atom in sp.atoms:
            bjdisp_raw = getattr(atom, "bjdisp", None)
            if bjdisp_raw is None:
                continue
            if not isinstance(bjdisp_raw, dict):
                continue

            lbl = atom.type_label
            if lbl in db:
                continue  # first encounter wins (same label appears in multiple species)

            db[lbl] = BjdispParams(
                type_label=lbl,
                alpha_iso=float(bjdisp_raw["alpha_iso"]),
                C6=float(bjdisp_raw["C6"]),
                s=float(bjdisp_raw.get("s", 1.40)),
                source=f"species_db/{sp_id}.yaml",
            )

    return db


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------

def _parse_config_bjdisp(config_bjdisp: Optional[Dict]) -> Dict[str, BjdispParams]:
    """
    Parse config.yaml [bjdisp] section into BjdispParams dict.

    Expected config format:
      bjdisp:
        Ir:
          alpha_iso: 31.4
          C6: 4209.986
          s: 1.40
        MyCustomAtom:
          alpha_iso: 10.0
          C6: 500.0
    """
    db: Dict[str, BjdispParams] = {}
    if not config_bjdisp:
        return db

    for type_label, fields in config_bjdisp.items():
        db[type_label] = BjdispParams(
            type_label=type_label,
            alpha_iso=float(fields["alpha_iso"]),
            C6=float(fields["C6"]),
            s=float(fields.get("s", 1.40)),
            source="config.yaml [bjdisp]",
        )
    return db


# ---------------------------------------------------------------------------
# Main lookup function
# ---------------------------------------------------------------------------

def get_bjdisp(
    type_label: str,
    mm_db: Dict[str, BjdispParams],
    qm_db: Dict[str, BjdispParams],
    config_db: Dict[str, BjdispParams],
) -> BjdispParams:
    """
    Look up bjdisp params for type_label with priority:
      1. MM species YAML (mm_db)
      2. QM params YAML  (qm_db)
      3. config.yaml fallback (config_db)
      4. KeyError with clear message

    Args:
        type_label  : atom type label (e.g. "Ow", "K", "Ir")
        mm_db       : from extract_mm_bjdisp_from_species()
        qm_db       : from load_qm_params_db()
        config_db   : from _parse_config_bjdisp()

    Returns:
        BjdispParams
    """
    if type_label in mm_db:
        return mm_db[type_label]
    if type_label in qm_db:
        return qm_db[type_label]
    if type_label in config_db:
        return config_db[type_label]

    # Friendly error message
    available = sorted(set(mm_db) | set(qm_db) | set(config_db))
    raise KeyError(
        f"\n[bjdisp] Parameter not found for type_label: '{type_label}'\n"
        f"Available in DB: {available}\n"
        f"Fix options:\n"
        f"  (A) Add to species YAML: species_db/<species>.yaml  →  atom.bjdisp: {{alpha_iso: ..., C6: ...}}\n"
        f"  (B) Add to QM params:    species_db/qm_params/<system>.yaml\n"
        f"  (C) Add to config:       config.yaml  →  bjdisp:\n"
        f"                             {type_label}:\n"
        f"                               alpha_iso: <value>  # a.u.\n"
        f"                               C6: <value>         # kcal/mol*Ang^6\n"
    )


# ---------------------------------------------------------------------------
# Pair coefficient calculation (geometric mean rule)
# ---------------------------------------------------------------------------

def compute_pair_coeff(
    qm_params: BjdispParams,
    mm_params: BjdispParams,
) -> tuple[float, float, float]:
    """
    Compute LAMMPS bjdisp pair_coeff values for a QM-MM atom pair.

    Geometric mean rule (JACS Au 2025):
        alpha_rms = sqrt(alpha_iso_QM * alpha_iso_MM)
        C6_combined = sqrt(C6_QM * C6_MM)
        s = mm_params.s   (scaling comes from the MM atom type)

    Returns:
        (alpha_rms, s, C6_combined)
    """
    alpha_rms = math.sqrt(qm_params.alpha_iso * mm_params.alpha_iso)
    C6_combined = math.sqrt(qm_params.C6 * mm_params.C6)
    s = mm_params.s
    return alpha_rms, s, C6_combined


# ---------------------------------------------------------------------------
# Build full bjdisp pair_coeff block for LAMMPS input
# ---------------------------------------------------------------------------

def build_bjdisp_table(
    mm_type_ids: Dict[str, int],      # type_label -> LAMMPS type id  (MM atoms only)
    qm_type_ids: Dict[str, int],      # type_label -> LAMMPS type id  (QM atoms only)
    mm_db: Dict[str, BjdispParams],
    qm_db: Dict[str, BjdispParams],
    config_db: Dict[str, BjdispParams],
) -> list[str]:
    """
    Generate all bjdisp pair_coeff lines needed for base.in.lammps.

    Lines generated:
      - QM-MM pairs  (every QM type × every MM type)
      - QM-QM pairs  (set to 0.0 — QM-QM dispersion handled by DFT)

    Format:
      pair_coeff  <mm_type> <qm_type> bjdisp <alpha_rms> <s> <C6>
      pair_coeff  <qm_type> <qm_type> bjdisp 0.0 0.0 0.0

    Returns list of strings (one per line), ready to join with newlines.
    """
    lines: list[str] = []

    lines.append("# QM-MM dispersion (bjdisp): alpha_rms [a.u.], s, C6 [kcal/mol*Ang^6]")
    lines.append("# Format: pair_coeff <MM_type> <QM_type> bjdisp <alpha_rms> <s> <C6>")

    for mm_label, mm_tid in sorted(mm_type_ids.items(), key=lambda x: x[1]):
        mm_p = get_bjdisp(mm_label, mm_db, qm_db, config_db)
        for qm_label, qm_tid in sorted(qm_type_ids.items(), key=lambda x: x[1]):
            qm_p = get_bjdisp(qm_label, mm_db, qm_db, config_db)
            alpha_rms, s, C6 = compute_pair_coeff(qm_p, mm_p)
            lines.append(
                f"pair_coeff  {mm_tid:3d} {qm_tid:3d} bjdisp "
                f"{alpha_rms:.10f} {s:.2f} {C6:.7f}"
                f"  # {mm_label}-{qm_label}"
            )

    lines.append("")
    lines.append("# QM-QM dispersion: set to 0 (handled inside DFT)")
    for qm_label_i, qm_tid_i in sorted(qm_type_ids.items(), key=lambda x: x[1]):
        for qm_label_j, qm_tid_j in sorted(qm_type_ids.items(), key=lambda x: x[1]):
            if qm_tid_i <= qm_tid_j:
                lines.append(
                    f"pair_coeff  {qm_tid_i:3d} {qm_tid_j:3d} bjdisp 0.0 0.0 0.0"
                    f"  # {qm_label_i}-{qm_label_j}"
                )

    return lines


# ---------------------------------------------------------------------------
# Convenience: build all three dbs from standard project layout
# ---------------------------------------------------------------------------

def load_all(
    species_db: Dict[str, Any],
    qm_params_dir: Path,
    config_bjdisp: Optional[Dict] = None,
) -> tuple[Dict[str, BjdispParams], Dict[str, BjdispParams], Dict[str, BjdispParams]]:
    """
    Load MM db, QM db, and config db in one call.

    Returns:
        (mm_db, qm_db, config_db)

    Usage:
        mm_db, qm_db, cfg_db = load_all(species_db, qm_params_dir, cfg.get("bjdisp"))
        params = get_bjdisp("Ir", mm_db, qm_db, cfg_db)
    """
    mm_db = extract_mm_bjdisp_from_species(species_db)
    qm_db = load_qm_params_db(qm_params_dir)
    cfg_db = _parse_config_bjdisp(config_bjdisp)
    return mm_db, qm_db, cfg_db
