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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import yaml


# C6 unit conversion: atomic units → kcal/mol·Å^6
_AU_TO_KCALMOL_ANG6 = 13.77928721

# Default BJ scaling per element when not provided in source.
# H of an OH-like donor uses 0.53; everything else uses 1.40.
_DEFAULT_S_BY_ELEMENT: Dict[str, float] = {"H": 0.53}
_DEFAULT_S_FALLBACK: float = 1.40


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

def load_qm_params_db(qm_params_dir: Path, filename: Optional[str] = None) -> Dict[str, BjdispParams]:
    """
    Load QM BJ-dispersion parameters from qm_params_dir.
    Returns dict: type_label -> BjdispParams.

    If filename is given (e.g. "IrO2" or "IrO2.yaml"), only that file is loaded.
    Otherwise all *.yaml files in the directory are loaded.

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

    if filename is not None:
        stem = Path(filename).stem
        target = qm_params_dir / f"{stem}.yaml"
        if not target.exists():
            available = [f.name for f in sorted(qm_params_dir.glob("*.yaml"))]
            raise FileNotFoundError(
                f"slab.qm_params_file: '{target.name}' not found in {qm_params_dir}. "
                f"Available: {available}"
            )
        yaml_files = [target]
    else:
        yaml_files = sorted(qm_params_dir.glob("*.yaml"))

    for yaml_file in yaml_files:
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
# Layer-file loader: bjparams_layer_avg.dat
# ---------------------------------------------------------------------------

@dataclass
class LayerEntry:
    """One row of bjparams_layer_avg.dat → one type_label group."""
    type_label: str     # e.g. "Ir_L01"
    element: str
    z_avg: float        # [Å] relative z (file column, lowest atom = 0)
    n_atoms: int
    params: BjdispParams


_Z_TOL_RE = re.compile(r"z_tol\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")


def _default_s_for(element: str) -> float:
    return _DEFAULT_S_BY_ELEMENT.get(element, _DEFAULT_S_FALLBACK)


def parse_layer_file(
    path: Path,
    default_z_tol: float = 0.20,
) -> Tuple[float, List[LayerEntry], Dict[str, BjdispParams]]:
    """
    Parse a bjparams_layer_avg.dat file.

    File format:
        # BJ dispersion parameters — layer-averaged (z_tol = 0.20 Ang)
        # Element      N    z_avg(Ang)    ALPHAscs_avg     C6_D3_avg
        # ----------------------------------------------------------
          O            4      0.000000          4.6230       10.5000
          Ir           8      1.226200         30.3319      305.5000
          ...

    Type label scheme: <Element>_L<NN>, 1-based per-element, z-ascending.
    C6 values are converted from atomic units → kcal/mol·Å^6 at load.
    s defaults to 0.53 for H, 1.40 for everything else.

    Returns
    -------
    (z_tol, entries, db):
        z_tol   : tolerance parsed from header, or *default_z_tol* if absent
        entries : list of LayerEntry for atom→layer matching
        db      : {type_label: BjdispParams} for bjdisp lookup
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    z_tol = default_z_tol
    rows: List[Tuple[str, int, float, float, float]] = []  # (el, N, z, alpha_au, C6_au)

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = _Z_TOL_RE.search(stripped)
            if m:
                z_tol = float(m.group(1))
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(
                f"[layer file {path.name}] expected 5 columns "
                f"(Element N z_avg alpha C6), got: {line!r}"
            )
        el = parts[0]
        n_atoms = int(parts[1])
        z_avg = float(parts[2])
        alpha_au = float(parts[3])
        c6_au = float(parts[4])
        rows.append((el, n_atoms, z_avg, alpha_au, c6_au))

    # Assign per-element labels in z-ascending order (stable)
    sorted_rows = sorted(enumerate(rows), key=lambda kv: (kv[1][0], kv[1][2]))
    label_by_orig_idx: Dict[int, str] = {}
    per_el_counter: Dict[str, int] = {}
    for orig_idx, (el, _n, _z, _a, _c) in sorted_rows:
        per_el_counter[el] = per_el_counter.get(el, 0) + 1
        label_by_orig_idx[orig_idx] = f"{el}_L{per_el_counter[el]:02d}"

    entries: List[LayerEntry] = []
    db: Dict[str, BjdispParams] = {}
    for orig_idx, (el, n_atoms, z_avg, alpha_au, c6_au) in enumerate(rows):
        lbl = label_by_orig_idx[orig_idx]
        params = BjdispParams(
            type_label=lbl,
            alpha_iso=alpha_au,
            C6=c6_au * _AU_TO_KCALMOL_ANG6,
            s=_default_s_for(el),
            source=f"{path.name}: layer {lbl} ({el}, z={z_avg:.3f} Å, N={n_atoms})",
        )
        db[lbl] = params
        entries.append(LayerEntry(
            type_label=lbl,
            element=el,
            z_avg=z_avg,
            n_atoms=n_atoms,
            params=params,
        ))
    return z_tol, entries, db


def assign_layer_labels(
    elements: Sequence[str],
    z_positions: Sequence[float],
    entries: Sequence[LayerEntry],
    z_tol: float,
) -> List[str]:
    """
    For each atom (element, Cartesian z), find the matching LayerEntry.

    Matching rule: same element AND |z_atom_rel - entry.z_avg| <= z_tol,
    where z_atom_rel = z_atom - min(z_positions).  If multiple entries match,
    the nearest by |Δz| wins.  Raises ValueError if any atom has no match.

    Returns per-atom list of type_label strings.
    """
    if len(elements) != len(z_positions):
        raise ValueError(
            f"assign_layer_labels: elements ({len(elements)}) and "
            f"z_positions ({len(z_positions)}) have different lengths"
        )

    z_min = min(z_positions) if z_positions else 0.0
    labels: List[str] = []
    misses: List[str] = []

    for i, (el, z_abs) in enumerate(zip(elements, z_positions)):
        z_rel = float(z_abs) - float(z_min)
        best: Optional[Tuple[float, LayerEntry]] = None
        for e in entries:
            if e.element != el:
                continue
            dz = abs(z_rel - e.z_avg)
            if dz <= z_tol and (best is None or dz < best[0]):
                best = (dz, e)
        if best is None:
            # Collect available z's for this element for the error message
            el_zs = sorted({round(e.z_avg, 3) for e in entries if e.element == el})
            misses.append(
                f"  atom {i}: element={el}, z_rel={z_rel:.3f} Å "
                f"(z_abs={z_abs:.3f}); available {el} layers at z_rel={el_zs}"
            )
            labels.append("")
        else:
            labels.append(best[1].type_label)

    if misses:
        raise ValueError(
            f"[layer assignment] {len(misses)} atom(s) could not be matched "
            f"to any layer (z_tol={z_tol} Å):\n" + "\n".join(misses[:10]) +
            (f"\n  ... ({len(misses)-10} more)" if len(misses) > 10 else "")
        )
    return labels


def layer_label_to_element(entries: Sequence[LayerEntry]) -> Dict[str, str]:
    """Mapping {type_label: element} for mass lookup on layer-derived labels."""
    return {e.type_label: e.element for e in entries}


# ---------------------------------------------------------------------------
# Convenience: build all three dbs from standard project layout
# ---------------------------------------------------------------------------

def load_all(
    species_db: Dict[str, Any],
    qm_params_dir: Optional[Path],
    config_bjdisp: Optional[Dict] = None,
    qm_params_file: Optional[str] = None,
) -> tuple[Dict[str, BjdispParams], Dict[str, BjdispParams], Dict[str, BjdispParams]]:
    """
    Load MM db, QM db, and config db in one call.

    Returns:
        (mm_db, qm_db, config_db)

    Pass qm_params_dir=None to skip loading the qm_params YAML database
    (e.g. when slab.bjparams_source is "layer_file").
    Pass qm_params_file to load only that file (e.g. "IrO2" or "IrO2.yaml");
    if omitted, all *.yaml files in qm_params_dir are loaded.
    """
    mm_db = extract_mm_bjdisp_from_species(species_db)
    qm_db = load_qm_params_db(qm_params_dir, filename=qm_params_file) if qm_params_dir is not None else {}
    cfg_db = _parse_config_bjdisp(config_bjdisp)
    return mm_db, qm_db, cfg_db
