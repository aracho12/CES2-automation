"""
lammps_input_writer.py — Generate base.in.lammps for DFT-CES2 QM/MM runs.

Generated file sections
-----------------------
  1. Initialization   — units, atom_style, boundary, pair_style, kspace, processors
  2. read_data        — reads data.file (with Masses, Atoms, Bonds, Angles, coeffs)
  3. Group defs       — SOLUTE (QM), OXYGEN, PROTON, WATER, per-ion groups, SOLVENT, regions
  4. LJ pair_coeff    — lj/cut/long: MM-MM (real params); QM-involved (zero)
  5. bjdisp pair_coeff— QM-MM (geometric-mean rule); QM-QM (zero); MM-MM (zero via omission)
  6. Gridforce fixes  — #CUBEPOSITION marker + fix gridforce template lines
  7. MD settings      — thermo, timestep, dump, fix shake, fix nvt, restart, run

Lookup for LJ params (per type_label)
--------------------------------------
  Priority:
    1. config.yaml [ces2][lj_params][<type_label>] → {epsilon, sigma}
    2. Built-in defaults for common species (TIP3P water, JC ions)
    3. Fallback: epsilon=0.0, sigma=1.0  (with a warning comment in output)

Cross-pair (i≠j) mixing rule: Lorentz-Berthelot
  eps_ij  = sqrt(eps_i * eps_j)
  sig_ij  = 0.5 * (sig_i + sig_j)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .bjdisp_db import load_all, build_bjdisp_table
from .species import Species


# ---------------------------------------------------------------------------
# Built-in LJ defaults: type_label → (epsilon [kcal/mol], sigma [Å])
# Sources:
#   TIP3P  — Jorgensen et al. 1983
#   Ions   — Joung & Cheatham 2008 (JC, TIP3P-parameterized)
#   OH-    — O treated same as water O (common approximation; override in config)
# ---------------------------------------------------------------------------
_DEFAULT_LJ: Dict[str, Tuple[float, float]] = {
    # TIP3P water
    "Ow":    (0.1521,   3.1507),
    "Hw":    (0.0000,   0.0000),
    # Hydroxide (approx — same as water O/H; override if you have better params)
    "O_oh":  (0.1521,   3.1507),
    "H_oh":  (0.0000,   0.0000),
    # Monovalent cations (JC / TIP3P)
    "Na":    (0.3526418, 2.1600),
    "K":     (0.4184,   3.3330),
    "Li":    (0.0279,   1.8250),
    "Rb":    (0.4748,   3.6560),
    "Cs":    (0.5000,   4.1430),
    # Monovalent anions (JC / TIP3P)
    "Cl":    (0.7200,   4.4170),
    "Br":    (0.7150,   4.8370),
    "F":     (0.7530,   3.1180),
    "I":     (0.6130,   5.4000),
    # Common polyatomic atoms (approximate; override for accurate work)
    "C":     (0.0860,   3.3997),
    "N":     (0.1700,   3.2500),
    "P":     (0.2000,   3.7400),
    "S":     (0.2500,   3.5640),
}


# ---------------------------------------------------------------------------
# Lorentz-Berthelot mixing
# ---------------------------------------------------------------------------

def _lb_mix(eps_i: float, sig_i: float, eps_j: float, sig_j: float) -> Tuple[float, float]:
    """Lorentz-Berthelot: geometric mean for epsilon, arithmetic mean for sigma."""
    return math.sqrt(eps_i * eps_j), 0.5 * (sig_i + sig_j)


def _get_lj(label: str, lj_db: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Return (eps, sig) for type_label; falls back to (0.0, 1.0) if unknown."""
    return lj_db.get(label, (0.0, 1.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_lammps_input(
    *,
    export_dir: Path,
    type_id_by_label: Dict[str, int],
    label_by_type_id: Dict[int, str],
    species_order: List[Tuple[str, int]],       # [(species_id, count), ...]
    species_db: Dict[str, Species],
    box,                                        # BoxMeta (has .Lx .Ly .z_el_lo .z_el_hi)
    bond_coeffs: Dict[int, Tuple[float, float]],   # already written in data.file
    angle_coeffs: Dict[int, Tuple[float, float]],
    qm_params_dir: Path,
    cfg: Dict[str, Any],
    n_mm: int,
) -> Path:
    """
    Write base.in.lammps to export_dir/base.in.lammps.

    Parameters
    ----------
    export_dir        Path where the file will be written (alongside data.file).
    type_id_by_label  {type_label: LAMMPS_type_id}  (from make_type_registry)
    label_by_type_id  {LAMMPS_type_id: type_label}
    species_order     [(species_id, molecule_count), ...] in PACKMOL order
    species_db        loaded species DB dict
    box               BoxMeta namedtuple/dataclass
    bond_coeffs       {bond_type_id: (k, r0)} — written in data.file (not repeated here)
    angle_coeffs      {angle_type_id: (k, theta0)}
    qm_params_dir     Path to species_db/qm_params/ directory
    cfg               raw config dict
    n_mm              total number of MM atoms (QM atoms start at atom n_mm+1)

    Returns
    -------
    Path of written file.
    """
    ces2_cfg   = cfg.get("ces2", {})
    recipe_cfg = cfg.get("electrolyte_recipe", {})

    # ------------------------------------------------------------------ #
    #  1. Pair-style parameters
    # ------------------------------------------------------------------ #
    lj_cut     = float(ces2_cfg.get("lj_cutoff",        12.0))
    coul_cut   = float(ces2_cfg.get("coulomb_cutoff",   12.0))
    kspace_acc = float(ces2_cfg.get("kspace_accuracy",  1.0e-5))
    bjd_a1     = float(ces2_cfg.get("bjdisp_a1",        1.40))
    bjd_a2     = float(ces2_cfg.get("bjdisp_a2",        0.50))
    bjd_s8     = float(ces2_cfg.get("bjdisp_s8",        2.10))

    # ------------------------------------------------------------------ #
    #  2. LJ parameter DB  (built-in defaults + user overrides)
    # ------------------------------------------------------------------ #
    lj_db: Dict[str, Tuple[float, float]] = dict(_DEFAULT_LJ)
    for lbl, vals in ces2_cfg.get("lj_params", {}).items():
        lj_db[lbl] = (float(vals["epsilon"]), float(vals["sigma"]))

    # ------------------------------------------------------------------ #
    #  3. Identify MM vs QM type labels
    # ------------------------------------------------------------------ #
    # MM labels = those coming from species in species_order
    mm_labels_set: set = set()
    mm_labels_ordered: List[str] = []   # stable order (first seen)
    for sid, _ in species_order:
        sp = species_db[sid]
        for a in sp.atoms:
            if a.type_label not in mm_labels_set:
                mm_labels_set.add(a.type_label)
                mm_labels_ordered.append(a.type_label)

    # QM labels = whatever is left in type_id_by_label (slab elements)
    qm_labels_ordered: List[str] = sorted(
        [lbl for lbl in type_id_by_label if lbl not in mm_labels_set],
        key=lambda l: type_id_by_label[l],
    )

    mm_type_ids: Dict[str, int] = {lbl: type_id_by_label[lbl] for lbl in mm_labels_ordered}
    qm_type_ids: Dict[str, int] = {lbl: type_id_by_label[lbl] for lbl in qm_labels_ordered}
    n_types = max(type_id_by_label.values()) if type_id_by_label else 0

    # ------------------------------------------------------------------ #
    #  4. bjdisp DB
    # ------------------------------------------------------------------ #
    mm_bjdisp, qm_bjdisp, cfg_bjdisp = load_all(
        species_db=species_db,
        qm_params_dir=qm_params_dir,
        config_bjdisp=cfg.get("bjdisp"),
    )

    # ------------------------------------------------------------------ #
    #  5. Identify water / ion type IDs for groups and SHAKE
    # ------------------------------------------------------------------ #
    water_sid   = recipe_cfg.get("water", {}).get("species_id", "water_tip3p")
    water_O_tid: Optional[int] = None
    water_H_tid: Optional[int] = None
    water_bond_type:  Optional[int] = None
    water_angle_type: Optional[int] = None

    if water_sid in species_db:
        wsp = species_db[water_sid]
        for a in wsp.atoms:
            if a.element == "O" and water_O_tid is None:
                water_O_tid = type_id_by_label.get(a.type_label)
            elif a.element == "H" and water_H_tid is None:
                water_H_tid = type_id_by_label.get(a.type_label)
        # Bond type for O-H bond (type id used in data.file) — first bond coeff
        if wsp.bond_coeffs:
            water_bond_type = min(wsp.bond_coeffs.keys())
        if wsp.angle_coeffs:
            water_angle_type = min(wsp.angle_coeffs.keys())

    # ------------------------------------------------------------------ #
    #  6. MD settings from config
    # ------------------------------------------------------------------ #
    md_cfg       = ces2_cfg.get("md", {})
    timestep     = float(md_cfg.get("timestep_fs",  0.5))
    n_steps      = int(  md_cfg.get("n_steps",      0))
    thermo_every = int(  md_cfg.get("thermo_every", 100))
    dump_every   = int(  md_cfg.get("dump_every",   100))
    nvt_temp     = float(md_cfg.get("temperature",  300.0))
    nvt_tdamp    = float(md_cfg.get("t_damp_fs",    100.0))
    restart_every = int( md_cfg.get("restart_every", 1000))
    shake_tol    = float(md_cfg.get("shake_tol",    1.0e-4))
    shake_iter   = int(  md_cfg.get("shake_iter",   100))

    # ------------------------------------------------------------------ #
    #  7. Assemble file lines
    # ------------------------------------------------------------------ #
    lines: List[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    def section(title: str) -> None:
        L()
        L(f"# {'='*60}")
        L(f"# {title}")
        L(f"# {'='*60}")

    # ── Header ──────────────────────────────────────────────────────────
    L("# base.in.lammps — DFT-CES2 QM/MM LAMMPS input")
    L("# Auto-generated by cesbuild  (lammps_input_writer.py)")
    L("# Do NOT edit pair_style/kspace lines by hand; regenerate instead.")
    L("#")
    L("# Runtime markers replaced by qmmm wrapper script:")
    L("#   ###dispf   → dispersion-force cube file path")
    L("#   ###qmxyz   → QM atom XYZ coordinates")
    L("#   #CUBEPOSITION → entire line replaced with gridforce fix commands")

    # ── Section 1: Initialization ────────────────────────────────────────
    section("1. Initialization")
    L()
    L("units           real")
    L("atom_style      full")
    L("boundary        p p f")
    L()
    # pair_style: lj/cut/long for MM electrostatics + bjdisp for QM-MM dispersion
    L(f"pair_style      hybrid/overlay"
      f"  lj/cut/long {lj_cut:.1f}"
      f"  bjdisp {bjd_a1:.2f} {bjd_a2:.2f} {bjd_s8:.2f}")
    L(f"kspace_style    pppm {kspace_acc:.1e}")
    L()
    L("processors      * * 1")

    # ── Section 2: Read data ─────────────────────────────────────────────
    section("2. Read data")
    L()
    L("read_data       data.file")

    # ── Section 3: Group definitions ─────────────────────────────────────
    section("3. Group definitions")

    # SOLUTE = QM (slab) atom types
    L()
    if qm_labels_ordered:
        qm_tid_str = " ".join(str(qm_type_ids[lbl]) for lbl in qm_labels_ordered)
        qm_lbl_str = " ".join(qm_labels_ordered)
        L(f"group    SOLUTE   type {qm_tid_str}   # QM slab: {qm_lbl_str}")
    else:
        L("group    SOLUTE   type 0   # WARNING: no QM type labels found")

    # Water groups
    L()
    L("# Water sub-groups (for SHAKE and diagnostics)")
    if water_O_tid is not None:
        o_lbl = label_by_type_id.get(water_O_tid, "?")
        L(f"group    OXYGEN   type {water_O_tid}   # {o_lbl} (water O)")
    if water_H_tid is not None:
        h_lbl = label_by_type_id.get(water_H_tid, "?")
        L(f"group    PROTON   type {water_H_tid}   # {h_lbl} (water H)")
    if water_O_tid is not None and water_H_tid is not None:
        L("group    WATER    union OXYGEN PROTON")

    # Per-species ion groups (non-water species)
    L()
    L("# Ion / solute groups")
    for sid, _ in species_order:
        if sid == water_sid:
            continue
        sp = species_db[sid]
        tids = sorted(set(type_id_by_label[a.type_label] for a in sp.atoms))
        tid_str  = " ".join(str(t) for t in tids)
        lbl_str  = " ".join(sorted(set(a.type_label for a in sp.atoms)))
        grp_name = sid.upper().replace("-", "_")[:12]   # LAMMPS group name limit
        L(f"group    {grp_name:<12s} type {tid_str}   # {sid}: {lbl_str}")

    # SOLVENT = all MM atoms
    L()
    L("group    SOLVENT  subtract all SOLUTE   # all MM (electrolyte) atoms")

    # ── Section 4: LJ pair_coeff ─────────────────────────────────────────
    section("4. LJ pair coefficients  (lj/cut/long)")
    L()
    L(f"# Lorentz-Berthelot mixing rule: eps_ij = sqrt(eps_i*eps_j), sig_ij = (sig_i+sig_j)/2")
    L(f"# QM-MM and QM-QM LJ = 0 (QM interactions handled by DFT)")
    L()

    # Sort all type ids
    all_tids = sorted(type_id_by_label.values())

    missing_lj: List[str] = []   # labels with no LJ params (will warn in file)

    for i in all_tids:
        lbl_i = label_by_type_id[i]
        is_qm_i = lbl_i in qm_type_ids

        for j in all_tids:
            if j < i:
                continue   # only upper triangle i <= j
            lbl_j = label_by_type_id[j]
            is_qm_j = lbl_j in qm_type_ids

            if is_qm_i or is_qm_j:
                # QM atom involved → zero LJ
                L(f"pair_coeff  {i:3d} {j:3d}  lj/cut/long  0.0000  1.0000"
                  f"   # {lbl_i}-{lbl_j}  [QM: zero LJ]")
            else:
                # MM-MM: use built-in defaults / user config
                eps_i, sig_i = _get_lj(lbl_i, lj_db)
                eps_j, sig_j = _get_lj(lbl_j, lj_db)
                if lbl_i not in lj_db and lbl_i not in missing_lj:
                    missing_lj.append(lbl_i)
                if lbl_j not in lj_db and lbl_j not in missing_lj:
                    missing_lj.append(lbl_j)
                eps_ij, sig_ij = _lb_mix(eps_i, sig_i, eps_j, sig_j)
                flag = "  [!DEFAULT FALLBACK]" if (lbl_i not in _DEFAULT_LJ or
                                                     lbl_j not in _DEFAULT_LJ) else ""
                L(f"pair_coeff  {i:3d} {j:3d}  lj/cut/long"
                  f"  {eps_ij:.7f}  {sig_ij:.4f}"
                  f"   # {lbl_i}-{lbl_j}{flag}")

    if missing_lj:
        L()
        L(f"# WARNING: LJ params not found for these type_labels (fallback 0,1 used):")
        for lbl in missing_lj:
            L(f"#   {lbl}  → add to config.yaml [ces2][lj_params]")

    # ── Section 5: bjdisp pair_coeff ──────────────────────────────────────
    section("5. bjdisp pair coefficients  (QM-MM dispersion)")
    L()
    if qm_type_ids:
        bjdisp_lines = build_bjdisp_table(
            mm_type_ids=mm_type_ids,
            qm_type_ids=qm_type_ids,
            mm_db=mm_bjdisp,
            qm_db=qm_bjdisp,
            config_db=cfg_bjdisp,
        )
        for bl in bjdisp_lines:
            L(bl)
    else:
        L("# (no QM types registered — bjdisp pair_coeff skipped)")

    # ── Section 6: Gridforce fixes + #CUBEPOSITION ────────────────────────
    section("6. CES2 gridforce fixes")
    L()
    L("# The qmmm wrapper script replaces the #CUBEPOSITION line below with")
    L("# actual 'fix gridforce ...' commands pointing to the QM cube files.")
    L("# Do not remove or rename the marker line.")
    L()
    L("#CUBEPOSITION")

    # ── Section 7: MD settings ────────────────────────────────────────────
    section("7. MD settings")
    L()
    L(f"timestep        {timestep}")
    L()
    L(f"thermo          {thermo_every}")
    L("thermo_style    custom step temp epair emol etotal press vol")
    L()
    L(f"dump            traj all atom {dump_every} traj.lammpstrj")
    L("dump_modify     traj sort id")
    L()

    # SHAKE for water O-H bonds and H-O-H angle
    if water_O_tid is not None and water_H_tid is not None:
        b_str = f"b {water_bond_type}"  if water_bond_type  is not None else "b 1"
        a_str = f"a {water_angle_type}" if water_angle_type is not None else "a 1"
        L(f"# SHAKE: rigid O-H bonds (bond type {water_bond_type})"
          f" and H-O-H angles (angle type {water_angle_type}) for water")
        L(f"fix   shake_water WATER shake {shake_tol} {shake_iter} 0 {b_str} {a_str}")
        L()

    # NVT thermostat on SOLVENT only
    seed_nvt = int(cfg.get("project", {}).get("seed", 4321)) + 1
    L(f"fix   nvt_solvent SOLVENT nvt temp {nvt_temp} {nvt_temp} {nvt_tdamp}")
    L()

    # Restart
    L(f"restart         {restart_every}  restart.lammps.a  restart.lammps.b")
    L()

    # Run
    L(f"run             {n_steps}")
    L()

    # ── Write file ────────────────────────────────────────────────────────
    out_path = export_dir / "base.in.lammps"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[lammps_input_writer] Written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: show LJ DB for debugging
# ---------------------------------------------------------------------------

def print_lj_db(lj_db: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    """Print the LJ parameter table (defaults if lj_db not provided)."""
    db = lj_db if lj_db is not None else _DEFAULT_LJ
    print(f"{'Label':<10}  {'epsilon':>10}  {'sigma':>8}  source")
    print("-" * 50)
    for lbl, (eps, sig) in sorted(db.items()):
        src = "built-in" if lbl in _DEFAULT_LJ else "user config"
        print(f"{lbl:<10}  {eps:10.7f}  {sig:8.4f}  {src}")
