"""
lammps_input_writer.py — Generate base.in.lammps for DFT-CES2 QM/MM runs.

Generated file sections
-----------------------
  1. Initialization   — units, atom_style, boundary, special_bonds, bond/angle_style,
                        pair_style (TIP4P/TIP3P + bjdisp), kspace + slab correction
  2. read_data        — reads data.file; reset_timestep 0; set water charges
  3. Group defs       — SOLUTE (QM), OXYGEN/PROTON/WATER, per-species ion groups,
                        SOLVENT, region blocks for toplayer/centerlayer
  4. LJ pair_coeff    — lj/cut/tip4p/long/opt (MM-MM real params, QM-involved = 0)
  5. bjdisp pair_coeff— QM-MM (geometric-mean from bjdisp_db); QM-QM = 0
  6. Bond/angle coeff — bond_coeff, angle_coeff for water (for TIP4P M-site)
  7. Gridforce fixes  — fix hGrid/oGrid (TIP4P) + fix per ion element (gridforce/net)
                        energy tracking variables; #CUBEPOSITION marker
  8. MD settings      — dispersion compute/dump, wall, shake, momentum, NVT,
                        thermo_style with Egrid variables, dump, restart, run

Water model note
----------------
The Research Notes (§2.4) use TIP4P-EW.  Set  ces2.water_model: TIP4P  (default)
in config.yaml.  TIP3P mode (lj/cut/long + pppm) is also supported but not recommended
for DFT-CES2.

Gridforce/net cube-index convention
------------------------------------
  The qmmm wrapper builds one LAMMPS cube file per MM "group":
    idx 0 → hydrogen (water H / H-containing species)
    idx 1 → oxygen   (water O / O-containing species)
    idx 2, 3, … → remaining unique elements (K, Na, Cl, Br, F, I, …)
  in the order they first appear in species_order (water excluded).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .bjdisp_db import load_all, build_bjdisp_table, BjdispParams
from .species import Species


# ---------------------------------------------------------------------------
# Built-in LJ defaults: type_label → (epsilon [kcal/mol], sigma [Å])
# TIP4P-EW water (used when water_model = TIP4P, default):
#   Ow: Horn et al. 2004 — 0.16275, 3.16435
#   Hw: 0.0, 0.0  (H has no LJ in TIP4P)
# Ions - JC for TIP4P-EW from J. Phys. Chem. B 112, 9020–9041 (2008).
# TIP3P water (used when water_model = TIP3P):
#   Ow: Jorgensen 1983 — 0.1521, 3.1507
# Override per type_label in config.yaml [ces2][lj_params].
# ---------------------------------------------------------------------------
_DEFAULT_LJ_TIP4P: Dict[str, Tuple[float, float]] = {
    # TIP4P-EW water
    # Source: Horn et al., J. Chem. Phys. 120, 9665 (2004)
    "Ow":    (0.16275000,  3.16435),
    "Hw":    (0.0000,      1.0000),   # eps=0; sigma set to 1 to avoid /0
    # Hydroxide (approximate — override in config for accurate work)
    "O_oh":  (0.16275000,  3.16435),
    "H_oh":  (0.0000,      1.0000),
    # Joung & Cheatham (2008) J. Phys. Chem. B 112, 9020 — TIP4P-Ew column
    # sigma = 2*(Rmin/2) / 2^(1/6)  (converted from Table 5)
    "Li":    (0.10398840,  1.43969),
    "Na":    (0.16843750,  2.18448),
    "K":     (0.27946510,  2.83306),
    "Rb":    (0.43314940,  3.04509),
    "Cs":    (0.39443180,  3.36403),
    "F":     (0.00157520,  4.52220),
    "Cl":    (0.01166150,  4.91776),
    "Br":    (0.03037730,  4.93202),
    "I":     (0.04170820,  5.25987),
}

# _DEFAULT_LJ_TIP3P: Dict[str, Tuple[float, float]] = {
#     # TIP3P water
#     "Ow":    (0.1521,      3.1507),
#     "Hw":    (0.0000,      1.0000),
#     "O_oh":  (0.1521,      3.1507),
#     "H_oh":  (0.0000,      1.0000),
#     # Joung-Cheatham ions (TIP3P-compatible)
#     "Na":    (0.3526418,   2.1600),
#     "K":     (0.4184,      3.3330),
#     "Li":    (0.0279,      1.8250),
#     "Rb":    (0.4748,      3.6560),
#     "Cs":    (0.5000,      4.1430),
#     "Cl":    (0.7200,      4.4170),
#     "Br":    (0.7150,      4.8370),
#     "F":     (0.7530,      3.1180),
#     "I":     (0.6130,      5.4000),
# }

# Additional common non-water/ion types (same for both models)
_DEFAULT_LJ_COMMON: Dict[str, Tuple[float, float]] = {
    "C":  (0.0860, 3.3997),
    "N":  (0.1700, 3.2500),
    "P":  (0.2000, 3.7400),
    "S":  (0.2500, 3.5640),
}


# ---------------------------------------------------------------------------
# Lorentz-Berthelot mixing
# ---------------------------------------------------------------------------
def _lb_mix(eps_i: float, sig_i: float, eps_j: float, sig_j: float) -> Tuple[float, float]:
    return math.sqrt(eps_i * eps_j), 0.5 * (sig_i + sig_j)


def _get_lj(label: str, lj_db: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
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
    box,                                        # BoxMeta
    bond_coeffs: Dict[int, Tuple[float, float]],
    angle_coeffs: Dict[int, Tuple[float, float]],
    qm_params_dir: Path,
    cfg: Dict[str, Any],
    n_mm: int,
    charged_params: Optional[Dict[str, float]] = None,
    extra_qm_bjdisp: Optional[Dict[str, BjdispParams]] = None,
) -> Path:
    """
    Write base.in.lammps to export_dir/base.in.lammps.
    Returns the path of the written file.
    """
    ces2_cfg   = cfg.get("ces2", {})
    recipe_cfg = cfg.get("electrolyte_recipe", {})
    cell_cfg   = cfg.get("cell", {})

    # ------------------------------------------------------------------ #
    #  Water model: TIP4P (default) or TIP3P
    # ------------------------------------------------------------------ #
    water_model = str(ces2_cfg.get("water_model", "TIP4P")).upper()
    use_tip4p   = (water_model == "TIP4P")

    # ------------------------------------------------------------------ #
    #  Pair-style parameters
    # ------------------------------------------------------------------ #
    lj_cut      = float(ces2_cfg.get("lj_cutoff",        12.0))
    kspace_acc  = float(ces2_cfg.get("kspace_accuracy",  1.0e-4))
    bjd_cutoff  = float(ces2_cfg.get("bjdisp_cutoff",     15.0))
    bjd_a1      = float(ces2_cfg.get("bjdisp_a1",        1.40))
    bjd_a2      = float(ces2_cfg.get("bjdisp_a2",        0.50))
    bjd_s8      = float(ces2_cfg.get("bjdisp_s8",        2.10))
    kspace_slab = float(ces2_cfg.get("kspace_slab",      3.0))
    tip4p_msite = float(ces2_cfg.get("tip4p_msite",      0.125))  # M-site dist Å

    # ------------------------------------------------------------------ #
    #  Supercell factor for gridforce/net
    # ------------------------------------------------------------------ #
    rep = list(cell_cfg.get("supercell", [1, 1, 1]))
    sc_factor = int(rep[0]) * int(rep[1]) * int(rep[2])

    # ------------------------------------------------------------------ #
    #  LJ parameter DB  (built-in defaults + user overrides)
    # ------------------------------------------------------------------ #
    lj_db: Dict[str, Tuple[float, float]] = dict(
        _DEFAULT_LJ_TIP4P if use_tip4p else _DEFAULT_LJ_TIP3P
    )
    lj_db.update(_DEFAULT_LJ_COMMON)
    for lbl, vals in ces2_cfg.get("lj_params", {}).items():
        lj_db[lbl] = (float(vals["epsilon"]), float(vals["sigma"]))

    # ------------------------------------------------------------------ #
    #  Identify MM vs QM type labels
    # ------------------------------------------------------------------ #
    mm_labels_set: set = set()
    mm_labels_ordered: List[str] = []
    for sid, _ in species_order:
        sp = species_db[sid]
        for a in sp.atoms:
            if a.type_label not in mm_labels_set:
                mm_labels_set.add(a.type_label)
                mm_labels_ordered.append(a.type_label)

    qm_labels_ordered: List[str] = sorted(
        [lbl for lbl in type_id_by_label if lbl not in mm_labels_set],
        key=lambda l: type_id_by_label[l],
    )

    mm_type_ids: Dict[str, int] = {lbl: type_id_by_label[lbl] for lbl in mm_labels_ordered}
    qm_type_ids: Dict[str, int] = {lbl: type_id_by_label[lbl] for lbl in qm_labels_ordered}

    # ------------------------------------------------------------------ #
    #  Water type IDs
    #  species_id is optional in config — auto-derived from water_model
    #  when not explicitly set.  Mapping: TIP4P→water_tip4p, TIP3P→water_tip3p
    # ------------------------------------------------------------------ #
    _WATER_MODEL_TO_SID = {"TIP4P": "water_tip4p", "TIP3P": "water_tip3p"}
    _default_water_sid  = _WATER_MODEL_TO_SID.get(water_model, "water_tip4p")
    water_sid    = recipe_cfg.get("water", {}).get("species_id") or _default_water_sid
    water_O_lbl: Optional[str] = None
    water_H_lbl: Optional[str] = None
    water_O_tid: Optional[int] = None
    water_H_tid: Optional[int] = None
    water_bond_type:  Optional[int] = None
    water_angle_type: Optional[int] = None
    water_OH_len:    float = 0.9572   # Å  default TIP4P-EW O-H bond
    water_HOH_ang:   float = 104.52   # deg

    if water_sid in species_db:
        wsp = species_db[water_sid]
        for a in wsp.atoms:
            if a.element == "O" and water_O_lbl is None:
                water_O_lbl = a.type_label
                water_O_tid = type_id_by_label.get(a.type_label)
            elif a.element == "H" and water_H_lbl is None:
                water_H_lbl = a.type_label
                water_H_tid = type_id_by_label.get(a.type_label)
        if wsp.bond_coeffs:
            bt = min(wsp.bond_coeffs.keys())
            water_bond_type = bt
            water_OH_len = float(wsp.bond_coeffs[bt][1])
        if wsp.angle_coeffs:
            at = min(wsp.angle_coeffs.keys())
            water_angle_type = at
            water_HOH_ang = float(wsp.angle_coeffs[at][1])

    # ------------------------------------------------------------------ #
    #  bjdisp DB
    # ------------------------------------------------------------------ #
    mm_bjdisp, qm_bjdisp, cfg_bjdisp = load_all(
        species_db=species_db,
        qm_params_dir=qm_params_dir,
        config_bjdisp=cfg.get("bjdisp"),
    )
    # Layer-file params override any generic qm_params entries with the same label.
    if extra_qm_bjdisp:
        qm_bjdisp = {**qm_bjdisp, **extra_qm_bjdisp}

    # ------------------------------------------------------------------ #
    #  MD settings from config
    # ------------------------------------------------------------------ #
    md_cfg        = ces2_cfg.get("md", {})
    timestep      = float(md_cfg.get("timestep_fs",  0.5))
    n_steps       = int(  md_cfg.get("n_steps",      0))
    thermo_every  = int(  md_cfg.get("thermo_every", 100))
    dump_every    = int(  md_cfg.get("dump_every",   1000))
    nvt_temp      = float(md_cfg.get("temperature",  300.0))
    nvt_tdamp     = float(md_cfg.get("t_damp_fs",    100.0))
    restart_every = int(  md_cfg.get("restart_every", 500000))
    shake_tol     = float(md_cfg.get("shake_tol",    1.0e-4))
    shake_iter    = int(  md_cfg.get("shake_iter",   20))
    shake_maxiter = int(  md_cfg.get("shake_maxiter", 500))
    _wall_buffer     = float(md_cfg.get("wall_buffer",      10.0))
    z_wall_hi        = float(md_cfg.get("z_wall_hi",    box.z_el_hi + _wall_buffer))
    prefix        = str(  ces2_cfg.get("prefix", "ces2"))

    # ------------------------------------------------------------------ #
    #  Cube index assignment for gridforce/net
    #  The grid command input order is:
    #    cube_coul_QM, cube_ind_QM, cube_QM_rho_hat[0], rho_hat[1], ...
    #  gridforce/net internally handles pot (cube 0) and ind (cube 1).
    #  The cubeID parameter in the fix command is a *relative* index into
    #  the rho_hat sub-array, NOT the absolute grid index.
    #  So H→0, O→1, Na→2, etc.  No offset needed.
    # ------------------------------------------------------------------ #
    _cube_offset = 0   # cubeID = relative index within rho_hat cubes
    cube_idx: Dict[str, int] = {}    # element_symbol → cube index
    if water_H_lbl:
        cube_idx["H"] = _cube_offset + 0
    if water_O_lbl:
        cube_idx["O"] = _cube_offset + 1
    next_cube = _cube_offset + 2
    for sid, _ in species_order:
        if sid == water_sid:
            continue
        sp = species_db[sid]
        for a in sp.atoms:
            el = a.element
            if el not in cube_idx:
                cube_idx[el] = next_cube
                next_cube += 1

    # ------------------------------------------------------------------ #
    #  Build file
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
    L("#")
    L(f"# Water model : {water_model}")
    L(f"# Supercell   : {rep[0]}x{rep[1]}x{rep[2]} → gridforce factor = {sc_factor}")
    L("#")
    L("# Runtime markers (replaced by qmmm_dftces2_charging_pts.sh):")
    L("#   ###qmxyz      → QM atom XYZ coordinates (in base.pw.in)")
    L("#   ###dispf      → dispersion forces       (in base.pw.in)")
    L("#   CUBEPOSITION  → replaced with  grid <cubes> ...  command")

    # ── Section 1: Initialization ────────────────────────────────────────
    section("1. Initialization")
    L()
    L("units           real")
    L("atom_style      full")
    L("boundary        p p f")
    L()
    L("dielectric      1")
    L("special_bonds   lj/coul 0.0 0.0 1.0   # 1-2 excluded, 1-3 excluded, 1-4 full")
    L()
    L("bond_style      harmonic")
    L("angle_style     harmonic")
    L("dihedral_style  none")
    L("improper_style  none")
    L()

    if use_tip4p:
        if water_O_tid is None or water_H_tid is None:
            L("# WARNING: water O or H type not found — TIP4P pair_style args may be wrong")
        O_tid = water_O_tid or 2
        H_tid = water_H_tid or 1
        bt    = water_bond_type  or 1
        at    = water_angle_type or 1
        L(f"# TIP4P-EW: O_type={O_tid} H_type={H_tid} bond_type={bt} angle_type={at}"
          f" M_dist={tip4p_msite} cutoff={lj_cut:.1f}")
        L(f"pair_style      hybrid/overlay lj/cut/tip4p/long/opt {O_tid} {H_tid} {bt} {at}"
          f" {tip4p_msite} {lj_cut:.1f} bjdisp {bjd_cutoff:.0f}")
        L(f"kspace_style    pppm/tip4p {kspace_acc:.1e}")
    else:
        L(f"pair_style      hybrid/overlay lj/cut/long {lj_cut:.1f} bjdisp {bjd_cutoff:.0f}")
        L(f"kspace_style    pppm {kspace_acc:.1e}")

    L(f"kspace_modify   slab {kspace_slab:.1f}   # 2D periodic slab correction")
    L()
    L("processors      * * 1   # no domain decomposition in z")

    # ── Section 2: Read data ─────────────────────────────────────────────
    section("2. Read data")
    L()
    L("read_data       data.file")

    # Optional: load pre-equilibrated positions+velocities from a dump file
    # (e.g. equilibrated.dump produced by md_relax).  Set ces2.initial_dump
    # in config to activate.  Timestep is taken from md_relax.equil_steps.
    initial_dump = str(ces2_cfg.get("initial_dump", "")).strip()
    if initial_dump:
        relax_cfg  = cfg.get("md_relax", {})
        dump_ts    = int(relax_cfg.get("equil_steps", 30000))
        L(f"read_dump       {initial_dump} {dump_ts} x y z vx vy vz box yes replace yes")

    L("reset_timestep  0")
    L()

    # Set TIP4P-Ew water charges — always override data.file values.
    # Reference: Horn et al., J. Chem. Phys. 120, 9665 (2004)
    # Safe TIP4P-Ew defaults; only accept species_db values that are
    # TIP4P-compatible (|q_O| > 1.0, q_H > 0.5) to guard against
    # accidentally reading TIP3P/SPC/SPC-E charges.
    _TIP4PEW_O, _TIP4PEW_H = -1.04844, 0.52422
    # Resolve TIP4P-Ew charges (used after group definitions below)
    _tip4p_charges = None
    if use_tip4p and water_O_tid is not None and water_H_tid is not None:
        o_charge, h_charge = _TIP4PEW_O, _TIP4PEW_H
        if water_sid in species_db:
            wsp = species_db[water_sid]
            db_o = next((a.charge for a in wsp.atoms if a.element == "O"), None)
            db_h = next((a.charge for a in wsp.atoms if a.element == "H"), None)
            if db_o is not None and abs(db_o) > 1.0:
                o_charge = db_o
            if db_h is not None and db_h > 0.5:
                h_charge = db_h
        _tip4p_charges = (h_charge, o_charge)

    # ── Section 3: Group definitions ─────────────────────────────────────
    section("3. Group definitions")

    L()
    L("# QM (slab) atoms = SOLUTE")
    if qm_labels_ordered:
        qm_tid_str = " ".join(str(qm_type_ids[lbl]) for lbl in qm_labels_ordered)
        qm_lbl_str = " ".join(qm_labels_ordered)
        L(f"group    SOLUTE   type {qm_tid_str}   # QM slab: {qm_lbl_str}")
    else:
        L("# WARNING: no QM type labels found")

    L()
    L("# Water sub-groups")
    if water_O_tid is not None:
        L(f"group    OXYGEN   type {water_O_tid}   # {water_O_lbl} (water O)")
    if water_H_tid is not None:
        L(f"group    PROTON   type {water_H_tid}   # {water_H_lbl} (water H)")
    if water_O_tid and water_H_tid:
        L("group    WATER    union OXYGEN PROTON")

    L()
    L("# Non-water species: one group per type_label (used by gridforce/net fixes)")
    L("# Polyatomic species (e.g. OH-) also get a species-level union group.")
    seen_lbl_grp: set = set()
    for sid, _ in species_order:
        if sid == water_sid:
            continue
        sp = species_db[sid]
        type_lbls_in_sp = sorted(set(a.type_label for a in sp.atoms),
                                  key=lambda l: type_id_by_label[l])
        grp_names_in_sp: List[str] = []
        for lbl in type_lbls_in_sp:
            grp = lbl.upper().replace("-", "_").replace(".", "_")[:12]
            grp_names_in_sp.append(grp)
            if lbl in seen_lbl_grp:
                continue            # already defined (type shared across species)
            seen_lbl_grp.add(lbl)
            tid = type_id_by_label[lbl]
            el  = next(a.element for a in sp.atoms if a.type_label == lbl)
            L(f"group    {grp:<12s} type {tid}   # {lbl} ({el})")
        # Polyatomic: add species-level union group for convenience
        if len(grp_names_in_sp) > 1:
            sp_grp = sid.upper().replace("-", "_")[:12]
            L(f"group    {sp_grp:<12s} union {' '.join(grp_names_in_sp)}   # all {sid}")

    L()
    L("group    SOLVENT  subtract all SOLUTE   # all MM atoms")

    # TIP4P-Ew water charges (must come AFTER group definitions)
    if _tip4p_charges is not None:
        h_charge, o_charge = _tip4p_charges
        L()
        L(f"# TIP4P-Ew water charges (overwrite data.file values)")
        L(f"set group PROTON charge {h_charge:.4f}")
        L(f"set group OXYGEN charge {o_charge:.4f}")

    # Region blocks for top/bottom surface layers (electrochemistry)
    # Use auto-calculated z_cutoff from charged_params when available;
    # otherwise fall back to manual region config.
    region_cfg = ces2_cfg.get("regions", {})

    if charged_params and charged_params.get("z_cutoff") is not None:
        # Auto-detected top layer: z_cutoff is the lower bound.
        # Upper bound = top_z_mean + 0.1 Å (small margin above highest atom).
        _z_cut = charged_params["z_cutoff"]
        _z_top_mean = charged_params.get("top_z_mean", _z_cut + 0.5)
        z_top_lo = _z_cut
        z_top_hi = _z_top_mean + 0.1
    else:
        z_top_lo = float(region_cfg.get("z_toplayer_lo", box.z_el_lo - 2.0))
        z_top_hi = float(region_cfg.get("z_toplayer_hi", box.z_el_lo + 0.5))

    z_cen_lo = float(region_cfg.get("z_centerlayer_lo", 0.0))
    z_cen_hi = float(region_cfg.get("z_centerlayer_hi", box.z_el_lo - 3.0))

    L()
    L("# Charged surface layer regions")
    L(f"region   toplayer    block 0 {box.Lx:.4f} 0 {box.Ly:.4f}"
      f" {z_top_lo:.4f} {z_top_hi:.4f} units box")
    L("group    top         region toplayer")
    L(f"region   centerlayer block 0 {box.Lx:.4f} 0 {box.Ly:.4f}"
      f" {z_cen_lo:.2f} {z_cen_hi:.2f} units box")
    L("group    center      region centerlayer")

    # ── set top-layer charge (charged system) ──────────────────────────────
    q_electrode = float(cfg.get("charge_control", {}).get("q_electrode_user_value", 0.0))
    if charged_params and q_electrode != 0.0:
        n_top = charged_params.get("n_top_atoms", 1)
        _dq_per_atom = q_electrode / n_top if n_top > 0 else 0.0
        L()
        L(f"# Charged system: distribute q_electrode={q_electrode:.6f} across {n_top} top-layer atoms")
        L(f"set group top charge {_dq_per_atom:.17g}")

    # ── Section 4: LJ pair_coeff ─────────────────────────────────────────
    section("4. LJ pair coefficients")
    L()
    lj_substyle = "lj/cut/tip4p/long/opt" if use_tip4p else "lj/cut/long"
    L(f"# Lorentz-Berthelot mixing; QM-involved pairs: epsilon=0")
    L()

    all_tids = sorted(type_id_by_label.values())
    missing_lj: List[str] = []

    for i in all_tids:
        lbl_i = label_by_type_id[i]
        is_qm_i = lbl_i in qm_type_ids
        for j in all_tids:
            if j < i:
                continue
            lbl_j = label_by_type_id[j]
            is_qm_j = lbl_j in qm_type_ids

            if is_qm_i or is_qm_j:
                L(f"pair_coeff  {i:3d} {j:3d}  {lj_substyle}  0.0000000  1.0000"
                  f"   # {lbl_i}-{lbl_j}  [QM: zero LJ]")
            else:
                eps_i, sig_i = _get_lj(lbl_i, lj_db)
                eps_j, sig_j = _get_lj(lbl_j, lj_db)
                for lbl in (lbl_i, lbl_j):
                    if lbl not in lj_db and lbl not in missing_lj:
                        missing_lj.append(lbl)
                eps_ij, sig_ij = _lb_mix(eps_i, sig_i, eps_j, sig_j)
                flag = "  [!NO DEFAULT]" if (lbl_i not in lj_db or lbl_j not in lj_db) else ""
                L(f"pair_coeff  {i:3d} {j:3d}  {lj_substyle}"
                  f"  {eps_ij:.7f}  {sig_ij:.5f}"
                  f"   # {lbl_i}-{lbl_j}{flag}")

    if missing_lj:
        L()
        L("# WARNING: no LJ defaults for these labels (fallback 0.0/1.0 used):")
        for lbl in missing_lj:
            L(f"#   {lbl}  → set in config.yaml [ces2][lj_params]")

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
        L("# (no QM type labels — bjdisp skipped)")

    # ── Section 6: Bond/angle coefficients ───────────────────────────────
    section("6. Bond / angle coefficients  (needed for TIP4P M-site)")
    L()
    L("# These mirror what is in data.file. Required here for TIP4P M-site computation.")
    for bt, (k, r0) in sorted(bond_coeffs.items()):
        L(f"bond_coeff      {bt}  {k:.1f}  {r0:.4f}")
    for at, (k, th) in sorted(angle_coeffs.items()):
        L(f"angle_coeff     {at}  {k:.1f}  {th:.2f}")

    # ── Section 7: Gridforce fixes + #CUBEPOSITION ────────────────────────
    section("7. CES2 gridforce/net fixes")
    L()
    L("# Format: fix NAME GROUP gridforce/net WEIGHT SC_FACTOR CUBE_IDX ELEMENT")
    L("#         [TIP4P O_type H_type bond_len angle M_dist O_cube_idx]")
    L(f"# Supercell factor = {rep[0]}×{rep[1]}×{rep[2]} = {sc_factor}")
    L()

    # Energy variable names for thermo_style
    egrid_vars: List[str] = []   # collect variable names

    # Water H (TIP4P)
    if water_H_tid is not None and water_O_tid is not None and use_tip4p:
        O_tid = water_O_tid
        H_tid = water_H_tid
        h_cube = cube_idx.get("H", 0)
        o_cube = cube_idx.get("O", 1)
        bt  = water_bond_type  or 1
        at  = water_angle_type or 1
        L(f"fix hGrid PROTON gridforce/net  -1  {sc_factor}  {h_cube}  H"
          f"  TIP4P {O_tid} {H_tid} {water_OH_len:.4f} {water_HOH_ang:.2f}"
          f" {tip4p_msite} {o_cube}")
        L(f"fix oGrid OXYGEN gridforce/net  -1  {sc_factor}  {o_cube}  O"
          f"  TIP4P {O_tid} {H_tid} {water_OH_len:.4f} {water_HOH_ang:.2f}"
          f" {tip4p_msite} {o_cube}")
        L("fix_modify hGrid energy yes")
        L("fix_modify oGrid energy yes")
        L()
        L("variable Egrid_H    equal f_hGrid[1]")
        L("variable EgridInd_H equal f_hGrid[2]")
        L("variable EgridRep_H equal f_hGrid[3]")
        L("variable Egrid_O    equal f_oGrid[1]")
        L("variable EgridInd_O equal f_oGrid[2]")
        L("variable EgridRep_O equal f_oGrid[3]")
        egrid_vars += ["v_Egrid_H", "v_EgridRep_H", "v_Egrid_O", "v_EgridRep_O",
                       "v_EgridInd_H", "v_EgridInd_O"]
    elif water_H_tid is not None and water_O_tid is not None:
        # TIP3P: simple gridforce
        h_cube = cube_idx.get("H", 0)
        o_cube = cube_idx.get("O", 1)
        L(f"fix hGrid PROTON gridforce/net  -1  {sc_factor}  {h_cube}  H")
        L(f"fix oGrid OXYGEN gridforce/net  -1  {sc_factor}  {o_cube}  O")
        L("fix_modify hGrid energy yes")
        L("fix_modify oGrid energy yes")
        L()
        L("variable Egrid_H    equal f_hGrid[1]")
        L("variable Egrid_O    equal f_oGrid[1]")
        egrid_vars += ["v_Egrid_H", "v_Egrid_O"]

    # Non-water gridforce fixes — one fix per unique type_label.
    # This correctly handles:
    #   - monoatomic ions  (K, Na, Cl …)       → one fix per species
    #   - polyatomic ions  (OH-, CO3-- …)      → one fix per type_label
    #     e.g. OH-: H_oh → HohGrid (cube 0), O_oh → OohGrid (cube 1)
    L()
    L("# Non-water species gridforce/net fixes (one per type_label)")
    seen_lbl_fix: set = set()
    for sid, _ in species_order:
        if sid == water_sid:
            continue
        sp = species_db[sid]
        for a in sp.atoms:
            lbl = a.type_label
            if lbl in seen_lbl_fix:
                continue
            seen_lbl_fix.add(lbl)
            el    = a.element
            c_idx = cube_idx.get(el, next_cube)
            grp   = lbl.upper().replace("-", "_").replace(".", "_")[:12]
            # fix name: strip non-alphanum chars, append "Grid"
            fix_tag  = re.sub(r"[^A-Za-z0-9]", "", lbl)[:8]
            fix_name = f"{fix_tag}Grid"
            # variable names: keep underscores, replace other non-alphanum
            var_sfx = re.sub(r"[^A-Za-z0-9_]", "_", lbl)
            var_e   = f"Egrid_{var_sfx}"
            var_ind = f"EgridInd_{var_sfx}"
            var_rep = f"EgridRep_{var_sfx}"
            L(f"fix {fix_name} {grp} gridforce/net  -1  {sc_factor}  {c_idx}  {el}")
            L(f"fix_modify {fix_name} energy yes")
            L(f"variable {var_e:<24s} equal f_{fix_name}[1]")
            L(f"variable {var_ind:<24s} equal f_{fix_name}[2]")
            L(f"variable {var_rep:<24s} equal f_{fix_name}[3]")
            egrid_vars += [f"v_{var_e}", f"v_{var_rep}", f"v_{var_ind}"]

    # #CUBEPOSITION marker
    L()
    L("# ------------------------------------------------------------------")
    L("# The qmmm wrapper replaces the marker below with grid command.")
    L("# Do NOT remove or rename this marker!")
    L("# ------------------------------------------------------------------")
    L()
    L("#CUBEPOSITION")

    # ── Section 8: MD settings ────────────────────────────────────────────
    section("8. MD settings")
    L()
    L(f"timestep        {timestep}")
    L()

    # Dispersion force compute + dump
    L("# Dispersion force tracking (QM←→MM via bjdisp)")
    L("compute fdisp SOLUTE force/tally SOLVENT")
    L("fix     showf all ave/atom 1 200 200 c_fdisp[*]")
    L(f"dump    dispTraj SOLUTE custom 100 dispf.ave id type xu yu zu f_showf[*]")
    L("dump_modify dispTraj sort id")
    L()
    L("# bjdisp energy (for thermo)")
    L("compute edisp all pair bjdisp")

    # Thermo — include ALL gridforce energy variables (water + all non-water types)
    L()
    L("thermo_style    custom step temp etotal ke pe evdwl ecoul elong press &")
    # Write gridforce variables 4 per continuation line for readability
    chunk_size = 4
    for i in range(0, len(egrid_vars), chunk_size):
        chunk = " ".join(egrid_vars[i : i + chunk_size])
        L(f"                {chunk} &")
    L("                c_edisp")
    L("thermo_modify   line multi format float %10.5f")
    L(f"thermo          {thermo_every}")
    L()

    # Wall, momentum, SHAKE, NVT
    L("# Keep solvent from escaping through fixed-z boundaries")
    L(f"fix   wallhi  SOLVENT wall/harmonic zhi {z_wall_hi:.2f} 1.0 1.0 5.0")
    L()
    L("# Remove spurious COM momentum drift")
    L("fix   momentum SOLVENT momentum 1 linear 1 1 0 angular")
    L()

    if water_O_tid is not None and water_H_tid is not None:
        b_str = f"b {water_bond_type}"  if water_bond_type  is not None else "b 1"
        a_str = f"a {water_angle_type}" if water_angle_type is not None else "a 1"
        L(f"# SHAKE: rigid O-H bonds and H-O-H angles for water")
        L(f"fix   shakeH SOLVENT shake {shake_tol} {shake_iter} {shake_maxiter}"
          f" {a_str} {b_str}")
        L()

    L(f"fix   nvt SOLVENT nvt temp {nvt_temp:.1f} {nvt_temp:.1f} {nvt_tdamp:.1f}")
    L()

    # Trajectory dump
    L(f"dump    traj all custom {dump_every} {prefix}.emd.lammpstrj"
      f" id type xu yu zu fx fy fz")
    L("dump_modify traj sort id")
    L()

    # Restart
    L(f"restart         {restart_every}  {prefix}.*.restart")
    L()
    L(f"run             {n_steps}")
    L()

    # ── Write ─────────────────────────────────────────────────────────────
    out_path = export_dir / "base.in.lammps"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[lammps_input_writer] Written: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Lightweight FF params for in.relax (pure MM, no QM, no gridforce)
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass

@_dataclass
class RelaxFFParams:
    """All force-field lines needed for a pure-MM in.relax script."""
    use_tip4p:           bool
    pair_style_line:     str          # full pair_style command (no bjdisp)
    kspace_line:         str          # kspace_style command
    lj_pair_coeff_lines: List[str]    # pair_coeff i j eps sig  (no substyle prefix)
    bond_coeff_lines:    List[str]
    angle_coeff_lines:   List[str]
    water_O_tid:         Optional[int]
    water_H_tid:         Optional[int]
    water_bond_type:     Optional[int]
    water_angle_type:    Optional[int]
    o_charge:            float
    h_charge:            float
    tip4p_msite:         float
    cutoff:              float


def collect_relax_ff_params(
    *,
    type_id_by_label:  Dict[str, int],
    label_by_type_id:  Dict[int, str],
    species_order:     List[Tuple[str, int]],
    species_db:        Dict[str, "Species"],
    bond_coeffs:       Dict[int, Tuple[float, float]],
    angle_coeffs:      Dict[int, Tuple[float, float]],
    cfg:               Dict[str, Any],
    relax_cutoff:      float = 10.0,
    relax_kspace_acc:  float = 1.0e-4,
) -> RelaxFFParams:
    """
    Collect a lightweight set of FF parameters for in.relax.
    Differences from base.in.lammps:
      - No bjdisp (QM-MM dispersion) — only lj/cut/tip4p/long
      - No gridforce/net
      - Shorter LJ cutoff (default 10 Å vs 12 Å)
      - Single pair_style (not hybrid/overlay) → no substyle prefix in pair_coeff
    """
    ces2_cfg   = cfg.get("ces2", {})
    recipe_cfg = cfg.get("electrolyte_recipe", {})

    water_model = str(ces2_cfg.get("water_model", "TIP4P")).upper()
    use_tip4p   = (water_model == "TIP4P")
    tip4p_msite = float(ces2_cfg.get("tip4p_msite", 0.125))

    # ---- LJ database ----
    lj_db: Dict[str, Tuple[float, float]] = dict(
        _DEFAULT_LJ_TIP4P if use_tip4p else _DEFAULT_LJ_TIP3P
    )
    lj_db.update(_DEFAULT_LJ_COMMON)
    for lbl, vals in ces2_cfg.get("lj_params", {}).items():
        lj_db[lbl] = (float(vals["epsilon"]), float(vals["sigma"]))

    # ---- Identify MM vs QM (slab) type labels ----
    mm_labels_set: set = set()
    for sid, _ in species_order:
        for a in species_db[sid].atoms:
            mm_labels_set.add(a.type_label)
    qm_type_ids = {lbl: tid for lbl, tid in type_id_by_label.items()
                   if lbl not in mm_labels_set}

    # ---- Water type IDs and charges ----
    # species_id is optional — auto-derived from water_model when not set.
    _WATER_MODEL_TO_SID = {"TIP4P": "water_tip4p", "TIP3P": "water_tip3p"}
    _default_water_sid  = _WATER_MODEL_TO_SID.get(water_model, "water_tip4p")
    water_sid      = recipe_cfg.get("water", {}).get("species_id") or _default_water_sid
    water_O_tid:   Optional[int] = None
    water_H_tid:   Optional[int] = None
    water_bond_type:  Optional[int] = None
    water_angle_type: Optional[int] = None
    o_charge = -1.04844   # TIP4P-Ew default (Horn et al. 2004)
    h_charge =  0.52422   # TIP4P-Ew default

    if water_sid in species_db:
        wsp = species_db[water_sid]
        for a in wsp.atoms:
            if a.element == "O" and water_O_tid is None:
                water_O_tid = type_id_by_label.get(a.type_label)
                # For TIP4P: only accept charge if TIP4P-compatible (|q_O| > 1.0)
                # to avoid accidentally using TIP3P/SPC/SPC-E values.
                if not use_tip4p or abs(a.charge) > 1.0:
                    o_charge = a.charge
            elif a.element == "H" and water_H_tid is None:
                water_H_tid = type_id_by_label.get(a.type_label)
                # For TIP4P: only accept charge if TIP4P-compatible (q_H > 0.5)
                if not use_tip4p or a.charge > 0.5:
                    h_charge = a.charge
        if wsp.bond_coeffs:
            water_bond_type  = min(wsp.bond_coeffs.keys())
        if wsp.angle_coeffs:
            water_angle_type = min(wsp.angle_coeffs.keys())

    # ---- pair_style and kspace (single style, no hybrid/overlay) ----
    if use_tip4p and water_O_tid and water_H_tid:
        O_tid = water_O_tid
        H_tid = water_H_tid
        bt    = water_bond_type  or 1
        at    = water_angle_type or 1
        pair_style_line = (
            f"pair_style      lj/cut/tip4p/long"
            f" {O_tid} {H_tid} {bt} {at} {tip4p_msite} {relax_cutoff:.1f}"
        )
        kspace_line = f"kspace_style    pppm/tip4p {relax_kspace_acc:.1e}"
    else:
        pair_style_line = f"pair_style      lj/cut/long {relax_cutoff:.1f}"
        kspace_line     = f"kspace_style    pppm {relax_kspace_acc:.1e}"

    # ---- pair_coeff lines (no substyle prefix — single pair_style) ----
    all_tids = sorted(type_id_by_label.values())
    lj_pair_coeff_lines: List[str] = []
    for i in all_tids:
        lbl_i   = label_by_type_id[i]
        is_qm_i = lbl_i in qm_type_ids
        for j in all_tids:
            if j < i:
                continue
            lbl_j   = label_by_type_id[j]
            is_qm_j = lbl_j in qm_type_ids
            if is_qm_i or is_qm_j:
                lj_pair_coeff_lines.append(
                    f"pair_coeff      {i:3d} {j:3d}  0.0000000  1.00000"
                    f"   # {lbl_i}-{lbl_j}  [QM frozen: zero LJ]"
                )
            else:
                eps_i, sig_i = _get_lj(lbl_i, lj_db)
                eps_j, sig_j = _get_lj(lbl_j, lj_db)
                eps_ij, sig_ij = _lb_mix(eps_i, sig_i, eps_j, sig_j)
                flag = "  [!NO DEFAULT]" if (lbl_i not in lj_db or lbl_j not in lj_db) else ""
                lj_pair_coeff_lines.append(
                    f"pair_coeff      {i:3d} {j:3d}  {eps_ij:.7f}  {sig_ij:.5f}"
                    f"   # {lbl_i}-{lbl_j}{flag}"
                )

    # ---- bond / angle coeff lines ----
    bond_coeff_lines  = [f"bond_coeff      {bt}  {k:.1f}  {r0:.4f}"
                         for bt, (k, r0) in sorted(bond_coeffs.items())]
    angle_coeff_lines = [f"angle_coeff     {at}  {k:.1f}  {th:.2f}"
                         for at, (k, th) in sorted(angle_coeffs.items())]

    return RelaxFFParams(
        use_tip4p=use_tip4p,
        pair_style_line=pair_style_line,
        kspace_line=kspace_line,
        lj_pair_coeff_lines=lj_pair_coeff_lines,
        bond_coeff_lines=bond_coeff_lines,
        angle_coeff_lines=angle_coeff_lines,
        water_O_tid=water_O_tid,
        water_H_tid=water_H_tid,
        water_bond_type=water_bond_type,
        water_angle_type=water_angle_type,
        o_charge=o_charge,
        h_charge=h_charge,
        tip4p_msite=tip4p_msite,
        cutoff=relax_cutoff,
    )


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
def print_lj_db(water_model: str = "TIP4P") -> None:
    db = _DEFAULT_LJ_TIP4P if water_model.upper() == "TIP4P" else _DEFAULT_LJ_TIP3P
    db = {**db, **_DEFAULT_LJ_COMMON}
    print(f"{'Label':<10}  {'epsilon':>10}  {'sigma':>8}  (water_model={water_model})")
    print("-" * 55)
    for lbl, (eps, sig) in sorted(db.items()):
        print(f"{lbl:<10}  {eps:10.7f}  {sig:8.5f}")
