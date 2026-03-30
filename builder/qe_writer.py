"""
qe_writer.py — Generate QE input files for DFT-CES2 QM/MM runs.

Files generated
---------------
  base.pw.in  — Quantum ESPRESSO SCF calculation (pw.x)
  base.pp.in  — Quantum ESPRESSO post-processing (pp.x → electrostatic potential cube)

Runtime markers (filled in by qmmm_dftces2_charging_pts.sh at each step)
-------------------------------------------------------------------------
  ###qmxyz   → replaced with QM atom cartesian positions (Å)
              placed after  ATOMIC_POSITIONS {angstrom}  card
  ###dispf   → replaced with QM dispersion forces (one per QM atom line)
              placed after  ATOMIC_FORCES  card

Unit-cell convention
--------------------
LAMMPS uses the full supercell (rep_x × rep_y × rep_z of the QM unit cell).
QE calculates one unit cell:
    a = Lx_slab / rep_x,   b = Ly_slab / rep_y,   c = Lz_slab / rep_z
where Lx_slab, Ly_slab, Lz_slab are the slab supercell cell parameters
(BEFORE z is expanded to include the electrolyte box).

nat in QE = n_qm_slab_total / (rep_x * rep_y * rep_z)

Pseudopotential config (config.yaml)
-------------------------------------
  qe:
    ecutwfc: 50.0                    # Ry  kinetic energy cutoff
    ecutrho: 400.0                   # Ry  density cutoff
    prefix:  "solute"
    outdir:  "./solute"
    pseudo_dir:  "./pseudo"           # directory with UPF files (relative to run dir)
    pseudo_set:  "sssp"              # built-in pseudopotential set to use as default
                                     # currently supported: "sssp" (SSSP PBE library)
    occupations: "smearing"          # or "fixed"
    smearing:    "mv"                # Methfessel-Paxton
    degauss:     0.02                # Ry
    k_points:    [1, 1, 1, 0, 0, 0] # Monkhorst-Pack grid + offsets
    emaxpos:     0.8                 # dipole correction (fraction of unit cell)
    edir:        3                   # z-direction dipole correction
    conv_thr:    1.0e-8              # SCF convergence (Ry)
    mixing_beta: 0.3
    electron_maxstep: 400
    diagonalization: "cg"
    pseudopotentials:                # optional per-element overrides (takes priority over pseudo_set)
      # Ir: "Ir_custom.UPF"         # uncomment to override an element
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

from .pseudo_db import resolve_pseudopotentials

# ---------------------------------------------------------------------------
# Atomic masses (standard atomic weight, a.u. / g/mol)
# ---------------------------------------------------------------------------
_ATOMIC_MASS: Dict[str, float] = {
    "H":  1.00794,  "He":  4.00260,
    "Li": 6.94100,  "Be":  9.01218,  "B":  10.8110,  "C":  12.0107,
    "N": 14.00670,  "O":  15.99940,  "F": 18.99840,  "Ne": 20.17970,
    "Na": 22.98977, "Mg": 24.30500,  "Al": 26.98154,  "Si": 28.08550,
    "P": 30.97376,  "S":  32.06500,  "Cl": 35.45300,  "Ar": 39.94800,
    "K": 39.09830,  "Ca": 40.07800,  "Sc": 44.95591,  "Ti": 47.86700,
    "V": 50.94150,  "Cr": 51.99610,  "Mn": 54.93805,  "Fe": 55.84500,
    "Co": 58.93320, "Ni": 58.69340,  "Cu": 63.54600,  "Zn": 65.38000,
    "Ga": 69.72300, "Ge": 72.64000,  "As": 74.92160,  "Se": 78.96000,
    "Br": 79.90400, "Kr": 83.79800,  "Rb": 85.46780,  "Sr": 87.62000,
    "Y": 88.90585,  "Zr": 91.22400,  "Nb": 92.90638,  "Mo": 95.96000,
    "Tc": 98.00000, "Ru": 101.0700,  "Rh": 102.9055,  "Pd": 106.4200,
    "Ag": 107.8682, "Cd": 112.4110,  "In": 114.8180,  "Sn": 118.7100,
    "Sb": 121.7600, "Te": 127.6000,  "I":  126.9045,  "Xe": 131.2930,
    "Cs": 132.9055, "Ba": 137.3270,
    "La": 138.9055, "Ce": 140.1160,  "Pr": 140.9077,  "Nd": 144.2420,
    "Pm": 145.0000, "Sm": 150.3600,  "Eu": 151.9640,  "Gd": 157.2500,
    "Tb": 158.9253, "Dy": 162.5000,  "Ho": 164.9303,  "Er": 167.2590,
    "Tm": 168.9342, "Yb": 173.0400,  "Lu": 174.9670,
    "Hf": 178.4900, "Ta": 180.9479,  "W":  183.8400,  "Re": 186.2070,
    "Os": 190.2300, "Ir": 192.2170,  "Pt": 195.0780,  "Au": 196.9665,
    "Hg": 200.5900, "Tl": 204.3833,  "Pb": 207.2000,  "Bi": 208.9804,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_qe_input(
    *,
    export_dir: Path,
    slab_cell: "np.ndarray",      # shape (3,3) — slab supercell cell matrix (Lx, Ly used for a,b)
    n_qm_total: int,              # total QM atoms in the *full* supercell
    qm_elements: List[str],       # unique element symbols in QM slab (ordered)
    box_z_total: float,           # full simulation box z span (zhi - zlo) from data.file
    cfg: Dict[str, Any],
) -> tuple:
    """
    Write base.pw.in and base.pp.in to export_dir.

    Parameters
    ----------
    export_dir   : directory to write output files
    slab_cell    : 3×3 cell matrix of the slab *supercell*.
                   Row 0 → a-vector (Lx), row 1 → b-vector (Ly) — used for QE a, b.
    n_qm_total   : total number of QM atoms in the supercell
    qm_elements  : unique element symbols appearing in QM slab, in desired ATOMIC_SPECIES order
    box_z_total  : full simulation box z span = z_hi - z_lo (must match data.file).
                   Used as QE c parameter so the electrostatic cell covers the entire box.
    cfg          : full config dict (reads cfg["qe"] and cfg["cell"])

    Returns
    -------
    (pw_path, pp_path) — Path objects of written files
    """
    import numpy as np

    cell_cfg = cfg.get("cell", {})
    qe_cfg   = cfg.get("qe",   {})

    rep = [int(x) for x in cell_cfg.get("supercell", [1, 1, 1])]
    sc_factor = rep[0] * rep[1] * rep[2]

    # ── Unit cell for QE ──────────────────────────────────────────────────
    # a, b : primitive cell lateral dimensions (supercell / rep)
    # c    : full simulation box z span (electrolyte + vacuum + buffer)
    #        so the QE cell matches the LAMMPS data.file box exactly.
    Lx_sc = float(slab_cell[0, 0])
    Ly_sc = float(slab_cell[1, 1])
    a_qe  = Lx_sc / rep[0]
    b_qe  = Ly_sc / rep[1]
    c_qe  = float(box_z_total)

    # ── Number of atoms in QE unit cell ───────────────────────────────────
    if n_qm_total % sc_factor != 0:
        print(f"[qe_writer] WARNING: n_qm_total={n_qm_total} not divisible by "
              f"sc_factor={sc_factor}. Rounding.")
    nat = n_qm_total // sc_factor
    ntyp = len(qm_elements)

    # ── QE parameters ─────────────────────────────────────────────────────
    ecutwfc     = float(qe_cfg.get("ecutwfc",     50.0))
    ecutrho     = float(qe_cfg.get("ecutrho",     400.0))
    prefix      = str(  qe_cfg.get("prefix",      "solute"))
    outdir      = str(  qe_cfg.get("outdir",      "./solute"))
    pseudo_dir  = str(  qe_cfg.get("pseudo_dir",  "./pseudo"))
    occupations = str(  qe_cfg.get("occupations", "smearing"))
    smearing    = str(  qe_cfg.get("smearing",    "mv"))
    degauss     = float(qe_cfg.get("degauss",     0.02))
    emaxpos     = float(qe_cfg.get("emaxpos",     0.8))
    edir        = int(  qe_cfg.get("edir",        3))
    conv_thr         = float(qe_cfg.get("conv_thr",          1.0e-8))
    mixing_beta      = float(qe_cfg.get("mixing_beta",       0.3))
    electron_maxstep = int(  qe_cfg.get("electron_maxstep",  400))
    diagonalization  = str(  qe_cfg.get("diagonalization",   "cg"))
    scf_must_converge = bool( qe_cfg.get("scf_must_converge", True))
    k_points    = list( qe_cfg.get("k_points",    [1, 1, 1, 0, 0, 0]))
    pseudo_set  = str(  qe_cfg.get("pseudo_set",  "sssp"))
    overrides: Dict[str, str] = qe_cfg.get("pseudopotentials", {})

    # ── Resolve pseudopotentials: built-in set + per-element overrides ────
    # Priority: qe.pseudopotentials (config) > pseudo_set built-in DB (default: sssp)
    pseudos = resolve_pseudopotentials(qm_elements, pseudo_set=pseudo_set, overrides=overrides)
    if pseudo_set == "sssp":
        print(f"[qe_writer] Using SSSP PBE pseudopotentials (override via qe.pseudopotentials)")

    # ── ATOMIC_SPECIES rows ───────────────────────────────────────────────
    atomic_species_lines: List[str] = []
    for el in qm_elements:
        mass = _ATOMIC_MASS.get(el, 0.0)
        if mass == 0.0:
            print(f"[qe_writer] WARNING: unknown atomic mass for element '{el}'")
        pseudo_file = pseudos[el]
        atomic_species_lines.append(f"  {el:<4s}  {mass:9.5f}  {pseudo_file}")

    # ======================================================================
    # base.pw.in
    # ======================================================================
    pw_lines: List[str] = []

    def PW(s: str = "") -> None:
        pw_lines.append(s)

    PW("! base.pw.in — DFT-CES2 QM/MM  (Quantum ESPRESSO SCF)")
    PW("! Auto-generated by cesbuild  (qe_writer.py)")
    PW("!")
    PW(f"! Supercell : {rep[0]}×{rep[1]}×{rep[2]}  →  QE unit cell a={a_qe:.4f}  b={b_qe:.4f}  c={c_qe:.4f} Å")
    PW(f"! nat (unit cell)  = {n_qm_total} / {sc_factor} = {nat}")
    PW("!")
    PW("! Runtime markers (replaced by qmmm_dftces2_charging_pts.sh):")
    PW("!   ###qmxyz  → QM atomic positions (one 'El x y z' line per QM atom)")
    PW("!   ###dispf  → dispersion forces   (one 'El fx fy fz' line per QM atom)")
    PW()

    # &CONTROL
    PW("&CONTROL")
    PW(f"  calculation = 'scf',")
    PW(f"  prefix      = '{prefix}',")
    PW(f"  outdir      = '{outdir}',")
    PW(f"  pseudo_dir  = '{pseudo_dir}',")
    PW( "  tprnfor     = .true.,")
    PW( "  tefield     = .true.,")
    PW( "  dipfield    = .true.,")
    PW("/")
    PW()

    # &SYSTEM
    PW("&SYSTEM")
    PW( "  ibrav     = 0,")
    PW(f"  nat       = {nat},")
    PW(f"  ntyp      = {ntyp},")
    PW(f"  ecutwfc   = {ecutwfc:.1f},")
    PW(f"  ecutrho   = {ecutrho:.1f},")
    PW(f"  occupations = '{occupations}',")
    if occupations == "smearing":
        PW(f"  smearing  = '{smearing}',")
        PW(f"  degauss   = {degauss},")
    PW(f"  edir      = {edir},")
    PW(f"  emaxpos   = {emaxpos},")
    PW("/")
    PW()

    # &ELECTRONS
    scf_conv_str = ".TRUE." if scf_must_converge else ".FALSE."
    PW("&ELECTRONS")
    PW(f"  conv_thr          = {conv_thr:.2e},")
    PW(f"  electron_maxstep  = {electron_maxstep},")
    PW(f"  mixing_beta       = {mixing_beta},")
    PW(f"  diagonalization   = '{diagonalization}',")
    PW(f"  scf_must_converge = {scf_conv_str},")
    PW( "  ! startingwfc / startingpot injected by qmmm script (= 'file' for step > 0)")
    PW("/")
    PW()

    # &IONS  (required by tprnfor even for scf)
    PW("&IONS")
    PW("/")
    PW()

    # ATOMIC_SPECIES
    PW("ATOMIC_SPECIES")
    for line in atomic_species_lines:
        PW(line)
    PW()

    # K_POINTS
    PW("K_POINTS {automatic}")
    k_str = "  ".join(str(int(k)) for k in k_points[:6])
    PW(f"  {k_str}")
    PW()

    # CELL_PARAMETERS
    PW("CELL_PARAMETERS {angstrom}")
    PW(f"  {a_qe:14.8f}   0.00000000   0.00000000")
    PW(f"   0.00000000  {b_qe:14.8f}   0.00000000")
    PW(f"   0.00000000   0.00000000  {c_qe:14.8f}")
    PW()

    # ATOMIC_POSITIONS — marker only (filled at runtime)
    PW("ATOMIC_POSITIONS {angstrom}")
    PW("###qmxyz")
    PW()

    # ATOMIC_FORCES — marker only (dispersion forces, filled at runtime)
    PW("ATOMIC_FORCES")
    PW("###dispf")
    PW()

    pw_path = export_dir / "base.pw.in"
    pw_path.write_text("\n".join(pw_lines) + "\n", encoding="utf-8")
    print(f"[qe_writer] Written: {pw_path}")

    # ======================================================================
    # base.pp.in  — electrostatic potential → solute.pot.cube
    # ======================================================================
    # QE pp.x is called twice by the wrapper:
    #   1. plot_num=11 → solute.pot.cube  (local potential, used as V_MM grid)
    #   2. plot_num=0  → solute.rho.cube  (electron density, optional diagnostic)
    # base.pp.in is for run #1 (plot_num=11).
    # ======================================================================
    pp_lines: List[str] = []

    def PP(s: str = "") -> None:
        pp_lines.append(s)

    PP("! base.pp.in — DFT-CES2 QM/MM  (Quantum ESPRESSO pp.x)")
    PP("! Auto-generated by cesbuild  (qe_writer.py)")
    PP("!")
    PP("! pp.x converts the SCF output to a cube file of the electrostatic")
    PP("! potential (plot_num=11 → solute.pot.cube).  The qmmm wrapper script")
    PP("! also generates solute.rho.cube (plot_num=0) for diagnostics.")
    PP()

    PP("&inputpp")
    PP(f"  prefix   = '{prefix}',")
    PP(f"  outdir   = '{outdir}',")
    PP( "  filplot  = 'solute.temp',")
    PP( "  plot_num = 11,")
    PP("/")
    PP()

    PP("&plot")
    PP("  nfile         = 1,")
    PP("  filepp(1)     = 'solute.temp',")
    PP("  weight(1)     = 1.0,")
    PP("  iflag         = 3,")
    PP("  output_format = 6,")
    PP("  fileout       = 'solute.pot.cube',")
    PP("/")

    pp_path = export_dir / "base.pp.in"
    pp_path.write_text("\n".join(pp_lines) + "\n", encoding="utf-8")
    print(f"[qe_writer] Written: {pp_path}")

    return pw_path, pp_path
