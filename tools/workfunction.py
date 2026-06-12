#!/usr/bin/env python3
"""
workfunction.py — DFT-CES2 Work Function & Electrode Voltage Analysis
=======================================================================
Parses QE output from a DFT-CES2 run directory and computes:
  • Work function  Φ = V_vac − E_Fermi
  • Absolute electrode potential  U_abs = Φ
  • Electrode voltage vs SHE  U_SHE = Φ − 4.44 V  (IUPAC reference)
  • Dipole correction amplitude (sawtooth V_amp, Ry)

Outputs
-------
  workfunction_summary.txt   — table of all key quantities
  workfunction_rawdata.csv   — z (Bohr), z (Ang), V (Ry), V (eV)  from pot.z.avg
  workfunction_profile.png   — planar-averaged potential plot
  workfunction_fermi.png     — QM Fermi-level convergence plot

Usage
-----
  python tools/workfunction.py [run_dir] [--cube PATH]

  # default: run_dir = current directory
  # If --cube is given, the script always planar-averages that cube and
  # overwrites pot.z.avg.  Otherwise, if pot.z.avg is missing in run_dir,
  # it automatically planar-averages total_pot_ortho.cube — the total
  # electrostatic potential
  # (QM solute V_H+V_bare + V_saw dipole correction + MM mobile-charge
  # potential, in Ry / electron convention) — along z and writes
  # pot.z.avg.  Use --cube to point at a specific cube file.

  # from repo root:
  python tools/workfunction.py test/04_Cs/run
  python tools/workfunction.py test/04_Cs/run --cube total_pot_ortho.cube

Cube-file visualization sub-command
-------------------------------------
  python tools/workfunction.py plot-cube FILE [FILE2 ...] [OPTIONS]

  Options:
    --axis {x,y,z,all}      Axis for planar average and 2D slice (default: z)
    --slice FRAC[,FRAC...]  Fractional cell position(s) for 2D slice
                            (0–1, default: 0.5; comma-separated for multiple)
    --unit {auto,ry,ha,ev,e_bohr3,e_ang3,raw}
                            Data unit — auto-detected from filename if omitted
    --no-avg                Skip planar-averaged profile plot
    --no-slice              Skip 2D slice heatmap plot
    --diff                  Plot difference (requires exactly 2 cube files):
                            FILE2 − FILE1.  Individual + difference profiles
                            are all overlaid on the same comparison figure.
    --output PREFIX         Output filename prefix (default: cubeplot_<stem>
                            placed next to the first cube file)

  Unit auto-detection heuristic (filename-based):
    *pot*, *v_*, *hartree*, *electro*  →  ry  (Ry → eV)
    *charge*, *density*, *rho*, *dens*, *ldos*  →  e_bohr3  (e/Bohr³ → e/Å³)
    otherwise  →  raw  (plotted as-is)

  Examples:
    # Potential cube — z planar average + XY slice at 50 % of cell
    python tools/workfunction.py plot-cube run/total_pot_ortho.cube

    # Charge density — all three axes, slices at 25 % and 75 %
    python tools/workfunction.py plot-cube run/density.cube \\
        --axis all --slice 0.25,0.75

    # Difference density — two cubes, z axis, slice at midpoint
    python tools/workfunction.py plot-cube run1/density.cube run2/density.cube \\
        --diff --axis z

    # Compare two potential cubes on the same 1D profile figure
    python tools/workfunction.py plot-cube run1/total_pot_ortho.cube \\
        run2/total_pot_ortho.cube --no-slice
"""

import sys
import os
import re
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime

# ── load shared plot style (tools/plot_setting.py) ──────────────────────────
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
try:
    import plot_setting as ps
    _C = ps.colors           # colour list
    _FS = ps.fontsize        # base font size
    _LS = ps.labelsize       # axis label size
    _LW = ps.linewidth       # base line width
    _FW, _FH = ps.figsize    # single-panel base size (3.5, 2.8)
except ImportError:
    ps = None
    _C  = ["#1A6FDF", "#F14040", "#37AD6B", "#B177DE", "#FEC211", "#515151"]
    _FS = 9;  _LS = 9;  _LW = 0.4;  _FW, _FH = 3.5, 2.8

# ─────────────────────────── constants ──────────────────────────────────────
RY2EV   = 13.6057039763  # 1 Ry in eV
BOHR2ANG = 0.529177210903  # 1 Bohr in Å
U_SHE_IUPAC    = 4.44   # V  (Li et al., Phys. Chem. Chem. Phys. 2015)
U_SHE_TRASATTI = 4.28   # V  (Trasatti, J. Electroanal. Chem. 1986)

# ─────────────────────────── helpers ────────────────────────────────────────

def parse_args(argv):
    """Parse: [run_dir] [--cube PATH]  (run_dir defaults to cwd)."""
    run_dir, cube = None, None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        elif a == "--cube" and i + 1 < len(argv):
            cube = argv[i + 1]; i += 2
        else:
            if run_dir is None:
                run_dir = a
            i += 1
    if run_dir is None:
        run_dir = os.getcwd()
    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: directory not found: {run_dir}")
    if cube and not os.path.isfile(cube):
        sys.exit(f"ERROR: cube file not found: {cube}")
    return os.path.abspath(run_dir), (os.path.abspath(cube) if cube else None)


# total_pot_ortho.cube = QM solute V_H+V_bare (ortho SCF, with MM Pauli
# repulsion via repA.cube) + V_saw (dipole correction) + (-2)·pot(MOBILE_final)
# (MM mobile-charge potential, Hartree→Ry and electron-sign converted).
# This is the full electrostatic potential an electron sees; its vacuum
# plateau is the correct V_vac for the work function.
CUBE_CANDIDATES = ("total_pot_ortho.cube",)


def load_cube(path):
    """Parse a Gaussian cube file.
    Returns: (origin_bohr, (nx,ny,nz), (vx,vy,vz) in Bohr, data grid[nx,ny,nz])."""
    with open(path) as f:
        f.readline(); f.readline()          # 2 comment lines
        parts  = f.readline().split()
        natoms = int(parts[0])
        origin = np.array([float(x) for x in parts[1:4]])

        def vox(line):
            p = line.split()
            return int(p[0]), np.array([float(x) for x in p[1:4]])

        nx, vx = vox(f.readline())
        ny, vy = vox(f.readline())
        nz, vz = vox(f.readline())
        for _ in range(abs(natoms)):
            f.readline()
        data = np.array(f.read().split(), dtype=np.float64)

    if data.size != nx * ny * nz:
        sys.exit(f"ERROR: cube data size mismatch. Expected {nx*ny*nz}, got {data.size}")
    return origin, (nx, ny, nz), (vx, vy, vz), data.reshape((nx, ny, nz))


def find_cube(run_dir):
    """Locate a suitable V_H+V_bare cube near run_dir."""
    for name in CUBE_CANDIDATES:
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    for qmdir in sorted(glob.glob(os.path.join(run_dir, "qm_*"))):
        for name in CUBE_CANDIDATES:
            p = os.path.join(qmdir, name)
            if os.path.isfile(p):
                return p
    return None


def cube_to_pot_z_avg(cube_path, out_path):
    """Planar-average a cube (Ry) along z and write pot.z.avg."""
    print(f"    Reading cube: {cube_path}")
    origin, (nx, ny, nz), (vx, vy, vz), grid = load_cube(cube_path)
    print(f"      grid = {nx}x{ny}x{nz},  dz = {vz[2]:.5f} Bohr")
    if abs(vz[0]) > 1e-6 or abs(vz[1]) > 1e-6 or abs(vx[2]) > 1e-6 or abs(vy[2]) > 1e-6:
        print("      WARNING: non-orthogonal z axis; planar average is approximate.")
    v_z    = grid.mean(axis=(0, 1))
    z_bohr = origin[2] + np.arange(nz) * vz[2]
    with open(out_path, "w") as f:
        f.write("# z (Bohr)        V (Ry, cube units)\n")
        for z, v in zip(z_bohr, v_z):
            f.write(f"{z:14.8f}  {v:18.10e}\n")
    print(f"      Wrote {out_path}  ({nz} z-points)")


def load_pot_z_avg(run_dir, cube_override=None):
    """Load pot.z.avg → arrays z_bohr, z_ang, v_ry, v_ev.
    If --cube PATH is given, always planar-average that cube and overwrite
    pot.z.avg in run_dir. Otherwise, if pot.z.avg is missing, planar-average
    total_pot_ortho.cube and write pot.z.avg in run_dir.
    """
    fpath = os.path.join(run_dir, "pot.z.avg")
    if cube_override:
        print("    --cube given — overwriting pot.z.avg from cube:")
        cube_to_pot_z_avg(cube_override, fpath)
    elif not os.path.isfile(fpath):
        cube_path = find_cube(run_dir)
        if not cube_path:
            sys.exit(
                f"ERROR: pot.z.avg not found in {run_dir}, and no cube file "
                f"({', '.join(CUBE_CANDIDATES)}) was found to derive it from.\n"
                f"       Pass --cube PATH to specify one explicitly."
            )
        print("    pot.z.avg missing — deriving from cube:")
        cube_to_pot_z_avg(cube_path, fpath)

    data = np.loadtxt(fpath, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        sys.exit("ERROR: unexpected format in pot.z.avg")
    z_bohr = data[:, 0]
    v_ry   = data[:, 1]
    z_ang  = z_bohr * BOHR2ANG
    v_ev   = v_ry   * RY2EV
    return z_bohr, z_ang, v_ry, v_ev


def parse_fermi_energies(run_dir):
    """Collect Fermi energies from qm_*/pw.out in iteration order."""
    pattern = os.path.join(run_dir, "qm_*", "pw.out")
    files   = sorted(glob.glob(pattern))
    results = []  # list of (label, E_fermi_eV)
    for fpath in files:
        label = os.path.basename(os.path.dirname(fpath))  # qm_0, qm_1, ...
        fermi = None
        with open(fpath) as f:
            for line in f:
                m = re.search(r"the Fermi energy is\s+([-\d.]+)\s*ev", line, re.I)
                if m:
                    fermi = float(m.group(1))
        if fermi is not None:
            results.append((label, fermi))
    if not results:
        sys.exit("ERROR: no Fermi energies found in qm_*/pw.out")
    return results


def parse_dipole_info(run_dir):
    """Parse dipole correction info from the last available pp.pot.out."""
    # Prefer main run dir, else most recent qm_N
    candidates = [os.path.join(run_dir, "pp.pot.out")] + \
        sorted(glob.glob(os.path.join(run_dir, "qm_*", "pp.pot.out")))[::-1]
    for fpath in candidates:
        if not os.path.isfile(fpath):
            continue
        info = {}
        with open(fpath) as f:
            for line in f:
                m = re.search(r"Dipole\s+([-\d.]+)\s+Ry au", line)
                if m:
                    info["dipole_ry"] = float(m.group(1))
                m = re.search(r"Dipole\s+[-\d.]+\s+Ry au,\s+([-\d.]+)\s+Debye", line)
                if m:
                    info["dipole_debye"] = float(m.group(1))
                m = re.search(r"Potential amp\.\s+([-\d.]+)\s+Ry", line)
                if m:
                    info["v_amp_ry"] = float(m.group(1))
                m = re.search(r"Total length\s+([-\d.]+)\s+bohr", line)
                if m:
                    info["total_length_bohr"] = float(m.group(1))
        if info:
            info["source"] = os.path.relpath(fpath, run_dir)
            return info
    return {}


def parse_cell_info(run_dir):
    """Get emaxpos and eopreg from the QE input (pw.in).

    emaxpos/eopreg define the dipole-correction sawtooth and MUST come from the
    actual QE input.  We read them from pw.in, preferring the real run input
    (``pw.in`` and the per-iteration ``qm_*/pw.in``) over the ``base.pw.in``
    template.  Only if no pw.in declares emaxpos do we fall back to QE defaults
    (emaxpos=0.5, eopreg=0.1) — and we warn loudly, because a wrong emaxpos
    puts the vacuum reference in the wrong region.
    """
    emaxpos = 0.5   # QE default when emaxpos is omitted from the input
    eopreg  = 0.1   # QE default sawtooth reset-region width (fractional)
    source  = None

    # Candidate inputs, in order of preference: the actual run input first,
    # the latest qm_*/pw.in next, then the base template as a last resort.
    candidates = [os.path.join(run_dir, "pw.in")]
    candidates += sorted(glob.glob(os.path.join(run_dir, "qm_*", "pw.in")))[::-1]
    candidates += [os.path.join(run_dir, "base.pw.in")]

    for pw_in in candidates:
        if not os.path.isfile(pw_in):
            continue
        with open(pw_in) as f:
            text = f.read()
        m = re.search(r"emaxpos\s*=\s*([\d.]+)", text)
        if not m:
            continue  # this pw.in has no emaxpos — keep looking
        emaxpos = float(m.group(1))
        me = re.search(r"eopreg\s*=\s*([\d.]+)", text)
        if me:
            eopreg = float(me.group(1))
        source = os.path.relpath(pw_in, run_dir)
        break

    if source is None:
        print(f"    WARNING: emaxpos not found in any pw.in under {run_dir}; "
              f"falling back to QE defaults emaxpos={emaxpos}, eopreg={eopreg}")
    else:
        print(f"    emaxpos/eopreg read from {source}")

    return {"emaxpos": emaxpos, "eopreg": eopreg, "source": source}


def determine_regions(run_dir, z_bohr):
    """Return region boundaries (Bohr) from build_summary or heuristic."""
    cell_z    = z_bohr[-1] + (z_bohr[1] - z_bohr[0])  # full cell length
    cell_info = parse_cell_info(run_dir)
    emaxpos   = cell_info["emaxpos"]
    eopreg    = cell_info["eopreg"]
    saw_peak  = emaxpos * cell_z                 # sawtooth discontinuity start
    saw_reset_end = (emaxpos + eopreg) * cell_z  # end of the reset region

    # Defaults (heuristic: look for large dip in potential)
    z_top_slab = z_bohr[0] + 25.0   # fallback
    z_el_hi    = saw_peak - 10.0     # fallback (leave a vacuum gap below saw)

    bsf = os.path.join(run_dir, "build_summary.json")
    if os.path.isfile(bsf):
        with open(bsf) as f:
            bs = json.load(f)
        box = bs.get("box", {})
        zt  = box.get("z_top_slab")  # Å
        zeh = box.get("z_el_hi")     # Å
        if zt  is not None: z_top_slab = zt  / BOHR2ANG
        if zeh is not None: z_el_hi    = zeh / BOHR2ANG

    return {
        "cell_z_bohr": cell_z,
        "emaxpos":     emaxpos,
        "eopreg":      eopreg,
        "saw_peak_bohr": saw_peak,
        "saw_reset_end_bohr": saw_reset_end,
        "z_slab_end_bohr": z_top_slab,
        "z_electrolyte_end_bohr": z_el_hi,
    }


def pick_vacuum_reference(z_bohr, v_ry, regions):
    """
    Return V_vac (Ry) from the flat vacuum plateau that lies BETWEEN the
    electrolyte/slab surface and the dipole sawtooth discontinuity (emaxpos).

    With QE's dipole correction (tefield/dipfield), the planar-averaged
    potential has a sawtooth discontinuity at emaxpos.  The TRUE vacuum the
    emitted electron sees is the flat plateau just OUTSIDE the surface and
    BEFORE that discontinuity (z in [electrolyte_end .. saw_peak]).

    The region AFTER emaxpos is NOT a usable vacuum reference: it is the
    artificial reset / wrap-around branch of the sawtooth (through PBC it
    belongs to the opposite face of the slab) and sits a full dipole step
    away.  Using it overestimates Φ by that dipole step.
    """
    saw_peak = regions["saw_peak_bohr"]
    z_el_end = regions["z_electrolyte_end_bohr"]

    # flat region: a few Bohr above the surface, a few Bohr below the sawtooth
    margin = 3.0  # Bohr — clears the surface ramp and the discontinuity edge
    lo = z_el_end + margin
    hi = saw_peak - margin
    mask = (z_bohr >= lo) & (z_bohr <= hi)
    if mask.sum() < 3:
        # Fallback: centred window in the electrolyte_end .. saw_peak gap
        mid = 0.5 * (z_el_end + saw_peak)
        lo, hi = mid - 3.0, mid + 3.0
        mask = (z_bohr >= lo) & (z_bohr <= hi)
    v_flat = v_ry[mask]
    return v_flat.mean(), v_flat.std(), lo, hi


# ─────────────────────────── plots ──────────────────────────────────────────

def _apply_minor_ticks(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="out")


def plot_potential(z_bohr, z_ang, v_ry, v_ev, regions, v_vac_ry,
                   v_vac_lo, v_vac_hi, fermi_ev, out_path):
    """Plot planar-averaged electrostatic potential with annotations."""
    cell_z    = regions["cell_z_bohr"]
    z_sl_end  = regions["z_slab_end_bohr"]
    z_el_end  = regions["z_electrolyte_end_bohr"]
    saw_peak  = regions["saw_peak_bohr"]
    saw_reset = regions.get("saw_reset_end_bohr", saw_peak)
    v_vac_ev  = v_vac_ry * RY2EV
    phi       = v_vac_ev - fermi_ev

    # colour aliases from plot_setting
    c_line   = _C[2]   # blue  – V(z) trace
    c_slab   = _C[1]   # red   – electrode shading
    c_elec   = _C[3]   # green – electrolyte shading
    c_vac    = _C[4]   # purple/yellow – vacuum shading
    c_vref   = _C[3]   # green – V_vac line
    c_efermi = _C[1]   # red   – E_Fermi line
    c_saw    = _C[5]   # grey  – emaxpos marker

    fig, axes = plt.subplots(
        1, 2,
        figsize=(_FW * 2.8, _FH * 1.8),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle("Planar-Averaged Total Electrostatic Potential  "
                 "(QM V$_H$+V$_{bare}$ + V$_{saw}$ + V$_{MM}$, ortho)",
                 fontsize=_FS, fontweight="bold", y=1.01)

    # ── Left panel: full z range ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(z_bohr, v_ev, color=c_line, lw=_LW * 2, label="V(z)")

    ax.axvspan(0,         z_sl_end,  alpha=0.12, color=c_slab, label="Electrode")
    ax.axvspan(z_sl_end,  z_el_end,  alpha=0.08, color=c_elec, label="Electrolyte")
    ax.axvspan(z_el_end,  saw_peak,  alpha=0.12, color=c_vac,  label="Vacuum (true)")
    ax.axvspan(saw_peak,  saw_reset, alpha=0.25, color=c_saw,  label="Sawtooth reset")
    ax.axvspan(saw_reset, cell_z,    alpha=0.06, color=c_saw,
               label="Wrap-around (artificial)")

    if ps:
        ps.draw_themed_line(v_vac_ev, ax, "horizontal")
        ps.draw_themed_line(fermi_ev, ax, "horizontal")
    ax.axhline(v_vac_ev, color=c_vref,   ls="--", lw=_LW * 2,
               label=f"V$_{{vac}}$ = {v_vac_ev:.4f} eV")
    ax.axhline(fermi_ev, color=c_efermi, ls="-.", lw=_LW * 2,
               label=f"E$_F$ = {fermi_ev:.4f} eV")
    ax.axvline(saw_peak, color=c_saw,    ls=":",  lw=_LW * 1.5, alpha=0.7,
               label=f"emaxpos ({saw_peak:.0f} Bohr)")

    ax.set_xlabel("z (Bohr)", fontsize=_LS)
    ax.set_ylabel("Electrostatic potential (eV)", fontsize=_LS)
    ax.set_xlim(z_bohr[0], z_bohr[-1])
    ax.set_title("Full cell", fontsize=_FS)
    ax.legend(fontsize=_FS - 1, loc="lower right", ncol=2)
    if ps:
        ps.set_ylim_top_margin(ax)
    _apply_minor_ticks(ax)

    ax2 = ax.twiny()
    ax2.set_xlim(z_ang[0], z_ang[-1])
    ax2.set_xlabel("z (Å)", fontsize=_LS - 1, color="gray")
    ax2.tick_params(axis="x", colors="gray", labelsize=_LS - 1)

    # ── Right panel: vacuum plateau zoom ────────────────────────────────────
    axr = axes[1]
    zoom_lo = max(0, z_el_end - 5)
    zoom_hi = cell_z
    mask_z  = (z_bohr >= zoom_lo) & (z_bohr <= zoom_hi)

    axr.plot(z_bohr[mask_z], v_ev[mask_z], color=c_line,
             lw=_LW * 2, marker="o", ms=2, label="V(z)")
    axr.axhline(v_vac_ev, color=c_vref,   ls="--", lw=_LW * 2,
                label=f"V$_{{vac}}$ = {v_vac_ev:.4f} eV")
    axr.axhline(fermi_ev, color=c_efermi, ls="-.", lw=_LW * 2,
                label=f"E$_F$ = {fermi_ev:.4f} eV")
    axr.axvspan(z_el_end, saw_peak,  alpha=0.12, color=c_vac)
    axr.axvspan(saw_peak, saw_reset, alpha=0.25, color=c_saw)
    axr.axvline(saw_peak, color=c_saw, ls=":", lw=_LW * 1.5, alpha=0.7)
    axr.axvspan(v_vac_lo, v_vac_hi, alpha=0.25, color=c_vref,
                label="Reference plateau (pre-sawtooth)")

    # Φ double-arrow annotation
    mid_y    = (v_vac_ev + fermi_ev) / 2
    arr_x    = zoom_lo + (zoom_hi - zoom_lo) * 0.80
    axr.annotate("", xy=(arr_x, fermi_ev), xytext=(arr_x, v_vac_ev),
                 arrowprops=dict(arrowstyle="<->", color="black",
                                 lw=_LW * 2))
    axr.text(arr_x + (zoom_hi - zoom_lo) * 0.03, mid_y,
             f"Φ = {phi:.3f} eV",
             va="center", ha="left", fontsize=_FS - 1, color="black",
             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                       ec="none", alpha=0.85))

    axr.set_xlabel("z (Bohr)", fontsize=_LS)
    axr.set_ylabel("Potential (eV)", fontsize=_LS)
    axr.set_xlim(zoom_lo, zoom_hi + (zoom_hi - zoom_lo) * 0.15)
    axr.set_title("Vacuum region (zoom)", fontsize=_FS)
    axr.legend(fontsize=_FS - 1, loc="lower left")
    if ps:
        ps.set_ylim_top_margin(axr)
    _apply_minor_ticks(axr)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_fermi_convergence(fermi_data, out_path):
    """Plot Fermi energy vs QM iteration."""
    labels = [d[0] for d in fermi_data]
    values = [d[1] for d in fermi_data]
    x      = np.arange(len(labels))

    c_line = _C[1]   # red – matches electrode colour in profile plot

    if ps:
        _plt = ps.pretty_plot(width=_FW * 1.6, height=_FH * 1.4)
        fig  = plt.gcf()
        ax   = fig.axes[0]
    else:
        fig, ax = plt.subplots(figsize=(_FW * 1.6, _FH * 1.4))

    ax.plot(x, values, "o-", color=c_line, lw=_LW * 2,
            ms=plt.rcParams.get("lines.markersize", 4) * 1.5,
            markeredgecolor=c_line, label="E$_F$")

    if ps:
        ps.draw_themed_line(values[-1], ax, "horizontal")
    ax.axhline(values[-1], color=_C[5], ls="--", lw=_LW * 1.5,
               label=f"Converged: {values[-1]:.4f} eV")

    # Point annotations
    for xi, (lbl, val) in zip(x, fermi_data):
        ax.annotate(f"{val:.4f}", xy=(xi, val),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=_FS - 1, color=c_line)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_LS)
    ax.set_xlabel("QM/MM iteration", fontsize=_LS)
    ax.set_ylabel("Fermi energy (eV)", fontsize=_LS)
    ax.set_title("QM Fermi Level Convergence", fontsize=_FS, fontweight="bold")
    ax.legend(fontsize=_FS)
    if ps:
        ps.set_ylim_top_margin(ax)
    _apply_minor_ticks(ax)

    # Δ label
    if len(values) >= 2:
        delta = abs(values[-1] - values[-2])
        ax.text(0.97, 0.04,
                f"Δ(last step) = {delta * 1000:.1f} meV",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_FS - 1, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────── output files ───────────────────────────────────

def write_raw_data(z_bohr, z_ang, v_ry, v_ev, out_path):
    """Write pot.z.avg data as well-formatted CSV."""
    header = "z_bohr,z_ang,V_ry,V_ev\n"
    with open(out_path, "w") as f:
        f.write(header)
        for zb, za, vr, ve in zip(z_bohr, z_ang, v_ry, v_ev):
            f.write(f"{zb:.6f},{za:.6f},{vr:.8f},{ve:.8f}\n")
    print(f"  Saved: {out_path}")


def write_summary_table(run_dir, fermi_data, regions, v_vac_ry, v_vac_std_ry,
                         dipole_info, out_path):
    """Write a human-readable summary table as a plain text file."""
    fermi_ev    = fermi_data[-1][1]
    v_vac_ev    = v_vac_ry * RY2EV
    phi         = v_vac_ev - fermi_ev
    u_abs       = phi
    u_she_iupac = phi - U_SHE_IUPAC
    u_she_tras  = phi - U_SHE_TRASATTI

    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    cell_z = regions["cell_z_bohr"]

    lines = []
    sep   = "─" * 62

    def row(item, value, unit="", note=""):
        note_str = f"  # {note}" if note else ""
        return f"  {item:<36s}  {value:>14s}  {unit:<6s}{note_str}"

    lines += [
        "=" * 62,
        "  DFT-CES2  Work Function & Electrode Voltage Summary",
        f"  Run dir : {os.path.relpath(run_dir)}",
        f"  Date    : {now}",
        "=" * 62,
        "",
        "── Cell & Dipole Correction ─────────────────────────────────",
        row("Cell z-length",        f"{cell_z:.4f}",           "Bohr"),
        row("Cell z-length",        f"{cell_z*BOHR2ANG:.4f}",  "Å"),
        row("emaxpos",              f"{regions['emaxpos']:.2f}",   ""),
        row("Sawtooth peak (z)",    f"{regions['saw_peak_bohr']:.4f}", "Bohr"),
    ]
    if dipole_info:
        v_amp_ry = dipole_info.get("v_amp_ry", float("nan"))
        lines += [
            row("Dipole correction  V_amp",
                f"{v_amp_ry:.4f}", "Ry",
                f"{v_amp_ry*RY2EV:.4f} eV"),
            row("Dipole (last run)",
                f"{dipole_info.get('dipole_ry', float('nan')):.4f}", "Ry au",
                f"{dipole_info.get('dipole_debye', float('nan')):.4f} Debye"),
        ]

    lines += [
        "",
        "── QM Fermi Level Convergence ───────────────────────────────",
    ]
    for label, ef in fermi_data:
        lines.append(row(f"  E_Fermi  ({label})", f"{ef:.4f}", "eV"))
    if len(fermi_data) >= 2:
        delta = abs(fermi_data[-1][1] - fermi_data[-2][1])
        lines.append(row("  Δ E_Fermi (last step)", f"{delta*1000:.2f}", "meV"))

    lines += [
        "",
        "── Electrostatic Potential Reference ────────────────────────",
        row("Vacuum plateau range",
            f"{regions['z_electrolyte_end_bohr']+3:.1f}–"
            f"{regions['saw_peak_bohr']-3:.1f}", "Bohr",
            "pre-sawtooth (electrolyte_end .. emaxpos)"),
        row("V_vac (mean)",   f"{v_vac_ry:.6f}", "Ry", f"{v_vac_ev:.6f} eV"),
        row("V_vac (std)",    f"{v_vac_std_ry:.6f}", "Ry",
            f"{v_vac_std_ry*RY2EV*1000:.2f} meV"),
        "",
        sep,
        "  RESULTS",
        sep,
        row("E_Fermi  (converged, last run)",  f"{fermi_ev:.4f}", "eV"),
        row("V_vac  (vacuum plateau)",          f"{v_vac_ev:.4f}", "eV"),
        row("Work function  Φ = V_vac − E_F",  f"{phi:.4f}",     "eV"),
        "",
        row("Absolute electrode potential",     f"{u_abs:.4f}",   "V"),
        row("U vs SHE  [ref 4.44 V, IUPAC]",  f"{u_she_iupac:+.4f}", "V_SHE"),
        row("U vs SHE  [ref 4.28 V, Trasatti]",f"{u_she_tras:+.4f}", "V_SHE"),
        sep,
        "",
        "Note: pot.z.avg = planar average of total_pot_ortho.cube",
        "      = QM V_H+V_bare (ortho SCF) + V_saw + MM mobile-charge potential.",
        "      V_vac is taken from the flat vacuum plateau BEFORE the sawtooth",
        "      discontinuity (electrolyte surface .. emaxpos).  The plateau",
        "      after emaxpos is the artificial reset/wrap-around branch and is",
        "      a full dipole step away -- it must NOT be used for Phi.",
        "      E_F is QE's Fermi level (pw.out), on the SAME electrostatic-",
        "      potential zero as the cube, so Phi = V_vac - E_F is well-defined.",
        "      SHE reference: Li et al. PCCP 17, 4647 (2015)  / Trasatti 1986.",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ══════════════════ cube-file visualization (plot-cube) ══════════════════════

# axis index map and the two complementary axes for each fixed axis
_AXIS_IDX  = {"x": 0, "y": 1, "z": 2}
# fix_ax → (h_ax_idx, v_ax_idx, h_name, v_name)
_AXIS_PAIR = {
    0: (1, 2, "y", "z"),   # fix x → horizontal=y, vertical=z
    1: (0, 2, "x", "z"),   # fix y → horizontal=x, vertical=z
    2: (0, 1, "x", "y"),   # fix z → horizontal=x, vertical=y
}


def _detect_cube_unit(path):
    """Heuristic: infer data unit from the cube filename."""
    name = os.path.basename(path).lower()
    if any(x in name for x in ("pot", "v_", "vkohn", "vxc", "hartree", "electro")):
        return "ry"
    if any(x in name for x in ("charge", "density", "rho", "dens", "ldos", "parchg")):
        return "e_bohr3"
    return "raw"


def _unit_display_label(unit):
    return {
        "ry":      "Potential (eV)",
        "ha":      "Potential (eV)",
        "ev":      "Potential (eV)",
        "e_bohr3": "Density (e/Å³)",
        "e_ang3":  "Density (e/Å³)",
        "raw":     "Value (a.u.)",
    }.get(unit, "Value (a.u.)")


def _convert_cube_values(values, unit):
    """Convert raw cube data to display units."""
    if unit == "ry":
        return values * RY2EV
    if unit == "ha":
        return values * RY2EV * 2.0
    if unit == "e_bohr3":
        return values / BOHR2ANG ** 3
    return values   # ev, e_ang3, raw: no conversion


def _axis_coords(origin, nvox, vvec, ai):
    """1D coordinate arrays (Bohr and Å) along axis index ai."""
    n     = nvox[ai]
    step  = float(np.linalg.norm(vvec[ai]))
    z_b   = origin[ai] + np.arange(n) * step
    return z_b, z_b * BOHR2ANG


def _cube_planar_avg(grid, ai):
    """Planar average of grid (nx,ny,nz) along axis ai; returns 1-D array."""
    other = tuple(j for j in range(3) if j != ai)
    return grid.mean(axis=other)


def _cube_slice_2d(grid, fix_ai, frac):
    """
    Extract a 2-D slice through grid by fixing axis fix_ai at fractional
    position frac (0–1).

    Returns
    -------
    sl_for_imshow : ndarray shape (n_v, n_h)
        Transposed so that imshow(origin='lower') gives
        h_name on x-axis and v_name on y-axis.
    h_ai, v_ai : int — horizontal / vertical axis indices
    h_name, v_name : str — axis labels
    idx : int — voxel index used for the slice
    """
    n   = grid.shape[fix_ai]
    idx = max(0, min(n - 1, int(round(frac * n))))
    h_ai, v_ai, h_name, v_name = _AXIS_PAIR[fix_ai]
    if fix_ai == 0:
        sl = grid[idx, :, :]   # shape (ny, nz)
    elif fix_ai == 1:
        sl = grid[:, idx, :]   # shape (nx, nz)
    else:
        sl = grid[:, :, idx]   # shape (nx, ny)
    # sl shape is (n_h, n_v) — transpose for imshow
    return sl.T, h_ai, v_ai, h_name, v_name, idx


def _imshow_extent_ang(origin, nvox, vvec, h_ai, v_ai):
    """[xmin, xmax, ymin, ymax] in Å for imshow extent."""
    def span(ai):
        step = float(np.linalg.norm(vvec[ai])) * BOHR2ANG
        o    = origin[ai] * BOHR2ANG
        return o, o + nvox[ai] * step
    x0, x1 = span(h_ai)
    y0, y1 = span(v_ai)
    return [x0, x1, y0, y1]


# ── cube-plot: planar-averaged 1-D profiles ──────────────────────────────────

def plot_cube_avg_profiles(entries, axis_str, out_prefix):
    """
    Plot planar-averaged 1-D profiles for one or more cube entries.

    Parameters
    ----------
    entries   : list of dicts with keys label, origin, nvox, vvec, grid, unit
    axis_str  : "x" | "y" | "z" | "all"
    out_prefix: output filename stem (axis name appended automatically)
    """
    axes_to_plot = ["x", "y", "z"] if axis_str == "all" else [axis_str]

    for ax_name in axes_to_plot:
        ai = _AXIS_IDX[ax_name]

        fig, ax = plt.subplots(figsize=(_FW * 2.2, _FH * 1.5))

        for k, entry in enumerate(entries):
            _, z_ang  = _axis_coords(entry["origin"], entry["nvox"],
                                     entry["vvec"], ai)
            avg       = _cube_planar_avg(entry["grid"], ai)
            avg_disp  = _convert_cube_values(avg, entry["unit"])
            ax.plot(z_ang, avg_disp, color=_C[k % len(_C)],
                    lw=_LW * 2, label=entry["label"])

        unit_lbl = _unit_display_label(entries[0]["unit"])
        ax.set_xlabel(f"{ax_name} (Å)", fontsize=_LS)
        ax.set_ylabel(unit_lbl, fontsize=_LS)
        ax.set_title(f"Planar-Averaged Profile  ({ax_name}-axis)",
                     fontsize=_FS, fontweight="bold")
        if len(entries) > 1:
            ax.legend(fontsize=_FS - 1, loc="best")
        if ps:
            ps.set_ylim_top_margin(ax)
        _apply_minor_ticks(ax)
        fig.tight_layout()

        out = f"{out_prefix}_avg_{ax_name}.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


# ── cube-plot: 2-D slice heatmap ─────────────────────────────────────────────

def plot_cube_slice_2d(entry, fix_axis_name, frac, out_prefix):
    """
    Plot a 2-D heatmap slice through a cube.

    Parameters
    ----------
    entry          : dict with label, origin, nvox, vvec, grid, unit
    fix_axis_name  : "x" | "y" | "z"
    frac           : float, fractional position along fix axis (0–1)
    out_prefix     : output filename stem
    """
    fix_ai = _AXIS_IDX[fix_axis_name]
    sl, h_ai, v_ai, h_name, v_name, idx = _cube_slice_2d(
        entry["grid"], fix_ai, frac)

    unit     = entry["unit"]
    sl_disp  = _convert_cube_values(sl, unit)
    extent   = _imshow_extent_ang(entry["origin"], entry["nvox"],
                                  entry["vvec"], h_ai, v_ai)

    # colour map & symmetric limits for potential; sequential for density
    is_pot  = unit in ("ry", "ha", "ev")
    cmap    = "RdBu_r" if is_pot else "viridis"
    if is_pot:
        vabs = float(np.percentile(np.abs(sl_disp), 98))
        vmin, vmax = -vabs, vabs
    else:
        vmin = float(np.percentile(sl_disp, 2))
        vmax = float(np.percentile(sl_disp, 98))

    # figure size proportional to physical cell dimensions
    w_ang = extent[1] - extent[0]
    h_ang = extent[3] - extent[2]
    ratio = w_ang / max(h_ang, 1e-6)
    fw    = min(_FW * 3.0, _FW * 1.8 * max(1.0, ratio))
    fh    = _FH * 1.8

    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.imshow(sl_disp, origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(_unit_display_label(unit), fontsize=_LS - 1)

    # annotate with actual position of the slice
    fix_coords_b, fix_coords_a = _axis_coords(
        entry["origin"], entry["nvox"], entry["vvec"], fix_ai)
    fix_pos_ang = float(fix_coords_a[idx]) if idx < len(fix_coords_a) else 0.0
    frac_pct    = int(round(frac * 100))

    ax.set_xlabel(f"{h_name} (Å)", fontsize=_LS)
    ax.set_ylabel(f"{v_name} (Å)", fontsize=_LS)
    ax.set_title(
        f"{entry['label']}  |  "
        f"{fix_axis_name} = {fix_pos_ang:.2f} Å  ({frac_pct} % of cell)",
        fontsize=_FS, fontweight="bold")

    fig.tight_layout()
    out = f"{out_prefix}_slice_{fix_axis_name}{frac_pct:03d}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── cube-plot: CLI parsing and orchestration ──────────────────────────────────

def parse_cube_plot_args(argv):
    """
    Parse CLI args for the 'plot-cube' sub-command.
    argv is everything that follows 'plot-cube' in sys.argv.
    """
    files       = []
    axis        = "z"
    slice_fracs = [0.5]
    unit        = "auto"
    no_avg      = False
    no_slice    = False
    diff        = False
    out_prefix  = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        elif a == "--axis" and i + 1 < len(argv):
            axis = argv[i + 1]; i += 2
        elif a == "--slice" and i + 1 < len(argv):
            try:
                slice_fracs = [float(x) for x in argv[i + 1].split(",")]
            except ValueError:
                sys.exit(f"ERROR: --slice value must be float(s), got: {argv[i+1]}")
            i += 2
        elif a == "--unit" and i + 1 < len(argv):
            unit = argv[i + 1]; i += 2
        elif a == "--no-avg":
            no_avg = True; i += 1
        elif a == "--no-slice":
            no_slice = True; i += 1
        elif a == "--diff":
            diff = True; i += 1
        elif a in ("--output", "-o") and i + 1 < len(argv):
            out_prefix = argv[i + 1]; i += 2
        elif not a.startswith("--"):
            files.append(os.path.abspath(a)); i += 1
        else:
            print(f"WARNING: unknown option ignored: {a}"); i += 1

    if not files:
        sys.exit("ERROR: plot-cube requires at least one cube file.\n"
                 "Usage: python tools/workfunction.py plot-cube FILE [FILE2 ...]")
    for f in files:
        if not os.path.isfile(f):
            sys.exit(f"ERROR: cube file not found: {f}")
    if diff and len(files) != 2:
        sys.exit("ERROR: --diff requires exactly 2 cube files.")
    valid_axes = ("x", "y", "z", "all")
    if axis not in valid_axes:
        sys.exit(f"ERROR: --axis must be one of {valid_axes}, got: '{axis}'")
    for frac in slice_fracs:
        if not (0.0 <= frac <= 1.0):
            sys.exit(f"ERROR: --slice values must be in [0, 1], got: {frac}")

    if out_prefix is None:
        stem       = os.path.splitext(os.path.basename(files[0]))[0]
        out_prefix = os.path.join(os.path.dirname(files[0]),
                                  "cubeplot_" + stem)

    return dict(files=files, axis=axis, slice_fracs=slice_fracs, unit=unit,
                no_avg=no_avg, no_slice=no_slice, diff=diff,
                out_prefix=out_prefix)


def main_plot_cube(argv_after):
    """Entry-point for the 'plot-cube' sub-command."""
    cfg = parse_cube_plot_args(argv_after)

    files    = cfg["files"]
    unit_arg = cfg["unit"]
    out_pfx  = cfg["out_prefix"]

    print(f"\n{'='*60}")
    print(f"  workfunction.py  →  plot-cube mode")
    print(f"  Output prefix    :  {out_pfx}")
    print(f"  Axis             :  {cfg['axis']}")
    print(f"  Slice position(s):  {cfg['slice_fracs']}")
    print(f"{'='*60}")

    # ── load cubes ────────────────────────────────────────────────────────
    entries = []
    for path in files:
        unit = unit_arg if unit_arg != "auto" else _detect_cube_unit(path)
        print(f"\n  Loading: {os.path.basename(path)}  →  unit: {unit}")
        origin, nvox, vvec, grid = load_cube(path)
        cell_ang = [nvox[i] * float(np.linalg.norm(vvec[i])) * BOHR2ANG
                    for i in range(3)]
        print(f"    grid : {nvox[0]} × {nvox[1]} × {nvox[2]}")
        print(f"    cell : {cell_ang[0]:.3f} × {cell_ang[1]:.3f} × "
              f"{cell_ang[2]:.3f} Å")
        print(f"    range: {_convert_cube_values(grid.min(), unit):.4g}"
              f" – {_convert_cube_values(grid.max(), unit):.4g}"
              f"  [{_unit_display_label(unit)}]")
        entries.append(dict(
            path=path,
            label=os.path.splitext(os.path.basename(path))[0],
            origin=origin, nvox=nvox, vvec=vvec, grid=grid, unit=unit,
        ))

    # ── difference mode ───────────────────────────────────────────────────
    if cfg["diff"]:
        e0, e1 = entries[0], entries[1]
        if e0["nvox"] != e1["nvox"]:
            sys.exit("ERROR: --diff requires cubes with identical grid dimensions.\n"
                     f"  {e0['label']} : {e0['nvox']}\n"
                     f"  {e1['label']} : {e1['nvox']}")
        diff_grid  = e1["grid"] - e0["grid"]
        diff_label = f"Δ ({e1['label']} − {e0['label']})"
        diff_entry = dict(
            path=None, label=diff_label,
            origin=e0["origin"], nvox=e0["nvox"], vvec=e0["vvec"],
            grid=diff_grid, unit=e0["unit"],
        )
        # For avg: show all three on one comparison figure
        entries_for_avg   = entries + [diff_entry]
        # For slices: only the difference
        entries_for_slice = [diff_entry]
        out_pfx_diff = out_pfx + "_diff"
        print(f"\n  Difference cube  :  {diff_label}")
        dv = _convert_cube_values(diff_grid, e0["unit"])
        print(f"    range: {dv.min():.4g} – {dv.max():.4g}"
              f"  [{_unit_display_label(e0['unit'])}]")
    else:
        entries_for_avg   = entries
        entries_for_slice = entries
        out_pfx_diff      = out_pfx

    # ── planar-averaged profiles ──────────────────────────────────────────
    if not cfg["no_avg"]:
        print(f"\n[1] Planar-averaged profile(s)  (axis = {cfg['axis']}) ...")
        if cfg["diff"]:
            # One comparison figure: both inputs + difference
            plot_cube_avg_profiles(entries_for_avg, cfg["axis"],
                                   out_pfx + "_comparison")
        else:
            plot_cube_avg_profiles(entries, cfg["axis"], out_pfx)

    # ── 2-D slice heatmaps ────────────────────────────────────────────────
    if not cfg["no_slice"]:
        axes_to_slice = (["x", "y", "z"] if cfg["axis"] == "all"
                         else [cfg["axis"]])
        print(f"\n[2] 2-D slice heatmap(s)  "
              f"(fix axis = {axes_to_slice}, "
              f"frac = {cfg['slice_fracs']}) ...")
        for entry in entries_for_slice:
            # per-cube output prefix when multiple non-diff cubes
            if not cfg["diff"] and len(entries) > 1:
                stem = out_pfx + "_" + entry["label"]
            else:
                stem = out_pfx_diff if cfg["diff"] else out_pfx
            for fix_ax in axes_to_slice:
                for frac in cfg["slice_fracs"]:
                    plot_cube_slice_2d(entry, fix_ax, frac, stem)

    print(f"\nDone.\n")


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    # ── sub-command dispatch ──────────────────────────────────────────────────
    # Check for 'plot-cube' as the first non-option argument.
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] == "plot-cube":
        main_plot_cube(raw_args[1:])
        return

    run_dir, cube_override = parse_args(sys.argv)
    print(f"\n{'='*55}")
    print(f"  workfunction.py  →  {run_dir}")
    print(f"{'='*55}")

    # 1. Load potential
    print("\n[1] Loading pot.z.avg ...")
    z_bohr, z_ang, v_ry, v_ev = load_pot_z_avg(run_dir, cube_override=cube_override)
    print(f"    {len(z_bohr)} grid points,  z = {z_bohr[0]:.3f}–{z_bohr[-1]:.3f} Bohr")

    # 2. Cell regions
    print("\n[2] Determining cell regions ...")
    regions = determine_regions(run_dir, z_bohr)
    print(f"    Cell z         = {regions['cell_z_bohr']:.4f} Bohr"
          f"  = {regions['cell_z_bohr']*BOHR2ANG:.4f} Å")
    print(f"    emaxpos/eopreg = {regions['emaxpos']:.2f} / {regions['eopreg']:.2f}")
    print(f"    Sawtooth peak  = {regions['saw_peak_bohr']:.2f} Bohr"
          f"  ({regions['saw_peak_bohr']*BOHR2ANG:.2f} Å)"
          f"  reset→{regions['saw_reset_end_bohr']*BOHR2ANG:.2f} Å")
    print(f"    Slab top       = {regions['z_slab_end_bohr']:.2f} Bohr"
          f"  ({regions['z_slab_end_bohr']*BOHR2ANG:.2f} Å)")
    print(f"    Electrolyte end= {regions['z_electrolyte_end_bohr']:.2f} Bohr"
          f"  ({regions['z_electrolyte_end_bohr']*BOHR2ANG:.2f} Å)")

    # 3. Vacuum reference
    print("\n[3] Picking vacuum reference ...")
    v_vac_ry, v_vac_std_ry, vlo, vhi = pick_vacuum_reference(z_bohr, v_ry, regions)
    v_vac_ev = v_vac_ry * RY2EV
    print(f"    Plateau range  = {vlo:.1f}–{vhi:.1f} Bohr"
          f"  ({vlo*BOHR2ANG:.1f}–{vhi*BOHR2ANG:.1f} Å)  [pre-sawtooth]")
    print(f"    V_vac (mean)   = {v_vac_ry:.6f} Ry  = {v_vac_ev:.4f} eV")
    print(f"    V_vac (std)    = {v_vac_std_ry:.6f} Ry  = {v_vac_std_ry*RY2EV*1000:.2f} meV")

    # 4. Fermi energies
    print("\n[4] Parsing Fermi energies ...")
    fermi_data = parse_fermi_energies(run_dir)
    for label, ef in fermi_data:
        print(f"    {label}: {ef:.4f} eV")
    fermi_ev = fermi_data[-1][1]

    # 5. Dipole correction
    print("\n[5] Parsing dipole correction ...")
    dipole_info = parse_dipole_info(run_dir)
    if dipole_info:
        v_amp = dipole_info.get("v_amp_ry", float("nan"))
        print(f"    V_amp = {v_amp:.4f} Ry = {v_amp*RY2EV:.4f} eV"
              f"  (source: {dipole_info.get('source','')})")
    else:
        print("    (no dipole info found)")

    # 6. Compute results
    phi         = v_vac_ev - fermi_ev
    u_she_iupac = phi - U_SHE_IUPAC
    u_she_tras  = phi - U_SHE_TRASATTI

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  E_Fermi (converged)   = {fermi_ev:.4f} eV")
    print(f"  V_vac   (plateau)     = {v_vac_ev:.4f} eV")
    print(f"  Work function  Φ      = {phi:.4f} eV")
    print(f"  U vs SHE [4.44 V]     = {u_she_iupac:+.4f} V_SHE")
    print(f"  U vs SHE [4.28 V]     = {u_she_tras:+.4f} V_SHE")
    print(f"{'='*55}\n")

    # 7. Write outputs
    print("[6] Writing outputs ...")
    base = run_dir  # write next to the run dir data

    write_raw_data(z_bohr, z_ang, v_ry, v_ev,
                   os.path.join(base, "workfunction_rawdata.csv"))

    write_summary_table(run_dir, fermi_data, regions,
                        v_vac_ry, v_vac_std_ry, dipole_info,
                        os.path.join(base, "workfunction_summary.txt"))

    plot_potential(z_bohr, z_ang, v_ry, v_ev, regions,
                   v_vac_ry, vlo, vhi, fermi_ev,
                   os.path.join(base, "workfunction_profile.png"))

    plot_fermi_convergence(fermi_data,
                           os.path.join(base, "workfunction_fermi.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
