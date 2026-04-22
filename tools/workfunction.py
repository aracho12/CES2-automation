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
  # If pot.z.avg is missing in run_dir, the script automatically planar-
  # averages a Gaussian cube of V_H+V_bare (solute.pot_ortho.cube or
  # solute.pot.cube) along z and writes pot.z.avg.  Use --cube to point
  # at a specific cube file.

  # from repo root:
  python tools/workfunction.py test/04_Cs/run
  python tools/workfunction.py test/04_Cs/run --cube solute.pot_ortho.cube
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


# Cube files containing raw V_H+V_bare from pp.x (no V_saw added).
# These are what workfunction.py needs for the planar average.
CUBE_CANDIDATES = ("solute.pot_ortho.cube", "solute.pot.cube")


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
    If pot.z.avg is missing, planar-average a cube (solute.pot_ortho.cube
    or solute.pot.cube — or --cube PATH) and write pot.z.avg in run_dir.
    """
    fpath = os.path.join(run_dir, "pot.z.avg")
    if not os.path.isfile(fpath):
        cube_path = cube_override or find_cube(run_dir)
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
    """Get cell z-length (Bohr) and emaxpos from pw.in or build_summary."""
    cell_z = None
    emaxpos = 0.8  # default

    # Try build_summary.json
    bsf = os.path.join(run_dir, "build_summary.json")
    if os.path.isfile(bsf):
        with open(bsf) as f:
            bs = json.load(f)
        box = bs.get("box", {})
        lz  = box.get("Lz")
        if lz:
            # build_summary Lz is in Å, but here the QE cell is larger (supercell)
            # so we trust the pot.z.avg z-range instead
            pass

    # Try pw.in for emaxpos and celldm
    for pw_in in [os.path.join(run_dir, "base.pw.in"),
                  os.path.join(run_dir, "pw.in"),
                  os.path.join(run_dir, "qm_1", "pw.in")]:
        if not os.path.isfile(pw_in):
            continue
        with open(pw_in) as f:
            text = f.read()
        m = re.search(r"emaxpos\s*=\s*([\d.]+)", text)
        if m:
            emaxpos = float(m.group(1))
        break

    return {"emaxpos": emaxpos}


def determine_regions(run_dir, z_bohr):
    """Return region boundaries (Bohr) from build_summary or heuristic."""
    cell_z    = z_bohr[-1] + (z_bohr[1] - z_bohr[0])  # full cell length
    emaxpos   = parse_cell_info(run_dir)["emaxpos"]
    saw_peak  = emaxpos * cell_z

    # Defaults (heuristic: look for large dip in potential)
    z_top_slab = z_bohr[0] + 25.0   # fallback
    z_el_hi    = saw_peak - 5.0      # fallback

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
        "saw_peak_bohr": saw_peak,
        "z_slab_end_bohr": z_top_slab,
        "z_electrolyte_end_bohr": z_el_hi,
    }


def pick_vacuum_reference(z_bohr, v_ry, regions):
    """
    Return V_vac (Ry) from the flat plateau AFTER the sawtooth peak.
    The vacuum region is [electrolyte_end .. cell_z].
    We skip the first ~5 Bohr of vacuum (ramp-up) and pick a flat plateau.
    """
    saw_peak = regions["saw_peak_bohr"]
    cell_z   = regions["cell_z_bohr"]

    # flat region: from saw_peak + 3 Bohr to cell end
    lo = saw_peak + 3.0
    hi = cell_z
    mask = (z_bohr >= lo) & (z_bohr <= hi)
    if mask.sum() < 3:
        # Fallback: last 10% of cell
        lo = 0.9 * cell_z
        mask = z_bohr >= lo
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
    cell_z   = regions["cell_z_bohr"]
    z_sl_end = regions["z_slab_end_bohr"]
    z_el_end = regions["z_electrolyte_end_bohr"]
    saw_peak = regions["saw_peak_bohr"]
    v_vac_ev = v_vac_ry * RY2EV
    phi      = v_vac_ev - fermi_ev

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
    fig.suptitle("Planar-Averaged Electrostatic Potential  (V$_H$ + V$_{bare}$)",
                 fontsize=_FS, fontweight="bold", y=1.01)

    # ── Left panel: full z range ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(z_bohr, v_ev, color=c_line, lw=_LW * 2, label="V(z)")

    ax.axvspan(0,        z_sl_end, alpha=0.12, color=c_slab,  label="Electrode")
    ax.axvspan(z_sl_end, z_el_end, alpha=0.08, color=c_elec,  label="Electrolyte")
    ax.axvspan(z_el_end, cell_z,   alpha=0.10, color=c_vac,   label="Vacuum")

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
    axr.axvspan(z_el_end, cell_z, alpha=0.10, color=c_vac)
    axr.axvspan(v_vac_lo, v_vac_hi, alpha=0.25, color=c_vref,
                label="Reference plateau")

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
            f"{regions['saw_peak_bohr']+3:.1f}–{cell_z:.1f}", "Bohr"),
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
        "Note: pot.z.avg contains V_H + V_bare (no V_saw).",
        "      V_vac is taken from the flat plateau after emaxpos.",
        "      SHE reference: Li et al. PCCP 17, 4647 (2015)  / Trasatti 1986.",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


# ─────────────────────────── main ───────────────────────────────────────────

def main():
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
    print(f"    emaxpos        = {regions['emaxpos']:.2f}")
    print(f"    Sawtooth peak  = {regions['saw_peak_bohr']:.2f} Bohr"
          f"  ({regions['saw_peak_bohr']*BOHR2ANG:.2f} Å)")
    print(f"    Slab top       = {regions['z_slab_end_bohr']:.2f} Bohr"
          f"  ({regions['z_slab_end_bohr']*BOHR2ANG:.2f} Å)")
    print(f"    Electrolyte end= {regions['z_electrolyte_end_bohr']:.2f} Bohr"
          f"  ({regions['z_electrolyte_end_bohr']*BOHR2ANG:.2f} Å)")

    # 3. Vacuum reference
    print("\n[3] Picking vacuum reference ...")
    v_vac_ry, v_vac_std_ry, vlo, vhi = pick_vacuum_reference(z_bohr, v_ry, regions)
    v_vac_ev = v_vac_ry * RY2EV
    print(f"    Plateau range  = {vlo:.1f}–{vhi:.1f} Bohr"
          f"  ({vlo*BOHR2ANG:.1f}–{vhi*BOHR2ANG:.1f} Å)")
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
