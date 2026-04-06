#!/usr/bin/env python3
"""
density_profile.py — Number density profiles ρ(z) from LAMMPS trajectory
==========================================================================
Computes planar-averaged number density profiles along z for water (O, H),
ions (Cs⁺, OH⁻), and any other atom types present in a DFT-CES2 run.

  ρ(z)  [Å⁻³]  =  N_atoms_in_slab(z, z+dz) / (Lx · Ly · dz · N_frames)

z is measured relative to the electrode surface (z_top_slab from
build_summary.json).  Atoms with unwrapped coordinates are first wrapped
back into the primary box.

Outputs (written to the run directory)
---------------------------------------
  density_profile_rawdata.csv   — z and ρ(z) for every species
  density_profile.png           — stacked density profile plot

Usage
-----
  python tools/density_profile.py [run_dir] [options]

Options
-------
  --traj   FILE   trajectory file (default: ces2.emd.lammpstrj in run_dir)
  --dz     FLOAT  bin width in Å (default: 0.1)
  --skip   INT    use every N-th frame (default: 1, i.e. all frames)
  --zlo    FLOAT  lower z-limit for plot in Å relative to electrode surface
  --zhi    FLOAT  upper z-limit for plot in Å relative to electrode surface
"""

import sys
import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ── shared plot style ────────────────────────────────────────────────────────
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
try:
    import plot_setting as ps
    _C  = ps.colors
    _FS = ps.fontsize
    _LS = ps.labelsize
    _LW = ps.linewidth
    _FW, _FH = ps.figsize
except ImportError:
    ps = None
    _C  = ["#1A6FDF","#F14040","#37AD6B","#B177DE","#FEC211","#515151",
           "#FF4081","#FB6501","#6699CC"]
    _FS = 9;  _LS = 9;  _LW = 0.4;  _FW, _FH = 3.5, 2.8


# ── atom-type catalogue ──────────────────────────────────────────────────────
# Default mapping expected from a DFT-CES2 run built with CsOH + TIP4P water
# on an Ir electrode.  Override via --types if needed.
#
#   key  : human-readable species name
#   types: set of LAMMPS integer type indices (1-based) that belong to it
#   label: legend / column header
DEFAULT_SPECIES = [
    # mol_mass: molecular mass [g/mol] used for mass-density conversion.
    #   water_O: each O = one H₂O molecule  → M = 18.015
    #   water_H: each H = ½ H₂O molecule    → M = 18.015/2  (gives same bulk ρ as water_O)
    #   Cs:      monatomic ion               → M = 132.905
    #   OH_O/H:  each O (or H) = one OH⁻    → M = 17.008
    {"name": "water_O", "types": {2}, "label": "Water O", "color_idx": 2, "mol_mass": 18.015},
    {"name": "water_H", "types": {1}, "label": "Water H", "color_idx": 8, "mol_mass": 9.008},
    {"name": "Cs",      "types": {3}, "label": "Cs⁺",     "color_idx": 1, "mol_mass": 132.905},
    {"name": "OH_O",    "types": {5}, "label": "OH⁻ (O)", "color_idx": 3, "mol_mass": 17.008},
    {"name": "OH_H",    "types": {4}, "label": "OH⁻ (H)", "color_idx": 7, "mol_mass": 17.008},
]

# ── unit-conversion constant ──────────────────────────────────────────────────
# ρ [g/cm³] = ρ_num [Å⁻³] × M [g/mol] × (10²⁴ Å³/cm³) / Nₐ
#           = ρ_num × M × 1.66054
_ANG3_TO_GCM3 = 1e24 / 6.02214076e23   # = 1.66054  (Å⁻³·(g/mol) → g/cm³)

# Bulk water reference density [g/cm³] at ~300 K
BULK_WATER_DENSITY = 1.0   # g/cm³ (expected value for sanity check)
# Corresponding bulk number density of water molecules [Å⁻³]
BULK_WATER_NUM = BULK_WATER_DENSITY / (18.015 * _ANG3_TO_GCM3)  # ≈ 0.03346 Å⁻³


# ── helpers ──────────────────────────────────────────────────────────────────

def find_run_dir(argv_rest):
    if argv_rest and os.path.isdir(argv_rest[0]):
        return os.path.abspath(argv_rest[0])
    return os.getcwd()


def load_build_summary(run_dir):
    bsf = os.path.join(run_dir, "build_summary.json")
    if os.path.isfile(bsf):
        with open(bsf) as f:
            return json.load(f)
    return {}


def detect_species_from_script(run_dir):
    """
    Try to read the atoms=(...) line from qmmm*.sh to build a type→element
    map and cross-check against DEFAULT_SPECIES.
    Returns the element list (1-based indexing) or None.
    """
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("qmmm") and fname.endswith(".sh"):
            path = os.path.join(run_dir, fname)
            with open(path) as f:
                text = f.read()
            m = re.search(r"atoms=\(([^)]+)\)", text)
            if m:
                elems = m.group(1).split()
                return elems   # index 0 → type 1
    return None


# ── trajectory parser ─────────────────────────────────────────────────────────

def iter_frames(traj_path, skip=1):
    """
    Generator: yields (timestep, box, atom_data) for every `skip`-th frame.

    box       : dict  {xlo, xhi, ylo, yhi, zlo, zhi}
    atom_data : numpy array  shape (N, 2)  columns [type, zu]
                (only type and z are needed for 1-D profiles)
    """
    frame_idx = -1
    with open(traj_path) as f:
        while True:
            # ── ITEM: TIMESTEP ──
            line = f.readline()
            if not line:
                return
            if "TIMESTEP" not in line:
                continue
            timestep = int(f.readline().strip())

            # ── NUMBER OF ATOMS ──
            f.readline()  # "ITEM: NUMBER OF ATOMS"
            n_atoms = int(f.readline().strip())

            # ── BOX BOUNDS ──
            f.readline()  # "ITEM: BOX BOUNDS ..."
            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())
            box = {"xlo": xlo, "xhi": xhi,
                   "ylo": ylo, "yhi": yhi,
                   "zlo": zlo, "zhi": zhi}

            # ── ATOMS header → find column indices ──
            header = f.readline().split()[2:]   # strip "ITEM:" and "ATOMS"
            try:
                col_id   = header.index("id")
                col_type = header.index("type")
                # prefer unwrapped (xu/zu), fall back to wrapped (x/z)
                col_z = (header.index("zu") if "zu" in header else
                         header.index("z"))
            except ValueError:
                # skip malformed frame
                for _ in range(n_atoms):
                    f.readline()
                continue

            frame_idx += 1

            # ── read atom lines ──
            types = np.empty(n_atoms, dtype=np.int32)
            z_arr = np.empty(n_atoms, dtype=np.float64)
            for i in range(n_atoms):
                tok = f.readline().split()
                types[i] = int(tok[col_type])
                z_arr[i] = float(tok[col_z])

            if frame_idx % skip != 0:
                continue

            yield timestep, box, types, z_arr


# ── density calculation ───────────────────────────────────────────────────────

def compute_density_profiles(traj_path, species, box_ref, z_ref,
                              dz=0.1, skip=1, z_range=None):
    """
    Parameters
    ----------
    traj_path : str
    species   : list of dicts  (name, types, label, color_idx)
    box_ref   : dict from build_summary['box']   (Lx, Ly needed)
    z_ref     : float  z-coordinate of electrode surface [Å] (absolute)
    dz        : float  bin width [Å]
    skip      : int    use every skip-th frame
    z_range   : (zlo_rel, zhi_rel) or None  – relative to z_ref

    Returns
    -------
    z_centers : 1-D array  (relative to z_ref, Å)
    profiles  : dict  name → 1-D density array [Å⁻³]
    meta      : dict  (n_frames, n_atoms, Lx, Ly, ...)
    """
    Lx = box_ref.get("Lx", 38.22)
    Ly = box_ref.get("Ly", 38.40)
    area = Lx * Ly  # Å²

    # bin edges relative to z_ref
    if z_range is None:
        zlo_rel = -5.0
        zhi_rel = box_ref.get("z_el_hi", 73.14) - z_ref + 2.0
    else:
        zlo_rel, zhi_rel = z_range

    edges = np.arange(zlo_rel, zhi_rel + dz, dz)
    n_bins = len(edges) - 1
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    # accumulators
    counts = {sp["name"]: np.zeros(n_bins, dtype=np.float64)
              for sp in species}
    n_frames = 0

    for ts, box, types, z_raw in iter_frames(traj_path, skip=skip):
        # wrap unwrapped z back into [zlo, zhi)
        Lz = box["zhi"] - box["zlo"]
        z_wrap = box["zlo"] + ((z_raw - box["zlo"]) % Lz)
        # shift to z_ref origin
        z_rel = z_wrap - z_ref

        for sp in species:
            mask = np.isin(types, list(sp["types"]))
            h, _ = np.histogram(z_rel[mask], bins=edges)
            counts[sp["name"]] += h

        n_frames += 1

    if n_frames == 0:
        sys.exit("ERROR: no frames read from trajectory.")

    print(f"    Frames processed: {n_frames}")

    # normalise → number density [Å⁻³]
    profiles = {}
    norm = area * dz * n_frames
    for sp in species:
        profiles[sp["name"]] = counts[sp["name"]] / norm

    # mass density profiles [g/cm³]
    mass_profiles = {}
    for sp in species:
        M = sp.get("mol_mass", 1.0)
        mass_profiles[sp["name"]] = profiles[sp["name"]] * M * _ANG3_TO_GCM3

    meta = {"n_frames": n_frames, "Lx": Lx, "Ly": Ly, "area": area,
            "dz": dz, "skip": skip}
    return z_centers, profiles, mass_profiles, meta


# ── output ────────────────────────────────────────────────────────────────────

def write_raw_data(z_centers, profiles, mass_profiles, species, meta, out_path):
    sp_names = [sp["name"] for sp in species]

    with open(out_path, "w") as f:
        f.write(f"# density_profile.py output\n")
        f.write(f"# frames={meta['n_frames']}  skip={meta['skip']}"
                f"  dz={meta['dz']:.3f} Ang"
                f"  Lx={meta['Lx']:.4f} Ang  Ly={meta['Ly']:.4f} Ang\n")
        f.write(f"# z: relative to electrode surface (Ang)\n")
        f.write(f"# rho_*: number density (Ang^-3)\n")
        f.write(f"# mass_*: mass density (g/cm^3)\n")
        num_cols  = [f"rho_{n}"  for n in sp_names]
        mass_cols = [f"mass_{n}" for n in sp_names]
        f.write(",".join(["z_ang"] + num_cols + mass_cols) + "\n")
        for i, z in enumerate(z_centers):
            num_vals  = [f"{profiles[n][i]:.8f}"      for n in sp_names]
            mass_vals = [f"{mass_profiles[n][i]:.8f}" for n in sp_names]
            f.write(",".join([f"{z:.4f}"] + num_vals + mass_vals) + "\n")

    print(f"  Saved: {out_path}")


def _apply_minor_ticks(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="out")


def plot_density_profiles(z_centers, profiles, mass_profiles, species, meta,
                          z_ref, box_ref, out_path):
    """
    Three-panel figure:
      Left   – all species, number density [Å⁻³]
                 + right y-axis showing water mass density [g/cm³]
                 + ρ = 1 g/cm³ reference line
      Middle – ion species only (enlarged)
      Right  – water mass density [g/cm³] alone (sanity-check panel)
    """
    z_el_hi_rel = box_ref.get("z_el_hi", 73.0) - z_ref

    ion_species   = [sp for sp in species if sp["name"] not in ("water_O", "water_H")]
    water_species = [sp for sp in species if sp["name"] in ("water_O", "water_H")]

    has_ions = bool(ion_species)
    n_panels = 3 if has_ions else 2

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(_FW * n_panels * 1.4, _FH * 1.6),
        sharey=False,
    )

    def _vlines(ax):
        ax.axvline(0, color=_C[5], ls="--", lw=_LW * 1.5,
                   alpha=0.8, label="Electrode surface")
        ax.axvline(z_el_hi_rel, color=_C[5], ls=":", lw=_LW * 1.5,
                   alpha=0.5, label=f"Electrolyte top")

    def _finish(ax, xlabel=True):
        if xlabel:
            ax.set_xlabel("z – z$_{surface}$ (Å)", fontsize=_LS)
        if ps:
            ps.set_ylim_top_margin(ax)
        _apply_minor_ticks(ax)

    # ── Panel 0: all species (number density) + water mass density (twin) ──
    ax0 = axes[0]
    for sp in species:
        ax0.plot(z_centers, profiles[sp["name"]],
                 color=_C[sp["color_idx"] % len(_C)],
                 lw=_LW * 2, label=sp["label"])
    _vlines(ax0)
    ax0.set_ylabel("Number density (Å$^{-3}$)", fontsize=_LS)
    ax0.set_title("All species", fontsize=_FS)
    ax0.legend(fontsize=_FS - 1, frameon=False, loc="upper right")
    _finish(ax0)

    # right y-axis: water mass density
    ax0r = ax0.twinx()
    w_mass = mass_profiles.get("water_O", np.zeros_like(z_centers))
    ax0r.plot(z_centers, w_mass,
              color=_C[2], lw=0, alpha=0)   # invisible – just to set scale
    ax0r.set_ylim(
        ax0.get_ylim()[0] * 18.015 * _ANG3_TO_GCM3,
        ax0.get_ylim()[1] * 18.015 * _ANG3_TO_GCM3,
    )
    ax0r.axhline(BULK_WATER_DENSITY, color=_C[2], ls="--",
                 lw=_LW * 1.5, alpha=0.6, label="1 g/cm³ (bulk)")
    ax0r.set_ylabel("Water mass density (g/cm³)", fontsize=_LS - 1, color=_C[2])
    ax0r.tick_params(axis="y", colors=_C[2], labelsize=_LS - 1)

    axes[0].text(
        0.99, 0.97,
        f"frames={meta['n_frames']}  dz={meta['dz']:.2f} Å",
        transform=axes[0].transAxes, ha="right", va="top",
        fontsize=_FS - 2, color="gray",
    )

    # ── Panel 1: ions (if present) ──
    if has_ions:
        ax1 = axes[1]
        for sp in ion_species:
            ax1.plot(z_centers, profiles[sp["name"]],
                     color=_C[sp["color_idx"] % len(_C)],
                     lw=_LW * 2, label=sp["label"])
        _vlines(ax1)
        ax1.set_ylabel("Number density (Å$^{-3}$)", fontsize=_LS)
        ax1.set_title("Ions (enlarged)", fontsize=_FS)
        ax1.legend(fontsize=_FS - 1, frameon=False)
        _finish(ax1)

    # ── Last panel: water mass density [g/cm³] sanity check ──
    axw = axes[-1]
    for sp in water_species:
        axw.plot(z_centers, mass_profiles[sp["name"]],
                 color=_C[sp["color_idx"] % len(_C)],
                 lw=_LW * 2, label=sp["label"])
    # bulk reference line
    axw.axhline(BULK_WATER_DENSITY, color=_C[1], ls="--", lw=_LW * 2,
                label=f"Bulk ref. = {BULK_WATER_DENSITY:.1f} g/cm³")
    _vlines(axw)
    axw.set_ylabel("Mass density (g/cm³)", fontsize=_LS)
    axw.set_title("Water mass density", fontsize=_FS)
    axw.legend(fontsize=_FS - 1, frameon=False)
    _finish(axw)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute z-density profiles from a LAMMPS DFT-CES2 trajectory."
    )
    p.add_argument("run_dir", nargs="?", default=None,
                   help="Path to run directory (default: current directory)")
    p.add_argument("--traj", default=None,
                   help="Trajectory file (default: ces2.emd.lammpstrj)")
    p.add_argument("--dz",   type=float, default=0.1,
                   help="Bin width in Å (default: 0.1)")
    p.add_argument("--skip", type=int,   default=1,
                   help="Use every N-th frame (default: 1 = all frames)")
    p.add_argument("--zlo",  type=float, default=None,
                   help="Lower z-limit relative to electrode surface (Å)")
    p.add_argument("--zhi",  type=float, default=None,
                   help="Upper z-limit relative to electrode surface (Å)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else os.getcwd()
    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: directory not found: {run_dir}")

    print(f"\n{'='*55}")
    print(f"  density_profile.py  →  {run_dir}")
    print(f"{'='*55}")

    # 1. Build summary
    print("\n[1] Loading build_summary.json ...")
    bs = load_build_summary(run_dir)
    box_ref = bs.get("box", {})
    z_ref   = box_ref.get("z_top_slab", 13.14)   # electrode surface, Å
    print(f"    Electrode surface z_ref = {z_ref:.4f} Å")
    print(f"    Lx = {box_ref.get('Lx',38.22):.4f} Å,  "
          f"Ly = {box_ref.get('Ly',38.40):.4f} Å")

    # 2. Species from script (informational)
    elems = detect_species_from_script(run_dir)
    if elems:
        print(f"\n[2] Detected element map from qmmm script:")
        for i, el in enumerate(elems, 1):
            print(f"    type {i:2d} → {el}")
    else:
        print("\n[2] Using default species mapping.")

    # 3. Trajectory
    traj = args.traj or os.path.join(run_dir, "ces2.emd.lammpstrj")
    if not os.path.isfile(traj):
        sys.exit(f"ERROR: trajectory not found: {traj}")
    print(f"\n[3] Trajectory: {os.path.relpath(traj, run_dir)}")
    print(f"    dz = {args.dz} Å,  skip = {args.skip}")

    z_range = None
    if args.zlo is not None or args.zhi is not None:
        zlo = args.zlo if args.zlo is not None else -5.0
        zhi = args.zhi if args.zhi is not None else (box_ref.get("z_el_hi", 73.14) - z_ref + 2.0)
        z_range = (zlo, zhi)
        print(f"    z range: {zlo:.1f} to {zhi:.1f} Å (relative to surface)")

    # 4. Compute profiles
    print("\n[4] Computing density profiles ...")
    z_centers, profiles, mass_profiles, meta = compute_density_profiles(
        traj, DEFAULT_SPECIES, box_ref, z_ref,
        dz=args.dz, skip=args.skip, z_range=z_range,
    )

    # Print peak info + water bulk sanity check
    print("\n    Peak densities:")
    for sp in DEFAULT_SPECIES:
        rho  = profiles[sp["name"]]
        mrho = mass_profiles[sp["name"]]
        if rho.max() > 0:
            z_peak = z_centers[np.argmax(rho)]
            print(f"    {sp['label']:12s}  ρ_max = {rho.max():.5f} Å⁻³"
                  f"  ({mrho.max():.4f} g/cm³)  at z = {z_peak:.2f} Å")

    # Water bulk sanity check: average over middle of electrolyte
    z_el_hi_rel = box_ref.get("z_el_hi", 73.14) - z_ref
    z_bulk_lo, z_bulk_hi = 20.0, min(50.0, z_el_hi_rel - 5.0)
    mask_bulk = (z_centers >= z_bulk_lo) & (z_centers <= z_bulk_hi)
    if mask_bulk.sum() > 0 and "water_O" in profiles:
        rho_bulk_num  = profiles["water_O"][mask_bulk].mean()
        rho_bulk_mass = mass_profiles["water_O"][mask_bulk].mean()
        deviation     = (rho_bulk_mass - BULK_WATER_DENSITY) / BULK_WATER_DENSITY * 100
        print(f"\n    ── Water bulk sanity check  "
              f"(z = {z_bulk_lo:.0f}–{z_bulk_hi:.0f} Å) ──")
        print(f"    ρ_bulk (number) = {rho_bulk_num:.5f} Å⁻³"
              f"  (expected ~{BULK_WATER_NUM:.5f} Å⁻³)")
        print(f"    ρ_bulk (mass)   = {rho_bulk_mass:.4f} g/cm³"
              f"  (expected ~{BULK_WATER_DENSITY:.1f} g/cm³)"
              f"  → deviation {deviation:+.1f} %")
        if abs(deviation) < 5:
            print(f"    ✓ Within 5 % of bulk water density — OK")
        else:
            print(f"    ⚠ Deviation > 5 % — check trajectory or box dimensions")

    # 5. Write outputs
    print("\n[5] Writing outputs ...")
    base = run_dir

    write_raw_data(z_centers, profiles, mass_profiles, DEFAULT_SPECIES, meta,
                   os.path.join(base, "density_profile_rawdata.csv"))

    plot_density_profiles(z_centers, profiles, mass_profiles, DEFAULT_SPECIES,
                          meta, z_ref, box_ref,
                          os.path.join(base, "density_profile.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
