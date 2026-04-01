#!/usr/bin/env python3
"""
ion_distance.py
===============
Compute distance between cation (e.g. Cs, Na, Li) and OH⁻ ions
over a LAMMPS trajectory, then visualize as time series and RDF.

The OH⁻ is represented by its O atom (type O_oh in CES2 = LAMMPS type 5).
Distances are minimum-image Cartesian distances in x/y (periodic) and z (non-periodic).

Outputs
-------
  <stem>_ion_dist.png   — time series (min / mean distance per frame) + running mean
  <stem>_ion_rdf.png    — pair RDF  (cation–O_oh)
  <stem>_ion_dist.csv   — per-frame distance data

Usage
-----
  # Auto-detect types from in.lammps
  python tools/ion_distance.py test/04_Cs/ces2.emd.lammpstrj

  # Explicit type IDs: cation type 3, OH-O type 5
  python tools/ion_distance.py traj.lammpstrj --cation-type 3 --oh-type 5

  # Multiple cation types (e.g. mixed solution)
  python tools/ion_distance.py traj.lammpstrj --cation-type 3 --oh-type 5 --cation-label Cs

  # Adjust RDF range and stride
  python tools/ion_distance.py traj.lammpstrj --r-max 12 --bins 200 --stride 5
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from lammpstrj_to_traj import (
    iter_frames, frame_to_atoms,
    auto_detect_type_map, DEFAULT_TYPE_MAP,
)

# ---------------------------------------------------------------------------
# Default type IDs for CES2 — overridden by auto-detection or CLI args
# ---------------------------------------------------------------------------
DEFAULT_CATION_TYPE = 3   # K or Cs
DEFAULT_OH_O_TYPE   = 5   # O_oh (oxygen of hydroxyl ion)
DEFAULT_OH_H_TYPE   = 4   # H_oh (hydrogen of hydroxyl ion)


# ---------------------------------------------------------------------------
# Core distance computation
# ---------------------------------------------------------------------------

def _min_image_dist(pos_a: np.ndarray, pos_b: np.ndarray,
                    cell: np.ndarray) -> np.ndarray:
    """
    Minimum-image pairwise distances between all pairs (a_i, b_j).
    Periodic in x, y only (z is slab non-periodic).

    Parameters
    ----------
    pos_a : (N, 3)
    pos_b : (M, 3)
    cell  : (3,) — [Lx, Ly, Lz]

    Returns
    -------
    (N, M) distance matrix
    """
    dv = pos_a[:, None, :] - pos_b[None, :, :]   # (N, M, 3)
    dv[:, :, 0] -= np.round(dv[:, :, 0] / cell[0]) * cell[0]
    dv[:, :, 1] -= np.round(dv[:, :, 1] / cell[1]) * cell[1]
    return np.sqrt(np.sum(dv * dv, axis=2))


def _kdtree_min_dist(pos_a: np.ndarray, pos_b: np.ndarray,
                     cell: np.ndarray) -> np.ndarray:
    """
    For each atom in pos_a, find the minimum distance to any atom in pos_b.
    Uses cKDTree for O(n log n) performance.
    Periodic in x, y; non-periodic in z.
    """
    Lx, Ly = cell[0], cell[1]
    z_big = max((pos_a[:, 2].max() - pos_a[:, 2].min()) * 10,
                (pos_b[:, 2].max() - pos_b[:, 2].min()) * 10,
                1e4)
    z_shift = min(pos_a[:, 2].min(), pos_b[:, 2].min()) - 1.0

    def _prep(xyz):
        p = xyz.copy()
        p[:, 0] = p[:, 0] % Lx
        p[:, 1] = p[:, 1] % Ly
        p[:, 2] = p[:, 2] - z_shift
        return p

    tree = cKDTree(_prep(pos_b), boxsize=[Lx, Ly, z_big])
    dists, _ = tree.query(_prep(pos_a), k=1)
    return dists   # (N,) — min dist from each A atom to nearest B atom


def compute_distances(
    lammpstrj: Path,
    cation_type: int,
    oh_o_type: int,
    type_map: Dict[int, str],
    r_max: float = 10.0,
    bins: int = 200,
    stride: int = 1,
    start: int = 0,
    end: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    For each frame compute:
      - min_dist  : minimum cation–O_oh distance (Å)
      - mean_dist : mean of all cation–O_oh minimum distances
      - coord_n   : mean coordination number of O_oh within r_max/2

    Also accumulates histogram data for a simple pair RDF.

    Returns a dict with keys:
      timesteps, min_dist, mean_dist, coord_n,
      rdf_r, rdf_g  (RDF bin centres and g(r))
    """
    timesteps  = []
    min_dists  = []
    mean_dists = []
    coord_ns   = []

    # RDF accumulation (simple histogram, normalised at the end)
    rdf_edges  = np.linspace(0.0, r_max, bins + 1)
    rdf_counts = np.zeros(bins, dtype=np.float64)
    n_cat_total   = 0
    n_ooh_total   = 0
    vol_total     = 0.0
    frames_used   = 0

    frame_idx = 0
    t_start   = time.perf_counter()

    for frame in iter_frames(lammpstrj):
        if frame_idx < start:
            frame_idx += 1
            continue
        if end is not None and frame_idx >= end:
            break
        if (frame_idx - start) % stride != 0:
            frame_idx += 1
            continue

        atoms  = frame_to_atoms(frame, type_map)
        cell   = atoms.cell.diagonal()
        pos    = atoms.positions.copy()
        types  = np.array(atoms.info["lammps_type"])

        cat_pos = pos[types == cation_type]
        ooh_pos = pos[types == oh_o_type]

        if len(cat_pos) == 0 or len(ooh_pos) == 0:
            frame_idx += 1
            continue

        # Per-cation minimum distance to any O_oh
        min_to_ooh = _kdtree_min_dist(cat_pos, ooh_pos, cell)

        timesteps.append(frame["timestep"])
        min_dists.append(float(min_to_ooh.min()))
        mean_dists.append(float(min_to_ooh.mean()))

        # Coordination: number of O_oh within r_max/2 of each cation
        coord_cutoff = r_max / 2.0
        coord = np.sum(min_to_ooh <= coord_cutoff)
        coord_ns.append(float(coord) / max(len(cat_pos), 1))

        # RDF histogram accumulation (cation → O_oh pairwise)
        # Use vectorised minimum-image distances for pairs within r_max
        Lx, Ly = cell[0], cell[1]
        z_big  = max((pos[:, 2].max() - pos[:, 2].min()) * 10, 1e4)
        z_shift = pos[:, 2].min() - 1.0

        def _prep(xyz):
            p = xyz.copy()
            p[:, 0] = p[:, 0] % Lx
            p[:, 1] = p[:, 1] % Ly
            p[:, 2] = p[:, 2] - z_shift
            return p

        tree_ooh = cKDTree(_prep(ooh_pos), boxsize=[Lx, Ly, z_big])
        # query_ball_point returns indices within r_max for each cation
        neighbours = tree_ooh.query_ball_point(_prep(cat_pos), r=r_max)

        for ci, nbrs in enumerate(neighbours):
            if not nbrs:
                continue
            nbr_arr = ooh_pos[nbrs]
            dv = cat_pos[ci] - nbr_arr
            dv[:, 0] -= np.round(dv[:, 0] / cell[0]) * cell[0]
            dv[:, 1] -= np.round(dv[:, 1] / cell[1]) * cell[1]
            d = np.sqrt(np.sum(dv * dv, axis=1))
            rdf_counts += np.histogram(d, bins=rdf_edges)[0]

        n_cat_total  += len(cat_pos)
        n_ooh_total  += len(ooh_pos)
        vol_total    += cell[0] * cell[1] * cell[2]
        frames_used  += 1
        frame_idx    += 1

        if verbose and frames_used % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  frame {frame_idx} | {frames_used} used "
                  f"| {frames_used/elapsed:.1f} fps "
                  f"| timestep {frame['timestep']}")

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"  Done: {frames_used} frames in {elapsed:.1f}s "
              f"({1000*elapsed/max(frames_used,1):.1f} ms/frame)")

    # Normalise RDF
    rdf_r = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
    dr    = rdf_edges[1] - rdf_edges[0]

    if frames_used > 0 and n_cat_total > 0 and vol_total > 0:
        n_cat_avg = n_cat_total / frames_used
        n_ooh_avg = n_ooh_total / frames_used
        vol_avg   = vol_total   / frames_used
        rho_ooh   = n_ooh_avg / vol_avg          # number density of O_oh
        shell_vol = 4.0 * np.pi * rdf_r**2 * dr
        ideal     = rho_ooh * shell_vol * n_cat_avg * frames_used
        ideal     = np.where(ideal > 0, ideal, 1.0)
        rdf_g     = rdf_counts / ideal
    else:
        rdf_g = np.zeros_like(rdf_r)

    return {
        "timesteps" : np.array(timesteps, dtype=int),
        "min_dist"  : np.array(min_dists),
        "mean_dist" : np.array(mean_dists),
        "coord_n"   : np.array(coord_ns),
        "rdf_r"     : rdf_r,
        "rdf_g"     : rdf_g,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centred running mean."""
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_results(
    data: Dict,
    output_ts: Path,
    output_rdf: Path,
    cation_label: str = "Cation",
    oh_label: str = "OH⁻",
    smooth_window: int = 20,
):
    """Generate time-series plot and RDF plot."""
    ts        = data["timesteps"]
    min_d     = data["min_dist"]
    mean_d    = data["mean_dist"]
    coord_n   = data["coord_n"]
    rdf_r     = data["rdf_r"]
    rdf_g     = data["rdf_g"]

    # ── Time series ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Distance panel
    ax1.plot(ts, min_d,  color="#4878CF", alpha=0.35, linewidth=0.8,
             label=f"Min dist ({cation_label}–{oh_label} O)")
    ax1.plot(ts, mean_d, color="#D65F5F", alpha=0.35, linewidth=0.8,
             label=f"Mean dist ({cation_label}–{oh_label} O)")
    smooth_min  = _running_mean(min_d,  smooth_window)
    smooth_mean = _running_mean(mean_d, smooth_window)
    ax1.plot(ts, smooth_min,  color="#4878CF", linewidth=1.8,
             label=f"Min (smoothed, w={smooth_window})")
    ax1.plot(ts, smooth_mean, color="#D65F5F", linewidth=1.8,
             label=f"Mean (smoothed)")
    ax1.set_xlabel("Timestep", fontsize=11)
    ax1.set_ylabel("Distance (Å)", fontsize=11)
    ax1.set_title(f"{cation_label}–{oh_label} Distance vs. Time", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # Coordination number panel
    ax2.plot(ts, coord_n, color="#6ACC65", alpha=0.35, linewidth=0.8)
    ax2.plot(ts, _running_mean(coord_n, smooth_window),
             color="#6ACC65", linewidth=1.8,
             label=f"Coord. N (smoothed, w={smooth_window})")
    ax2.set_xlabel("Timestep", fontsize=11)
    ax2.set_ylabel(f"# {oh_label} per {cation_label}", fontsize=11)
    ax2.set_title(f"{cation_label}–{oh_label} Coordination Number vs. Time", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    output_ts.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_ts, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved time series → {output_ts}")

    # ── RDF ────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rdf_r, rdf_g, color="#4878CF", linewidth=1.8,
            label=f"g(r)  {cation_label}–{oh_label} O")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Annotate first peak
    if len(rdf_g) > 5:
        peak_idx = int(np.argmax(rdf_g[5:])) + 5   # skip r < ~0.25 Å
        ax.axvline(rdf_r[peak_idx], color="red", linestyle=":",
                   linewidth=1.0, alpha=0.7,
                   label=f"1st peak @ {rdf_r[peak_idx]:.2f} Å")

    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_title(f"RDF — {cation_label}–{oh_label} O", fontsize=13)
    ax.set_xlim(0, rdf_r[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    output_rdf.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(output_rdf, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved RDF plot      → {output_rdf}")


def save_csv(data: Dict, output: Path, cation_label: str, oh_label: str):
    """Save per-frame distance data as CSV."""
    header = "timestep,min_dist_A,mean_dist_A,coord_n"
    rows = np.column_stack([
        data["timesteps"],
        data["min_dist"],
        data["mean_dist"],
        data["coord_n"],
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, rows, delimiter=",",
               header=f"# {cation_label}-{oh_label} distances\n{header}",
               comments="", fmt="%d,%.6f,%.6f,%.6f")
    print(f"Saved CSV data      → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute cation–OH⁻ distances from a LAMMPS trajectory."
    )
    parser.add_argument("lammpstrj", type=Path, help="LAMMPS dump file")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: same directory as input file)"
    )
    parser.add_argument(
        "--lammps-input", type=Path, default=None,
        help="LAMMPS input file for type map auto-detection"
    )
    parser.add_argument(
        "--cation-type", type=int, default=None,
        help=f"LAMMPS type ID of the cation (default: {DEFAULT_CATION_TYPE})"
    )
    parser.add_argument(
        "--oh-type", type=int, default=None,
        help=f"LAMMPS type ID of O in OH⁻ (default: {DEFAULT_OH_O_TYPE})"
    )
    parser.add_argument(
        "--cation-label", type=str, default=None,
        help="Label for cation in plots (default: auto from type map, e.g. 'Cs')"
    )
    parser.add_argument(
        "--oh-label", type=str, default="OH⁻",
        help="Label for OH⁻ in plots (default: 'OH⁻')"
    )
    parser.add_argument(
        "--r-max", type=float, default=10.0,
        help="Max distance for RDF and coordination (Å, default: 10.0)"
    )
    parser.add_argument(
        "--bins", type=int, default=200,
        help="Number of RDF bins (default: 200)"
    )
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Running-mean window for time series plots (default: 20 frames)"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Use every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="First frame index, 0-based (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Last frame index, exclusive (default: all)"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip saving CSV file"
    )
    args = parser.parse_args()

    if not args.lammpstrj.exists():
        print(f"ERROR: {args.lammpstrj} not found", file=sys.stderr)
        sys.exit(1)

    # --- Type map ---
    candidates = [
        args.lammps_input,
        args.lammpstrj.parent / "in.lammps",
        args.lammpstrj.parent.parent / "in.lammps",
    ]
    type_map = dict(DEFAULT_TYPE_MAP)
    for c in candidates:
        if c and c.exists():
            type_map = auto_detect_type_map(c)
            print(f"Type map from {c}: {type_map}")
            break
    else:
        print(f"Using default type map: {type_map}")

    # --- Resolve type IDs ---
    cation_type = args.cation_type or DEFAULT_CATION_TYPE
    oh_o_type   = args.oh_type     or DEFAULT_OH_O_TYPE

    # Auto-label from type map
    if args.cation_label:
        cation_label = args.cation_label
    else:
        cation_label = type_map.get(cation_type, str(cation_type))

    oh_label = args.oh_label

    print(f"Cation: type {cation_type} ({cation_label})")
    print(f"OH⁻ O : type {oh_o_type}   ({oh_label})")
    print(f"r_max={args.r_max} Å, bins={args.bins}, stride={args.stride}")
    if args.start or args.end:
        print(f"Frame range: [{args.start}, {args.end})")
    print()

    # --- Output paths ---
    stem    = args.lammpstrj.stem
    outdir  = args.output_dir or args.lammpstrj.parent
    out_ts  = outdir / f"{stem}_{cation_label.lower()}_oh_dist.png"
    out_rdf = outdir / f"{stem}_{cation_label.lower()}_oh_rdf.png"
    out_csv = outdir / f"{stem}_{cation_label.lower()}_oh_dist.csv"

    # --- Compute ---
    data = compute_distances(
        args.lammpstrj,
        cation_type=cation_type,
        oh_o_type=oh_o_type,
        type_map=type_map,
        r_max=args.r_max,
        bins=args.bins,
        stride=args.stride,
        start=args.start,
        end=args.end,
    )

    if len(data["timesteps"]) == 0:
        print("ERROR: no frames processed — check type IDs and frame range.", file=sys.stderr)
        sys.exit(1)

    # Summary stats
    print(f"\nSummary ({len(data['timesteps'])} frames):")
    print(f"  Min distance  — min: {data['min_dist'].min():.3f} Å  "
          f"mean: {data['min_dist'].mean():.3f} Å  "
          f"max: {data['min_dist'].max():.3f} Å")
    print(f"  Mean distance — min: {data['mean_dist'].min():.3f} Å  "
          f"mean: {data['mean_dist'].mean():.3f} Å")
    print(f"  Coord. number — mean: {data['coord_n'].mean():.2f}")
    peak_idx = int(np.argmax(data["rdf_g"][5:])) + 5 if len(data["rdf_g"]) > 5 else 0
    print(f"  RDF 1st peak  @ {data['rdf_r'][peak_idx]:.2f} Å  "
          f"(g(r) = {data['rdf_g'][peak_idx]:.2f})")

    # --- Save ---
    plot_results(data, out_ts, out_rdf,
                 cation_label=cation_label,
                 oh_label=oh_label,
                 smooth_window=args.smooth)

    if not args.no_csv:
        save_csv(data, out_csv, cation_label, oh_label)


if __name__ == "__main__":
    main()
