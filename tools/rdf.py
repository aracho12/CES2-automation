#!/usr/bin/env python3
"""
rdf.py — Radial Distribution Function calculator for LAMMPS trajectories
=========================================================================
Uses freud (C++ backend) for fast neighbor-list based RDF computation.

Benchmark (ces2.emd.lammpstrj, 10365 atoms):
  Ow-Ow  (~2850 atoms/type): ~6 ms/frame  → full 1274 frames in ~8 s
  Hw-Ow  (~5700+2850 atoms): ~12 ms/frame → full 1274 frames in ~15 s
  Speedup vs pure NumPy: ~22×

Usage
-----
  # All default pairs, all frames
  python tools/rdf.py test/ces2.emd.lammpstrj

  # Custom pairs (LAMMPS type IDs)
  python tools/rdf.py test/ces2.emd.lammpstrj --pairs "2-2 1-2 3-2 6-2"

  # Subset of frames & custom output
  python tools/rdf.py test/ces2.emd.lammpstrj --stride 5 --start 100 --end 500 -o output/rdf.png

  # Adjust r_max and bin count
  python tools/rdf.py test/ces2.emd.lammpstrj --r-max 12 --bins 300

  # Process every mm_N directory under a QM/MM run directory
  python tools/rdf.py --mm . --pairs "5-2 3-2"
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── local import ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from lammpstrj_to_traj import (
    iter_frames, frame_to_atoms,
    auto_detect_type_map, DEFAULT_TYPE_MAP,
)

# ── Default LAMMPS type labels for CES2 ──────────────────────────────────────
TYPE_LABELS = {
    1: "Hw",
    2: "Ow",
    3: "Li",
    4: "H_oh",
    5: "O_oh",
    6: "Ir",
    7: "O_surf",
}

# Default pair type IDs to compute
DEFAULT_PAIR_TYPES: List[Tuple[int, int]] = [
    (2, 2),
    (1, 2),
    (3, 2),
    (6, 2),
    (6, 7),
    (5, 2),
]


def parse_type_label_map(lammps_input: Path) -> Dict[int, str]:
    """Extract LAMMPS type→CES2 type_label mapping from group comments."""
    if not lammps_input.exists():
        return {}

    group_pattern = re.compile(
        r"^\s*group\s+\S+\s+type\s+([\d\s]+)(?:#\s*(.+))?$",
        re.IGNORECASE,
    )
    label_map: Dict[int, str] = {}

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue
            type_ids = [int(x) for x in m.group(1).split()]
            comment = m.group(2) or ""
            text = comment.split(":", 1)[1] if ":" in comment else comment
            text = re.sub(r"\([^)]*\)", " ", text)
            labels = [tok for tok in re.split(r"[\s,]+", text.strip()) if tok]
            labels = [
                tok for tok in labels
                if tok.lower() not in {"all", "water", "qm", "slab", "atoms"}
            ]
            if len(labels) >= len(type_ids):
                for tid, label in zip(type_ids, labels):
                    label_map[tid] = label
            elif len(type_ids) == 1 and labels:
                label_map[type_ids[0]] = labels[0]

    return label_map


def label_for_type(type_id: int, type_labels: Optional[Dict[int, str]] = None) -> str:
    """Return the best display label for a LAMMPS type ID."""
    if type_labels and type_id in type_labels:
        return type_labels[type_id]
    return TYPE_LABELS.get(type_id, str(type_id))


def build_pairs(
    pair_types: List[Tuple[int, int]],
    type_labels: Optional[Dict[int, str]] = None,
) -> List[Tuple[int, int, str]]:
    """Attach display labels to pair type IDs."""
    return [
        (t1, t2, f"{label_for_type(t1, type_labels)}-{label_for_type(t2, type_labels)}")
        for t1, t2 in pair_types
    ]


def parse_pair_types(pair_str: str) -> List[Tuple[int, int]]:
    """
    Parse a pair string like '2-2 1-2 3-2' into [(2,2), (1,2), (3,2)].
    """
    result = []
    for token in pair_str.split():
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Bad pair spec '{token}': expected 'T1-T2'")
        t1, t2 = int(parts[0]), int(parts[1])
        result.append((t1, t2))
    return result


def find_mm_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Find mm_N directories sorted by N."""
    mm_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and re.match(r"^mm_(\d+)$", d.name):
            mm_dirs.append((int(d.name.split("_")[1]), d))
    return sorted(mm_dirs, key=lambda x: x[0])


def find_lammpstrj_in_mm_dir(mm_dir: Path) -> Optional[Path]:
    """Find the trajectory to use for RDF inside one mm_N directory."""
    for pattern in ("*.emd.lammpstrj", "*.lammpstrj"):
        matches = sorted(mm_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_lammps_input(traj_path: Path, explicit: Optional[Path] = None) -> Optional[Path]:
    """Find the LAMMPS input near a trajectory path."""
    candidates = [
        explicit,
        traj_path.parent / "in.lammps",
        traj_path.parent.parent / "in.lammps",
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def compute_rdf(
    lammpstrj: Path,
    pairs: List[Tuple[int, int, str]],
    type_map: Dict[int, str],
    r_max: float = 10.0,
    bins: int = 200,
    stride: int = 1,
    start: int = 0,
    end: int = None,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute RDF for each pair using freud.

    Returns
    -------
    dict: label → (r_centers, g_r)
      r_centers : np.ndarray, shape (bins,)   — bin centres in Å
      g_r       : np.ndarray, shape (bins,)   — g(r) averaged over frames
    """
    try:
        import freud
    except ImportError:
        print(
            "ERROR: rdf.py requires the 'freud' Python package. "
            "Install it in this environment before running RDF calculations.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialise one freud RDF object per pair
    rdf_objs = {}
    for t1, t2, label in pairs:
        rdf_objs[label] = freud.density.RDF(bins=bins, r_max=r_max)

    frame_count = 0
    used_frames = 0
    t_start = time.perf_counter()

    for frame in iter_frames(lammpstrj):
        if frame_count < start:
            frame_count += 1
            continue
        if end is not None and frame_count >= end:
            break
        if (frame_count - start) % stride != 0:
            frame_count += 1
            continue

        atoms = frame_to_atoms(frame, type_map)
        box_arr = atoms.cell.diagonal()          # [Lx, Ly, Lz]
        pos     = atoms.positions.copy()
        types   = np.array(atoms.info["lammps_type"])

        # freud requires positions centred in [-L/2, L/2]
        pos_c = pos - box_arr / 2.0

        fbox = freud.box.Box(Lx=box_arr[0], Ly=box_arr[1], Lz=box_arr[2])

        for t1, t2, label in pairs:
            pts1 = pos_c[types == t1].astype(np.float32)
            pts2 = pos_c[types == t2].astype(np.float32)
            if len(pts1) == 0 or len(pts2) == 0:
                continue
            rdf_objs[label].compute(
                (fbox, pts2),
                query_points=pts1,
                reset=False,
            )

        used_frames += 1
        frame_count += 1

        if verbose and used_frames % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = used_frames / elapsed
            print(f"  frame {frame_count} | {used_frames} used "
                  f"| {rate:.1f} frames/s "
                  f"| timestep {frame['timestep']}")

    elapsed = time.perf_counter() - t_start
    if verbose:
        if used_frames:
            print(f"  Done: {used_frames} frames in {elapsed:.1f} s "
                  f"({elapsed/used_frames*1000:.1f} ms/frame)")
        else:
            print(f"  Done: 0 frames in {elapsed:.1f} s")

    # Extract results
    results = {}
    for _, _, label in pairs:
        rdf = rdf_objs[label]
        results[label] = (rdf.bin_centers.copy(), rdf.rdf.copy())

    return results


def plot_rdf(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output: Path,
    title: str = "Radial Distribution Functions",
):
    """Plot all RDFs on a single figure and save to output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    for idx, (label, (r, gr)) in enumerate(results.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(r, gr, color=colors[idx % 10], linewidth=1.5, label=label)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("r (Å)", fontsize=11)
        ax.set_ylabel("g(r)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, r[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {output}")


def save_data(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output: Path,
):
    """Save RDF data as a single CSV: r, g(r)_pair1, g(r)_pair2, ..."""
    labels = list(results.keys())
    r = results[labels[0]][0]

    header = "r_Angstrom," + ",".join(labels)
    data = np.column_stack([r] + [gr for _, gr in results.values()])

    np.savetxt(output, data, delimiter=",", header=header, comments="")
    print(f"Saved data  → {output}")


def resolve_type_context(
    lammpstrj: Path,
    lammps_input: Optional[Path] = None,
) -> Tuple[Dict[int, str], Dict[int, str], Optional[Path]]:
    """Resolve type→element and type→label maps for one trajectory."""
    input_path = find_lammps_input(lammpstrj, lammps_input)
    type_map = DEFAULT_TYPE_MAP
    type_labels = dict(TYPE_LABELS)
    if input_path:
        type_map = auto_detect_type_map(input_path)
        detected_labels = parse_type_label_map(input_path)
        type_labels.update(detected_labels)
        print(f"Type map from {input_path}: {type_map}")
        if detected_labels:
            print(f"Type labels from {input_path}: {detected_labels}")
    return type_map, type_labels, input_path


def run_single_rdf(
    lammpstrj: Path,
    pair_types: List[Tuple[int, int]],
    args,
    output: Optional[Path] = None,
    title_prefix: str = "RDF",
) -> bool:
    """Compute and write RDF outputs for one trajectory."""
    if not lammpstrj.exists():
        print(f"ERROR: {lammpstrj} not found", file=sys.stderr)
        return False

    type_map, type_labels, _ = resolve_type_context(lammpstrj, args.lammps_input)
    pairs = build_pairs(pair_types, type_labels)

    print(f"Trajectory: {lammpstrj}")
    print(f"Pairs: {[p[2] for p in pairs]}")
    print(f"r_max={args.r_max} Å, bins={args.bins}, stride={args.stride}")
    if args.start or args.end:
        print(f"Frame range: [{args.start}, {args.end})")
    print()

    stem = lammpstrj.stem
    plot_out = output or lammpstrj.parent / f"{stem}_rdf.png"
    csv_out = plot_out.with_suffix(".csv")

    results = compute_rdf(
        lammpstrj,
        pairs,
        type_map,
        r_max=args.r_max,
        bins=args.bins,
        stride=args.stride,
        start=args.start,
        end=args.end,
    )

    n_frames_label = (
        f"frames {args.start}–{args.end or 'end'}, stride {args.stride}"
    )
    plot_rdf(results, plot_out, title=f"{title_prefix} — {stem} ({n_frames_label})")
    if not args.no_csv:
        save_data(results, csv_out)
    return True


def run_mm_rdf(args, pair_types: List[Tuple[int, int]]) -> None:
    """Compute RDF for every mm_N directory under args.mm."""
    run_dir = args.mm.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    mm_dirs = find_mm_dirs(run_dir)
    if not mm_dirs:
        print(f"ERROR: no mm_N directories found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"RDF mm_N sweep")
    print(f"Run directory: {run_dir}")
    print(f"mm_N found: {len(mm_dirs)}")
    print()

    processed = 0
    skipped = 0
    for step, mm_dir in mm_dirs:
        traj = find_lammpstrj_in_mm_dir(mm_dir)
        if traj is None:
            print(f"mm_{step}: no .lammpstrj found, skipping")
            skipped += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"mm_{step}: {traj.name}")
        print(f"{'=' * 60}")
        output = mm_dir / f"{args.prefix}.png"
        ok = run_single_rdf(
            traj,
            pair_types,
            args,
            output=output,
            title_prefix=f"RDF mm_{step}",
        )
        processed += int(ok)
        skipped += int(not ok)

    print(f"\nDone. Processed {processed} mm_N directories, skipped {skipped}.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RDF from a LAMMPS .lammpstrj file using freud."
    )
    parser.add_argument("lammpstrj", type=Path, nargs="?", help="LAMMPS dump file")
    parser.add_argument(
        "--mm", type=Path, default=None, metavar="RUN_DIR",
        help="Process every mm_N directory under RUN_DIR, e.g. --mm .",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output plot path (default: <lammpstrj_stem>_rdf.png)",
    )
    parser.add_argument(
        "--lammps-input", type=Path, default=None,
        help="LAMMPS input file for atom type→element mapping",
    )
    parser.add_argument(
        "--pairs", type=str, default=None,
        help="Pairs to compute, e.g. '2-2 1-2 3-2 6-2'. "
             "Uses LAMMPS type IDs. Default: all CES2 pairs.",
    )
    parser.add_argument("--r-max",  type=float, default=10.0, help="Max r in Å (default: 10)")
    parser.add_argument("--bins",   type=int,   default=200,  help="Number of bins (default: 200)")
    parser.add_argument("--stride", type=int,   default=1,    help="Use every Nth frame")
    parser.add_argument("--start",  type=int,   default=0,    help="First frame index (0-based)")
    parser.add_argument("--end",    type=int,   default=None, help="Last frame index (exclusive)")
    parser.add_argument("--no-csv", action="store_true",      help="Skip saving CSV data file")
    parser.add_argument("--prefix", type=str, default="rdf",
                        help="Output prefix for --mm mode (default: rdf)")
    args = parser.parse_args()

    if args.mm is not None and args.lammpstrj is not None:
        print("ERROR: provide either a trajectory file or --mm, not both.", file=sys.stderr)
        sys.exit(1)
    if args.mm is not None and args.output is not None:
        print("ERROR: --output is only supported for single trajectory mode.", file=sys.stderr)
        sys.exit(1)
    if args.mm is None and args.lammpstrj is None:
        print("ERROR: provide a trajectory file or --mm RUN_DIR.", file=sys.stderr)
        sys.exit(1)

    # --- Pairs ---
    pair_types = parse_pair_types(args.pairs) if args.pairs else DEFAULT_PAIR_TYPES

    if args.mm is not None:
        run_mm_rdf(args, pair_types)
    else:
        ok = run_single_rdf(args.lammpstrj, pair_types, args, output=args.output)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
