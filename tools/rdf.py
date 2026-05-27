#!/usr/bin/env python3
"""
rdf.py — Radial Distribution Function calculator for LAMMPS/ASE trajectories
============================================================================
Uses freud (C++ backend) for fast neighbor-list based RDF computation.

Benchmark (ces2.emd.lammpstrj, 10365 atoms):
  Ow-Ow  (~2850 atoms/type): ~6 ms/frame  → full 1274 frames in ~8 s
  Hw-Ow  (~5700+2850 atoms): ~12 ms/frame → full 1274 frames in ~15 s
  Speedup vs pure NumPy: ~22×

Usage
-----
  # All default pairs, all frames
  python tools/rdf.py test/ces2.emd.lammpstrj
  python tools/rdf.py test/ces2.emd.wrapped.traj

  # Custom pairs (LAMMPS type IDs)
  python tools/rdf.py test/ces2.emd.lammpstrj --pairs "2-2 1-2 3-2 6-2"
  python tools/rdf.py test/ces2.emd.wrapped.traj --pairs "O_oh-K"

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
import plot_setting

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
    3: "K",
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


def parse_pair_types(
    pair_str: str,
    type_labels: Optional[Dict[int, str]] = None,
) -> List[Tuple[int, int]]:
    """
    Parse pair specs like '2-2 O_oh-K' into [(2,2), (5,3)].
    """
    label_to_type = {}
    if type_labels:
        label_to_type = {label: tid for tid, label in type_labels.items()}

    def _resolve(part: str) -> int:
        if part.isdigit():
            return int(part)
        if part in label_to_type:
            return int(label_to_type[part])
        known = ", ".join(sorted(label_to_type)) if label_to_type else "none"
        raise ValueError(
            f"Bad pair spec '{part}': expected a LAMMPS type ID or known "
            f"type_label. Known labels: {known}"
        )

    result = []
    for token in pair_str.split():
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Bad pair spec '{token}': expected 'T1-T2' or 'LABEL1-LABEL2'")
        t1, t2 = _resolve(parts[0]), _resolve(parts[1])
        result.append((t1, t2))
    return result


def find_mm_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Find mm_N directories sorted by N."""
    mm_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and re.match(r"^mm_(\d+)$", d.name):
            mm_dirs.append((int(d.name.split("_")[1]), d))
    return sorted(mm_dirs, key=lambda x: x[0])


def find_rdf_traj_in_mm_dir(mm_dir: Path) -> Optional[Path]:
    """Find the trajectory to use for RDF inside one mm_N directory."""
    for pattern in ("*.wrapped.traj", "*.traj", "*.emd.lammpstrj", "*.lammpstrj"):
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


def infer_type_labels_from_traj(traj_path: Path) -> Dict[int, str]:
    """Infer simple type labels from the first ASE .traj frame's symbols."""
    if traj_path.suffix.lower() != ".traj":
        return {}
    try:
        from ase.io.trajectory import Trajectory as ASETraj
    except ImportError:
        return {}

    traj = ASETraj(str(traj_path), mode="r")
    try:
        if len(traj) == 0:
            return {}
        atoms = traj[0]
        if "lammps_type" not in atoms.info:
            return {}
        inferred: Dict[int, str] = {}
        symbols = atoms.get_chemical_symbols()
        types = atoms.info["lammps_type"]
        for tid, sym in zip(types, symbols):
            tid = int(tid)
            if tid not in inferred:
                inferred[tid] = sym
        # Preserve known CES2 water/OH/slab labels; use symbols for variable ions.
        for tid in (1, 2, 4, 5, 6, 7):
            if tid in TYPE_LABELS:
                inferred[tid] = TYPE_LABELS[tid]
        return inferred
    finally:
        traj.close()


def iter_rdf_frames(traj_path: Path, type_map: Dict[int, str]):
    """Yield frame data needed by freud from .lammpstrj or ASE .traj input."""
    suffix = traj_path.suffix.lower()
    if suffix == ".lammpstrj":
        for frame in iter_frames(traj_path):
            atoms = frame_to_atoms(frame, type_map)
            yield {
                "timestep": frame.get("timestep"),
                "box": atoms.cell.diagonal(),
                "positions": atoms.positions.copy(),
                "types": np.array(atoms.info["lammps_type"]),
            }
    elif suffix == ".traj":
        try:
            from ase.io.trajectory import Trajectory as ASETraj
        except ImportError:
            print(
                "ERROR: .traj input requires ASE. Install ase in this environment.",
                file=sys.stderr,
            )
            sys.exit(1)

        traj = ASETraj(str(traj_path), mode="r")
        try:
            for atoms in traj:
                if "lammps_type" not in atoms.info:
                    print(
                        "ERROR: .traj input is missing atoms.info['lammps_type']. "
                        "Convert from .lammpstrj with tools/lammpstrj_to_traj.py.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                yield {
                    "timestep": atoms.info.get("timestep"),
                    "box": atoms.cell.diagonal(),
                    "positions": atoms.positions.copy(),
                    "types": np.array(atoms.info["lammps_type"]),
                }
        finally:
            traj.close()
    else:
        print(
            f"ERROR: unsupported trajectory format '{traj_path.suffix}'. "
            "Use .lammpstrj or .traj.",
            file=sys.stderr,
        )
        sys.exit(1)


def compute_rdf(
    traj_path: Path,
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
        from freud.density import RDF
        from freud.box import Box
    except ImportError:
        print(
            "ERROR: rdf.py requires freud-analysis, imported as 'freud'.\n"
            "The PyPI package named 'freud' is a different package and does "
            "not provide freud.density.\n"
            "Recommended pip fix:\n"
            "  pip uninstall freud\n"
            "  pip install freud-analysis\n"
            "  pip install 'prompt-toolkit>=3.0.30,<3.1.0'\n"
            "Conda alternative:\n"
            "  conda install -c conda-forge freud",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialise one freud RDF object per pair
    rdf_objs = {}
    for t1, t2, label in pairs:
        rdf_objs[label] = RDF(bins=bins, r_max=r_max)

    frame_count = 0
    used_frames = 0
    t_start = time.perf_counter()

    for frame in iter_rdf_frames(traj_path, type_map):
        if frame_count < start:
            frame_count += 1
            continue
        if end is not None and frame_count >= end:
            break
        if (frame_count - start) % stride != 0:
            frame_count += 1
            continue

        box_arr = frame["box"]          # [Lx, Ly, Lz]
        pos     = frame["positions"]
        types   = frame["types"]

        # freud requires positions centred in [-L/2, L/2]
        pos_c = pos - box_arr / 2.0

        fbox = Box(Lx=box_arr[0], Ly=box_arr[1], Lz=box_arr[2])

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
    traj_path: Path,
    lammps_input: Optional[Path] = None,
) -> Tuple[Dict[int, str], Dict[int, str], Optional[Path]]:
    """Resolve type→element and type→label maps for one trajectory."""
    input_path = find_lammps_input(traj_path, lammps_input)
    type_map = DEFAULT_TYPE_MAP
    type_labels = dict(TYPE_LABELS)
    inferred_labels = infer_type_labels_from_traj(traj_path)
    if inferred_labels:
        type_labels.update(inferred_labels)
    if input_path:
        type_map = auto_detect_type_map(input_path)
        detected_labels = parse_type_label_map(input_path)
        type_labels.update(detected_labels)
        print(f"Type map from {input_path}: {type_map}")
        if detected_labels:
            print(f"Type labels from {input_path}: {detected_labels}")
    elif inferred_labels:
        print(f"Type labels inferred from {traj_path.name}: {inferred_labels}")
    return type_map, type_labels, input_path


def run_single_rdf(
    traj_path: Path,
    args,
    output: Optional[Path] = None,
    title_prefix: str = "RDF",
) -> bool:
    """Compute and write RDF outputs for one trajectory."""
    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found", file=sys.stderr)
        return False

    type_map, type_labels, _ = resolve_type_context(traj_path, args.lammps_input)
    pair_types = (
        parse_pair_types(args.pairs, type_labels)
        if args.pairs else DEFAULT_PAIR_TYPES
    )
    pairs = build_pairs(pair_types, type_labels)

    print(f"Trajectory: {traj_path}")
    print(f"Pairs: {[p[2] for p in pairs]}")
    print(f"r_max={args.r_max} Å, bins={args.bins}, stride={args.stride}")
    if args.start or args.end:
        print(f"Frame range: [{args.start}, {args.end})")
    print()

    stem = traj_path.stem
    plot_out = output or traj_path.parent / f"{stem}_rdf.png"
    csv_out = plot_out.with_suffix(".csv")

    results = compute_rdf(
        traj_path,
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


def run_mm_rdf(args) -> None:
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
        traj = find_rdf_traj_in_mm_dir(mm_dir)
        if traj is None:
            print(f"mm_{step}: no .traj/.lammpstrj found, skipping")
            skipped += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"mm_{step}: {traj.name}")
        print(f"{'=' * 60}")
        output = mm_dir / f"{args.prefix}.png"
        ok = run_single_rdf(
            traj,
            args,
            output=output,
            title_prefix=f"RDF mm_{step}",
        )
        processed += int(ok)
        skipped += int(not ok)

    print(f"\nDone. Processed {processed} mm_N directories, skipped {skipped}.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute RDF from a LAMMPS .lammpstrj or ASE .traj file using freud."
    )
    parser.add_argument("traj", type=Path, nargs="?",
                        help="Trajectory file (.lammpstrj or .traj)")
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
        help="Pairs to compute, e.g. '2-2 1-2' or 'O_oh-K'. "
             "Uses LAMMPS type IDs or type_labels. Default: all CES2 pairs.",
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

    if args.mm is not None and args.traj is not None:
        print("ERROR: provide either a trajectory file or --mm, not both.", file=sys.stderr)
        sys.exit(1)
    if args.mm is not None and args.output is not None:
        print("ERROR: --output is only supported for single trajectory mode.", file=sys.stderr)
        sys.exit(1)
    if args.mm is None and args.traj is None:
        print("ERROR: provide a trajectory file or --mm RUN_DIR.", file=sys.stderr)
        sys.exit(1)

    if args.mm is not None:
        run_mm_rdf(args)
    else:
        ok = run_single_rdf(args.traj, args, output=args.output)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
