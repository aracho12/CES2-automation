#!/usr/bin/env python3
"""
view_lammps.py  — LAMMPS dump / trajectory viewer with correct element mapping

Supports:
  .dump         single-frame or multi-frame LAMMPS custom dump
  .lammpstrj    LAMMPS trajectory (same format, multi-frame)

Usage examples
--------------
  # Open in ASE GUI directly
  python view_lammps.py minimized.dump
  python view_lammps.py relax.lammpstrj
  python view_lammps.py min_traj.lammpstrj --data data.file

  # Convert to .traj only (no GUI)
  python view_lammps.py relax.lammpstrj --no-gui --save traj

  # Show info only (no GUI, no save)
  python view_lammps.py minimized.dump --info

  # Load specific frames (Python slice syntax)
  python view_lammps.py relax.lammpstrj --frames "0:100:10"  # every 10th frame
  python view_lammps.py relax.lammpstrj --frames "-1"         # last frame only

  # Manual element mapping (when data.file is unavailable)
  python view_lammps.py minimized.dump --types 1=H 2=O 3=K 4=H 5=O 6=Ir 7=O
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# ── Element matching ───────────────────────────────────────────────────────────

def find_data_file(start_dir: Path) -> Path | None:
    """Search for data.file in current dir and up to 3 parent dirs."""
    for d in [start_dir] + list(start_dir.parents)[:3]:
        for name in ("data.file", "data.lammps", "structure.data"):
            p = d / name
            if p.exists():
                return p
    return None


def read_type_map_from_data(data_file: Path) -> dict[int, str]:
    """Read Masses section → {type_int: element_symbol}."""
    from ase.data import chemical_symbols, atomic_masses

    type_map: dict[int, str] = {}
    in_masses = False
    comment_map: dict[int, str] = {}  # from inline comments like  # Ow

    with open(data_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if stripped == "":
                    continue
                # Stop at next section header
                parts = stripped.split()
                if len(parts) < 2:
                    break
                try:
                    t = int(parts[0])
                    mass = float(parts[1])
                except ValueError:
                    break

                # Match mass → element
                diffs = np.abs(atomic_masses - mass)
                diffs[0] = 1e9
                elem = chemical_symbols[int(np.argmin(diffs))]

                # Try to extract label from comment (e.g. "  1.008  # Hw")
                if "#" in line:
                    comment = line.split("#", 1)[1].strip().split()[0]  # first token
                    comment_map[t] = comment  # store label for display

                type_map[t] = elem

    return type_map, comment_map


def parse_manual_types(type_strs: list[str]) -> dict[int, str]:
    """Parse  "1=H"  "2=O"  etc. into {int: str}."""
    result = {}
    for s in type_strs:
        if "=" not in s:
            raise ValueError(f"Expected format N=ELEM, got: '{s}'")
        k, v = s.split("=", 1)
        result[int(k.strip())] = v.strip()
    return result


# ── ASE helpers ───────────────────────────────────────────────────────────────

def apply_type_map(atoms, type_map: dict[int, str]) -> None:
    """Fix element symbols — handles ASE type-as-atomic-number issue."""
    type_key = next(
        (k for k in ("type", "lammps_type", "atom_types") if k in atoms.arrays),
        None,
    )
    types = atoms.arrays[type_key] if type_key else atoms.numbers
    symbols = [type_map.get(int(t), "X") for t in types]
    atoms.set_chemical_symbols(symbols)


def read_dump(
    dump_file: Path,
    frames_slice: str = ":",
    type_map: dict[int, str] | None = None,
    verbose: bool = True,
) -> list:
    from ase.io import read

    if verbose:
        print(f"Reading '{dump_file}' ...")

    frames = read(str(dump_file), format="lammps-dump-text", index=frames_slice)
    if not isinstance(frames, list):
        frames = [frames]

    if verbose:
        print(f"  {len(frames)} frame(s), {len(frames[0])} atoms/frame")

    if type_map:
        for atoms in frames:
            apply_type_map(atoms, type_map)
    return frames


# ── Output helpers ────────────────────────────────────────────────────────────

def save_traj(frames: list, out_path: Path, verbose: bool = True) -> None:
    from ase.io.trajectory import Trajectory

    if verbose:
        print(f"Writing '{out_path}' ...")
    with Trajectory(str(out_path), "w") as traj:
        for atoms in frames:
            traj.write(atoms)
    if verbose:
        print(f"Done! → ase gui {out_path}")


def print_info(frames: list, type_map: dict, comment_map: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"Frames  : {len(frames)}")
    print(f"Atoms   : {len(frames[0])}")
    print(f"\nType → element map:")
    for t in sorted(type_map.keys()):
        label = f"  ({comment_map[t]})" if t in comment_map else ""
        print(f"  type {t} → {type_map[t]}{label}")
    print(f"\nElement counts (last frame):")
    counts = Counter(frames[-1].get_chemical_symbols())
    for sym, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {sym:3s}: {n}")

    # Box info from last frame
    cell = frames[-1].get_cell()
    if cell is not None:
        a, b, c = cell.lengths()
        print(f"\nBox (last frame): {a:.3f} × {b:.3f} × {c:.3f} Å")
    print(f"{'─'*50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="view_lammps.py",
        description="View LAMMPS .dump / .lammpstrj files with ASE GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )
    p.add_argument("dump", help=".dump or .lammpstrj file")
    p.add_argument(
        "--data", metavar="FILE",
        help="LAMMPS data.file for element mapping (auto-searched if omitted)"
    )
    p.add_argument(
        "--types", nargs="+", metavar="N=ELEM",
        help="Manual type mapping, e.g. --types 1=H 2=O 3=K (overrides data.file)"
    )
    p.add_argument(
        "--frames", default=":", metavar="SLICE",
        help="Frames to load as Python slice (default: all). "
             "Examples: ':' all, '-1' last, '0:50' first 50, '::10' every 10th"
    )
    p.add_argument(
        "--no-gui", action="store_true",
        help="Do not open ASE GUI (useful with --save or --info)"
    )
    p.add_argument(
        "--save", nargs="?", const="traj", choices=["traj", "extxyz"],
        help="Save converted trajectory. 'traj' (default) or 'extxyz'"
    )
    p.add_argument(
        "--info", action="store_true",
        help="Print frame / atom / element info and exit"
    )
    p.add_argument(
        "--repeat", nargs=3, type=int, metavar=("NX","NY","NZ"),
        help="Tile the unit cell for display, e.g. --repeat 2 2 1"
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dump_file = Path(args.dump)
    if not dump_file.exists():
        print(f"Error: file not found: {dump_file}", file=sys.stderr)
        sys.exit(1)

    # ── Build type map ──────────────────────────────────────────────────
    type_map: dict[int, str] = {}
    comment_map: dict[int, str] = {}

    if args.types:
        type_map = parse_manual_types(args.types)
        print(f"Using manual type map: {type_map}")
    else:
        data_path = Path(args.data) if args.data else find_data_file(dump_file.parent)
        if data_path and data_path.exists():
            type_map, comment_map = read_type_map_from_data(data_path)
            print(f"Type map loaded from: '{data_path}'")
        else:
            print("Warning: no data.file found — elements will show as LAMMPS type numbers.")
            print("  Use --data <file>  or  --types 1=H 2=O ...  to fix this.")

    # ── Read frames ─────────────────────────────────────────────────────
    frames = read_dump(dump_file, frames_slice=args.frames,
                       type_map=type_map or None, verbose=True)

    # ── Info ─────────────────────────────────────────────────────────────
    print_info(frames, type_map, comment_map)

    if args.info:
        return

    # ── Optionally tile ──────────────────────────────────────────────────
    if args.repeat:
        nx, ny, nz = args.repeat
        frames = [atoms.repeat((nx, ny, nz)) for atoms in frames]
        print(f"Tiled {nx}×{ny}×{nz} → {len(frames[0])} atoms")

    # ── Save ─────────────────────────────────────────────────────────────
    if args.save:
        if args.save == "traj":
            out = dump_file.with_suffix(".traj")
            save_traj(frames, out)
        elif args.save == "extxyz":
            from ase.io import write as ase_write
            out = dump_file.with_suffix(".xyz")
            ase_write(str(out), frames, format="extxyz")
            print(f"Saved '{out}'")

    # ── ASE GUI ──────────────────────────────────────────────────────────
    if not args.no_gui:
        from ase.visualize import view
        print(f"Opening ASE GUI ({len(frames)} frame(s)) ...")
        view(frames)


if __name__ == "__main__":
    main()
