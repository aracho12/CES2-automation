#!/usr/bin/env python3
"""
view_lammps.py  — LAMMPS structure / trajectory viewer with correct element mapping

Supports:
  data.file     LAMMPS data file (structure, single frame)
  .dump         single-frame or multi-frame LAMMPS custom dump
  .lammpstrj    LAMMPS trajectory (same format, multi-frame)

Usage examples
--------------
  # Open data.file directly in ASE GUI
  python view_lammps.py data.file
  python view_lammps.py equilibrated.data

  # Open dump / trajectory in ASE GUI
  python view_lammps.py minimized.dump
  python view_lammps.py relax.lammpstrj

  # data.file for element mapping (auto-searched if omitted)
  python view_lammps.py relax.lammpstrj --data data.file

  # Convert to .traj only (no GUI)
  python view_lammps.py relax.lammpstrj --no-gui --save traj

  # Show info only (no GUI, no save)
  python view_lammps.py minimized.dump --info

  # Load specific frames (Python slice syntax)
  python view_lammps.py relax.lammpstrj --frames "0:100:10"  # every 10th frame
  python view_lammps.py relax.lammpstrj --frames "-1"         # last frame only

  # Manual element mapping (when data.file is unavailable)
  python view_lammps.py minimized.dump --types 1=H 2=O 3=K 4=H 5=O 6=Ir 7=O

  # Tile the supercell for display
  python view_lammps.py data.file --repeat 2 2 1
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# ── File-type detection ────────────────────────────────────────────────────────

_DATA_FILE_NAMES = {"data.file", "data.lammps", "structure.data"}
_DUMP_SUFFIXES   = {".dump", ".lammpstrj", ".lammps"}

def _is_data_file(p: Path) -> bool:
    """Heuristic: named data.file / *.data / *.lammps OR contains 'atoms' header."""
    if p.name in _DATA_FILE_NAMES or p.suffix in (".data",):
        return True
    if p.suffix in _DUMP_SUFFIXES:
        return False
    # Peek at first non-empty line — LAMMPS data files start with a comment
    # then a blank line, then atom count; dump files start with "ITEM: TIMESTEP"
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ITEM:"):
                    return False
                if line:
                    # data files: first non-blank line is a free-form comment
                    # next meaningful content will be "N atoms" — check a few lines
                    for line2 in f:
                        if "atoms" in line2 and not line2.startswith("ITEM"):
                            return True
                        if line2.strip().startswith("ITEM:"):
                            return False
                    break
    except Exception:
        pass
    return False


# ── Element matching ───────────────────────────────────────────────────────────

def find_data_file(start_dir: Path) -> Path | None:
    """Search for data.file in current dir and up to 3 parent dirs."""
    for d in [start_dir] + list(start_dir.parents)[:3]:
        for name in _DATA_FILE_NAMES:
            p = d / name
            if p.exists():
                return p
    return None


def _mass_to_elem(mass: float) -> str:
    from ase.data import chemical_symbols, atomic_masses
    diffs = np.abs(atomic_masses - mass)
    diffs[0] = 1e9   # exclude placeholder index 0
    return chemical_symbols[int(np.argmin(diffs))]


def read_type_map_from_data(data_file: Path) -> tuple[dict[int, str], dict[int, str]]:
    """
    Read Masses section → ({type_int: element_symbol}, {type_int: label_comment}).

    Also returns Z_of_type dict (type → atomic number) for ASE lammps-data reader.
    """
    type_map: dict[int, str] = {}
    comment_map: dict[int, str] = {}
    in_masses = False

    with open(data_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "Masses":
                in_masses = True
                continue
            if in_masses:
                if stripped == "":
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    break
                try:
                    t    = int(parts[0])
                    mass = float(parts[1])
                except ValueError:
                    break

                type_map[t] = _mass_to_elem(mass)

                if "#" in line:
                    label = line.split("#", 1)[1].strip().split()[0]
                    comment_map[t] = label

    return type_map, comment_map


def parse_manual_types(type_strs: list[str]) -> dict[int, str]:
    """Parse "1=H" "2=O" etc. → {int: str}."""
    result = {}
    for s in type_strs:
        if "=" not in s:
            raise ValueError(f"Expected format N=ELEM, got: '{s}'")
        k, v = s.split("=", 1)
        result[int(k.strip())] = v.strip()
    return result


# ── ASE read helpers ───────────────────────────────────────────────────────────

def apply_type_map(atoms, type_map: dict[int, str]) -> None:
    """Fix element symbols — handles ASE type-as-atomic-number issue."""
    type_key = next(
        (k for k in ("type", "lammps_type", "atom_types") if k in atoms.arrays),
        None,
    )
    types   = atoms.arrays[type_key] if type_key else atoms.numbers
    symbols = [type_map.get(int(t), "X") for t in types]
    atoms.set_chemical_symbols(symbols)


def read_data_file(
    data_file: Path,
    type_map:  dict[int, str],
    verbose:   bool = True,
) -> list:
    """Read a LAMMPS data.file → single-element list of Atoms."""
    from ase.io import read
    from ase.data import atomic_numbers

    if verbose:
        print(f"Reading data file '{data_file}' ...")

    # Build Z_of_type for ASE: {lammps_type: atomic_number}
    Z_of_type = {t: atomic_numbers.get(sym, 1) for t, sym in type_map.items()}

    atoms = read(str(data_file), format="lammps-data",
                 Z_of_type=Z_of_type, style="full")

    if verbose:
        print(f"  1 frame, {len(atoms)} atoms")

    return [atoms]


def read_dump(
    dump_file:    Path,
    frames_slice: str = ":",
    type_map:     dict[int, str] | None = None,
    verbose:      bool = True,
) -> list:
    """Read a LAMMPS dump / lammpstrj file → list of Atoms frames."""
    from ase.io import read

    if verbose:
        print(f"Reading dump '{dump_file}' ...")

    frames = read(str(dump_file), format="lammps-dump-text", index=frames_slice)
    if not isinstance(frames, list):
        frames = [frames]

    if len(frames) == 0:
        print(f"Error: no frames read from '{dump_file}'.", file=sys.stderr)
        print("  If this is a data.file, pass it directly (auto-detected by filename).", file=sys.stderr)
        sys.exit(1)

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


def print_info(
    frames: list,
    type_map: dict,
    comment_map: dict,
    file_kind: str,
) -> None:
    print(f"\n{'─'*52}")
    print(f"File type : {file_kind}")
    print(f"Frames    : {len(frames)}")
    print(f"Atoms     : {len(frames[0])}")

    if type_map:
        print(f"\nType → element map:")
        for t in sorted(type_map.keys()):
            label = f"  ({comment_map[t]})" if t in comment_map else ""
            print(f"  type {t} → {type_map[t]}{label}")

    print(f"\nElement counts (last frame):")
    counts = Counter(frames[-1].get_chemical_symbols())
    for sym, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {sym:3s}: {n}")

    cell = frames[-1].get_cell()
    if cell is not None and cell.any():
        a, b, c = cell.lengths()
        print(f"\nBox (last frame): {a:.4f} × {b:.4f} × {c:.4f} Å")
    print(f"{'─'*52}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="view_lammps.py",
        description="View LAMMPS data.file / .dump / .lammpstrj with ASE GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        help="LAMMPS file: data.file, *.data, *.dump, or *.lammpstrj",
    )
    p.add_argument(
        "--data", metavar="FILE",
        help="LAMMPS data.file for element mapping (auto-searched if omitted). "
             "Only needed when reading a dump/traj file.",
    )
    p.add_argument(
        "--types", nargs="+", metavar="N=ELEM",
        help="Manual type mapping, e.g. --types 1=H 2=O 3=K (overrides data.file)",
    )
    p.add_argument(
        "--frames", default=":", metavar="SLICE",
        help="Frames to load — Python slice syntax (dump/traj only). "
             "Examples: ':' all, '-1' last, '0:50' first 50, '::10' every 10th",
    )
    p.add_argument(
        "--no-gui", action="store_true",
        help="Do not open ASE GUI (useful with --save or --info)",
    )
    p.add_argument(
        "--save", nargs="?", const="traj", choices=["traj", "extxyz"],
        help="Save converted trajectory: 'traj' (default) or 'extxyz'",
    )
    p.add_argument(
        "--info", action="store_true",
        help="Print frame / atom / element info and exit (no GUI)",
    )
    p.add_argument(
        "--repeat", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
        help="Tile the unit cell for display, e.g. --repeat 2 2 1",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # ── Detect file type ────────────────────────────────────────────────
    is_data = _is_data_file(input_file)
    file_kind = "LAMMPS data file" if is_data else "LAMMPS dump/trajectory"

    # ── Build type map ──────────────────────────────────────────────────
    type_map:    dict[int, str] = {}
    comment_map: dict[int, str] = {}

    if args.types:
        type_map = parse_manual_types(args.types)
        print(f"Using manual type map: {type_map}")
    else:
        # For data.file: Masses section is inside it — parse from itself
        # For dump: look for a separate data.file
        data_path: Path | None = None
        if is_data:
            data_path = input_file
        elif args.data:
            data_path = Path(args.data)
        else:
            data_path = find_data_file(input_file.parent)

        if data_path and data_path.exists():
            type_map, comment_map = read_type_map_from_data(data_path)
            if not is_data:
                print(f"Type map loaded from: '{data_path}'")
        else:
            print("Warning: no data.file found — elements may be incorrect.")
            print("  Use --data <file>  or  --types 1=H 2=O ...  to fix this.")

    # ── Read file ───────────────────────────────────────────────────────
    if is_data:
        if not type_map:
            print("Error: could not build type map from data.file Masses section.",
                  file=sys.stderr)
            sys.exit(1)
        frames = read_data_file(input_file, type_map, verbose=True)
    else:
        frames = read_dump(input_file, frames_slice=args.frames,
                           type_map=type_map or None, verbose=True)

    # ── Info ─────────────────────────────────────────────────────────────
    print_info(frames, type_map, comment_map, file_kind)

    if args.info:
        return

    # ── Optionally tile ──────────────────────────────────────────────────
    if args.repeat:
        nx, ny, nz = args.repeat
        frames = [atoms.repeat((nx, ny, nz)) for atoms in frames]
        print(f"Tiled {nx}×{ny}×{nz} → {len(frames[0])} atoms")

    # ── Save ─────────────────────────────────────────────────────────────
    if args.save:
        out_stem = input_file.stem
        if args.save == "traj":
            out = input_file.with_suffix(".traj")
            save_traj(frames, out)
        elif args.save == "extxyz":
            from ase.io import write as ase_write
            out = input_file.with_suffix(".xyz")
            ase_write(str(out), frames, format="extxyz")
            print(f"Saved '{out}'")

    # ── ASE GUI ──────────────────────────────────────────────────────────
    if not args.no_gui:
        from ase.visualize import view
        print(f"Opening ASE GUI ({len(frames)} frame(s)) ...")
        view(frames)


if __name__ == "__main__":
    main()
