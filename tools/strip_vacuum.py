#!/usr/bin/env python3
"""
strip_vacuum.py  —  Remove vacuum from LAMMPS data.file / dump / traj
                    and write back as extxyz (combined.xyz style)

Typical use-case
----------------
LAMMPS slab simulations use boundary p p f with a vacuum gap in z.
This script strips that vacuum so the box matches the original combined.xyz.

The z-cell is reset to:   max_atom_z - min_atom_z + z_buffer  (default 1.0 Å)
Atoms are shifted so that   min_atom_z  → 0.0  (optional, default ON).
PBC is set to T T T in the output (periodic in all directions).

Usage
-----
  # data.file → xyz  (strips vacuum, shifts atoms, pbc T T T)
  python strip_vacuum.py data.file

  # equilibrated.data → xyz
  python strip_vacuum.py equilibrated.data

  # trajectory (strip vacuum from every frame)
  python strip_vacuum.py relax.lammpstrj

  # Custom output name
  python strip_vacuum.py data.file -o stripped.xyz

  # Keep vacuum (no resize), just convert format
  python strip_vacuum.py data.file --no-strip

  # Don't shift atoms to z=0
  python strip_vacuum.py data.file --no-shift

  # z_buffer other than 1.0 Å
  python strip_vacuum.py data.file --buffer 2.0
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np


# ── Element mapping (shared with view_lammps.py) ───────────────────────────────

def read_type_map(data_file: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Read Masses section → ({type: symbol}, {type: label})."""
    from ase.data import chemical_symbols, atomic_masses

    type_map:    dict[int, str] = {}
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
                diffs    = np.abs(atomic_masses - mass)
                diffs[0] = 1e9
                type_map[t] = chemical_symbols[int(np.argmin(diffs))]
                if "#" in line:
                    label = line.split("#", 1)[1].strip().split()[0]
                    comment_map[t] = label

    return type_map, comment_map


def parse_manual_types(type_strs: list[str]) -> dict[int, str]:
    result = {}
    for s in type_strs:
        k, v = s.split("=", 1)
        result[int(k.strip())] = v.strip()
    return result


def find_data_file(start_dir: Path) -> Path | None:
    for d in [start_dir] + list(start_dir.parents)[:3]:
        for name in ("data.file", "data.lammps", "structure.data"):
            p = d / name
            if p.exists():
                return p
    return None


# ── Read helpers ──────────────────────────────────────────────────────────────

_DATA_NAMES = {"data.file", "data.lammps", "structure.data"}

def _is_data_file(p: Path) -> bool:
    if p.name in _DATA_NAMES or p.suffix == ".data":
        return True
    if p.suffix in (".dump", ".lammpstrj"):
        return False
    try:
        with open(p) as f:
            for line in f:
                if line.strip().startswith("ITEM:"):
                    return False
                if line.strip():
                    for line2 in f:
                        if "atoms" in line2 and not line2.startswith("ITEM"):
                            return True
                        if line2.strip().startswith("ITEM:"):
                            return False
                    break
    except Exception:
        pass
    return False


def read_frames(
    input_file: Path,
    type_map:   dict[int, str],
    frames_slice: str = ":",
) -> list:
    from ase.io import read
    from ase.data import atomic_numbers

    Z_of_type = {t: atomic_numbers.get(sym, 1) for t, sym in type_map.items()}

    if _is_data_file(input_file):
        atoms = read(str(input_file), format="lammps-data",
                     Z_of_type=Z_of_type, style="full")
        return [atoms]
    else:
        frames = read(str(input_file), format="lammps-dump-text",
                      index=frames_slice)
        if not isinstance(frames, list):
            frames = [frames]
        # fix element mapping
        for atoms in frames:
            type_key = next(
                (k for k in ("type", "lammps_type", "atom_types")
                 if k in atoms.arrays), None,
            )
            types   = atoms.arrays[type_key] if type_key else atoms.numbers
            symbols = [type_map.get(int(t), "X") for t in types]
            atoms.set_chemical_symbols(symbols)
        return frames


# ── Vacuum stripping ──────────────────────────────────────────────────────────

def strip_vacuum_z(atoms, z_buffer: float = 1.0, shift_to_zero: bool = True):
    """
    Remove z vacuum from an Atoms object.

    - Sets cell[2][2] = (max_z - min_z) + z_buffer
    - Optionally translates atoms so min_z = 0
    - Sets pbc = [True, True, True]
    """
    atoms = atoms.copy()
    pos   = atoms.get_positions()
    z     = pos[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())

    if shift_to_zero and z_min != 0.0:
        pos[:, 2] -= z_min
        atoms.set_positions(pos)
        z_max -= z_min
        z_min  = 0.0

    # Rebuild cell: keep Lx, Ly, set Lz = atom_span + buffer
    cell    = atoms.get_cell().array.copy()
    old_lz  = float(cell[2][2])
    new_lz  = (z_max - z_min) + z_buffer
    cell[2][2] = new_lz
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, True])

    return atoms, old_lz, new_lz


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="strip_vacuum.py",
        description="Strip z vacuum from LAMMPS data.file / dump → extxyz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="LAMMPS data.file, *.data, *.dump, or *.lammpstrj")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="Output file (default: <input>.xyz)")
    p.add_argument("--data", metavar="FILE",
                   help="LAMMPS data.file for element mapping (dump/traj only)")
    p.add_argument("--types", nargs="+", metavar="N=ELEM",
                   help="Manual type mapping, e.g. --types 1=H 2=O 3=K")
    p.add_argument("--frames", default=":", metavar="SLICE",
                   help="Frames to process (dump/traj only). Default: all")
    p.add_argument("--no-strip", action="store_true",
                   help="Skip vacuum removal — only convert format")
    p.add_argument("--no-shift", action="store_true",
                   help="Keep original z positions (don't shift min_z → 0)")
    p.add_argument("--buffer", type=float, default=1.0, metavar="Å",
                   help="z buffer added above atoms after stripping (default: 1.0 Å)")
    p.add_argument("--gui", action="store_true",
                   help="Open ASE GUI after conversion")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # ── Type map ──────────────────────────────────────────────────────────
    type_map:    dict[int, str] = {}
    comment_map: dict[int, str] = {}

    if args.types:
        type_map = parse_manual_types(args.types)
    else:
        if _is_data_file(input_file):
            data_path = input_file
        elif args.data:
            data_path = Path(args.data)
        else:
            data_path = find_data_file(input_file.parent)

        if data_path and data_path.exists():
            type_map, comment_map = read_type_map(data_path)
            print(f"Type map from: '{data_path}'")
            for t in sorted(type_map.keys()):
                label = f"  ({comment_map[t]})" if t in comment_map else ""
                print(f"  type {t} → {type_map[t]}{label}")
        else:
            print("Warning: no data.file found for element mapping.", file=sys.stderr)

    # ── Read ──────────────────────────────────────────────────────────────
    print(f"\nReading '{input_file}' ...")
    frames = read_frames(input_file, type_map, frames_slice=args.frames)
    print(f"  {len(frames)} frame(s), {len(frames[0])} atoms")

    # ── Strip vacuum ──────────────────────────────────────────────────────
    if not args.no_strip:
        stripped = []
        for atoms in frames:
            new_atoms, old_lz, new_lz = strip_vacuum_z(
                atoms,
                z_buffer=args.buffer,
                shift_to_zero=not args.no_shift,
            )
            stripped.append(new_atoms)
        frames = stripped

        cell = frames[0].get_cell()
        lx, ly, lz = cell.lengths()
        print(f"\nVacuum stripped:")
        print(f"  Old Lz = {old_lz:.4f} Å")
        print(f"  New Lz = {new_lz:.4f} Å  (max_atom_z + {args.buffer:.1f} Å buffer)")
        print(f"  Box    = {lx:.4f} × {ly:.4f} × {lz:.4f} Å")
    else:
        cell = frames[0].get_cell()
        lx, ly, lz = cell.lengths()
        print(f"\nBox: {lx:.4f} × {ly:.4f} × {lz:.4f} Å  (vacuum not stripped)")

    # ── Element counts ────────────────────────────────────────────────────
    counts = Counter(frames[-1].get_chemical_symbols())
    print(f"\nElement counts:")
    for sym, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {sym:3s}: {n}")

    # ── Write extxyz ──────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else input_file.with_suffix(".xyz")
    from ase.io import write as ase_write
    ase_write(str(out_path), frames, format="extxyz")
    print(f"\nSaved → '{out_path}'")
    print(f"  (open with: ase gui {out_path})")

    # ── Optional GUI ──────────────────────────────────────────────────────
    if args.gui:
        from ase.visualize import view
        print("Opening ASE GUI ...")
        view(frames)


if __name__ == "__main__":
    main()
