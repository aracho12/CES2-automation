#!/usr/bin/env python3
"""
lammpstrj_to_traj.py
====================
Convert a LAMMPS dump trajectory (.lammpstrj) to an ASE trajectory (.traj)
that can be viewed with `ase gui`.

Usage
-----
  # Basic conversion (auto-detect atom types from a LAMMPS input file)
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj

  # Specify output path
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj -o output/ces2.traj

  # Override type→element mapping explicitly
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj \\
      --type-map "1:H 2:O 3:K 4:H 5:O 6:Ir 7:O"

  # Convert only a subset of frames (e.g. every 10th frame)
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj --stride 10

  # Count frames only (no conversion)
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj --count-only

  # View directly with ASE GUI after conversion
  python tools/lammpstrj_to_traj.py test/ces2.emd.lammpstrj --view
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Default type→element mapping for CES2 system
#   1: Hw  (water H)
#   2: Ow  (water O)
#   3: K
#   4: H_oh (hydroxyl H)
#   5: O_oh (hydroxyl O)
#   6: Ir
#   7: O   (surface O)
# ---------------------------------------------------------------------------
DEFAULT_TYPE_MAP = {
    1: "H",
    2: "O",
    3: "K",
    4: "H",
    5: "O",
    6: "Ir",
    7: "O",
}


def parse_type_map_str(s: str) -> Dict[int, str]:
    """Parse '1:H 2:O 3:K ...' into {1: 'H', 2: 'O', 3: 'K', ...}."""
    result = {}
    for token in s.split():
        t, elem = token.split(":")
        result[int(t)] = elem
    return result


def auto_detect_type_map(lammps_input: Path) -> Dict[int, str]:
    """
    Try to extract type→element mapping from a LAMMPS input file by reading
    group lines of the form:
      group  NAME  type N [M ...]  # label (element) ...

    Strategy:
      1. Start with DEFAULT_TYPE_MAP as base.
      2. Parse each group line; extract all valid element symbols from the
         entire comment (including inside parentheses).
      3. When a group has exactly one type ID and at least one element found,
         map that type to the first element.
      4. When a group has multiple type IDs and the same number of elements
         found in the comment, map them 1-to-1 in order.
      5. Auto-detected entries override DEFAULT_TYPE_MAP; the rest stay.
    """
    element_symbols = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    }

    # Start from the default; auto-detect will override specific entries
    type_map = dict(DEFAULT_TYPE_MAP)

    if not lammps_input.exists():
        return type_map

    # Matches: group  GROUPNAME  type  1 [2 3 ...]  # comment text
    group_pattern = re.compile(
        r"^\s*group\s+\S+\s+type\s+([\d\s]+)#\s*(.+)$", re.IGNORECASE
    )
    # Extract element symbols from a comment string.
    # Handles both bare symbols (Ir, O, K) and parenthesised ones like (water O).
    # We collect all tokens that are valid element symbols.
    elem_token = re.compile(r"\b([A-Z][a-z]?)\b")

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue

            type_ids = [int(x) for x in m.group(1).split()]
            comment = m.group(2)

            # Collect all valid element symbols from the comment, in order
            elems = [tok for tok in elem_token.findall(comment)
                     if tok in element_symbols]

            if not elems:
                continue

            if len(type_ids) == 1:
                # Single type → use first element found
                type_map[type_ids[0]] = elems[0]
            elif len(type_ids) == len(elems):
                # Multiple types with matching number of elements → pair 1-to-1
                for tid, elem in zip(type_ids, elems):
                    type_map[tid] = elem
            else:
                # Ambiguous: all types share the first element found
                for tid in type_ids:
                    type_map[tid] = elems[0]

    return type_map


def count_frames(lammpstrj: Path) -> int:
    """Count the number of frames in a LAMMPS dump file."""
    count = 0
    with open(lammpstrj) as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                count += 1
    return count


def iter_frames(lammpstrj: Path):
    """
    Generator that yields one frame at a time as a dict:
      {
        'timestep': int,
        'n_atoms': int,
        'box': np.ndarray (3x2, [[xlo,xhi],[ylo,yhi],[zlo,zhi]]),
        'columns': list[str],
        'data': np.ndarray (n_atoms, n_cols),
      }
    """
    with open(lammpstrj) as f:
        while True:
            # --- TIMESTEP ---
            line = f.readline()
            if not line:
                return  # EOF
            if "TIMESTEP" not in line:
                continue
            timestep = int(f.readline().strip())

            # --- NUMBER OF ATOMS ---
            f.readline()  # "ITEM: NUMBER OF ATOMS"
            n_atoms = int(f.readline().strip())

            # --- BOX BOUNDS ---
            f.readline()  # "ITEM: BOX BOUNDS ..."
            box = np.zeros((3, 2))
            for i in range(3):
                vals = f.readline().split()
                box[i] = [float(vals[0]), float(vals[1])]

            # --- ATOMS header ---
            atoms_header = f.readline()  # "ITEM: ATOMS id type xu yu zu ..."
            columns = atoms_header.split()[2:]  # skip "ITEM:" and "ATOMS"

            # --- atom data ---
            rows = []
            for _ in range(n_atoms):
                rows.append(f.readline().split())
            data = np.array(rows, dtype=float)

            yield {
                "timestep": timestep,
                "n_atoms": n_atoms,
                "box": box,
                "columns": columns,
                "data": data,
            }


def frame_to_atoms(frame: dict, type_map: Dict[int, str]) -> Atoms:
    """Convert a parsed frame dict to an ASE Atoms object."""
    cols = frame["columns"]
    data = frame["data"]
    box = frame["box"]

    col_idx = {c: i for i, c in enumerate(cols)}

    # Atom types → element symbols
    types = data[:, col_idx["type"]].astype(int)
    symbols = [type_map.get(t, "X") for t in types]

    # Positions (prefer xu/yu/zu = unwrapped, fall back to x/y/z)
    if "xu" in col_idx:
        pos_cols = ["xu", "yu", "zu"]
    elif "x" in col_idx:
        pos_cols = ["x", "y", "z"]
    else:
        raise KeyError("No position columns found in dump file.")

    positions = data[:, [col_idx[c] for c in pos_cols]]

    # Cell: orthorhombic assumed (LAMMPS pp pp ff boundary)
    cell = np.diag(box[:, 1] - box[:, 0])
    pbc = [True, True, False]  # ff boundary → non-periodic in z

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )

    # Store forces via SinglePointCalculator so they persist in .traj files
    if "fx" in col_idx and "fy" in col_idx and "fz" in col_idx:
        forces = data[:, [col_idx["fx"], col_idx["fy"], col_idx["fz"]]]
        calc = SinglePointCalculator(atoms, forces=forces)
        atoms.calc = calc

    # Store timestep as info
    atoms.info["timestep"] = frame["timestep"]
    atoms.info["lammps_type"] = types.tolist()

    return atoms


def wrap_molecules(atoms: Atoms, oh_cutoff: float = 1.3) -> Atoms:
    """
    Wrap atoms into the periodic simulation box while keeping molecules intact.

    Algorithm:
      1. Wrap all atoms individually into the box (x, y only; z is non-periodic).
      2. For each H atom that is bonded to an O (within oh_cutoff Å using minimum-
         image convention), shift the H by the nearest lattice vector so that the
         O-H distance is minimised — i.e., both atoms end up on the same side of
         every periodic boundary.

    This ensures H2O and OH groups are never split across the box boundary.
    Standalone atoms (K, Ir, surface O) are simply wrapped in step 1.

    Parameters
    ----------
    atoms     : ASE Atoms object with pbc=[True, True, False]
    oh_cutoff : Maximum O-H bond length in Å (default 1.3 Å)

    Returns
    -------
    A new Atoms object with wrapped positions (forces/info preserved).
    """
    atoms = atoms.copy()
    cell = atoms.cell.diagonal()          # orthorhombic: [Lx, Ly, Lz]
    pbc_mask = np.array([1.0, 1.0, 0.0]) # wrap only in x, y

    pos = atoms.positions.copy()
    origin = atoms.cell.scaled_positions(pos)  # fractional coords

    # --- Step 1: wrap all atoms into [0, 1) in x and y ---
    origin[:, :2] = origin[:, :2] % 1.0
    pos = atoms.cell.cartesian_positions(origin)

    symbols = atoms.get_chemical_symbols()
    lammps_types = atoms.info.get("lammps_type", None)

    # Build index sets for H and O atoms
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    o_indices  = [i for i, s in enumerate(symbols) if s == "O"]

    if h_indices and o_indices:
        o_pos = pos[o_indices]  # shape (n_O, 3)

        for hi in h_indices:
            h_pos = pos[hi]

            # Vector from each O to this H (Cartesian)
            dv = h_pos - o_pos                          # (n_O, 3)

            # Apply minimum-image correction in periodic directions
            for dim in range(3):
                if pbc_mask[dim] and cell[dim] > 0:
                    dv[:, dim] -= np.round(dv[:, dim] / cell[dim]) * cell[dim]

            # Distance to each O after minimum-image
            dist = np.linalg.norm(dv, axis=1)           # (n_O,)
            nearest_idx = int(np.argmin(dist))

            if dist[nearest_idx] <= oh_cutoff:
                # Place H relative to its bonded O using the minimum-image vector
                pos[hi] = o_pos[nearest_idx] + dv[nearest_idx]

    atoms.positions = pos

    # Re-wrap H atoms that may have drifted slightly outside [0, Lx/Ly]
    # after being repositioned relative to O (only in periodic dims)
    frac = atoms.cell.scaled_positions(atoms.positions)
    frac[:, :2] = frac[:, :2] % 1.0
    atoms.positions = atoms.cell.cartesian_positions(frac)

    return atoms


def convert(
    lammpstrj: Path,
    output: Path,
    type_map: Dict[int, str],
    stride: int = 1,
    wrap: bool = False,
    verbose: bool = True,
) -> int:
    """
    Convert lammpstrj → ASE .traj.
    Returns the number of frames written.
    """
    traj = Trajectory(str(output), mode="w")
    written = 0
    frame_idx = 0

    for frame in iter_frames(lammpstrj):
        if frame_idx % stride == 0:
            atoms = frame_to_atoms(frame, type_map)
            if wrap:
                atoms = wrap_molecules(atoms)
            traj.write(atoms)
            written += 1
            if verbose and written % 100 == 0:
                print(f"  Written {written} frames (timestep {frame['timestep']})...")
        frame_idx += 1

    traj.close()
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LAMMPS .lammpstrj dump to an ASE .traj file."
    )
    parser.add_argument("lammpstrj", type=Path, help="Input LAMMPS dump file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .traj file (default: same name as input with .traj extension)"
    )
    parser.add_argument(
        "--lammps-input", type=Path, default=None,
        help="LAMMPS input file to auto-detect atom type→element mapping"
    )
    parser.add_argument(
        "--type-map", type=str, default=None,
        help="Manual type→element map, e.g. '1:H 2:O 3:K 4:H 5:O 6:Ir 7:O'"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Write every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--wrap", action="store_true",
        help=(
            "Wrap atoms into the simulation box for clean visualization. "
            "Molecules (H2O, OH) are kept intact by minimum-image bond detection. "
            "Only x/y are wrapped (z is non-periodic in this system)."
        )
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Only count frames, do not convert"
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Open ASE GUI after conversion"
    )

    args = parser.parse_args()

    lammpstrj = args.lammpstrj
    if not lammpstrj.exists():
        print(f"ERROR: file not found: {lammpstrj}", file=sys.stderr)
        sys.exit(1)

    # --- Count frames ---
    print(f"Counting frames in {lammpstrj} ...")
    n_frames = count_frames(lammpstrj)
    print(f"  Total frames: {n_frames:,}")

    if args.count_only:
        return

    # --- Determine type map ---
    if args.type_map:
        type_map = parse_type_map_str(args.type_map)
        print(f"Using manual type map: {type_map}")
    elif args.lammps_input:
        type_map = auto_detect_type_map(args.lammps_input)
        print(f"Auto-detected type map from {args.lammps_input}: {type_map}")
    else:
        # Try to find in.lammps in the same or parent directory
        candidates = [
            lammpstrj.parent / "in.lammps",
            lammpstrj.parent.parent / "in.lammps",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found:
            type_map = auto_detect_type_map(found)
            print(f"Auto-detected type map from {found}: {type_map}")
        else:
            type_map = DEFAULT_TYPE_MAP
            print(f"Using default CES2 type map: {type_map}")

    # --- Output path ---
    if args.output:
        output = args.output
    else:
        output = lammpstrj.with_suffix(".traj")
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- Convert ---
    expected = (n_frames + args.stride - 1) // args.stride
    print(f"\nConverting to {output}")
    print(f"  Stride: every {args.stride} frame(s) → ~{expected:,} frames to write")
    print()

    if args.wrap:
        print("  Molecule-aware wrapping: ON (O-H cutoff 1.3 Å, wrapping x/y only)")
    written = convert(lammpstrj, output, type_map, stride=args.stride, wrap=args.wrap)

    print(f"\nDone. Wrote {written:,} frames → {output}")
    print(f"View with:  ase gui {output}")

    if args.view:
        from ase.visualize import view as ase_view
        from ase.io import read as ase_read
        print("Opening ASE GUI ...")
        images = ase_read(str(output), index=":")
        ase_view(images)


if __name__ == "__main__":
    main()
