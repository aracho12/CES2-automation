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

  # Extract average trajectories from the last 2 QM/MM steps
  python tools/lammpstrj_to_traj.py --qmmm-average /path/to/qmmm_run_dir

  # Extract last 3 steps, with stride and custom output
  python tools/lammpstrj_to_traj.py --qmmm-average /path/to/run_dir \\
      --last-steps 3 --stride 5 -o analysis/average_last3.traj

  # Convert every mm_N/ces2.emd.lammpstrj into a wrapped .traj in place
  python tools/lammpstrj_to_traj.py --qmmm-wrap-each /path/to/qmmm_run_dir
"""

import argparse
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Default type→element mapping for CES2 system.
# Only water types (1: Hw, 2: Ow) are fixed across all systems.
# Types 3+ depend on the electrolyte recipe and are auto-detected from
# in.lammps. If in.lammps is unavailable, unknown types appear as "X".
# ---------------------------------------------------------------------------
DEFAULT_TYPE_MAP = {
    1: "H",   # Hw (water H) — always type 1
    2: "O",   # Ow (water O) — always type 2
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
    # Handles bare symbols (Ir, O, K), parenthesised ones like (water O),
    # and underscore-prefixed labels like H_ads, O_oh where the element
    # appears before the underscore.
    # We split the comment into whitespace-delimited tokens and check each.
    def _extract_elements(comment: str) -> list:
        """Extract valid element symbols from a comment string, in order."""
        elems = []
        for token in re.split(r"[\s()\[\]:,]+", comment):
            if not token:
                continue
            # Handle "X_label" patterns: extract prefix before underscore
            base = token.split("_")[0] if "_" in token else token
            # Must be 1-2 chars, first uppercase, optional lowercase
            if re.fullmatch(r"[A-Z][a-z]?", base) and base in element_symbols:
                elems.append(base)
        return elems

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue

            type_ids = [int(x) for x in m.group(1).split()]
            comment = m.group(2)

            # Collect all valid element symbols from the comment, in order
            elems = _extract_elements(comment)

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

    symbols = np.array(atoms.get_chemical_symbols())

    # Build index arrays for H and O atoms
    h_indices = np.where(symbols == "H")[0]
    o_indices = np.where(symbols == "O")[0]

    if len(h_indices) > 0 and len(o_indices) > 0:
        Lx, Ly = cell[0], cell[1]
        o_pos = pos[o_indices]             # (n_O, 3)
        h_pos_all = pos[h_indices]         # (n_H, 3)

        # --- Use cKDTree for fast O(n log n) nearest-neighbor search ---
        # Positions are already wrapped into [0, L) by step 1.
        # cKDTree supports PBC natively via boxsize — all coords must be
        # in [0, boxsize).  For z (non-periodic) use a huge box so it
        # never wraps.
        z_big = max(abs(pos[:, 2].max() - pos[:, 2].min()) * 10, 1e4)
        # Shift z so all values are in [0, z_big)
        z_min = pos[:, 2].min() - 1.0
        o_tree = o_pos.copy()
        o_tree[:, 2] -= z_min
        h_tree = h_pos_all.copy()
        h_tree[:, 2] -= z_min

        tree = cKDTree(o_tree, boxsize=[Lx, Ly, z_big])
        dists, o_local_idx = tree.query(h_tree, k=1)

        bonded = dists <= oh_cutoff
        bonded_h = np.where(bonded)[0]

        if len(bonded_h) > 0:
            bonded_o_local = o_local_idx[bonded_h]
            # Compute actual displacement with minimum-image in Cartesian
            dv = h_pos_all[bonded_h] - o_pos[bonded_o_local]  # (n_bonded, 3)
            dv[:, 0] -= np.round(dv[:, 0] / Lx) * Lx
            dv[:, 1] -= np.round(dv[:, 1] / Ly) * Ly
            pos[h_indices[bonded_h]] = o_pos[bonded_o_local] + dv

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
    wrap: bool = True,
    verbose: bool = True,
) -> int:
    """
    Convert lammpstrj → ASE .traj.
    Returns the number of frames written.
    """
    traj = Trajectory(str(output), mode="w")
    written = 0
    frame_idx = 0
    t_start = time.perf_counter()
    t_parse = 0.0
    t_wrap = 0.0
    t_write = 0.0

    for frame in iter_frames(lammpstrj):
        if frame_idx % stride == 0:
            t0 = time.perf_counter()
            atoms = frame_to_atoms(frame, type_map)
            t1 = time.perf_counter()
            t_parse += t1 - t0

            if wrap:
                atoms = wrap_molecules(atoms)
                t2 = time.perf_counter()
                t_wrap += t2 - t1
            else:
                t2 = t1

            traj.write(atoms)
            t3 = time.perf_counter()
            t_write += t3 - t2

            written += 1
            if verbose and written % 100 == 0:
                elapsed = time.perf_counter() - t_start
                fps = written / elapsed if elapsed > 0 else 0
                print(f"  Written {written} frames (timestep {frame['timestep']}) "
                      f"[{elapsed:.1f}s elapsed, {fps:.1f} frames/s]")
        frame_idx += 1

    traj.close()
    t_total = time.perf_counter() - t_start

    if verbose:
        print(f"\n  Timing breakdown:")
        print(f"    Total         : {t_total:8.2f} s")
        print(f"    Parse frames  : {t_parse:8.2f} s ({100*t_parse/t_total:.1f}%)")
        if wrap:
            print(f"    Wrap molecules: {t_wrap:8.2f} s ({100*t_wrap/t_total:.1f}%)")
        print(f"    Write .traj   : {t_write:8.2f} s ({100*t_write/t_total:.1f}%)")
        overhead = t_total - t_parse - t_wrap - t_write
        print(f"    I/O + overhead : {overhead:8.2f} s ({100*overhead/t_total:.1f}%)")
        if written > 0:
            print(f"    Per frame     : {1000*t_total/written:.2f} ms/frame")

    return written


def find_mm_dirs(run_dir: Path) -> List[Path]:
    """
    Find mm_N directories in a QM/MM run directory, sorted by step index.
    Returns list of (step_index, mm_dir_path) sorted ascending.
    """
    mm_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and re.match(r"^mm_(\d+)$", d.name):
            step = int(d.name.split("_")[1])
            mm_dirs.append((step, d))
    mm_dirs.sort(key=lambda x: x[0])
    return mm_dirs


def find_lammpstrj_in_dir(d: Path) -> Optional[Path]:
    """Find the *.emd.lammpstrj (or any *.lammpstrj) in a directory."""
    # Prefer *.emd.lammpstrj (the standard CES2 trajectory dump)
    emd_files = list(d.glob("*.emd.lammpstrj"))
    if emd_files:
        return emd_files[0]
    # Fallback: any .lammpstrj
    all_files = list(d.glob("*.lammpstrj"))
    if all_files:
        return all_files[0]
    return None


def convert_qmmm_average(
    run_dir: Path,
    output: Path,
    type_map: Dict[int, str],
    last_steps: int = 2,
    stride: int = 1,
    wrap: bool = True,
    verbose: bool = True,
) -> int:
    """
    Extract average trajectories from the last N QM/MM steps and write to a
    single .traj file.

    In CES2 QM/MM, each step's mm_{N}/ directory contains *.lammpstrj from the
    averaging LAMMPS run (the equil trajectory is overwritten by the average run
    since LAMMPS opens dump files fresh).  This function concatenates trajectories
    from the last `last_steps` mm directories into one .traj.

    Parameters
    ----------
    run_dir    : QM/MM run directory containing mm_0/, mm_1/, ... subdirectories
    output     : output .traj file path
    type_map   : LAMMPS type → element mapping
    last_steps : number of last QM/MM steps to include (default: 2)
    stride     : write every Nth frame (default: 1)
    wrap       : apply molecule-aware wrapping (default: True)
    verbose    : print progress info

    Returns
    -------
    Total number of frames written.
    """
    mm_dirs = find_mm_dirs(run_dir)
    if not mm_dirs:
        print(f"ERROR: No mm_N/ directories found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    total_steps = len(mm_dirs)
    if last_steps > total_steps:
        print(f"WARNING: Requested last {last_steps} steps but only {total_steps} "
              f"mm_N/ directories found. Using all {total_steps} steps.")
        last_steps = total_steps

    selected = mm_dirs[-last_steps:]
    step_indices = [s[0] for s in selected]
    step_dirs = [s[1] for s in selected]

    if verbose:
        print(f"QM/MM run directory: {run_dir}")
        print(f"  Total mm_N/ directories found: {total_steps}")
        print(f"  Selecting last {last_steps} steps: {step_indices}")
        print()

    # Locate lammpstrj files in each selected mm dir
    trj_files: List[Path] = []
    for step_idx, mm_dir in selected:
        trj = find_lammpstrj_in_dir(mm_dir)
        if trj is None:
            print(f"  WARNING: No .lammpstrj found in {mm_dir}, skipping step {step_idx}")
            continue
        n_frames = count_frames(trj)
        if verbose:
            print(f"  mm_{step_idx}: {trj.name} ({n_frames:,} frames)")
        trj_files.append(trj)

    if not trj_files:
        print("ERROR: No .lammpstrj files found in any selected mm_N/ directory.",
              file=sys.stderr)
        sys.exit(1)

    # Convert all selected trajectories into one .traj
    if verbose:
        print(f"\nConverting to {output}")
        if wrap:
            print("  Molecule-aware wrapping: ON")
        print(f"  Stride: every {stride} frame(s)")
        print()

    output.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(output), mode="w")
    total_written = 0
    t_start = time.perf_counter()

    for trj_file in trj_files:
        frame_idx = 0
        step_name = trj_file.parent.name
        for frame in iter_frames(trj_file):
            if frame_idx % stride == 0:
                atoms = frame_to_atoms(frame, type_map)
                # Tag QM/MM step in atoms.info for downstream analysis
                atoms.info["qmmm_step"] = step_name
                if wrap:
                    atoms = wrap_molecules(atoms)
                traj.write(atoms)
                total_written += 1
                if verbose and total_written % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    fps = total_written / elapsed if elapsed > 0 else 0
                    print(f"  Written {total_written} frames "
                          f"({step_name}, timestep {frame['timestep']}) "
                          f"[{elapsed:.1f}s, {fps:.1f} fr/s]")
            frame_idx += 1

    traj.close()
    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\nDone. Wrote {total_written:,} frames → {output}")
        print(f"  Steps included: {step_indices}")
        print(f"  Elapsed: {elapsed:.1f} s")
        if total_written > 0:
            print(f"  Per frame: {1000*elapsed/total_written:.2f} ms/frame")
        print(f"View with:  ase gui {output}")

    return total_written


def convert_qmmm_wrap_each(
    run_dir: Path,
    type_map: Dict[int, str],
    stride: int = 1,
    wrap: bool = True,
    verbose: bool = True,
) -> int:
    """
    Convert each mm_N/*.emd.lammpstrj trajectory into its own wrapped .traj.

    The output is written next to the source trajectory as
    <source>.wrapped.traj, for example:
      mm_12/ces2.emd.lammpstrj -> mm_12/ces2.emd.wrapped.traj

    Returns the total number of frames written across all converted files.
    """
    mm_dirs = find_mm_dirs(run_dir)
    if not mm_dirs:
        print(f"ERROR: No mm_N/ directories found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"QM/MM run directory: {run_dir}")
        print(f"  Total mm_N/ directories found: {len(mm_dirs)}")
        if wrap:
            print("  Molecule-aware wrapping: ON")
        else:
            print("  Molecule-aware wrapping: OFF")
        print(f"  Stride: every {stride} frame(s)")
        print()

    total_written = 0
    converted = 0
    skipped = 0
    t_start = time.perf_counter()

    for step_idx, mm_dir in mm_dirs:
        trj_file = find_lammpstrj_in_dir(mm_dir)
        if trj_file is None:
            print(f"  WARNING: No .lammpstrj found in {mm_dir}, skipping step {step_idx}")
            skipped += 1
            continue

        output = trj_file.with_suffix(".wrapped.traj")
        n_frames = count_frames(trj_file)
        expected = (n_frames + stride - 1) // stride

        if verbose:
            print(f"  mm_{step_idx}: {trj_file.name} ({n_frames:,} frames)"
                  f" -> {output.name} (~{expected:,} frames)")

        written = convert(
            lammpstrj=trj_file,
            output=output,
            type_map=type_map,
            stride=stride,
            wrap=wrap,
            verbose=False,
        )
        total_written += written
        converted += 1

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\nDone. Converted {converted} mm_N directories, skipped {skipped}.")
        print(f"  Total frames written: {total_written:,}")
        print(f"  Elapsed: {elapsed:.1f} s")
        if total_written > 0:
            print(f"  Per frame: {1000*elapsed/total_written:.2f} ms/frame")

    return total_written


def _resolve_type_map(args, search_dir: Path) -> Dict[int, str]:
    """Resolve type→element mapping from args, searching in search_dir as fallback."""
    if args.type_map:
        type_map = parse_type_map_str(args.type_map)
        print(f"Using manual type map: {type_map}")
        return type_map

    if args.lammps_input:
        type_map = auto_detect_type_map(args.lammps_input)
        print(f"Auto-detected type map from {args.lammps_input}: {type_map}")
        return type_map

    # Auto-search for in.lammps or base.in.lammps
    candidates = [
        search_dir / "base.in.lammps",
        search_dir / "in.lammps",
        search_dir / "export" / "base.in.lammps",
    ]
    for c in candidates:
        if c.exists():
            type_map = auto_detect_type_map(c)
            print(f"Auto-detected type map from {c}: {type_map}")
            return type_map

    print(f"Using default CES2 type map: {DEFAULT_TYPE_MAP}")
    return dict(DEFAULT_TYPE_MAP)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LAMMPS .lammpstrj dump to an ASE .traj file."
    )
    parser.add_argument(
        "lammpstrj", type=Path, nargs="?", default=None,
        help="Input LAMMPS dump file (not needed with a QM/MM directory mode)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .traj file (default: auto-generated from input name)"
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
        "--wrap", action="store_true", default=True,
        help=(
            "Wrap atoms into the simulation box for clean visualization (default: ON). "
            "Molecules (H2O, OH) are kept intact by minimum-image bond detection. "
            "Only x/y are wrapped (z is non-periodic in this system)."
        )
    )
    parser.add_argument(
        "--no-wrap", action="store_true",
        help="Disable molecule-aware wrapping."
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Only count frames, do not convert"
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Open ASE GUI after conversion"
    )

    # ── QM/MM average trajectory extraction ──────────────────────────────
    parser.add_argument(
        "--qmmm-average", type=Path, default=None, metavar="RUN_DIR",
        help=(
            "Extract only the averaging-phase LAMMPS trajectories from the last N "
            "QM/MM steps.  RUN_DIR is the QM/MM run directory containing mm_0/, "
            "mm_1/, ... subdirectories.  Each mm_N/ holds the *.lammpstrj from the "
            "averaging run (equil trajectory is overwritten).  Combine with "
            "--last-steps to control how many steps to include."
        )
    )
    parser.add_argument(
        "--qmmm-wrap-each", type=Path, default=None, metavar="RUN_DIR",
        help=(
            "Convert each mm_N/*.emd.lammpstrj in a QM/MM run directory into a "
            "separate wrapped .traj next to the source file.  The default output "
            "name is <input>.wrapped.traj, e.g. ces2.emd.wrapped.traj."
        )
    )
    parser.add_argument(
        "--last-steps", type=int, default=2,
        help="Number of last QM/MM steps to include (default: 2, used with --qmmm-average)"
    )

    args = parser.parse_args()
    do_wrap = args.wrap and not args.no_wrap

    if args.qmmm_average is not None and args.qmmm_wrap_each is not None:
        parser.error("use only one of --qmmm-average or --qmmm-wrap-each")

    # =================================================================
    # Mode 1: --qmmm-average  (extract average trajectories from QM/MM)
    # =================================================================
    if args.qmmm_average is not None:
        run_dir = args.qmmm_average
        if not run_dir.is_dir():
            print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
            sys.exit(1)

        type_map = _resolve_type_map(args, search_dir=run_dir)

        if args.output:
            output = args.output
        else:
            output = run_dir / f"average_last{args.last_steps}.traj"
        output.parent.mkdir(parents=True, exist_ok=True)

        written = convert_qmmm_average(
            run_dir=run_dir,
            output=output,
            type_map=type_map,
            last_steps=args.last_steps,
            stride=args.stride,
            wrap=do_wrap,
        )

        if args.view and written > 0:
            from ase.visualize import view as ase_view
            from ase.io import read as ase_read
            print("Opening ASE GUI ...")
            images = ase_read(str(output), index=":")
            ase_view(images)

        return

    # =================================================================
    # Mode 2: --qmmm-wrap-each  (convert every mm_N trajectory in place)
    # =================================================================
    if args.qmmm_wrap_each is not None:
        run_dir = args.qmmm_wrap_each
        if not run_dir.is_dir():
            print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
            sys.exit(1)

        if args.output:
            parser.error("--output is not supported with --qmmm-wrap-each")

        type_map = _resolve_type_map(args, search_dir=run_dir)
        convert_qmmm_wrap_each(
            run_dir=run_dir,
            type_map=type_map,
            stride=args.stride,
            wrap=do_wrap,
        )

        return

    # =================================================================
    # Mode 3: Single file conversion (original behavior)
    # =================================================================
    if args.lammpstrj is None:
        parser.error("a lammpstrj file is required (or use --qmmm-average)")

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
    type_map = _resolve_type_map(args, search_dir=lammpstrj.parent)

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

    if do_wrap:
        print("  Molecule-aware wrapping: ON (O-H cutoff 1.3 Å, wrapping x/y only)")
    else:
        print("  Molecule-aware wrapping: OFF")
    written = convert(lammpstrj, output, type_map, stride=args.stride, wrap=do_wrap)

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
