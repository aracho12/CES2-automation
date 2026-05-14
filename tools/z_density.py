#!/usr/bin/env python3
"""
z_density.py — Element- or type_label-resolved number-density profiles ρ(z)
===========================================================================
Computes planar-averaged number density along the z-axis for selected
elements or CES2/LAMMPS type_labels from a LAMMPS dump (.lammpstrj) or ASE
trajectory (.traj) file.

  ρ(z) [Å⁻³] = N_in_bin(z, z+dz) / (Lx · Ly · dz · N_frames)

By default, **all solvent elements** (everything except electrode metals)
are plotted.  You can override with ``--elements O H K`` etc.

Outputs (written next to the trajectory)
-----------------------------------------
  z_density_rawdata.csv   — z and ρ(z) for every element
  z_density.png           — density profile plot

Usage examples
--------------
  # All solvent elements, skip first 100 frames, stride 5
  python tools/z_density.py path/to/ces2.emd.lammpstrj --skip 100 --stride 5

  # Only O and H from a .traj file
  python tools/z_density.py md.traj --elements O H --dz 0.05

  # Separate water O from OH- O using LAMMPS type_labels from in.lammps
  python tools/z_density.py ces2.emd.lammpstrj --type-labels O_oh Ow

  # Provide LAMMPS type→element map explicitly for .lammpstrj
  python tools/z_density.py ces2.emd.lammpstrj --type-map "1:H 2:O 3:Cs 4:H 5:O 6:Ir 7:O"

  # Auto-detect type map from in.lammps (searched automatically)
  python tools/z_density.py run_dir/ces2.emd.lammpstrj

  # QM/SOLUTE atoms are excluded by default if in.lammps is found.
  # To include all atoms (including QM slab):
  python tools/z_density.py ces2.emd.lammpstrj --all-atoms

  # Exclude specific LAMMPS types manually (e.g. types 6,7 = Ir, slab O)
  python tools/z_density.py ces2.emd.lammpstrj --exclude-types 6 7

  # Quick scan: frame count, element composition, box size (no density calc)
  python tools/z_density.py ces2.emd.lammpstrj --info
  python tools/z_density.py md.traj --info

  # Process every mm_N directory in a QM/MM run and write an evolution plot
  python tools/z_density.py --qmmm-mm /path/to/qmmm_run_dir
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

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
    _C  = ["#515151", "#F14040", "#1A6FDF", "#37AD6B", "#B177DE",
           "#FEC211", "#999999", "#FF4081", "#FB6501", "#6699CC"]
    _FS = 9; _LS = 9; _LW = 0.4; _FW, _FH = 3.5, 2.8

# ── atomic masses (g/mol) ────────────────────────────────────────────────────
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Zr": 91.224,
    "Mo": 95.95, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cs": 132.905, "Ba": 137.327, "La": 138.905,
    "Ce": 140.116, "Ir": 192.217, "Pt": 195.084, "Au": 196.967,
}

# Electrode / slab metals — excluded from "solvent" by default
ELECTRODE_ELEMENTS: Set[str] = {
    "Ir", "Pt", "Au", "Ru", "Rh", "Pd", "Ag", "Cu", "Ni", "Co", "Fe",
    "Ti", "Zr", "Mo", "La", "Ce",
}

# ── unit conversion ──────────────────────────────────────────────────────────
_ANG3_TO_GCM3 = 1e24 / 6.02214076e23   # ρ[g/cm³] = ρ_num[Å⁻³] × M × this


# ═══════════════════════════════════════════════════════════════════════════════
#  LAMMPS type-map detection (reused from lammpstrj_to_traj.py)
# ═══════════════════════════════════════════════════════════════════════════════

_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}

DEFAULT_TYPE_MAP: Dict[int, str] = {1: "H", 2: "O"}


def parse_type_map_str(s: str) -> Dict[int, str]:
    """Parse '1:H 2:O 3:K ...' → {1: 'H', 2: 'O', 3: 'K', ...}."""
    result = {}
    for token in s.split():
        t, elem = token.split(":")
        result[int(t)] = elem
    return result


def parse_type_label_map_str(s: str) -> Dict[int, str]:
    """Parse '1:Hw 2:Ow 5:O_oh ...' → {1: 'Hw', 2: 'Ow', ...}."""
    return parse_type_map_str(s)


def _label_to_element(label: str) -> Optional[str]:
    """Best-effort element inference from a CES2 type_label."""
    if label in _ELEMENT_SYMBOLS:
        return label
    prefix = re.split(r"[_\-.]", label, maxsplit=1)[0]
    if prefix in _ELEMENT_SYMBOLS:
        return prefix
    # Common water labels such as Ow/Hw are not valid element symbols as-is.
    if len(prefix) >= 2 and prefix[0] in {"H", "O"} and prefix[1].islower():
        return prefix[0]
    return None


def _extract_type_labels_from_comment(comment: str, type_ids: List[int]) -> List[str]:
    """Extract CES2 type_labels from LAMMPS group comments."""
    text = comment.split(":", 1)[1] if ":" in comment else comment
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = [tok for tok in re.split(r"[\s,]+", text.strip()) if tok]

    labels = []
    for tok in tokens:
        clean = tok.strip()
        if not clean:
            continue
        if clean.lower() in {"all", "water", "qm", "slab", "atoms"}:
            continue
        labels.append(clean)

    if len(labels) >= len(type_ids):
        return labels[:len(type_ids)]
    if len(type_ids) == 1 and labels:
        return [labels[0]]
    return []


def auto_detect_type_label_map(lammps_input: Path) -> Dict[int, str]:
    """Extract LAMMPS type→CES2 type_label mapping from a LAMMPS input file."""
    type_label_map: Dict[int, str] = {}
    if not lammps_input.exists():
        return type_label_map

    group_pattern = re.compile(
        r"^\s*group\s+\S+\s+type\s+([\d\s]+)(?:#\s*(.+))?$",
        re.IGNORECASE,
    )

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue
            type_ids = [int(x) for x in m.group(1).split()]
            comment = m.group(2) or ""
            labels = _extract_type_labels_from_comment(comment, type_ids)
            if len(labels) == len(type_ids):
                for tid, label in zip(type_ids, labels):
                    type_label_map[tid] = label
            elif len(type_ids) == 1 and labels:
                type_label_map[type_ids[0]] = labels[0]

    return type_label_map


def auto_detect_type_map(lammps_input: Path) -> Dict[int, str]:
    """Extract type→element mapping from a LAMMPS input file."""
    type_map = dict(DEFAULT_TYPE_MAP)
    if not lammps_input.exists():
        return type_map

    group_pattern = re.compile(
        r"^\s*group\s+\S+\s+type\s+([\d\s]+)#\s*(.+)$", re.IGNORECASE
    )

    def _extract_elements(comment: str) -> list:
        elems = []
        for token in re.split(r"[\s()\[\]:,]+", comment):
            if not token:
                continue
            base = token.split("_")[0] if "_" in token else token
            if re.fullmatch(r"[A-Z][a-z]?", base) and base in _ELEMENT_SYMBOLS:
                elems.append(base)
        return elems

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue
            type_ids = [int(x) for x in m.group(1).split()]
            elems = _extract_elements(m.group(2))
            if not elems:
                continue
            if len(type_ids) == 1:
                type_map[type_ids[0]] = elems[0]
            elif len(type_ids) == len(elems):
                for tid, elem in zip(type_ids, elems):
                    type_map[tid] = elem
            else:
                for tid in type_ids:
                    type_map[tid] = elems[0]

    return type_map


def _find_lammps_input(traj_path: Path) -> Optional[Path]:
    """Search for in.lammps near the trajectory file."""
    for d in [traj_path.parent, traj_path.parent.parent]:
        p = d / "in.lammps"
        if p.exists():
            return p
    return None


def detect_solute_types(lammps_input: Path) -> Set[int]:
    """
    Parse in.lammps to find SOLUTE/QM atom types.

    Looks for patterns like:
      group  SOLUTE   type 6 7   # QM slab: Ir O
      group  QM       type 6 7
    """
    if not lammps_input.exists():
        return set()

    solute_pattern = re.compile(
        r"^\s*group\s+(SOLUTE|QM|QM_ATOMS|SLAB)\s+type\s+([\d\s]+)",
        re.IGNORECASE,
    )

    with open(lammps_input) as f:
        for line in f:
            m = solute_pattern.match(line.split("#")[0].rstrip() + " "
                                     if "#" not in line
                                     else line)
            # simpler: match against the full line
            m = solute_pattern.match(line)
            if m:
                type_ids = {int(x) for x in m.group(2).split()}
                return type_ids
    return set()


def find_mm_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Find mm_N directories in a QM/MM run directory, sorted by step index."""
    mm_dirs = []
    for d in run_dir.iterdir():
        if d.is_dir() and re.match(r"^mm_(\d+)$", d.name):
            step = int(d.name.split("_")[1])
            mm_dirs.append((step, d))
    mm_dirs.sort(key=lambda x: x[0])
    return mm_dirs


def find_density_traj_in_mm_dir(mm_dir: Path) -> Optional[Path]:
    """Find the trajectory to use for z-density inside one mm_N directory."""
    patterns = [
        "*.wrapped.traj",
        "*.emd.lammpstrj",
        "*.lammpstrj",
        "*.traj",
    ]
    for pattern in patterns:
        matches = sorted(mm_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  --info : trajectory scan (no density calculation)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_lammpstrj_info(path: Path, type_map: Dict[int, str]) -> dict:
    """Scan a .lammpstrj file and return summary info."""
    from collections import Counter
    n_frames = 0
    timesteps: List[int] = []
    n_atoms = 0
    elem_counts: Counter = Counter()
    type_counts: Counter = Counter()
    box_first = None
    box_last = None
    z_min_global = np.inf
    z_max_global = -np.inf

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "TIMESTEP" not in line:
                continue
            ts = int(f.readline().strip())
            timesteps.append(ts)

            f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS
            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())
            box = {"Lx": xhi - xlo, "Ly": yhi - ylo, "Lz": zhi - zlo,
                   "xlo": xlo, "xhi": xhi, "ylo": ylo, "yhi": yhi,
                   "zlo": zlo, "zhi": zhi}
            if box_first is None:
                box_first = box
            box_last = box

            header = f.readline().split()[2:]
            try:
                col_type = header.index("type")
                col_z = header.index("zu") if "zu" in header else header.index("z")
            except ValueError:
                for _ in range(n_atoms):
                    f.readline()
                n_frames += 1
                continue

            # only count element composition on first frame to keep it fast
            if n_frames == 0:
                for i in range(n_atoms):
                    tok = f.readline().split()
                    t = int(tok[col_type])
                    type_counts[t] += 1
                    elem_counts[type_map.get(t, f"X(type{t})")] += 1
                    z_val = float(tok[col_z])
                    if z_val < z_min_global:
                        z_min_global = z_val
                    if z_val > z_max_global:
                        z_max_global = z_val
            else:
                for _ in range(n_atoms):
                    f.readline()

            n_frames += 1
            if n_frames % 500 == 0:
                print(f"    ... scanning frame {n_frames}")

    # timestep interval
    if len(timesteps) >= 2:
        dt = timesteps[1] - timesteps[0]
    else:
        dt = None

    return {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "timesteps_first": timesteps[0] if timesteps else None,
        "timesteps_last": timesteps[-1] if timesteps else None,
        "timestep_interval": dt,
        "type_map": type_map,
        "type_counts": dict(type_counts),
        "elem_counts": dict(elem_counts),
        "box_first": box_first,
        "box_last": box_last,
        "z_min": z_min_global,
        "z_max": z_max_global,
    }


def scan_traj_info(path: Path) -> dict:
    """Scan an ASE .traj file and return summary info."""
    from collections import Counter
    from ase.io.trajectory import Trajectory as ASETraj

    traj = ASETraj(str(path), mode="r")
    n_frames = len(traj)

    # read first and last frame
    first = traj[0]
    last = traj[n_frames - 1] if n_frames > 1 else first

    symbols = first.get_chemical_symbols()
    elem_counts = dict(Counter(symbols))

    cell_first = first.cell.diagonal()
    cell_last = last.cell.diagonal()

    z_all = first.positions[:, 2]
    z_min = float(z_all.min())
    z_max = float(z_all.max())

    ts_first = first.info.get("timestep", None)
    ts_last = last.info.get("timestep", None)
    dt = None
    if n_frames >= 2:
        second = traj[1]
        ts_second = second.info.get("timestep", None)
        if ts_first is not None and ts_second is not None:
            dt = ts_second - ts_first

    traj.close()

    return {
        "n_frames": n_frames,
        "n_atoms": len(first),
        "timesteps_first": ts_first,
        "timesteps_last": ts_last,
        "timestep_interval": dt,
        "elem_counts": dict(elem_counts),
        "box_first": {"Lx": cell_first[0], "Ly": cell_first[1], "Lz": cell_first[2]},
        "box_last": {"Lx": cell_last[0], "Ly": cell_last[1], "Lz": cell_last[2]},
        "pbc": list(first.pbc),
        "z_min": z_min,
        "z_max": z_max,
    }


def print_info(info: dict, traj_path: Path):
    """Pretty-print trajectory info."""
    print(f"\n{'=' * 60}")
    print(f"  z_density.py --info")
    print(f"{'=' * 60}")
    print(f"  File       : {traj_path}")
    print(f"  Format     : {traj_path.suffix}")
    fsize = traj_path.stat().st_size
    if fsize > 1e9:
        print(f"  File size  : {fsize / 1e9:.2f} GB")
    elif fsize > 1e6:
        print(f"  File size  : {fsize / 1e6:.1f} MB")
    else:
        print(f"  File size  : {fsize / 1e3:.1f} KB")

    print(f"\n  ── Frames ──")
    print(f"  Total frames : {info['n_frames']}")
    print(f"  N atoms      : {info['n_atoms']}")
    if info["timesteps_first"] is not None:
        print(f"  Timestep     : {info['timesteps_first']} → {info['timesteps_last']}")
    if info["timestep_interval"] is not None:
        print(f"  Interval     : {info['timestep_interval']} steps between frames")

    print(f"\n  ── Element composition (from first frame) ──")
    elem_counts = info["elem_counts"]
    total = sum(elem_counts.values())
    # separate solvent vs electrode
    solvent_elems = {}
    electrode_elems = {}
    for el, cnt in sorted(elem_counts.items(), key=lambda x: -x[1]):
        if el.split("(")[0] in ELECTRODE_ELEMENTS:
            electrode_elems[el] = cnt
        else:
            solvent_elems[el] = cnt

    if solvent_elems:
        print(f"  Solvent:")
        for el, cnt in sorted(solvent_elems.items(), key=lambda x: -x[1]):
            print(f"    {el:10s}  {cnt:6d}  ({100 * cnt / total:5.1f} %)")
    if electrode_elems:
        print(f"  Electrode:")
        for el, cnt in sorted(electrode_elems.items(), key=lambda x: -x[1]):
            print(f"    {el:10s}  {cnt:6d}  ({100 * cnt / total:5.1f} %)")

    # LAMMPS type counts if available
    if "type_counts" in info:
        print(f"\n  ── LAMMPS type → element map ──")
        tm = info.get("type_map", {})
        for t, cnt in sorted(info["type_counts"].items()):
            el = tm.get(t, "?")
            print(f"    type {t:2d} → {el:4s}  ({cnt:6d} atoms)")

    print(f"\n  ── Box dimensions ──")
    b = info["box_first"]
    print(f"  Lx = {b['Lx']:.4f} Å,  Ly = {b['Ly']:.4f} Å,  Lz = {b['Lz']:.4f} Å")
    if "pbc" in info:
        print(f"  PBC = {info['pbc']}")
    print(f"  z range (first frame) : {info['z_min']:.2f} → {info['z_max']:.2f} Å")

    # skip/stride suggestions
    nf = info["n_frames"]
    print(f"\n  ── Suggested skip / stride ──")
    if nf > 200:
        skip_10 = nf // 10
        skip_20 = nf // 5
        print(f"  To discard first 10%: --skip {skip_10}")
        print(f"  To discard first 20%: --skip {skip_20}")
    if nf > 100:
        for target in [200, 500, 1000]:
            s = max(1, nf // target)
            actual = (nf + s - 1) // s
            if s > 1:
                print(f"  ~{target} frames: --stride {s}  (→ {actual} frames)")
    if nf > 200:
        skip_sug = nf // 10
        stride_sug = max(1, (nf - skip_sug) // 500)
        actual = ((nf - skip_sug) + stride_sug - 1) // stride_sug
        print(f"  Recommended : --skip {skip_sug} --stride {stride_sug}"
              f"  (→ {actual} frames)")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame iterators — .lammpstrj and .traj
# ═══════════════════════════════════════════════════════════════════════════════

def iter_lammpstrj(path: Path, type_map: Dict[int, str],
                   skip: int = 0, stride: int = 1,
                   exclude_types: Optional[Set[int]] = None):
    """
    Yield (labels, z_coords, Lx, Ly, Lz) per selected frame.

    Parameters
    ----------
    skip          : discard the first `skip` frames (equilibration).
    stride        : after skipping, yield every `stride`-th frame.
    exclude_types : LAMMPS type IDs to drop (e.g. QM slab atoms).
    """
    frame_idx = 0
    accepted = 0
    _excl = exclude_types or set()
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if "TIMESTEP" not in line:
                continue
            _ts = f.readline()  # timestep value (unused)

            f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS
            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())

            header = f.readline().split()[2:]  # strip "ITEM:" "ATOMS"
            try:
                col_type = header.index("type")
                col_z = header.index("zu") if "zu" in header else header.index("z")
            except ValueError:
                for _ in range(n_atoms):
                    f.readline()
                frame_idx += 1
                continue

            # read atom data regardless (to advance file pointer)
            types_raw = np.empty(n_atoms, dtype=np.int32)
            z_arr = np.empty(n_atoms, dtype=np.float64)
            for i in range(n_atoms):
                tok = f.readline().split()
                types_raw[i] = int(tok[col_type])
                z_arr[i] = float(tok[col_z])

            # skip / stride logic
            if frame_idx < skip:
                frame_idx += 1
                continue
            if (frame_idx - skip) % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            # filter out excluded types
            if _excl:
                keep = np.array([t not in _excl for t in types_raw])
                types_raw = types_raw[keep]
                z_arr = z_arr[keep]

            elements = np.array([type_map.get(t, "X") for t in types_raw])
            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo

            # wrap z into [zlo, zhi)
            z_wrap = zlo + ((z_arr - zlo) % Lz)

            accepted += 1
            yield elements, z_wrap, Lx, Ly, Lz


def iter_traj(path: Path, skip: int = 0, stride: int = 1,
              type_label_map: Optional[Dict[int, str]] = None,
              exclude_types: Optional[Set[int]] = None):
    """
    Yield (labels, z_coords, Lx, Ly, Lz) per selected frame from ASE .traj.

    Parameters
    ----------
    type_label_map : when provided, replace chemical symbols with CES2
                     type_labels using atoms.info["lammps_type"].
    exclude_types : LAMMPS type IDs to drop. Uses atoms.info["lammps_type"]
                    stored by lammpstrj_to_traj.py.
    """
    from ase.io.trajectory import Trajectory as ASETraj

    traj = ASETraj(str(path), mode="r")
    n_total = len(traj)
    _excl = exclude_types or set()

    for frame_idx in range(n_total):
        if frame_idx < skip:
            continue
        if (frame_idx - skip) % stride != 0:
            continue

        atoms = traj[frame_idx]
        elements = np.array(atoms.get_chemical_symbols())
        z_arr = atoms.positions[:, 2].copy()
        cell = atoms.cell.diagonal()
        Lx, Ly, Lz = cell[0], cell[1], cell[2]
        lmp_types = None
        if "lammps_type" in atoms.info:
            lmp_types = np.array(atoms.info["lammps_type"])

        if type_label_map is not None:
            if lmp_types is None:
                traj.close()
                sys.exit(
                    "ERROR: --type-labels for .traj requires lammps_type metadata. "
                    "Convert from .lammpstrj with tools/lammpstrj_to_traj.py."
                )
            elements = np.array([type_label_map.get(int(t), f"type{int(t)}")
                                 for t in lmp_types])

        # filter out excluded types via stored lammps_type info
        if _excl and lmp_types is not None:
            keep = np.array([t not in _excl for t in lmp_types])
            elements = elements[keep]
            z_arr = z_arr[keep]

        # wrap z if periodic in z
        if atoms.pbc[2]:
            zlo = 0.0
            z_arr = zlo + ((z_arr - zlo) % Lz)

        yield elements, z_arr, Lx, Ly, Lz

    traj.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Core density calculation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_z_density(
    traj_path: Path,
    type_map: Optional[Dict[int, str]],
    target_elements: Optional[List[str]],
    target_type_labels: Optional[List[str]] = None,
    type_label_map: Optional[Dict[int, str]] = None,
    dz: float = 0.1,
    skip: int = 0,
    stride: int = 1,
    zlo: Optional[float] = None,
    zhi: Optional[float] = None,
    exclude_types: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """
    Compute z-density profiles.

    Returns
    -------
    z_centers      : bin centres [Å]
    num_profiles   : {element_or_type_label: ρ(z) [Å⁻³]}
    mass_profiles  : {element_or_type_label: ρ(z) [g/cm³]}
    meta           : dict with n_frames, Lx, Ly, etc.
    """
    suffix = traj_path.suffix.lower()
    use_type_labels = target_type_labels is not None

    if exclude_types:
        print(f"    Excluding LAMMPS types: {sorted(exclude_types)}")

    # ── choose iterator ──
    if suffix == ".lammpstrj":
        if type_map is None:
            lmp_in = _find_lammps_input(traj_path)
            if lmp_in:
                type_map = auto_detect_type_map(lmp_in)
                print(f"    Auto-detected type map from {lmp_in}")
            else:
                type_map = dict(DEFAULT_TYPE_MAP)
                print(f"    Using default type map (water only): {type_map}")
        print(f"    Type map: {type_map}")
        if use_type_labels:
            if type_label_map is None:
                lmp_in = _find_lammps_input(traj_path)
                if lmp_in:
                    type_label_map = auto_detect_type_label_map(lmp_in)
                    print(f"    Auto-detected type_label map from {lmp_in}")
            if not type_label_map:
                sys.exit(
                    "ERROR: --type-labels requires --type-label-map or an in.lammps "
                    "with group comments containing type_labels."
                )
            print(f"    Type_label map: {type_label_map}")
        if exclude_types:
            excl_elems = [f"{t}→{type_map.get(t,'?')}" for t in sorted(exclude_types)]
            print(f"    Excluded: {', '.join(excl_elems)}")
        frame_map = type_label_map if use_type_labels else type_map
        frame_iter = iter_lammpstrj(traj_path, frame_map, skip=skip, stride=stride,
                                    exclude_types=exclude_types)
    elif suffix == ".traj":
        if use_type_labels and not type_label_map:
            sys.exit(
                "ERROR: --type-labels for .traj requires --type-label-map or "
                "--lammps-input so LAMMPS types can be mapped to type_labels."
            )
        frame_iter = iter_traj(traj_path, skip=skip, stride=stride,
                               type_label_map=type_label_map if use_type_labels else None,
                               exclude_types=exclude_types)
    else:
        sys.exit(f"ERROR: unsupported file format '{suffix}'. Use .lammpstrj or .traj")

    # ── first pass: detect elements & z-range if not given ──
    # We stream frames, accumulating histograms on the fly.
    # On the first frame we discover which elements are present and set up bins.

    counts: Dict[str, np.ndarray] = {}
    edges: Optional[np.ndarray] = None
    all_elements: Set[str] = set()
    n_frames = 0
    Lx_sum = 0.0
    Ly_sum = 0.0
    n_bins = 0

    t0 = time.perf_counter()

    for elements, z_arr, Lx, Ly, Lz in frame_iter:
        # ── first frame: set up bins ──
        if edges is None:
            all_elements = set(elements)
            z_min = zlo if zlo is not None else z_arr.min() - 1.0
            z_max = zhi if zhi is not None else z_arr.max() + 1.0
            edges = np.arange(z_min, z_max + dz, dz)
            n_bins = len(edges) - 1

            # determine target labels
            if use_type_labels:
                target_elements = target_type_labels
                print(f"    Target type_labels: {target_elements}")
            elif target_elements is None:
                # all solvent = everything minus electrode metals
                solvent = sorted(all_elements - ELECTRODE_ELEMENTS - {"X"})
                if not solvent:
                    solvent = sorted(all_elements - {"X"})
                target_elements = solvent
                print(f"    Auto-detected solvent elements: {target_elements}")
            else:
                print(f"    Target elements: {target_elements}")

            for el in target_elements:
                counts[el] = np.zeros(n_bins, dtype=np.float64)
        else:
            all_elements |= set(elements)

        Lx_sum += Lx
        Ly_sum += Ly

        for el in target_elements:
            mask = elements == el
            if mask.any():
                h, _ = np.histogram(z_arr[mask], bins=edges)
                counts[el] += h

        n_frames += 1
        if n_frames % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    ... {n_frames} frames processed ({elapsed:.1f} s)")

    if n_frames == 0:
        sys.exit("ERROR: no frames read (check --skip / --stride values).")

    elapsed = time.perf_counter() - t0
    print(f"    Total frames: {n_frames}  ({elapsed:.1f} s)")

    # ── normalise ──
    Lx_avg = Lx_sum / n_frames
    Ly_avg = Ly_sum / n_frames
    area = Lx_avg * Ly_avg
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    num_profiles: Dict[str, np.ndarray] = {}
    mass_profiles: Dict[str, np.ndarray] = {}
    norm = area * dz * n_frames

    label_to_element: Dict[str, str] = {}
    if type_label_map and type_map:
        for tid, label in type_label_map.items():
            elem = type_map.get(tid)
            if elem:
                label_to_element[label] = elem

    for el in target_elements:
        num_profiles[el] = counts[el] / norm
        mass_elem = label_to_element.get(el, el)
        if mass_elem not in ATOMIC_MASS:
            mass_elem = _label_to_element(el) or mass_elem
        M = ATOMIC_MASS.get(mass_elem, 1.0)
        mass_profiles[el] = num_profiles[el] * M * _ANG3_TO_GCM3

    meta = {
        "n_frames": n_frames,
        "skip": skip,
        "stride": stride,
        "dz": dz,
        "Lx": Lx_avg,
        "Ly": Ly_avg,
        "area": area,
        "all_elements": sorted(all_elements),
        "target_elements": target_elements,
        "profile_mode": "type_label" if use_type_labels else "element",
        "label_to_element": label_to_element,
    }
    return z_centers, num_profiles, mass_profiles, meta


# ═══════════════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(z_centers, num_profiles, mass_profiles, meta, out_path: Path):
    elems = meta["target_elements"]
    key_name = "type_label" if meta.get("profile_mode") == "type_label" else "element"
    with open(out_path, "w") as f:
        f.write(f"# z_density.py output\n")
        f.write(f"# profile_mode={key_name}\n")
        f.write(f"# frames={meta['n_frames']}  skip={meta['skip']}"
                f"  stride={meta['stride']}  dz={meta['dz']:.3f} Ang\n")
        f.write(f"# Lx={meta['Lx']:.4f} Ang  Ly={meta['Ly']:.4f} Ang\n")
        f.write(f"# rho_*: number density [Ang^-3]   mass_*: mass density [g/cm^3]\n")
        num_cols  = [f"rho_{e}" for e in elems]
        mass_cols = [f"mass_{e}" for e in elems]
        f.write(",".join(["z_ang"] + num_cols + mass_cols) + "\n")
        for i, z in enumerate(z_centers):
            nv = [f"{num_profiles[e][i]:.8f}" for e in elems]
            mv = [f"{mass_profiles[e][i]:.8f}" for e in elems]
            f.write(",".join([f"{z:.4f}"] + nv + mv) + "\n")
    print(f"  Saved: {out_path}")


def plot_profiles(z_centers, num_profiles, mass_profiles, meta, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    elems = meta["target_elements"]
    key_name = "type_label" if meta.get("profile_mode") == "type_label" else "element"

    fig, axes = plt.subplots(1, 2, figsize=(_FW * 2.4, _FH * 1.4))

    # ── left panel: number density ──
    ax = axes[0]
    for i, el in enumerate(elems):
        ax.plot(z_centers, num_profiles[el],
                color=_C[(i + 1) % len(_C)], lw=_LW * 2, label=el)
    ax.set_xlabel("z (Å)", fontsize=_LS)
    ax.set_ylabel("Number density (Å$^{-3}$)", fontsize=_LS)
    ax.set_title(f"Number density ρ(z) by {key_name}", fontsize=_FS)
    ax.legend(fontsize=_FS - 1, frameon=False, ncol=max(1, len(elems) // 4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if ps:
        ps.set_ylim_top_margin(ax)

    # ── right panel: mass density ──
    ax = axes[1]
    for i, el in enumerate(elems):
        ax.plot(z_centers, mass_profiles[el],
                color=_C[(i + 1) % len(_C)], lw=_LW * 2, label=el)
    ax.set_xlabel("z (Å)", fontsize=_LS)
    ax.set_ylabel("Mass density (g/cm³)", fontsize=_LS)
    ax.set_title(f"Mass density ρ(z) by {key_name}", fontsize=_FS)
    ax.legend(fontsize=_FS - 1, frameon=False, ncol=max(1, len(elems) // 4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if ps:
        ps.set_ylim_top_margin(ax)

    # annotation
    axes[0].text(
        0.99, 0.97,
        f"frames={meta['n_frames']}  skip={meta['skip']}  "
        f"stride={meta['stride']}  dz={meta['dz']:.2f} Å",
        transform=axes[0].transAxes, ha="right", va="top",
        fontsize=_FS - 2, color="gray",
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_qmmm_evolution_csv(
    results: List[dict],
    elements: List[str],
    out_path: Path,
):
    """Write long-format per-mm density profiles for downstream analysis."""
    key_name = "type_label"
    if results:
        key_name = (
            "type_label"
            if results[0]["meta"].get("profile_mode") == "type_label"
            else "element"
        )
    with open(out_path, "w") as f:
        f.write("# z_density.py QM/MM mm_N evolution output\n")
        f.write("# rho: number density [Ang^-3], mass: mass density [g/cm^3]\n")
        f.write(f"mm_step,mm_dir,z_ang,{key_name},rho,mass\n")
        for result in results:
            step = result["step"]
            mm_name = result["mm_dir"].name
            z_centers = result["z_centers"]
            num_profiles = result["num_profiles"]
            mass_profiles = result["mass_profiles"]
            for el in elements:
                if el not in num_profiles:
                    continue
                rho = num_profiles[el]
                mass = mass_profiles[el]
                for z, rv, mv in zip(z_centers, rho, mass):
                    f.write(f"{step},{mm_name},{z:.4f},{el},{rv:.8f},{mv:.8f}\n")
    print(f"  Saved: {out_path}")


def plot_qmmm_evolution(
    results: List[dict],
    elements: List[str],
    dz: float,
    out_path: Path,
):
    """Plot element- or type_label-resolved rho(z) evolution across mm_N steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    steps = [r["step"] for r in results]
    z_min = min(float(r["z_centers"][0] - 0.5 * dz) for r in results)
    z_max = max(float(r["z_centers"][-1] + 0.5 * dz) for r in results)
    z_common = np.arange(z_min + 0.5 * dz, z_max, dz)

    n_elems = len(elements)
    ncols = 2 if n_elems > 1 else 1
    nrows = (n_elems + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_FW * 1.7 * ncols, _FH * 1.25 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for idx, el in enumerate(elements):
        ax = axes[idx // ncols][idx % ncols]
        matrix = np.full((len(z_common), len(results)), np.nan, dtype=float)
        for col, result in enumerate(results):
            profile = result["num_profiles"].get(el)
            if profile is None:
                continue
            z_src = result["z_centers"]
            matrix[:, col] = np.interp(
                z_common,
                z_src,
                profile,
                left=np.nan,
                right=np.nan,
            )

        vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 0.0
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, len(steps) - 0.5, z_common[0], z_common[-1]),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
        )
        ax.set_title(f"{el} number density", fontsize=_FS)
        ax.set_ylabel("z (Å)", fontsize=_LS)
        ax.set_xticks(range(len(steps)))
        tick_step = max(1, len(steps) // 8)
        labels = [str(s) if i % tick_step == 0 else "" for i, s in enumerate(steps)]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=_FS - 1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=_FS - 2)
        cbar.set_label("Å$^{-3}$", fontsize=_FS - 1)

    for idx in range(n_elems, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("mm step", fontsize=_LS)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def resolve_type_map_for_path(args, traj_path: Path) -> Tuple[Optional[Dict[int, str]], Optional[Path]]:
    """Resolve the LAMMPS type map for one trajectory, when needed."""
    type_map = None
    lmp_in_path = args.lammps_input or _find_lammps_input(traj_path)
    if args.type_map:
        type_map = parse_type_map_str(args.type_map)
    elif lmp_in_path:
        type_map = auto_detect_type_map(lmp_in_path)
    return type_map, lmp_in_path


def resolve_type_label_map_for_path(
    args,
    lmp_in_path: Optional[Path],
) -> Optional[Dict[int, str]]:
    """Resolve the LAMMPS type→type_label map when type_label profiles are requested."""
    if not args.type_labels:
        return None
    if args.type_label_map:
        return parse_type_label_map_str(args.type_label_map)
    if lmp_in_path:
        return auto_detect_type_label_map(lmp_in_path)
    return None


def resolve_exclude_types(args, lmp_in_path: Optional[Path]) -> Optional[Set[int]]:
    """Resolve LAMMPS types excluded from density calculations."""
    if args.exclude_types:
        return set(args.exclude_types)
    if args.all_atoms:
        return None
    if lmp_in_path:
        solute_types = detect_solute_types(lmp_in_path)
        if solute_types:
            return solute_types
    return None


def run_qmmm_mm_density(args) -> None:
    """Compute per-mm_N z-density plots and a root-level evolution plot."""
    run_dir = args.qmmm_mm.resolve()
    if not run_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {run_dir}")

    mm_dirs = find_mm_dirs(run_dir)
    if not mm_dirs:
        sys.exit(f"ERROR: no mm_N directories found in {run_dir}")

    print(f"\n{'=' * 60}")
    print(f"  z_density.py — QM/MM mm_N z-density sweep")
    print(f"{'=' * 60}")
    print(f"  Run directory : {run_dir}")
    print(f"  mm_N found    : {len(mm_dirs)}")
    print(f"  skip={args.skip}  stride={args.stride}  dz={args.dz} Å")

    results: List[dict] = []
    skipped = 0

    for step, mm_dir in mm_dirs:
        traj_path = find_density_traj_in_mm_dir(mm_dir)
        if traj_path is None:
            print(f"\nmm_{step}: no trajectory found, skipping")
            skipped += 1
            continue

        print(f"\nmm_{step}: {traj_path.name}")
        type_map, lmp_in_path = resolve_type_map_for_path(args, traj_path)
        type_label_map = resolve_type_label_map_for_path(args, lmp_in_path)
        exclude_types = resolve_exclude_types(args, lmp_in_path)
        if type_map:
            print(f"  Type map: {type_map}")
        if type_label_map:
            print(f"  Type_label map: {type_label_map}")
        if exclude_types:
            print(f"  Excluding LAMMPS types: {sorted(exclude_types)}")

        z_centers, num_profiles, mass_profiles, meta = compute_z_density(
            traj_path,
            type_map=type_map,
            target_elements=args.elements,
            target_type_labels=args.type_labels,
            type_label_map=type_label_map,
            dz=args.dz,
            skip=args.skip,
            stride=args.stride,
            zlo=args.zlo,
            zhi=args.zhi,
            exclude_types=exclude_types,
        )

        prefix = args.prefix
        write_csv(z_centers, num_profiles, mass_profiles, meta,
                  mm_dir / f"{prefix}_rawdata.csv")
        plot_profiles(z_centers, num_profiles, mass_profiles, meta,
                      mm_dir / f"{prefix}.png")

        results.append({
            "step": step,
            "mm_dir": mm_dir,
            "traj_path": traj_path,
            "z_centers": z_centers,
            "num_profiles": num_profiles,
            "mass_profiles": mass_profiles,
            "meta": meta,
        })

    if not results:
        sys.exit("ERROR: no mm_N trajectories were processed.")

    elements = sorted({el for r in results for el in r["meta"]["target_elements"]})
    print(f"\n[Summary] Writing mm_N evolution outputs to {run_dir} ...")
    write_qmmm_evolution_csv(
        results,
        elements,
        run_dir / f"{args.prefix}_mm_evolution_rawdata.csv",
    )
    plot_qmmm_evolution(
        results,
        elements,
        args.dz,
        run_dir / f"{args.prefix}_mm_evolution.png",
    )

    print(f"\nDone. Processed {len(results)} mm_N directories, skipped {skipped}.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute element- or type_label-resolved z-density profiles ρ(z) from "
            ".lammpstrj or .traj trajectory files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/z_density.py run/ces2.emd.lammpstrj --skip 100 --stride 5\n"
            "  python tools/z_density.py md.traj --elements O H K\n"
            "  python tools/z_density.py ces2.emd.lammpstrj --type-labels O_oh Ow\n"
            "  python tools/z_density.py ces2.emd.lammpstrj --all-atoms\n"
            "  python tools/z_density.py ces2.emd.lammpstrj --exclude-types 6 7\n"
            "  python tools/z_density.py --qmmm-mm run_dir --elements O H Cs\n"
            "  python tools/z_density.py run/ces2.emd.lammpstrj "
            '--type-map "1:H 2:O 3:Cs 4:H 5:O 6:Ir 7:O"\n'
        ),
    )
    p.add_argument("traj", type=Path, nargs="?", default=None,
                   help="Trajectory file (.lammpstrj or .traj)")
    p.add_argument("--qmmm-mm", type=Path, default=None, metavar="RUN_DIR",
                   help=("Process every mm_N directory in a QM/MM run. Each "
                         "mm_N gets its own z_density outputs, and RUN_DIR gets "
                         "profile-wise mm-step evolution plots."))
    p.add_argument("--info", action="store_true",
                   help="Scan trajectory and print summary (frames, elements, "
                        "box, suggested skip/stride). No density calculation.")
    p.add_argument("--elements", nargs="+", default=None,
                   help="Element symbols to analyse (default: all solvent elements)")
    p.add_argument("--type-labels", nargs="+", default=None,
                   help=("CES2/LAMMPS type_labels to analyse separately, e.g. "
                         "O_oh Ow. Uses in.lammps group comments or "
                         "--type-label-map. Cannot be combined with --elements."))
    p.add_argument("--dz", type=float, default=0.1,
                   help="Bin width in Å (default: 0.1)")
    p.add_argument("--skip", type=int, default=0,
                   help="Discard the first N frames (equilibration, default: 0)")
    p.add_argument("--stride", type=int, default=1,
                   help="After skipping, use every N-th frame (default: 1 = all)")
    p.add_argument("--zlo", type=float, default=None,
                   help="Lower z-limit for binning [Å] (default: auto from data)")
    p.add_argument("--zhi", type=float, default=None,
                   help="Upper z-limit for binning [Å] (default: auto from data)")
    p.add_argument("--exclude-types", nargs="+", type=int, default=None,
                   help=("LAMMPS type IDs to exclude (e.g. QM slab atoms). "
                         "Example: --exclude-types 6 7"))
    p.add_argument("--all-atoms", action="store_true",
                   help=("Include ALL atoms including SOLUTE/QM slab. "
                         "By default, SOLUTE types are auto-excluded if "
                         "in.lammps is found (solvent-only mode)."))
    p.add_argument("--type-map", type=str, default=None,
                   help=("Manual LAMMPS type→element map for .lammpstrj, "
                         "e.g. '1:H 2:O 3:Cs 4:H 5:O 6:Ir 7:O'"))
    p.add_argument("--type-label-map", type=str, default=None,
                   help=("Manual LAMMPS type→type_label map, e.g. "
                         "'1:Hw 2:Ow 3:Li 4:H_oh 5:O_oh 6:Ir 7:O'"))
    p.add_argument("--lammps-input", type=Path, default=None,
                   help="LAMMPS input file for auto-detecting type→element map")
    p.add_argument("-o", "--outdir", type=Path, default=None,
                   help="Output directory (default: same as trajectory)")
    p.add_argument("--prefix", type=str, default="z_density",
                   help="Output file prefix (default: z_density)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.elements and args.type_labels:
        sys.exit("ERROR: use either --elements or --type-labels, not both.")

    if args.qmmm_mm is not None:
        if args.traj is not None:
            sys.exit("ERROR: provide either a trajectory file or --qmmm-mm, not both.")
        if args.info:
            sys.exit("ERROR: --info is only supported for single trajectory mode.")
        if args.outdir is not None:
            sys.exit("ERROR: --outdir is not supported with --qmmm-mm.")
        run_qmmm_mm_density(args)
        return

    if args.traj is None:
        sys.exit("ERROR: trajectory file is required unless --qmmm-mm is used.")

    traj_path = args.traj.resolve()
    if not traj_path.exists():
        sys.exit(f"ERROR: file not found: {traj_path}")

    out_dir = args.outdir or traj_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  z_density.py — Element/type_label-resolved z-density profiles")
    print(f"{'=' * 60}")
    print(f"  Trajectory : {traj_path}")
    print(f"  Format     : {traj_path.suffix}")
    print(f"  skip={args.skip}  stride={args.stride}  dz={args.dz} Å")

    # ── type map / type_label map ──
    type_map = None
    type_label_map = None
    lmp_in_path = args.lammps_input or _find_lammps_input(traj_path)
    if args.type_map:
        type_map = parse_type_map_str(args.type_map)
        print(f"  Manual type map: {type_map}")
    elif lmp_in_path:
        type_map = auto_detect_type_map(lmp_in_path)
        print(f"  Auto-detected type map from {lmp_in_path}")

    if args.type_labels:
        if args.type_label_map:
            type_label_map = parse_type_label_map_str(args.type_label_map)
            print(f"  Manual type_label map: {type_label_map}")
        elif lmp_in_path:
            type_label_map = auto_detect_type_label_map(lmp_in_path)
            print(f"  Auto-detected type_label map from {lmp_in_path}")
        if not type_label_map:
            sys.exit(
                "ERROR: --type-labels requires --type-label-map or an in.lammps "
                "with group comments containing type_labels."
            )

    # ── resolve exclude_types ──
    # Default: solvent-only (auto-exclude SOLUTE/QM types from in.lammps).
    # Override with --all-atoms to include everything, or --exclude-types for manual control.
    exclude_types: Optional[Set[int]] = None
    if args.exclude_types:
        exclude_types = set(args.exclude_types)
        print(f"  Excluding LAMMPS types: {sorted(exclude_types)}")
    elif not args.all_atoms:
        # solvent-only by default
        if lmp_in_path:
            solute_types = detect_solute_types(lmp_in_path)
            if solute_types:
                exclude_types = solute_types
                if type_map:
                    excl_info = [f"{t}→{type_map.get(t,'?')}" for t in sorted(solute_types)]
                else:
                    excl_info = [str(t) for t in sorted(solute_types)]
                print(f"  Solvent-only (default): excluding SOLUTE types "
                      f"{', '.join(excl_info)}  (from {lmp_in_path.name})")
                print(f"  Use --all-atoms to include SOLUTE/QM atoms")
            else:
                print(f"  No SOLUTE group found in {lmp_in_path} — using all atoms")
        # if no in.lammps found, silently use all atoms
    else:
        print(f"  --all-atoms: including all atoms (no type exclusion)")

    # ── --info mode: scan and exit ──
    if args.info:
        print(f"\n  Scanning trajectory ...")
        if traj_path.suffix.lower() == ".lammpstrj":
            if type_map is None:
                if lmp_in_path:
                    type_map = auto_detect_type_map(lmp_in_path)
                else:
                    type_map = dict(DEFAULT_TYPE_MAP)
            info = scan_lammpstrj_info(traj_path, type_map)
        else:
            info = scan_traj_info(traj_path)
        print_info(info, traj_path)
        return

    # ── compute ──
    print(f"\n[1] Computing density profiles ...")
    z_centers, num_profiles, mass_profiles, meta = compute_z_density(
        traj_path,
        type_map=type_map,
        target_elements=args.elements,
        target_type_labels=args.type_labels,
        type_label_map=type_label_map,
        dz=args.dz,
        skip=args.skip,
        stride=args.stride,
        zlo=args.zlo,
        zhi=args.zhi,
        exclude_types=exclude_types,
    )

    # ── peak info ──
    print(f"\n    Peak densities:")
    for el in meta["target_elements"]:
        rho = num_profiles[el]
        mrho = mass_profiles[el]
        if rho.max() > 0:
            z_peak = z_centers[np.argmax(rho)]
            print(f"    {el:4s}  ρ_max = {rho.max():.5f} Å⁻³"
                  f"  ({mrho.max():.4f} g/cm³)  at z = {z_peak:.2f} Å")

    # ── write outputs ──
    prefix = args.prefix
    print(f"\n[2] Writing outputs to {out_dir} ...")
    write_csv(z_centers, num_profiles, mass_profiles, meta,
              out_dir / f"{prefix}_rawdata.csv")
    plot_profiles(z_centers, num_profiles, mass_profiles, meta,
                  out_dir / f"{prefix}.png")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
