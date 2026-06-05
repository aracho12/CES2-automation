#!/usr/bin/env python3
"""
water_orientation.py — Water dipole and O-H orientation profiles vs. z
======================================================================

Computes water orientational order parameters from a LAMMPS custom dump
(.lammpstrj/.dump) or an ASE .traj file:

  P1(z) = <cos(theta)>
  P2(z) = <(3 cos^2(theta) - 1) / 2>

For the water dipole, the vector is O -> midpoint(H1,H2).  For O-H bonds,
the vector is O -> H.  z-binning uses the parent water oxygen position.

Outputs
-------
  water_orientation_rawdata.csv
      z, counts, P1/P2 for dipole and O-H vectors
  water_orientation_oh_angle_distribution.csv
      long-format O-H theta distribution P(theta | z)
  water_orientation.png
      dipole P1/P2, O-H P1/P2, and O-H angle heatmap

Usage examples
--------------
  python tools/water_orientation.py run/ces2.emd.lammpstrj --skip 100 --stride 5
  python tools/water_orientation.py run/ces2.emd.wrapped.traj --dz 0.2
  python tools/water_orientation.py run/ces2.emd.lammpstrj --water-o-types 2 --water-h-types 1
  python tools/water_orientation.py --qmmm-mm /path/to/qmmm_run --skip 100 --stride 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

# ── local imports ────────────────────────────────────────────────────────────
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ces2_water_orientation_cache"
for _env_name, _subdir in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
    if not os.environ.get(_env_name):
        _cache_dir = _CACHE_ROOT / _subdir
        try:
            _cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ[_env_name] = str(_cache_dir)
        except OSError:
            pass

_DEFAULT_COLORS = ["#515151", "#F14040", "#1A6FDF", "#37AD6B", "#B177DE", "#FEC211"]
_DEFAULT_FS = 9
_DEFAULT_LS = 9
_DEFAULT_LW = 0.4
_DEFAULT_FIGSIZE = (3.5, 2.8)

try:
    from lammpstrj_to_traj import iter_frames
except ImportError as exc:
    raise SystemExit(
        "ERROR: could not import tools/lammpstrj_to_traj.py. "
        "Run this script from the CES2-automation repository."
    ) from exc


def _load_plot_style():
    """Load shared plot style lazily so --help/CSV-only runs stay lightweight."""
    try:
        import plot_setting as ps

        return (
            ps.colors,
            ps.fontsize,
            ps.labelsize,
            ps.linewidth,
            ps.figsize,
        )
    except ImportError:
        return (
            _DEFAULT_COLORS,
            _DEFAULT_FS,
            _DEFAULT_LS,
            _DEFAULT_LW,
            _DEFAULT_FIGSIZE,
        )


DEFAULT_WATER_H_TYPES: Set[int] = {1}
DEFAULT_WATER_O_TYPES: Set[int] = {2}


@dataclass
class OrientationFrame:
    """One trajectory frame in the minimal form needed for orientation analysis."""

    timestep: Optional[int]
    positions: np.ndarray
    types: np.ndarray
    box_lo: np.ndarray
    box_lengths: np.ndarray


@dataclass
class FrameOrientations:
    """Per-frame orientation samples before z-binning."""

    dipole_z: np.ndarray
    dipole_cos: np.ndarray
    oh_z: np.ndarray
    oh_cos: np.ndarray
    n_water_o: int
    n_paired_water: int
    n_skipped_water: int


@dataclass
class OrientationResult:
    """Binned orientation profiles and angle distributions."""

    z_edges: np.ndarray
    z_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    n_water: np.ndarray
    n_oh: np.ndarray
    p1_dipole: np.ndarray
    p2_dipole: np.ndarray
    theta_dipole_mean: np.ndarray
    p1_oh: np.ndarray
    p2_oh: np.ndarray
    theta_oh_mean: np.ndarray
    oh_angle_counts: np.ndarray
    oh_angle_prob_density: np.ndarray
    meta: Dict[str, object]


# ═══════════════════════════════════════════════════════════════════════════════
#  LAMMPS input helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _find_lammps_input(traj_path: Path) -> Optional[Path]:
    """Search for in.lammps/base.in.lammps near a trajectory file."""
    candidates = [
        traj_path.parent / "in.lammps",
        traj_path.parent / "base.in.lammps",
        traj_path.parent.parent / "in.lammps",
        traj_path.parent.parent / "base.in.lammps",
        traj_path.parent / "export" / "base.in.lammps",
        traj_path.parent.parent / "export" / "base.in.lammps",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def detect_water_types(lammps_input: Path) -> Tuple[Set[int], Set[int]]:
    """
    Detect water oxygen/proton LAMMPS types from generated CES2 group lines.

    Preferred generated form:
      group    OXYGEN   type 2   # Ow (water O)
      group    PROTON   type 1   # Hw (water H)
    """
    water_o: Set[int] = set()
    water_h: Set[int] = set()
    if not lammps_input.exists():
        return water_o, water_h

    group_pattern = re.compile(
        r"^\s*group\s+(\S+)\s+type\s+([\d\s]+)(?:#\s*(.+))?$",
        re.IGNORECASE,
    )

    with open(lammps_input) as f:
        for line in f:
            m = group_pattern.match(line)
            if not m:
                continue
            group_name = m.group(1).upper()
            type_ids = {int(x) for x in m.group(2).split()}
            comment = (m.group(3) or "").lower()

            if group_name == "OXYGEN":
                water_o |= type_ids
            elif group_name == "PROTON":
                water_h |= type_ids
            elif "water o" in comment or re.search(r"\bow[\w-]*\b", comment):
                water_o |= type_ids
            elif "water h" in comment or re.search(r"\bhw[\w-]*\b", comment):
                water_h |= type_ids

    return water_o, water_h


def resolve_water_types(
    water_o_types: Optional[Sequence[int]],
    water_h_types: Optional[Sequence[int]],
    lammps_input: Optional[Path],
) -> Tuple[Set[int], Set[int], str]:
    """Resolve water atom type selectors from CLI, LAMMPS input, or defaults."""
    source = "defaults"
    detected_o: Set[int] = set()
    detected_h: Set[int] = set()
    if lammps_input:
        detected_o, detected_h = detect_water_types(lammps_input)
        if detected_o or detected_h:
            source = str(lammps_input)

    o_types = set(water_o_types) if water_o_types else detected_o
    h_types = set(water_h_types) if water_h_types else detected_h

    if not o_types:
        o_types = set(DEFAULT_WATER_O_TYPES)
    if not h_types:
        h_types = set(DEFAULT_WATER_H_TYPES)

    if water_o_types or water_h_types:
        source = "command line"

    return o_types, h_types, source


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory iterators
# ═══════════════════════════════════════════════════════════════════════════════


def _position_columns(columns: Sequence[str]) -> List[str]:
    """Choose Cartesian position columns from a LAMMPS custom dump header."""
    col_set = set(columns)
    for names in (("xu", "yu", "zu"), ("x", "y", "z")):
        if all(name in col_set for name in names):
            return list(names)
    raise KeyError("No Cartesian position columns found; expected xu/yu/zu or x/y/z.")


def iter_lammps_orientation_frames(
    dump_path: Path,
    skip: int = 0,
    stride: int = 1,
    end: Optional[int] = None,
) -> Iterable[OrientationFrame]:
    """Yield OrientationFrame objects from a LAMMPS custom dump."""
    for frame in iter_frames(dump_path, stride=stride, start=skip, stop=end):
        cols = frame["columns"]
        col_idx = {name: i for i, name in enumerate(cols)}
        pos_cols = _position_columns(cols)
        if "type" not in col_idx:
            raise KeyError("LAMMPS dump is missing the required 'type' column.")

        data = frame["data"]
        types = data[:, col_idx["type"]].astype(np.int32)
        positions = data[:, [col_idx[name] for name in pos_cols]].astype(np.float64)
        box = np.asarray(frame["box"], dtype=np.float64)

        yield OrientationFrame(
            timestep=frame.get("timestep"),
            positions=positions,
            types=types,
            box_lo=box[:, 0],
            box_lengths=box[:, 1] - box[:, 0],
        )


def iter_traj_orientation_frames(
    traj_path: Path,
    skip: int = 0,
    stride: int = 1,
    end: Optional[int] = None,
    allow_symbol_fallback: bool = False,
) -> Iterable[OrientationFrame]:
    """Yield OrientationFrame objects from an ASE .traj file."""
    try:
        from ase.io.trajectory import Trajectory as ASETraj
    except ImportError:
        raise SystemExit("ERROR: .traj input requires ASE. Install ase first.")

    traj = ASETraj(str(traj_path), mode="r")
    try:
        n_total = len(traj)
        stop = n_total if end is None else min(end, n_total)
        for frame_idx in range(skip, stop, stride):
            atoms = traj[frame_idx]
            if "lammps_type" in atoms.info:
                types = np.asarray(atoms.info["lammps_type"], dtype=np.int32)
            elif allow_symbol_fallback:
                symbols = np.asarray(atoms.get_chemical_symbols())
                types = np.zeros(len(symbols), dtype=np.int32)
                types[symbols == "H"] = 1
                types[symbols == "O"] = 2
            else:
                raise SystemExit(
                    "ERROR: .traj input is missing atoms.info['lammps_type']. "
                    "Convert from .lammpstrj with tools/lammpstrj_to_traj.py, "
                    "or pass --allow-symbol-fallback for simple water-only .traj files."
                )

            cell = atoms.cell.diagonal().astype(np.float64)
            yield OrientationFrame(
                timestep=atoms.info.get("timestep"),
                positions=atoms.positions.astype(np.float64, copy=True),
                types=types,
                box_lo=np.zeros(3, dtype=np.float64),
                box_lengths=cell,
            )
    finally:
        traj.close()


def iter_orientation_frames(
    traj_path: Path,
    skip: int = 0,
    stride: int = 1,
    end: Optional[int] = None,
    allow_symbol_fallback: bool = False,
) -> Iterable[OrientationFrame]:
    """Dispatch to the proper trajectory iterator."""
    if traj_path.suffix.lower() == ".traj":
        yield from iter_traj_orientation_frames(
            traj_path,
            skip=skip,
            stride=stride,
            end=end,
            allow_symbol_fallback=allow_symbol_fallback,
        )
    else:
        yield from iter_lammps_orientation_frames(
            traj_path,
            skip=skip,
            stride=stride,
            end=end,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Orientation calculation
# ═══════════════════════════════════════════════════════════════════════════════


def _wrap_xy(
    positions: np.ndarray,
    box_lo: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Wrap x/y positions into the primary cell; leave z unchanged."""
    wrapped = positions.copy()
    for axis in (0, 1):
        length = float(box_lengths[axis])
        if length > 0:
            wrapped[:, axis] = box_lo[axis] + ((wrapped[:, axis] - box_lo[axis]) % length)
    return wrapped


def _minimum_image_xy(vecs: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention in x/y only."""
    out = vecs.copy()
    for axis in (0, 1):
        length = float(box_lengths[axis])
        if length > 0:
            out[:, axis] -= np.round(out[:, axis] / length) * length
    return out


def _safe_cos_z(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cos(theta_z) and a finite-vector mask."""
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    cos_z = np.full(len(vectors), np.nan, dtype=np.float64)
    cos_z[valid] = vectors[valid, 2] / norms[valid]
    cos_z = np.clip(cos_z, -1.0, 1.0)
    return cos_z, valid


def measure_frame_orientations(
    frame: OrientationFrame,
    water_o_types: Set[int],
    water_h_types: Set[int],
    oh_cutoff: float = 1.35,
) -> FrameOrientations:
    """Pair water O atoms with their two nearest water H atoms and measure cos(theta)."""
    types = np.asarray(frame.types, dtype=np.int32)
    o_idx = np.where(np.isin(types, list(water_o_types)))[0]
    h_idx = np.where(np.isin(types, list(water_h_types)))[0]

    empty = np.array([], dtype=np.float64)
    if len(o_idx) == 0 or len(h_idx) < 2:
        return FrameOrientations(empty, empty, empty, empty, len(o_idx), 0, len(o_idx))

    pos = _wrap_xy(frame.positions, frame.box_lo, frame.box_lengths)
    o_pos = pos[o_idx]
    h_pos = pos[h_idx]

    lx, ly = float(frame.box_lengths[0]), float(frame.box_lengths[1])
    shifts = []
    x_shifts = (-lx, 0.0, lx) if lx > 0 else (0.0,)
    y_shifts = (-ly, 0.0, ly) if ly > 0 else (0.0,)
    for sx in x_shifts:
        for sy in y_shifts:
            shifts.append((sx, sy, 0.0))

    image_positions = np.concatenate([h_pos + np.asarray(s) for s in shifts], axis=0)
    image_to_h_local = np.tile(np.arange(len(h_idx)), len(shifts))
    tree = cKDTree(image_positions)
    k_query = min(max(12, 2 * len(shifts)), len(image_positions))
    dists, image_hits = tree.query(o_pos, k=k_query)
    if k_query == 1:
        dists = dists[:, None]
        image_hits = image_hits[:, None]

    dipole_vectors: List[np.ndarray] = []
    oh_vectors: List[np.ndarray] = []
    z_dipole: List[float] = []
    z_oh: List[float] = []
    skipped = 0

    for o_local, (dist_row, hit_row) in enumerate(zip(dists, image_hits)):
        selected_h: List[int] = []
        for dist, hit in zip(dist_row, hit_row):
            if not np.isfinite(dist) or dist > oh_cutoff:
                continue
            h_local = int(image_to_h_local[int(hit)])
            if h_local not in selected_h:
                selected_h.append(h_local)
            if len(selected_h) == 2:
                break

        if len(selected_h) != 2:
            skipped += 1
            continue

        vecs = h_pos[selected_h] - o_pos[o_local]
        vecs = _minimum_image_xy(vecs, frame.box_lengths)
        dipole_vec = 0.5 * (vecs[0] + vecs[1])

        dipole_vectors.append(dipole_vec)
        oh_vectors.extend([vecs[0], vecs[1]])
        z0 = float(o_pos[o_local, 2])
        z_dipole.append(z0)
        z_oh.extend([z0, z0])

    if not dipole_vectors:
        return FrameOrientations(empty, empty, empty, empty, len(o_idx), 0, skipped)

    dipole_arr = np.vstack(dipole_vectors)
    oh_arr = np.vstack(oh_vectors)
    dipole_cos, dipole_valid = _safe_cos_z(dipole_arr)
    oh_cos, oh_valid = _safe_cos_z(oh_arr)

    dipole_z = np.asarray(z_dipole, dtype=np.float64)[dipole_valid]
    oh_z_arr = np.asarray(z_oh, dtype=np.float64)[oh_valid]

    return FrameOrientations(
        dipole_z=dipole_z,
        dipole_cos=dipole_cos[dipole_valid],
        oh_z=oh_z_arr,
        oh_cos=oh_cos[oh_valid],
        n_water_o=len(o_idx),
        n_paired_water=len(dipole_z),
        n_skipped_water=skipped + int((~dipole_valid).sum()),
    )


def _make_edges_from_first_frame(
    sample_z: np.ndarray,
    dz: float,
    zlo: Optional[float],
    zhi: Optional[float],
) -> np.ndarray:
    """Create z-bin edges from user limits or first-frame water oxygen range."""
    if zlo is None:
        if len(sample_z) == 0:
            raise ValueError("No water oxygen z positions available for automatic zlo.")
        zlo_val = float(np.nanmin(sample_z)) - dz
    else:
        zlo_val = float(zlo)

    if zhi is None:
        if len(sample_z) == 0:
            raise ValueError("No water oxygen z positions available for automatic zhi.")
        zhi_val = float(np.nanmax(sample_z)) + dz
    else:
        zhi_val = float(zhi)

    if not np.isfinite(zlo_val) or not np.isfinite(zhi_val) or zhi_val <= zlo_val:
        raise ValueError(f"Bad z range: zlo={zlo_val}, zhi={zhi_val}")

    n_bins = max(1, int(np.ceil((zhi_val - zlo_val) / dz)))
    return zlo_val + dz * np.arange(n_bins + 1, dtype=np.float64)


def _add_binned_samples(
    edges: np.ndarray,
    z_values: np.ndarray,
    values: np.ndarray,
    counts: Optional[np.ndarray],
    sums: np.ndarray,
) -> None:
    """Accumulate one scalar sample per z value into counts/sums arrays."""
    if len(z_values) == 0:
        return
    bin_idx = np.searchsorted(edges, z_values, side="right") - 1
    n_bins = len(sums)
    valid = (
        (bin_idx >= 0)
        & (bin_idx < n_bins)
        & np.isfinite(z_values)
        & np.isfinite(values)
    )
    if not valid.any():
        return
    if counts is not None:
        np.add.at(counts, bin_idx[valid], 1.0)
    np.add.at(sums, bin_idx[valid], values[valid])


def _profile_average(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return sums/counts with NaN in empty bins."""
    out = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def compute_water_orientation(
    traj_path: Path,
    water_o_types: Set[int],
    water_h_types: Set[int],
    dz: float = 0.1,
    skip: int = 0,
    stride: int = 1,
    end: Optional[int] = None,
    zlo: Optional[float] = None,
    zhi: Optional[float] = None,
    theta_bins: int = 90,
    oh_cutoff: float = 1.35,
    allow_symbol_fallback: bool = False,
    verbose: bool = True,
) -> OrientationResult:
    """Compute binned water dipole and O-H orientational order parameters."""
    if dz <= 0:
        raise ValueError("dz must be > 0")
    if theta_bins < 1:
        raise ValueError("theta_bins must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if skip < 0:
        raise ValueError("skip must be >= 0")

    theta_edges = np.linspace(0.0, 180.0, theta_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dtheta = float(theta_edges[1] - theta_edges[0])

    z_edges: Optional[np.ndarray] = None
    z_centers: Optional[np.ndarray] = None

    n_water = n_oh = None
    sum_dipole_cos = sum_dipole_p2 = sum_dipole_theta = None
    sum_oh_cos = sum_oh_p2 = sum_oh_theta = None
    oh_angle_counts = None

    frames_read = 0
    total_water_o = 0
    total_paired_water = 0
    total_skipped_water = 0
    t0 = time.perf_counter()

    frame_iter = iter_orientation_frames(
        traj_path,
        skip=skip,
        stride=stride,
        end=end,
        allow_symbol_fallback=allow_symbol_fallback,
    )

    for frame in frame_iter:
        orient = measure_frame_orientations(
            frame,
            water_o_types=water_o_types,
            water_h_types=water_h_types,
            oh_cutoff=oh_cutoff,
        )
        total_water_o += orient.n_water_o
        total_paired_water += orient.n_paired_water
        total_skipped_water += orient.n_skipped_water

        if z_edges is None:
            if len(orient.dipole_z) == 0:
                frames_read += 1
                continue
            z_edges = _make_edges_from_first_frame(orient.dipole_z, dz, zlo, zhi)
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            n_bins = len(z_centers)

            n_water = np.zeros(n_bins, dtype=np.float64)
            n_oh = np.zeros(n_bins, dtype=np.float64)
            sum_dipole_cos = np.zeros(n_bins, dtype=np.float64)
            sum_dipole_p2 = np.zeros(n_bins, dtype=np.float64)
            sum_dipole_theta = np.zeros(n_bins, dtype=np.float64)
            sum_oh_cos = np.zeros(n_bins, dtype=np.float64)
            sum_oh_p2 = np.zeros(n_bins, dtype=np.float64)
            sum_oh_theta = np.zeros(n_bins, dtype=np.float64)
            oh_angle_counts = np.zeros((n_bins, theta_bins), dtype=np.float64)

        dipole_p2 = 0.5 * (3.0 * orient.dipole_cos**2 - 1.0)
        dipole_theta = np.degrees(np.arccos(np.clip(orient.dipole_cos, -1.0, 1.0)))
        oh_p2 = 0.5 * (3.0 * orient.oh_cos**2 - 1.0)
        oh_theta = np.degrees(np.arccos(np.clip(orient.oh_cos, -1.0, 1.0)))

        _add_binned_samples(
            z_edges, orient.dipole_z, orient.dipole_cos, n_water, sum_dipole_cos
        )
        _add_binned_samples(
            z_edges, orient.dipole_z, dipole_p2, None, sum_dipole_p2
        )
        _add_binned_samples(
            z_edges, orient.dipole_z, dipole_theta, None, sum_dipole_theta
        )

        _add_binned_samples(z_edges, orient.oh_z, orient.oh_cos, n_oh, sum_oh_cos)
        _add_binned_samples(z_edges, orient.oh_z, oh_p2, None, sum_oh_p2)
        _add_binned_samples(z_edges, orient.oh_z, oh_theta, None, sum_oh_theta)

        if len(orient.oh_z) > 0:
            hist2d, _, _ = np.histogram2d(
                orient.oh_z,
                oh_theta,
                bins=[z_edges, theta_edges],
            )
            oh_angle_counts += hist2d

        frames_read += 1
        if verbose and frames_read % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    ... {frames_read} frames processed ({elapsed:.1f} s)")

    if z_edges is None or z_centers is None:
        raise SystemExit(
            "ERROR: no water molecules could be paired. Check --water-o-types, "
            "--water-h-types, --oh-cutoff, and frame selection."
        )

    assert n_water is not None
    assert n_oh is not None
    assert sum_dipole_cos is not None
    assert sum_dipole_p2 is not None
    assert sum_dipole_theta is not None
    assert sum_oh_cos is not None
    assert sum_oh_p2 is not None
    assert sum_oh_theta is not None
    assert oh_angle_counts is not None

    p1_dipole = _profile_average(sum_dipole_cos, n_water)
    p2_dipole = _profile_average(sum_dipole_p2, n_water)
    theta_dipole_mean = _profile_average(sum_dipole_theta, n_water)
    p1_oh = _profile_average(sum_oh_cos, n_oh)
    p2_oh = _profile_average(sum_oh_p2, n_oh)
    theta_oh_mean = _profile_average(sum_oh_theta, n_oh)

    row_sums = oh_angle_counts.sum(axis=1, keepdims=True)
    oh_angle_prob_density = np.full_like(oh_angle_counts, np.nan, dtype=np.float64)
    np.divide(
        oh_angle_counts,
        row_sums * dtheta,
        out=oh_angle_prob_density,
        where=row_sums > 0,
    )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    Total frames: {frames_read} ({elapsed:.1f} s)")
        print(f"    Water O seen: {total_water_o}")
        print(f"    Waters paired: {total_paired_water}")
        if total_skipped_water:
            print(f"    Waters skipped by pairing/cutoff: {total_skipped_water}")

    meta: Dict[str, object] = {
        "traj_path": str(traj_path),
        "frames": frames_read,
        "skip": skip,
        "stride": stride,
        "end": end,
        "dz": dz,
        "theta_bins": theta_bins,
        "theta_bin_width_deg": dtheta,
        "oh_cutoff": oh_cutoff,
        "water_o_types": sorted(water_o_types),
        "water_h_types": sorted(water_h_types),
        "total_water_o": total_water_o,
        "total_paired_water": total_paired_water,
        "total_skipped_water": total_skipped_water,
    }

    return OrientationResult(
        z_edges=z_edges,
        z_centers=z_centers,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        n_water=n_water,
        n_oh=n_oh,
        p1_dipole=p1_dipole,
        p2_dipole=p2_dipole,
        theta_dipole_mean=theta_dipole_mean,
        p1_oh=p1_oh,
        p2_oh=p2_oh,
        theta_oh_mean=theta_oh_mean,
        oh_angle_counts=oh_angle_counts,
        oh_angle_prob_density=oh_angle_prob_density,
        meta=meta,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt_float(value: float) -> str:
    """Compact CSV float formatting with NaN preserved."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.8f}"


def write_profile_csv(result: OrientationResult, out_path: Path) -> None:
    """Write z-resolved P1/P2 profiles."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# water_orientation.py output\n")
        f.write("# dipole vector = O -> midpoint(H1,H2)\n")
        f.write("# OH vector = O -> H\n")
        f.write("# theta is measured against +z; z binning uses water O z position\n")
        f.write(
            f"# frames={result.meta['frames']} skip={result.meta['skip']} "
            f"stride={result.meta['stride']} dz={result.meta['dz']} Ang "
            f"oh_cutoff={result.meta['oh_cutoff']} Ang\n"
        )
        f.write(
            "# P1=<cos(theta)>; P2=<0.5*(3*cos(theta)^2-1)>; "
            "empty bins are nan\n"
        )
        f.write(
            "z_ang,n_water,n_oh,P1_dipole,P2_dipole,"
            "theta_dipole_mean_deg,P1_oh,P2_oh,theta_oh_mean_deg\n"
        )
        for i, z in enumerate(result.z_centers):
            row = [
                f"{z:.4f}",
                str(int(result.n_water[i])),
                str(int(result.n_oh[i])),
                _fmt_float(result.p1_dipole[i]),
                _fmt_float(result.p2_dipole[i]),
                _fmt_float(result.theta_dipole_mean[i]),
                _fmt_float(result.p1_oh[i]),
                _fmt_float(result.p2_oh[i]),
                _fmt_float(result.theta_oh_mean[i]),
            ]
            f.write(",".join(row) + "\n")
    print(f"  Saved: {out_path}")


def write_angle_csv(result: OrientationResult, out_path: Path) -> None:
    """Write long-format O-H angle distribution P(theta | z)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# water_orientation.py O-H angle distribution\n")
        f.write("# prob_density_deg^-1 is normalized within each z bin\n")
        f.write("z_ang,theta_deg,count,prob_density_deg^-1\n")
        for iz, z in enumerate(result.z_centers):
            for itheta, theta in enumerate(result.theta_centers):
                count = result.oh_angle_counts[iz, itheta]
                prob = result.oh_angle_prob_density[iz, itheta]
                f.write(
                    f"{z:.4f},{theta:.4f},{int(count)},"
                    f"{_fmt_float(prob)}\n"
                )
    print(f"  Saved: {out_path}")


def plot_orientation(result: OrientationResult, out_path: Path, show_title: bool = False) -> None:
    """Plot dipole/O-H P1/P2 profiles and the O-H angle heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    colors, fontsize, labelsize, linewidth, figsize = _load_plot_style()
    fig_width, fig_height = figsize

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(fig_width * 2.45, fig_height * 2.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18])
    ax_dip = fig.add_subplot(gs[0, 0])
    ax_oh = fig.add_subplot(gs[0, 1], sharex=ax_dip)
    ax_heat = fig.add_subplot(gs[1, :], sharex=ax_dip)

    ax_dip.axhline(0.0, color="0.7", lw=max(linewidth, 0.5), ls="--")
    ax_dip.plot(result.z_centers, result.p1_dipole,
                color=colors[1], lw=linewidth * 2, label="P1")
    ax_dip.plot(result.z_centers, result.p2_dipole,
                color=colors[2], lw=linewidth * 2, label="P2")
    ax_dip.set_ylabel("Dipole order", fontsize=labelsize)
    if show_title:
        ax_dip.set_title("Water dipole", fontsize=fontsize)
    ax_dip.legend(fontsize=fontsize - 1, frameon=False)
    ax_dip.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_dip.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax_oh.axhline(0.0, color="0.7", lw=max(linewidth, 0.5), ls="--")
    ax_oh.plot(result.z_centers, result.p1_oh,
               color=colors[3], lw=linewidth * 2, label="P1")
    ax_oh.plot(result.z_centers, result.p2_oh,
               color=colors[4], lw=linewidth * 2, label="P2")
    ax_oh.set_ylabel("O-H order", fontsize=labelsize)
    if show_title:
        ax_oh.set_title("O-H bonds", fontsize=fontsize)
    ax_oh.legend(fontsize=fontsize - 1, frameon=False)
    ax_oh.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_oh.yaxis.set_minor_locator(AutoMinorLocator(2))

    mesh = ax_heat.pcolormesh(
        result.z_edges,
        result.theta_edges,
        result.oh_angle_prob_density.T,
        shading="auto",
        cmap="viridis",
    )
    ax_heat.set_xlabel("z (Å)", fontsize=labelsize)
    ax_heat.set_ylabel(r"O-H $\theta$ (deg)", fontsize=labelsize)
    ax_heat.set_ylim(0, 180)
    ax_heat.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_heat.yaxis.set_minor_locator(AutoMinorLocator(2))
    cbar = fig.colorbar(mesh, ax=ax_heat, pad=0.015)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    cbar.set_label(r"P($\theta$ | z) (deg$^{-1}$)", fontsize=fontsize - 1)

    ax_dip.text(
        0.99,
        0.97,
        f"frames={result.meta['frames']}  dz={result.meta['dz']} Å",
        transform=ax_dip.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize - 2,
        color="0.4",
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  QM/MM directory sweep
# ═══════════════════════════════════════════════════════════════════════════════


def find_mm_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Find mm_N directories sorted by step index."""
    mm_dirs = []
    for path in run_dir.iterdir():
        if path.is_dir() and re.match(r"^mm_(\d+)$", path.name):
            mm_dirs.append((int(path.name.split("_", 1)[1]), path))
    return sorted(mm_dirs, key=lambda x: x[0])


def find_orientation_traj_in_mm_dir(mm_dir: Path) -> Optional[Path]:
    """Find the trajectory to use inside one mm_N directory."""
    patterns = ("*.wrapped.traj", "*.emd.lammpstrj", "*.lammpstrj", "*.traj", "*.dump")
    for pattern in patterns:
        matches = sorted(mm_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def run_single_orientation(
    traj_path: Path,
    args: argparse.Namespace,
    outdir: Optional[Path] = None,
    title_prefix: Optional[str] = None,
) -> bool:
    """Compute and write water orientation outputs for one trajectory."""
    if not traj_path.exists():
        print(f"ERROR: file not found: {traj_path}", file=sys.stderr)
        return False

    lmp_in = args.lammps_input or _find_lammps_input(traj_path)
    water_o, water_h, type_source = resolve_water_types(
        args.water_o_types,
        args.water_h_types,
        lmp_in,
    )

    out_dir = outdir or args.outdir or traj_path.parent
    prefix = args.prefix
    profile_csv = out_dir / f"{prefix}_rawdata.csv"
    angle_csv = out_dir / f"{prefix}_oh_angle_distribution.csv"
    plot_path = out_dir / f"{prefix}.png"

    print(f"\n{'=' * 60}")
    print("  water_orientation.py — Water orientational order")
    print(f"{'=' * 60}")
    if title_prefix:
        print(f"  {title_prefix}")
    print(f"  Trajectory : {traj_path}")
    print(f"  Water O types: {sorted(water_o)}")
    print(f"  Water H types: {sorted(water_h)}")
    print(f"  Type source : {type_source}")
    print(
        f"  skip={args.skip} stride={args.stride} dz={args.dz} Å "
        f"OH cutoff={args.oh_cutoff} Å"
    )

    result = compute_water_orientation(
        traj_path=traj_path,
        water_o_types=water_o,
        water_h_types=water_h,
        dz=args.dz,
        skip=args.skip,
        stride=args.stride,
        end=args.end,
        zlo=args.zlo,
        zhi=args.zhi,
        theta_bins=args.theta_bins,
        oh_cutoff=args.oh_cutoff,
        allow_symbol_fallback=args.allow_symbol_fallback,
        verbose=True,
    )

    write_profile_csv(result, profile_csv)
    write_angle_csv(result, angle_csv)
    if not args.no_plot:
        plot_orientation(result, plot_path, show_title=args.plot_title)
    return True


def run_qmmm_orientation(args: argparse.Namespace) -> None:
    """Run orientation analysis in every mm_N directory under a QM/MM run."""
    run_dir = args.qmmm_mm.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"ERROR: not a directory: {run_dir}")

    mm_dirs = find_mm_dirs(run_dir)
    if not mm_dirs:
        raise SystemExit(f"ERROR: no mm_N directories found in {run_dir}")

    print(f"QM/MM water orientation sweep: {run_dir}")
    print(f"mm_N directories found: {len(mm_dirs)}")

    processed = 0
    skipped = 0
    for step, mm_dir in mm_dirs:
        traj = find_orientation_traj_in_mm_dir(mm_dir)
        if traj is None:
            print(f"mm_{step}: no trajectory found, skipping")
            skipped += 1
            continue
        ok = run_single_orientation(
            traj,
            args,
            outdir=mm_dir,
            title_prefix=f"mm_{step}",
        )
        processed += int(ok)
        skipped += int(not ok)

    print(f"\nDone. Processed {processed} mm_N directories, skipped {skipped}.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute water dipole P1/P2 and O-H orientation distributions vs. z "
            "from .lammpstrj/.dump or .traj files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/water_orientation.py run/ces2.emd.lammpstrj --skip 100 --stride 5\n"
            "  python tools/water_orientation.py run/ces2.emd.wrapped.traj --dz 0.2\n"
            "  python tools/water_orientation.py run/ces2.emd.lammpstrj "
            "--water-o-types 2 --water-h-types 1\n"
            "  python tools/water_orientation.py --qmmm-mm run_dir --skip 100 --stride 5\n"
        ),
    )
    p.add_argument("traj", type=Path, nargs="?", default=None,
                   help="Trajectory file (.lammpstrj/.dump or .traj)")
    p.add_argument("--qmmm-mm", type=Path, default=None, metavar="RUN_DIR",
                   help="Process every mm_N directory in a QM/MM run directory.")
    p.add_argument("--lammps-input", type=Path, default=None,
                   help="LAMMPS input file for auto-detecting water atom types.")
    p.add_argument("--water-o-types", nargs="+", type=int, default=None,
                   help="LAMMPS type IDs for water oxygen atoms (default: auto, then 2).")
    p.add_argument("--water-h-types", nargs="+", type=int, default=None,
                   help="LAMMPS type IDs for water hydrogen atoms (default: auto, then 1).")
    p.add_argument("--oh-cutoff", type=float, default=1.35,
                   help="Maximum O-H distance for water pairing in Å (default: 1.35).")
    p.add_argument("--dz", type=float, default=0.1,
                   help="z-bin width in Å (default: 0.1).")
    p.add_argument("--zlo", type=float, default=None,
                   help="Lower z-limit for binning in Å (default: first-frame water range).")
    p.add_argument("--zhi", type=float, default=None,
                   help="Upper z-limit for binning in Å (default: first-frame water range).")
    p.add_argument("--theta-bins", type=int, default=90,
                   help="Number of O-H angle bins from 0 to 180 degrees (default: 90).")
    p.add_argument("--skip", type=int, default=0,
                   help="Discard the first N frames (default: 0).")
    p.add_argument("--stride", type=int, default=1,
                   help="After skipping, use every N-th frame (default: 1).")
    p.add_argument("--end", type=int, default=None,
                   help="Last frame index, exclusive, before stride selection (default: end).")
    p.add_argument("--allow-symbol-fallback", action="store_true",
                   help=("For .traj without lammps_type metadata, treat all H as type 1 "
                         "and all O as type 2. Best for water-only trajectories."))
    p.add_argument("--plot-title", action="store_true",
                   help="Show subplot titles in generated figures (default: off).")
    p.add_argument("--no-plot", action="store_true",
                   help="Write CSV files only.")
    p.add_argument("-o", "--outdir", type=Path, default=None,
                   help="Output directory (default: same as trajectory).")
    p.add_argument("--prefix", type=str, default="water_orientation",
                   help="Output file prefix (default: water_orientation).")
    args = p.parse_args()

    if args.traj is not None and args.qmmm_mm is not None:
        p.error("provide either a trajectory file or --qmmm-mm, not both")
    if args.traj is None and args.qmmm_mm is None:
        p.error("provide a trajectory file or --qmmm-mm RUN_DIR")
    if args.outdir is not None and args.qmmm_mm is not None:
        p.error("--outdir is not supported with --qmmm-mm")
    if args.dz <= 0:
        p.error("--dz must be > 0")
    if args.oh_cutoff <= 0:
        p.error("--oh-cutoff must be > 0")
    if args.theta_bins < 1:
        p.error("--theta-bins must be >= 1")
    if args.skip < 0:
        p.error("--skip must be >= 0")
    if args.stride < 1:
        p.error("--stride must be >= 1")
    if args.end is not None and args.end <= args.skip:
        p.error("--end must be greater than --skip")

    return args


def main() -> None:
    args = parse_args()
    if args.qmmm_mm is not None:
        run_qmmm_orientation(args)
        return

    traj_path = args.traj.resolve()
    run_single_orientation(traj_path, args)


if __name__ == "__main__":
    main()
