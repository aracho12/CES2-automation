from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from ase import Atoms
from ase.io import read, write

def is_orthogonal_cell(cell: np.ndarray, tol: float = 1e-8) -> bool:
    a, b, c = cell[0], cell[1], cell[2]
    return (abs(np.dot(a,b)) < tol) and (abs(np.dot(a,c)) < tol) and (abs(np.dot(b,c)) < tol)


def _detect_format(path: str) -> Optional[str]:
    """Return an ASE format hint based on filename, or None for auto-detect."""
    name = Path(path).name.lower()
    if name.endswith((".in", ".pwi")) or name.startswith("pw."):
        return "espresso-in"
    if name.endswith((".out", ".pwo")) or name.endswith(".pw.out"):
        return "espresso-out"
    return None  # ASE auto-detects VASP CONTCAR/POSCAR by filename


def read_structure(path: str, fmt: Optional[str] = None) -> Atoms:
    """
    Read a slab structure file. Supports VASP (CONTCAR/POSCAR) and
    Quantum ESPRESSO (pw.x input/output) via ASE. The format is auto-
    detected from the filename; pass `fmt` to override.
    """
    if fmt is None:
        fmt = _detect_format(path)
    if fmt is None:
        return read(path)
    return read(path, format=fmt)


def read_vasp(path: str) -> Atoms:
    """Backwards-compatible alias for read_structure."""
    return read_structure(path)

def write_vasp(path: str, atoms: Atoms) -> None:
    write(path, atoms, vasp5=True, direct=True)

def write_xyz(path: str, atoms: Atoms) -> None:
    write(path, atoms)

def make_supercell(atoms: Atoms, rep: Tuple[int,int,int]) -> Atoms:
    # Wrap the primitive cell into [0, 1) fractional coordinates before tiling.
    # Without this, atoms that VASP placed just outside the -x/-y boundary
    # (e.g. frac = -0.0004) survive repeat() as the ix=0 copy with a negative
    # Cartesian coordinate.  sc.wrap() then folds that copy to the *top* of the
    # supercell (e.g. x ≈ 3·Lx), which lands 3 QE unit-cells outside the cell
    # and causes pw.x to crash.  Pre-wrapping ensures ix=0 copies stay near
    # x ≈ 0, well within the QE unit cell.
    atoms_copy = atoms.copy()
    atoms_copy.wrap()
    sc = atoms_copy.repeat(rep)
    sc.wrap()
    return sc
