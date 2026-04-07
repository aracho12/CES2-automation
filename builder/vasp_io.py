from __future__ import annotations
from typing import Tuple
import numpy as np
from ase import Atoms
from ase.io import read, write

def is_orthogonal_cell(cell: np.ndarray, tol: float = 1e-8) -> bool:
    a, b, c = cell[0], cell[1], cell[2]
    return (abs(np.dot(a,b)) < tol) and (abs(np.dot(a,c)) < tol) and (abs(np.dot(b,c)) < tol)

def read_vasp(path: str) -> Atoms:
    return read(path)

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
