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
    sc = atoms.repeat(rep)
    sc.wrap()
    return sc
