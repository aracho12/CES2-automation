from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ase import Atoms

@dataclass
class DataFileFormat:
    title: str = "data"

def write_data_file_reference_style(path: Path,
                                   atoms: Atoms,
                                   atom_types: List[int],
                                   charges: List[float],
                                   masses_by_type: Dict[int, float],
                                   bond_coeffs: Optional[Dict[int, Tuple[float, float]]],
                                   angle_coeffs: Optional[Dict[int, Tuple[float, float]]],
                                   bonds: List[Tuple[int,int,int]],
                                   angles: List[Tuple[int,int,int,int]],
                                   fmt: DataFileFormat = DataFileFormat(),
                                   vacuum_z: float = 20.0,
                                   z_buffer_lo: float = 1.0) -> None:
    cell = atoms.cell.array
    Lx, Ly, Lz = float(cell[0,0]), float(cell[1,1]), float(cell[2,2])

    n_atoms = len(atoms)
    n_atom_types = max(atom_types) if atom_types else 0
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_bond_types = max([bt for bt,_,_ in bonds]) if bonds else 0
    n_angle_types = max([at for at,_,_,_ in angles]) if angles else 0

    pos = atoms.positions

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{fmt.title}\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_atom_types} atom types\n")
        f.write(f"{n_bonds} bonds\n")
        f.write(f"{n_bond_types} bond types\n")
        f.write(f"{n_angles} angles\n")
        f.write(f"{n_angle_types} angle types\n\n")

        # boundary p p f: all atoms must be within [zlo, zhi] at read_data.
        # zlo = -z_buffer_lo: small buffer for slab atoms near z=0.
        # zhi = max(Lz, max_atom_z + 1.0) + vacuum_z: contains all atoms
        #       + vacuum gap so water cannot see slab bottom through PBC.
        import numpy as np
        z_coords = pos[:, 2] if len(pos) > 0 else np.array([0.0])
        z_hi = max(Lz, float(z_coords.max()) + 1.0) + vacuum_z
        z_lo = -z_buffer_lo
        f.write(f"0.0      {Lx:.4f}  xlo xhi\n")
        f.write(f"0.0      {Ly:.4f}  ylo yhi\n")
        f.write(f"{z_lo:.1f}     {z_hi:.4f}  zlo zhi\n\n")

        f.write("Masses\n\n")
        for t in range(1, n_atom_types+1):
            f.write(f"{t} {masses_by_type.get(t, 0.0):.6f}\n")
        f.write("\n")

        if bond_coeffs:
            f.write("Bond Coeffs\n\n")
            for bt in sorted(bond_coeffs.keys()):
                k, r0 = bond_coeffs[bt]
                f.write(f"{bt} {k} {r0}\n")
            f.write("\n")

        if angle_coeffs:
            f.write("Angle Coeffs\n\n")
            for at in sorted(angle_coeffs.keys()):
                k, th = angle_coeffs[at]
                f.write(f"{at} {k} {th}\n")
            f.write("\n")

        f.write("Atoms\n\n")
        for i in range(n_atoms):
            aid = i+1
            t = int(atom_types[i])
            q = float(charges[i])
            x,y,z = pos[i]
            f.write(f"{aid} 0 {t} {q} {x:.6f} {y:.6f} {z:.6f} 0 0 0\n")
        f.write("\n")

        if n_bonds:
            f.write("Bonds\n\n")
            for bi,(bt,i,j) in enumerate(bonds, start=1):
                f.write(f"{bi} {bt} {i} {j}\n")
            f.write("\n")

        if n_angles:
            f.write("Angles\n\n")
            for ai,(at,i,j,k) in enumerate(angles, start=1):
                f.write(f"{ai} {at} {i} {j} {k}\n")
            f.write("\n")
