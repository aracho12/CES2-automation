from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ase import Atoms

@dataclass
class BoxMeta:
    Lx: float
    Ly: float
    Lz: float
    z_top_slab: float
    z_el_lo: float
    z_el_hi: float
    vacuum_z: float = 20.0      # Å vacuum gap above water for boundary p p f
    z_buffer_lo: float = 1.0    # Å buffer below slab (zlo = -z_buffer_lo)
    # Full simulation-box z extent (set after data.file is built in main.py).
    # Needed by wall-clamp logic that maps QE emaxpos (fractional, 0..1) into
    # absolute LAMMPS z so the upper SOLVENT wall stays below the dipole-correction
    # region of the cube.
    box_z_total: Optional[float] = None
    box_zlo:     Optional[float] = None
    box_zhi:     Optional[float] = None

def compute_box_meta(slab: Atoms, z_gap: float, thickness: float,
                     z_margin_top: float,
                     vacuum_z: float = 20.0,
                     z_buffer_lo: float = 1.0) -> BoxMeta:
    cell = slab.cell.array
    Lx, Ly, Lz = float(cell[0,0]), float(cell[1,1]), float(cell[2,2])
    z_top = float(np.max(slab.positions[:,2]))
    z_el_lo = z_top + float(z_gap)
    #z_el_hi = min(z_el_lo + float(thickness), Lz - float(z_margin_top))
    z_el_hi = z_top + float(thickness)
    if z_el_hi <= z_el_lo + 1.0:
        raise ValueError(f"Electrolyte region too thin: z_el_lo={z_el_lo:.3f}, z_el_hi={z_el_hi:.3f}, Lz={Lz:.3f}")
    return BoxMeta(Lx=Lx, Ly=Ly, Lz=Lz, z_top_slab=z_top,
                   z_el_lo=z_el_lo, z_el_hi=z_el_hi,
                   vacuum_z=vacuum_z, z_buffer_lo=z_buffer_lo)
