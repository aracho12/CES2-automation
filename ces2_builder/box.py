from __future__ import annotations
from dataclasses import dataclass
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

def compute_box_meta(slab: Atoms, z_gap: float, thickness: float, z_margin_top: float) -> BoxMeta:
    cell = slab.cell.array
    Lx, Ly, Lz = float(cell[0,0]), float(cell[1,1]), float(cell[2,2])
    z_top = float(np.max(slab.positions[:,2]))
    z_el_lo = z_top + float(z_gap)
    #z_el_hi = min(z_el_lo + float(thickness), Lz - float(z_margin_top))
    z_el_hi = z_top + float(thickness)
    if z_el_hi <= z_el_lo + 1.0:
        raise ValueError(f"Electrolyte region too thin: z_el_lo={z_el_lo:.3f}, z_el_hi={z_el_hi:.3f}, Lz={Lz:.3f}")
    return BoxMeta(Lx=Lx, Ly=Ly, Lz=Lz, z_top_slab=z_top, z_el_lo=z_el_lo, z_el_hi=z_el_hi)
