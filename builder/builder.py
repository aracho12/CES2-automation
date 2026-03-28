from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from ase import Atoms
from ase.io import read, write

from .species import Species
from .lammps_writer import write_data_file_reference_style, DataFileFormat

ATOMIC_MASS = {
    "H": 1.008,
    "Li": 6.941,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.0983,
    "Ca": 40.078,
    "Br": 79.904,
    "Rb": 85.468,
    "I": 126.90447,
    "Cs": 132.905,
    "Ir": 192.217,
}

def write_species_xyz(path: Path, sp: Species) -> None:
    lines = [str(sp.natoms), sp.name]
    for a in sp.atoms:
        x,y,z = a.xyz
        lines.append(f"{a.element}  {x:.6f}  {y:.6f}  {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def make_type_registry(species_order: List[str], species_db: Dict[str, Species], slab_elements: List[str]) -> Tuple[Dict[str,int], Dict[int,str]]:
    """
    Create stable type ids:
      - iterate species_order, then within species atoms in order, assign by atom.type_label
      - then slab elements (alphabetical) as type_label = element symbol
    Returns:
      type_id_by_label, label_by_type_id
    """
    type_id_by_label: Dict[str,int] = {}
    label_by_type_id: Dict[int,str] = {}
    next_id = 1

    def add_label(lbl: str):
        nonlocal next_id
        if lbl not in type_id_by_label:
            type_id_by_label[lbl] = next_id
            label_by_type_id[next_id] = lbl
            next_id += 1

    # MM types: iterate species in the given order, but within each species
    # we optionally prioritize hydrogen-like labels first, then oxygen-like,
    # then everything else. This ensures, e.g., for TIP3P water that H is
    # type 1, O is type 2, and any additional MM atom types (ions, organics)
    # follow as 3, 4, ... while QM slab elements are appended afterwards.
    element_priority = {
        "H": 0,
        "O": 1,
    }

    for sid in species_order:
        sp = species_db[sid]
        # Stable sort: by (priority, original_index)
        indexed_atoms = list(enumerate(sp.atoms))
        indexed_atoms.sort(key=lambda t: (element_priority.get(t[1].element, 10), t[0]))
        for _, a in indexed_atoms:
            add_label(a.type_label)

    for el in sorted(set(slab_elements)):
        add_label(el)

    return type_id_by_label, label_by_type_id

def assign_mm_types_charges_by_order(mm_atoms: Atoms,
                                    species_plan: List[Tuple[str,int]],
                                    species_db: Dict[str, Species],
                                    type_id_by_label: Dict[str,int]) -> Tuple[List[int], List[float], List[Tuple[str, int, int]]]:
    """
    species_plan: [(species_id, count)] in the exact PACKMOL structure order
    Returns:
      types, charges for mm_atoms (length n_mm)
      mm_slices: list of (species_id, start_idx0, end_idx0_exclusive) for each molecule instance
    Assumes PACKMOL output preserves the order of structures as listed (practical assumption).
    """
    types: List[int] = []
    charges: List[float] = []
    slices: List[Tuple[str,int,int]] = []

    idx = 0
    for sid, count in species_plan:
        sp = species_db[sid]
        for _ in range(count):
            start = idx
            for a in sp.atoms:
                types.append(type_id_by_label[a.type_label])
                charges.append(float(a.charge))
                idx += 1
            end = idx
            slices.append((sid, start, end))

    if idx != len(mm_atoms):
        raise ValueError(f"MM parsing mismatch: expected {idx} atoms from species_plan but mm_atoms has {len(mm_atoms)}. "
                         f"Check packmol ordering or species natoms/counts.")
    return types, charges, slices

def build_mm_connectivity(mm_slices: List[Tuple[str,int,int]],
                          species_db: Dict[str, Species]) -> Tuple[List[Tuple[int,int,int]], List[Tuple[int,int,int,int]],
                                                                   Dict[int,Tuple[float,float]], Dict[int,Tuple[float,float]]]:
    """
    Generate global bonds/angles (1-based indices) for MM atoms by shifting species-local connectivity.
    Also aggregates bond/angle coeffs (max of provided; later you can override by FF system).
    """
    bonds: List[Tuple[int,int,int]] = []
    angles: List[Tuple[int,int,int,int]] = []
    bond_coeffs: Dict[int,Tuple[float,float]] = {}
    angle_coeffs: Dict[int,Tuple[float,float]] = {}

    for sid, start0, end0 in mm_slices:
        sp = species_db[sid]
        offset = start0  # 0-based
        # coeffs
        for bt, (k,r0) in sp.bond_coeffs.items():
            bond_coeffs.setdefault(bt, (k,r0))
        for at, (k,th) in sp.angle_coeffs.items():
            angle_coeffs.setdefault(at, (k,th))
        # connectivity
        for bt,i,j in sp.bonds:
            gi = offset + (i-1) + 1  # to 1-based global
            gj = offset + (j-1) + 1
            bonds.append((bt, gi, gj))
        for at,i,j,k in sp.angles:
            gi = offset + (i-1) + 1
            gj = offset + (j-1) + 1
            gk = offset + (k-1) + 1
            angles.append((at, gi, gj, gk))

    return bonds, angles, bond_coeffs, angle_coeffs

def apply_slab_charging(slab: Atoms, q_electrode: float) -> List[float]:
    n = len(slab)
    dq = float(q_electrode) / float(n) if n else 0.0
    return [dq]*n

def masses_by_type_from_labels(label_by_type_id: Dict[int,str],
                               type_id_by_label: Dict[str,int],
                               species_db: Dict[str,Species],
                               extra_label_to_element: Optional[Dict[str,str]] = None) -> Dict[int,float]:
    """
    Resolve mass by type_label:
      - if type_label is element symbol, use atomic mass
      - otherwise try to infer from first atom that uses that type_label in species_db
      - extra_label_to_element: optional override map for custom QM type_labels
        (e.g. {"O_ads": "O", "H_ads": "H"} from slab.type_label_overrides)
    """
    # build reverse map: type_label -> element
    label_to_element: Dict[str,str] = {}
    for sp in species_db.values():
        for a in sp.atoms:
            label_to_element.setdefault(a.type_label, a.element)
    # apply caller-supplied overrides (custom QM type_labels not in species_db)
    if extra_label_to_element:
        label_to_element.update(extra_label_to_element)

    out: Dict[int,float] = {}
    for tid, lbl in label_by_type_id.items():
        el = lbl if lbl in ATOMIC_MASS else label_to_element.get(lbl, None)
        if el is None:
            raise ValueError(f"Cannot infer atomic mass for type_label '{lbl}'. "
                             f"Add it to species_db or to slab.type_label_overrides in config.")
        out[tid] = float(ATOMIC_MASS.get(el, 0.0))
    return out
