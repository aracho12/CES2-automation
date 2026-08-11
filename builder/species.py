from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml

@dataclass
class AtomDef:
    element: str
    type_label: str
    charge: Optional[float]
    xyz: Tuple[float, float, float]
    bjdisp: Optional[Dict[str, float]] = None  # {alpha_iso, C6, s} or None
    # Optional LJ alias: keep a distinct LAMMPS type_label while inheriting
    # epsilon/sigma from another label in the active water-model FF database.
    lj_type_label: Optional[str] = None

@dataclass
class Species:
    id: str
    name: str
    net_charge: float
    atoms: List[AtomDef]
    # connectivity in *molecule-local* 1-based indices
    bonds: List[Tuple[int, int, int]]   # (bond_type, i, j)
    angles: List[Tuple[int, int, int, int]]  # (angle_type, i, j, k)
    # optional: coeffs for reference-style blocks
    bond_coeffs: Dict[int, Tuple[float, float]]  # k, r0
    angle_coeffs: Dict[int, Tuple[float, float]] # k, theta0
    # Optional model tag for species whose geometry/charges depend on a
    # force-field set (currently hydroxide_*).
    forcefield_set: Optional[str] = None

    @property
    def natoms(self) -> int:
        return len(self.atoms)

def _as_float(x: Any) -> float:
    return float(x)

def load_species_yaml(path: str | Path) -> Species:
    p = Path(path)
    y = yaml.safe_load(p.read_text(encoding="utf-8"))

    sid = str(y["id"])
    name = str(y.get("name", sid))
    netq = float(y.get("net_charge", 0.0))
    forcefield_set_raw = y.get("forcefield_set")
    forcefield_set = str(forcefield_set_raw) if forcefield_set_raw is not None else None

    atoms_raw = y["atoms"]
    atoms: List[AtomDef] = []
    for a in atoms_raw:
        element = str(a["element"])
        type_label = str(a.get("type_label", element))
        lj_type_label_raw = a.get("lj_type_label", None)
        lj_type_label = str(lj_type_label_raw) if lj_type_label_raw is not None else None
        charge = a.get("charge", None)
        if charge is not None:
            charge = float(charge)
        x, yy, z = a["xyz"]
        # bjdisp: load if present and not null
        bjdisp_raw = a.get("bjdisp", None)
        bjdisp: Optional[Dict[str, float]] = None
        if isinstance(bjdisp_raw, dict):
            bjdisp = {k: float(v) for k, v in bjdisp_raw.items()}
        atoms.append(AtomDef(
            element=element,
            type_label=type_label,
            charge=charge,
            xyz=(float(x), float(yy), float(z)),
            bjdisp=bjdisp,
            lj_type_label=lj_type_label,
        ))

    # charge completion
    scheme = y.get("charge_scheme", "explicit_or_zero")
    if scheme == "uniform_from_net_charge":
        # distribute remaining charge (net - sum(explicit)) uniformly on atoms with charge None
        explicit_sum = sum(a.charge for a in atoms if a.charge is not None)
        missing = [i for i,a in enumerate(atoms) if a.charge is None]
        if missing:
            per = (netq - explicit_sum) / float(len(missing))
            for i in missing:
                atoms[i].charge = per
        else:
            # if all explicit, trust them (may not sum exactly)
            pass
    else:
        # set None -> 0.0
        for i,a in enumerate(atoms):
            if a.charge is None:
                atoms[i].charge = 0.0

    bonds = []
    for b in y.get("connectivity", {}).get("bonds", []):
        bt, i, j = b
        bonds.append((int(bt), int(i), int(j)))

    angles = []
    for a in y.get("connectivity", {}).get("angles", []):
        at, i, j, k = a
        angles.append((int(at), int(i), int(j), int(k)))

    bond_coeffs = {}
    for bt, vals in (y.get("coeffs", {}).get("bond_coeffs", {}) or {}).items():
        k, r0 = vals
        bond_coeffs[int(bt)] = (float(k), float(r0))

    angle_coeffs = {}
    for at, vals in (y.get("coeffs", {}).get("angle_coeffs", {}) or {}).items():
        k, th = vals
        angle_coeffs[int(at)] = (float(k), float(th))

    return Species(
        id=sid, name=name, net_charge=netq,
        atoms=atoms, bonds=bonds, angles=angles,
        bond_coeffs=bond_coeffs, angle_coeffs=angle_coeffs,
        forcefield_set=forcefield_set,
    )

def load_species_db(db_dir: str | Path) -> Dict[str, Species]:
    d = Path(db_dir)
    out: Dict[str, Species] = {}
    for p in sorted(d.glob("*.yaml")):
        sp = load_species_yaml(p)
        out[sp.id] = sp
    if not out:
        raise FileNotFoundError(f"No species YAML found in {d}")
    return out


def validate_hydroxide_species_models(
    counts: Dict[str, int],
    species_db: Dict[str, Species],
    selected_set: str,
) -> Optional[float]:
    """Validate active OH species and return their common YAML bond length."""
    mismatches: List[Tuple[str, str]] = []
    bond_lengths: set[float] = set()
    for sid, count in counts.items():
        if int(count) <= 0:
            continue
        sp = species_db[sid]
        if sp.forcefield_set is None or not sp.forcefield_set.startswith("hydroxide_"):
            continue
        if sp.forcefield_set != selected_set:
            mismatches.append((sid, sp.forcefield_set))
            continue
        for bond_type, i, j in sp.bonds:
            ai, aj = sp.atoms[i - 1], sp.atoms[j - 1]
            li = ai.lj_type_label or ai.type_label
            lj = aj.lj_type_label or aj.type_label
            if {li, lj} == {"O_oh", "H_oh"}:
                if bond_type not in sp.bond_coeffs:
                    raise KeyError(
                        f"Species '{sid}' O_oh-H_oh bond type {bond_type} has no bond_coeff"
                    )
                bond_lengths.add(float(sp.bond_coeffs[bond_type][1]))
    if mismatches:
        details = ", ".join(f"{sid} ({model})" for sid, model in mismatches)
        raise ValueError(
            f"ces2.hydroxide_model selected '{selected_set}', but active species "
            f"use a different model: {details}. Select matching species YAML IDs."
        )
    if len(bond_lengths) > 1:
        raise ValueError(
            f"Active hydroxide species have inconsistent bond lengths: {sorted(bond_lengths)}"
        )
    return next(iter(bond_lengths), None)
