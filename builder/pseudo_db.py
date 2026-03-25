"""
pseudo_db.py — Built-in pseudopotential databases for QE pw.x.

Currently ships the SSSP (Standard Solid-State Pseudopotentials) PBE set
as collected from the SSSP library (https://www.materialscloud.org/discover/sssp).
All filenames match the files distributed with the SSSP efficiency/precision sets.

Usage in qe_writer:
    from .pseudo_db import get_pseudo_file
    upf = get_pseudo_file("Ir", pseudo_set="sssp")  # → "Ir_pbe_v1.2.uspp.F.UPF"
"""
from __future__ import annotations
from typing import Optional

# ---------------------------------------------------------------------------
# SSSP PBE set — element symbol (capitalised) → UPF filename
# Source: SSSP library, PBE functional
# ---------------------------------------------------------------------------
_SSSP_PBE: dict[str, str] = {
    "Ac": "Ac.us.z_11.ld1.psl.v1.0.0-high.upf",
    "Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Am": "Am.paw.z_17.ld1.uni-marburg.v0.upf",
    "Ar": "Ar_ONCV_PBE-1.1.oncvpsp.upf",
    "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
    "At": "At.us.z_17.ld1.psl.v1.0.0-high.upf",
    "Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",
    "B":  "b_pbe_v1.4.uspp.F.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Be": "be_pbe_v1.4.uspp.F.UPF",
    "Bi": "Bi_pbe_v1.uspp.F.UPF",
    "Bk": "Bk.paw.z_19.ld1.uni-marburg.v0.upf",
    "Br": "br_pbe_v1.4.uspp.F.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca_pbe_v1.uspp.F.UPF",
    "Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF",
    "Ce": "Ce.paw.z_12.atompaw.wentzcovitch.v1.2.upf",
    "Cf": "Cf.paw.z_20.ld1.uni-marburg.v0.upf",
    "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    "Cm": "Cm.paw.z_18.ld1.uni-marburg.v0.upf",
    "Co": "Co_pbe_v1.2.uspp.F.UPF",
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "Cs": "Cs_pbe_v1.uspp.F.UPF",
    "Cu": "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf",
    "Dy": "Dy.paw.z_20.atompaw.wentzcovitch.v1.2.upf",
    "Er": "Er.paw.z_22.atompaw.wentzcovitch.v1.2.upf",
    "Es": "Es.paw.z_21.ld1.uni-marburg.v0.upf",
    "Eu": "Eu.paw.z_17.atompaw.wentzcovitch.v1.2.upf",
    "F":  "f_pbe_v1.4.uspp.F.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Fm": "Fm.paw.z_22.ld1.uni-marburg.v0.upf",
    "Fr": "Fr.paw.z_19.ld1.psl.v1.0.0-high.upf",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Gd": "Gd.paw.z_18.atompaw.wentzcovitch.v1.2.upf",
    "Ge": "ge_pbe_v1.4.uspp.F.UPF",
    "H":  "H.pbe-rrkjus_psl.1.0.0.UPF",
    "He": "He_ONCV_PBE-1.0.oncvpsp.upf",
    "Hf": "Hf-sp.oncvpsp.upf",
    "Hg": "Hg_ONCV_PBE-1.0.oncvpsp.upf",
    "Ho": "Ho.paw.z_21.atompaw.wentzcovitch.v1.2.upf",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF",
    "In": "In.pbe-dn-rrkjus_psl.0.2.2.UPF",
    "Ir": "Ir_pbe_v1.2.uspp.F.UPF",
    "K":  "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Kr": "Kr_ONCV_PBE-1.0.oncvpsp.upf",
    "La": "La.paw.z_11.atompaw.wentzcovitch.v1.2.upf",
    "Li": "li_pbe_v1.4.uspp.F.UPF",
    "Lr": "Lr.paw.z_25.ld1.uni-marburg.v0.upf",
    "Lu": "Lu.paw.z_25.atompaw.wentzcovitch.v1.2.upf",
    "Md": "Md.paw.z_23.ld1.uni-marburg.v0.upf",
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "Mn": "mn_pbe_v1.5.uspp.F.UPF",
    "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
    "N":  "N.pbe-n-radius_5.UPF",
    "Na": "na_pbe_v1.5.uspp.F.UPF",
    "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
    "Nd": "Nd.paw.z_14.atompaw.wentzcovitch.v1.2.upf",
    "Ne": "Ne_ONCV_PBE-1.0.oncvpsp.upf",
    "Ni": "ni_pbe_v1.4.uspp.F.UPF",
    "No": "No.paw.z_24.ld1.uni-marburg.v0.upf",
    "Np": "Np.paw.z_15.ld1.uni-marburg.v0.upf",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
    "Os": "Os_pbe_v1.2.uspp.F.UPF",
    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Pa": "Pa.paw.z_13.ld1.uni-marburg.v0.upf",
    "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
    "Pd": "Pd_ONCV_PBE-1.0.oncvpsp.upf",
    "Pm": "Pm.paw.z_15.atompaw.wentzcovitch.v1.2.upf",
    "Po": "Po.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "Pr": "Pr.paw.z_13.atompaw.wentzcovitch.v1.2.upf",
    "Pt": "pt_pbe_v1.4.uspp.F.UPF",
    "Pu": "Pu.paw.z_16.ld1.uni-marburg.v0.upf",
    "Ra": "Ra.paw.z_20.ld1.psl.v1.0.0-high.upf",
    "Rb": "Rb_ONCV_PBE-1.0.oncvpsp.upf",
    "Re": "Re_pbe_v1.2.uspp.F.UPF",
    "Rh": "Rh_ONCV_PBE-1.0.oncvpsp.upf",
    "Rn": "Rn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ru": "Ru_ONCV_PBE-1.0.oncvpsp.upf",
    "S":  "s_pbe_v1.4.uspp.F.UPF",
    "Sb": "sb_pbe_v1.4.uspp.F.UPF",
    "Sc": "Sc_ONCV_PBE-1.0.oncvpsp.upf",
    "Se": "Se_pbe_v1.uspp.F.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Sm": "Sm.paw.z_16.atompaw.wentzcovitch.v1.2.upf",
    "Sn": "Sn_pbe_v1.uspp.F.UPF",
    "Sr": "Sr_pbe_v1.uspp.F.UPF",
    "Ta": "Ta_pbe_v1.uspp.F.UPF",
    "Tb": "Tb.paw.z_19.atompaw.wentzcovitch.v1.2.upf",
    "Tc": "Tc_ONCV_PBE-1.0.oncvpsp.upf",
    "Te": "Te_pbe_v1.uspp.F.UPF",
    "Th": "Th.paw.z_12.ld1.uni-marburg.v0.upf",
    "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "Tl": "Tl_pbe_v1.2.uspp.F.UPF",
    "Tm": "Tm.paw.z_23.atompaw.wentzcovitch.v1.2.upf",
    "U":  "U.paw.z_14.ld1.uni-marburg.v0.upf",
    "V":  "v_pbe_v1.4.uspp.F.UPF",
    "W":  "W_pbe_v1.2.uspp.F.UPF",
    "Xe": "Xe_ONCV_PBE-1.1.oncvpsp.upf",
    "Y":  "Y_pbe_v1.uspp.F.UPF",
    "Yb": "Yb.paw.z_24.atompaw.wentzcovitch.v1.2.upf",
    "Zn": "Zn_pbe_v1.uspp.F.UPF",
    "Zr": "Zr_pbe_v1.uspp.F.UPF",
}

# Registry of named sets
_PSEUDO_SETS: dict[str, dict[str, str]] = {
    "sssp": _SSSP_PBE,
}


def get_pseudo_file(element: str, pseudo_set: str = "sssp") -> Optional[str]:
    """Return the UPF filename for *element* from the named built-in set.

    Returns None if the element or set is not found.
    """
    db = _PSEUDO_SETS.get(pseudo_set.lower())
    if db is None:
        return None
    # Try exact match first, then title-case fallback
    return db.get(element) or db.get(element.capitalize())


def resolve_pseudopotentials(
    elements: list[str],
    pseudo_set: str = "sssp",
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build element → UPF filename mapping.

    Priority: overrides (from config qe.pseudopotentials) > built-in pseudo_set.
    Elements not found in either source get a placeholder warning string.
    """
    overrides = overrides or {}
    result: dict[str, str] = {}
    missing: list[str] = []

    for el in elements:
        if el in overrides:
            result[el] = overrides[el]
        else:
            upf = get_pseudo_file(el, pseudo_set)
            if upf:
                result[el] = upf
            else:
                missing.append(el)
                result[el] = f"{el}.UPF   # !!! not found in {pseudo_set} — specify manually"

    if missing:
        print(f"[pseudo_db] WARNING: no pseudopotential found for {missing} "
              f"in set '{pseudo_set}'. Add entries to qe.pseudopotentials in config.")

    return result
