from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

N_A = 6.02214076e23
M_H2O = 18.01528

@dataclass
class Counts:
    # species_id -> molecule count
    counts: Dict[str, int]
    V_L_approx: float
    Q_total: float

def water_volume_L(n_water: int, rho_g_ml: float) -> float:
    n_w = n_water / N_A
    mass_g = n_w * M_H2O
    return (mass_g / rho_g_ml) * 1e-3

def counts_from_salts(V_L: float, salts: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in salts:
        conc = float(s["concentration_M"])
        n_units = int(round(conc * V_L * N_A))
        stoich: Dict[str, int] = s["stoich"]  # species_id -> coefficient
        for sid, coeff in stoich.items():
            out[sid] = out.get(sid, 0) + int(coeff) * n_units
    return out

def compute_total_charge(counts: Dict[str,int], net_charge_by_species: Dict[str,float], q_electrode: float) -> float:
    q = float(q_electrode)
    for sid, n in counts.items():
        q += float(net_charge_by_species[sid]) * int(n)
    return q

def adjust_with_counterions(counts: Dict[str,int],
                            net_charge_by_species: Dict[str,float],
                            q_electrode: float,
                            q_target_total: float,
                            counterion_pool: List[str],
                            max_pair_search: int = 5000) -> Tuple[Dict[str,int], float]:
    """
    Try to satisfy Q_total == q_target_total by adding species from counterion_pool.
    Strategy:
      1) try single species that divides deltaQ
      2) try pair combination (a,b) small search
    """
    counts2 = dict(counts)
    q_current = compute_total_charge(counts2, net_charge_by_species, q_electrode)
    delta = float(q_target_total) - q_current
    if abs(delta) < 1e-8:
        return counts2, q_current

    # charges of pool
    pool = [(sid, float(net_charge_by_species[sid])) for sid in counterion_pool]
    # remove zero-charge entries
    pool = [(sid, z) for sid, z in pool if abs(z) > 1e-12]
    if not pool:
        raise ValueError("counterion_pool is empty or has only neutral species.")

    # 1) single species exact
    for sid, z in pool:
        n = delta / z
        if abs(n - round(n)) < 1e-10:
            nint = int(round(n))
            if nint < 0:
                continue
            counts2[sid] = counts2.get(sid, 0) + nint
            q_final = compute_total_charge(counts2, net_charge_by_species, q_electrode)
            if abs(q_final - q_target_total) < 1e-6:
                return counts2, q_final

    # 2) pair search: find n_a, n_b >= 0 such that n_a*z_a + n_b*z_b = delta
    # brute-force small n for first species
    for sid_a, z_a in pool:
        for sid_b, z_b in pool:
            if abs(z_b) < 1e-12:
                continue
            # sweep n_a
            for n_a in range(0, max_pair_search+1):
                rem = delta - n_a*z_a
                n_b = rem / z_b
                if abs(n_b - round(n_b)) < 1e-10:
                    n_bi = int(round(n_b))
                    if n_bi < 0:
                        continue
                    counts3 = dict(counts2)
                    counts3[sid_a] = counts3.get(sid_a, 0) + n_a
                    counts3[sid_b] = counts3.get(sid_b, 0) + n_bi
                    q_final = compute_total_charge(counts3, net_charge_by_species, q_electrode)
                    if abs(q_final - q_target_total) < 1e-6:
                        return counts3, q_final

    raise ValueError(
        f"Cannot satisfy total charge exactly with available counterions. "
        f"deltaQ={delta}. Consider using 1-valent counterions or integer electrode charge."
    )
