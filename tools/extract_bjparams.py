#!/usr/bin/env python3
"""
extract_bjparams.py — Extract BJ dispersion parameters:
                      vdW forcefield (FISCS) from VASP OUTCAR
                      and DFTD3 C6 coefficients from d3.out

Usage:
  python extract_bjparams.py [OUTCAR]          (default: ./OUTCAR)

Output:
  fiscs.out            — full per-atom table with column header
  fiscs-average.out    — per-element averages (C6ts, ALPHAscs, C6 in kcal/mol·Å^6)
  atoms_fiscs.json     — ASE Atoms JSON with ALPHAscs as initial_charges
  atoms_c6.json        — ASE Atoms JSON with DFTD3 C6 as initial_charges
  bjparams_zsorted.dat — element / z / ALPHAscs / C6(DFTD3) sorted by z-coordinate
"""

import sys
import re
import os
import subprocess
from collections import OrderedDict

# ─── Configuration ───────────────────────────────────────────────────────────
DFTD3_BIN = "/home/jthlol/program/dftd3_program/origin/test/a.out"
DFTD3_ARGS = ["-func", "hf", "-bj", "-pbc"]

# Unit conversion: Ha·a0^6 → kcal/mol·Å^6
# 1 Ha = 627.5094740631 kcal/mol ; 1 a0 = 0.529177 Å
# factor = 627.5094740631 * 0.529177^6 = 13.77928721
CONV_AU_TO_KCAL = 13.77928721


# ─── Read structure ──────────────────────────────────────────────────────────

def read_structure():
    """Read CONTCAR or POSCAR with ASE. Returns Atoms object or None."""
    from ase.io import read
    for posfile in ['CONTCAR', 'POSCAR']:
        if os.path.isfile(posfile):
            atoms = read(posfile)
            print(f"[structure] Read {posfile}  ({len(atoms)} atoms)")
            return atoms
    print("[structure] WARNING: No POSCAR/CONTCAR found.")
    return None


# ─── FISCS extraction from OUTCAR ────────────────────────────────────────────

def parse_outcar_vdw(outcar_path):
    """
    Parse vdW forcefield parameters from VASP OUTCAR.
    Returns list of dicts with keys:
      element, c6ts, r0ts, alphats, r0scs, alphascs, relvol
    Only keeps the LAST block (in case of ionic relaxation).
    """
    pattern = re.compile(
        r'^\s+([A-Z][a-z]?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
        r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    )
    atoms_data = []
    in_block = False

    with open(outcar_path, 'r') as f:
        for line in f:
            if 'Parameters of vdW forcefield' in line:
                atoms_data = []  # reset — keep only last block
                in_block = True
                continue
            if in_block:
                m = pattern.match(line)
                if m:
                    atoms_data.append({
                        'element':  m.group(1),
                        'c6ts':     float(m.group(2)),
                        'r0ts':     float(m.group(3)),
                        'alphats':  float(m.group(4)),
                        'r0scs':    float(m.group(5)),
                        'alphascs': float(m.group(6)),
                        'relvol':   float(m.group(7)),
                    })
    return atoms_data


def write_fiscs_out(atoms_data, filepath='fiscs.out'):
    """Write full per-atom table (same format as shell script)."""
    with open(filepath, 'w') as f:
        f.write(f"{'Element':<8s}  {'C6ts(au)':>16s}  {'R0ts(au)':>14s}  "
                f"{'ALPHAts(au)':>12s}  {'R0scs(au)':>12s}  {'ALPHAscs(au)':>12s}  "
                f"{'RELVOL':>10s}\n")
        f.write("-" * 92 + "\n")
        for a in atoms_data:
            f.write(f"  {a['element']:<6s}  {a['c6ts']:>14.4f}  {a['r0ts']:>14.4f}  "
                    f"{a['alphats']:>12.4f}  {a['r0scs']:>12.4f}  {a['alphascs']:>12.4f}  "
                    f"{a['relvol']:>10.4f}\n")
    print(f"[fiscs] Written: {filepath}  ({len(atoms_data)} atoms)")


def write_fiscs_average(atoms_data, filepath='fiscs-average.out'):
    """Write per-element averages (same format as shell script)."""
    elem_order = []
    elem_data = OrderedDict()

    for a in atoms_data:
        el = a['element']
        if el not in elem_data:
            elem_order.append(el)
            elem_data[el] = {'count': 0, 'c6ts': 0.0, 'r0ts': 0.0,
                             'alphascs': 0.0, 'relvol': 0.0}
        d = elem_data[el]
        d['count'] += 1
        d['c6ts']     += a['c6ts']
        d['r0ts']     += a['r0ts']
        d['alphascs'] += a['alphascs']
        d['relvol']   += a['relvol']

    with open(filepath, 'w') as f:
        f.write(f"{'Element':<8s}  {'N':>5s}  {'C6ts_avg(au)':>14s}  "
                f"{'ALPHAscs_avg(au)':>15s}  {'RELVOL_avg':>14s}  "
                f"{'C6ts_avg(kcal/mol*A6)':>18s}\n")
        f.write("-" * 92 + "\n")
        for el in elem_order:
            d = elem_data[el]
            n = d['count']
            c6_avg     = d['c6ts'] / n
            alpha_avg  = d['alphascs'] / n
            relvol_avg = d['relvol'] / n
            f.write(f"{el:<8s}  {n:>5d}  {c6_avg:>14.3f}  {alpha_avg:>15.3f}  "
                    f"{relvol_avg:>14.3f}  {c6_avg * CONV_AU_TO_KCAL:>18.3f}\n")

    print(f"[fiscs] Written: {filepath}")


def write_atoms_json(atoms, charges, json_path, label="initial_charges"):
    """Set initial_charges on ASE Atoms and write as JSON."""
    from ase.io import write

    if atoms is None:
        print(f"[json] WARNING: No structure loaded, skipping {json_path}")
        return
    if len(atoms) != len(charges):
        print(f"[json] WARNING: atom count mismatch "
              f"(structure={len(atoms)}, data={len(charges)}), skipping {json_path}")
        return

    atoms_copy = atoms.copy()
    atoms_copy.set_initial_charges(charges)
    write(json_path, atoms_copy)
    print(f"[json] Written: {json_path}  ({label} → initial_charges)")


# ─── DFTD3 C6 extraction ─────────────────────────────────────────────────────

def run_dftd3(poscar='POSCAR'):
    """Run DFTD3 and return output filepath."""
    if not os.path.isfile(poscar):
        print(f"[dftd3] No {poscar} found, skipping DFTD3.")
        return None
    if not os.path.isfile(DFTD3_BIN):
        print(f"[dftd3] DFTD3 binary not found: {DFTD3_BIN}")
        return None

    cmd = [DFTD3_BIN, poscar] + DFTD3_ARGS
    print(f"[dftd3] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        with open('d3.out', 'w') as f:
            f.write(result.stdout)
        print("[dftd3] Written: d3.out")
        return 'd3.out'
    except Exception as e:
        print(f"[dftd3] ERROR: {e}")
        return None


def parse_d3_c6(d3_path):
    """
    Parse DFTD3 output for per-atom C6(AA) values.
    Returns list of C6 values in atom order.
    """
    c6_values = []
    in_table = False

    atom_pattern = re.compile(
        r'^\s*\d+\s+'           # atom index
        r'[\d.\-]+\s+'          # x
        r'[\d.\-]+\s+'          # y
        r'[\d.\-]+\s+'          # z
        r'[a-zA-Z]+\s+'         # element
        r'[\d.]+\s+'            # R0(AA)
        r'[\d.]+\s+'            # CN
        r'([\d.]+)'             # C6(AA)
    )

    with open(d3_path, 'r') as f:
        for line in f:
            if '#' in line and 'XYZ' in line and 'C6' in line:
                in_table = True
                continue
            if in_table:
                m = atom_pattern.match(line)
                if m:
                    c6_values.append(float(m.group(1)))
                elif line.strip() and not atom_pattern.match(line):
                    if c6_values:
                        break

    return c6_values


# ─── Z-sorted combined table ─────────────────────────────────────────────────

def write_zsorted_table(atoms, alphascs_list, c6_list, filepath='bjparams_zsorted.dat'):
    """
    Write a table sorted by z-coordinate with columns:
      index(original)  element  z(Å)  ALPHAscs(au)  C6_D3(au)
    """
    if atoms is None:
        print(f"[zsorted] WARNING: No structure, skipping {filepath}")
        return

    n = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()  # Å

    # Build per-atom records
    records = []
    for i in range(n):
        rec = {
            'idx':     i + 1,
            'element': symbols[i],
            'z':       positions[i][2],
            'alpha':   alphascs_list[i] if (alphascs_list and i < len(alphascs_list)) else None,
            'c6':      c6_list[i]       if (c6_list       and i < len(c6_list))       else None,
        }
        records.append(rec)

    # Sort by z
    records.sort(key=lambda r: r['z'])

    with open(filepath, 'w') as f:
        f.write(f"# BJ dispersion parameters sorted by z-coordinate\n")
        f.write(f"# {'idx':>4s}  {'Element':<8s}  {'z(Ang)':>12s}  "
                f"{'ALPHAscs(au)':>14s}  {'C6_D3(au)':>12s}\n")
        f.write("#" + "-" * 60 + "\n")
        for r in records:
            alpha_str = f"{r['alpha']:>14.4f}" if r['alpha'] is not None else f"{'N/A':>14s}"
            c6_str    = f"{r['c6']:>12.4f}"    if r['c6']    is not None else f"{'N/A':>12s}"
            f.write(f"  {r['idx']:>4d}  {r['element']:<8s}  {r['z']:>12.6f}  "
                    f"{alpha_str}  {c6_str}\n")

    print(f"[zsorted] Written: {filepath}  ({n} atoms, sorted by z)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    outcar = sys.argv[1] if len(sys.argv) > 1 else 'OUTCAR'

    # Read structure once
    atoms = read_structure()

    # ── Part 1: FISCS from OUTCAR ──
    alphascs_list = []
    if os.path.isfile(outcar):
        atoms_data = parse_outcar_vdw(outcar)
        if not atoms_data:
            print(f"[fiscs] No vdW forcefield parameters found in {outcar}")
        else:
            write_fiscs_out(atoms_data)
            write_fiscs_average(atoms_data)
            alphascs_list = [a['alphascs'] for a in atoms_data]
            write_atoms_json(atoms, alphascs_list, 'atoms_fiscs.json', label='ALPHAscs')
    else:
        print(f"[fiscs] OUTCAR not found: {outcar}, skipping FISCS extraction.")

    # ── Part 2: DFTD3 C6 ──
    c6_list = []
    d3_path = 'd3.out'
    if not os.path.isfile(d3_path):
        d3_path = run_dftd3()

    if d3_path and os.path.isfile(d3_path):
        c6_list = parse_d3_c6(d3_path)
        if c6_list:
            print(f"[dftd3] Extracted {len(c6_list)} C6 values")
            write_atoms_json(atoms, c6_list, 'atoms_c6.json', label='C6')
        else:
            print("[dftd3] No C6 values found in d3.out")

    # ── Part 3: Z-sorted combined table ──
    if alphascs_list or c6_list:
        write_zsorted_table(atoms, alphascs_list, c6_list, 'bjparams_zsorted.dat')

    print("[extract_bjparams] Done.")


if __name__ == '__main__':
    main()
