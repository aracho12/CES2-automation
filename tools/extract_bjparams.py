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
  bjparams_zsorted.dat      — element / z / ALPHAscs / C6(DFTD3) sorted by z-coordinate
  bjparams_elem_zsorted.dat — same columns, sorted by element first then z-coordinate
  bjparams_layer_avg.dat    — layer-averaged (element+z within 0.2Å grouped & averaged)
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

def _build_records(atoms, alphascs_list, c6_list):
    """Build per-atom record list from structure + parameter lists."""
    n = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()  # Å

    records = []
    for i in range(n):
        records.append({
            'idx':     i + 1,
            'element': symbols[i],
            'z':       positions[i][2],
            'alpha':   alphascs_list[i] if (alphascs_list and i < len(alphascs_list)) else None,
            'c6':      c6_list[i]       if (c6_list       and i < len(c6_list))       else None,
        })
    return records


def _write_dat(records, filepath, header_comment):
    """Write records to a .dat file with standard column format."""
    with open(filepath, 'w') as f:
        f.write(f"# {header_comment}\n")
        f.write(f"# {'idx':>4s}  {'Element':<8s}  {'z(Ang)':>12s}  "
                f"{'ALPHAscs(au)':>14s}  {'C6_D3(au)':>12s}\n")
        f.write("#" + "-" * 60 + "\n")
        for r in records:
            alpha_str = f"{r['alpha']:>14.4f}" if r['alpha'] is not None else f"{'N/A':>14s}"
            c6_str    = f"{r['c6']:>12.4f}"    if r['c6']    is not None else f"{'N/A':>12s}"
            f.write(f"  {r['idx']:>4d}  {r['element']:<8s}  {r['z']:>12.6f}  "
                    f"{alpha_str}  {c6_str}\n")
    print(f"[table] Written: {filepath}  ({len(records)} atoms)")


def write_zsorted_table(atoms, alphascs_list, c6_list, filepath='bjparams_zsorted.dat'):
    """
    Write table sorted purely by z-coordinate.
    """
    if atoms is None:
        print(f"[zsorted] WARNING: No structure, skipping {filepath}")
        return

    records = _build_records(atoms, alphascs_list, c6_list)
    records.sort(key=lambda r: r['z'])
    _write_dat(records, filepath, "BJ dispersion parameters sorted by z-coordinate")


def write_element_zsorted_table(atoms, alphascs_list, c6_list, filepath='bjparams_elem_zsorted.dat'):
    """
    Write table sorted by element first, then by z-coordinate within each element.
    """
    if atoms is None:
        print(f"[elem_zsorted] WARNING: No structure, skipping {filepath}")
        return

    records = _build_records(atoms, alphascs_list, c6_list)
    records.sort(key=lambda r: (r['element'], r['z']))
    _write_dat(records, filepath, "BJ dispersion parameters sorted by element, then z-coordinate")


def write_layer_averaged_table(atoms, alphascs_list, c6_list,
                               filepath='bjparams_layer_avg.dat', z_tol=0.2):
    """
    Group atoms by (element, z-layer) and average ALPHAscs / C6 per layer.
    Atoms of the same element within z_tol (Å) of each other belong to the
    same layer.  Layers are sorted by z, then by element within the same z.

    Algorithm:
      1. Sort records by (element, z)
      2. Walk through each element group; start a new layer whenever
         Δz > z_tol from the current layer's first atom.
      3. Average z, alpha, c6 within each layer.
      4. Sort resulting layers by (z_avg, element) for final output.
    """
    if atoms is None:
        print(f"[layer_avg] WARNING: No structure, skipping {filepath}")
        return

    records = _build_records(atoms, alphascs_list, c6_list)
    records.sort(key=lambda r: (r['element'], r['z']))

    # ── Group into layers per element ──
    layers = []  # list of {element, n, z_avg, alpha_avg, c6_avg}

    i = 0
    while i < len(records):
        el = records[i]['element']
        # Collect all records of same element (already sorted by z)
        j = i
        while j < len(records) and records[j]['element'] == el:
            j += 1
        elem_recs = records[i:j]

        # Sub-group by z-layer within this element
        layer_start = 0
        while layer_start < len(elem_recs):
            z_ref = elem_recs[layer_start]['z']
            layer_end = layer_start + 1
            while layer_end < len(elem_recs) and (elem_recs[layer_end]['z'] - z_ref) <= z_tol:
                layer_end += 1

            chunk = elem_recs[layer_start:layer_end]
            n = len(chunk)
            z_avg = sum(r['z'] for r in chunk) / n

            alphas = [r['alpha'] for r in chunk if r['alpha'] is not None]
            c6s    = [r['c6']    for r in chunk if r['c6']    is not None]

            layers.append({
                'element':   el,
                'n':         n,
                'z_avg':     z_avg,
                'alpha_avg': sum(alphas) / len(alphas) if alphas else None,
                'c6_avg':    sum(c6s)    / len(c6s)    if c6s    else None,
            })
            layer_start = layer_end

        i = j

    # Sort layers by z_avg, then element
    layers.sort(key=lambda L: (L['z_avg'], L['element']))

    # ── Write ──
    with open(filepath, 'w') as f:
        f.write(f"# BJ dispersion parameters — layer-averaged (z_tol = {z_tol:.2f} Ang)\n")
        f.write(f"# {'Element':<8s}  {'N':>4s}  {'z_avg(Ang)':>12s}  "
                f"{'ALPHAscs_avg':>14s}  {'C6_D3_avg':>12s}\n")
        f.write("#" + "-" * 60 + "\n")
        for L in layers:
            a_str = f"{L['alpha_avg']:>14.4f}" if L['alpha_avg'] is not None else f"{'N/A':>14s}"
            c_str = f"{L['c6_avg']:>12.4f}"    if L['c6_avg']    is not None else f"{'N/A':>12s}"
            f.write(f"  {L['element']:<8s}  {L['n']:>4d}  {L['z_avg']:>12.6f}  "
                    f"{a_str}  {c_str}\n")

    print(f"[layer_avg] Written: {filepath}  ({len(layers)} layers, z_tol={z_tol} Ang)")


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

    # ── Part 3: Sorted combined tables ──
    if alphascs_list or c6_list:
        write_zsorted_table(atoms, alphascs_list, c6_list, 'bjparams_zsorted.dat')
        write_element_zsorted_table(atoms, alphascs_list, c6_list, 'bjparams_elem_zsorted.dat')
        write_layer_averaged_table(atoms, alphascs_list, c6_list, 'bjparams_layer_avg.dat')

    print("[extract_bjparams] Done.")


if __name__ == '__main__':
    main()
