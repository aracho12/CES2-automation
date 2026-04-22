#!/usr/bin/env python3
"""cube_to_potzavg.py — planar-average a Gaussian cube file of an
electrostatic potential along z and write a QE-style  pot.z.avg
(columns: z_bohr, V_Ry) for workfunction.py.

Assumes an orthorhombic grid (QE pp.x, iflag=3, output_format=6 on
an orthogonal cell).  Cube values are taken as-is; DFT-CES2 pp.x
writes V_H + V_bare in Ry, which is what workfunction.py expects.

Usage
-----
    python cube_to_potzavg.py <cube_file> [output_file]

    # default output: pot.z.avg next to the cube file
"""
import os
import sys
import numpy as np


def read_cube(path):
    with open(path) as f:
        f.readline(); f.readline()  # 2 comment lines
        parts = f.readline().split()
        natoms = int(parts[0])
        origin = np.array([float(x) for x in parts[1:4]])

        def vox(line):
            p = line.split()
            return int(p[0]), np.array([float(x) for x in p[1:4]])

        nx, vx = vox(f.readline())
        ny, vy = vox(f.readline())
        nz, vz = vox(f.readline())

        for _ in range(abs(natoms)):
            f.readline()

        data = np.array(f.read().split(), dtype=np.float64)

    if data.size != nx * ny * nz:
        sys.exit(f"ERROR: cube data size mismatch. Expected {nx*ny*nz}, got {data.size}")

    grid = data.reshape((nx, ny, nz))
    return origin, (nx, ny, nz), (vx, vy, vz), grid


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    cube_path = sys.argv[1]
    if not os.path.isfile(cube_path):
        sys.exit(f"ERROR: file not found: {cube_path}")

    out_path = sys.argv[2] if len(sys.argv) >= 3 else \
        os.path.join(os.path.dirname(os.path.abspath(cube_path)), "pot.z.avg")

    print(f"Reading: {cube_path}")
    origin, (nx, ny, nz), (vx, vy, vz), grid = read_cube(cube_path)
    print(f"  grid       = {nx} x {ny} x {nz}")
    print(f"  origin     = {origin} Bohr")
    print(f"  vx / vy / vz (Bohr) = {vx} / {vy} / {vz}")

    if abs(vz[0]) > 1e-6 or abs(vz[1]) > 1e-6 or abs(vx[2]) > 1e-6 or abs(vy[2]) > 1e-6:
        print("WARNING: cell has non-orthogonal z axis; planar average may be approximate.")

    v_z    = grid.mean(axis=(0, 1))
    dz     = vz[2]
    z_bohr = origin[2] + np.arange(nz) * dz

    print(f"Writing: {out_path}")
    with open(out_path, "w") as f:
        f.write("# z (Bohr)        V (Ry, cube units)\n")
        for z, v in zip(z_bohr, v_z):
            f.write(f"{z:14.8f}  {v:18.10e}\n")

    print(f"Done. {nz} z-points written  (z = {z_bohr[0]:.3f}–{z_bohr[-1]:.3f} Bohr).")


if __name__ == "__main__":
    main()
