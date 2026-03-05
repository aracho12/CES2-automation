from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import subprocess

@dataclass
class StructureReq:
    species_id: str
    xyz_file: str
    count: int
    zmin: float

@dataclass
class PackmolJob:
    binary: str
    tolerance: float
    maxit: int
    seed: int
    Lx: float
    Ly: float
    z_lo: float
    z_hi: float
    output_xyz: str
    structures: List[StructureReq]

def write_packmol_input(path: Path, job: PackmolJob) -> None:
    xlo, ylo, xhi, yhi = 0.0, 0.0, job.Lx, job.Ly

    def inside_box(zmin: float) -> str:
        return f"inside box {xlo:.6f} {ylo:.6f} {max(zmin, job.z_lo):.6f} {xhi:.6f} {yhi:.6f} {job.z_hi:.6f}"

    lines = []
    lines += [f"tolerance {job.tolerance}"]
    lines += [f"maxit {job.maxit}"]
    lines += ["filetype xyz"]
    lines += [f"seed {job.seed}"]
    lines += [f"output {job.output_xyz}", ""]

    for s in job.structures:
        if s.count <= 0:
            continue
        lines += [f"structure {s.xyz_file}",
                  f"  number {s.count}",
                  f"  {inside_box(s.zmin)}",
                  "end structure", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_packmol(packmol_bin: str, inp_path: Path, cwd: Path) -> None:
    with open(inp_path, "r") as f:
        res = subprocess.run([packmol_bin], stdin=f,
                         text=True, cwd=str(cwd),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"PACKMOL failed (code={res.returncode}).\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
