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

@dataclass
class PackmolResult:
    """Summary of a packmol run extracted from its stdout."""
    converged:        bool    # True if "SOLUTION FOUND" is present
    obj_final:        float   # final objective function value (0.0 = perfect)
    n_iter:           int     # number of iterations used
    stdout:           str     # full packmol stdout (saved to packmol.log)


def _parse_packmol_output(stdout: str) -> PackmolResult:
    """Extract convergence info from packmol stdout."""
    converged  = "SOLUTION FOUND" in stdout
    obj_final  = 0.0
    n_iter     = 0
    for line in stdout.splitlines():
        # "  Objective function value at solution: 0.00000E+00"
        if "Objective function" in line and ":" in line:
            try:
                obj_final = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        # "  Number of GENCAN iterations:" or "  Function evaluations:"
        if "Number of GENCAN iterations" in line and ":" in line:
            try:
                n_iter = int(line.split(":")[-1].strip())
            except ValueError:
                pass
    return PackmolResult(converged=converged, obj_final=obj_final,
                         n_iter=n_iter, stdout=stdout)


def run_packmol(packmol_bin: str, inp_path: Path, cwd: Path) -> PackmolResult:
    """Run packmol and return a PackmolResult with convergence info.

    Raises RuntimeError on non-zero exit code.
    The full stdout is always saved to  <cwd>/packmol.log  for post-mortem
    inspection regardless of success or failure.
    """
    with open(inp_path, "r") as f:
        res = subprocess.run([packmol_bin], stdin=f,
                             text=True, cwd=str(cwd),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Always save the full log
    (cwd / "packmol.log").write_text(res.stdout, encoding="utf-8")

    if res.returncode != 0:
        raise RuntimeError(
            f"PACKMOL failed (code={res.returncode}).\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return _parse_packmol_output(res.stdout)
