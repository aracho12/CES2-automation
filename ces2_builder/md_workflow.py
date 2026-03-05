from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json

@dataclass
class MDOutputs:
    in_relax: str
    ff_dir: str
    sbatch: str
    pbs: str
    run_sh: str

def write_placeholder_ff(ff_dir: Path) -> None:
    ff_dir.mkdir(parents=True, exist_ok=True)
    (ff_dir/"pair_coeff.in").write_text(
        "# Placeholder: define nonbonded terms here\n"
        "# Example:\n"
        "# pair_style lj/cut/coul/long 10.0\n"
        "# pair_modify mix arithmetic\n"
        "# pair_coeff * * 0.0 1.0\n",
        encoding="utf-8"
    )
    (ff_dir/"bond_coeff.in").write_text(
        "# Placeholder: bonded terms\n"
        "# Example:\n"
        "bond_style harmonic\n"
        "# bond_coeff 1 450.0 0.9572\n",
        encoding="utf-8"
    )
    (ff_dir/"angle_coeff.in").write_text(
        "# Placeholder: angle terms\n"
        "# Example:\n"
        "# angle_style harmonic\n"
        "# angle_coeff 1 55.0 104.52\n",
        encoding="utf-8"
    )
    (ff_dir/"kspace.in").write_text(
        "# Placeholder: long-range electrostatics\n"
        "# Example:\n"
        "# kspace_style pppm 1e-4\n",
        encoding="utf-8"
    )

def write_in_relax(path: Path,
                   data_file: str = "data.file",
                   ff_dir: str = "ff",
                   timestep_fs: float = 1.0,
                   min_etol: float = 1e-4,
                   min_ftol: float = 1e-6,
                   min_maxiter: int = 5000,
                   min_maxeval: int = 10000,
                   t_start: float = 10.0,
                   t_stop: float = 300.0,
                   tdamp_fs: float = 100.0,
                   nvt_steps: int = 20000,
                   dump_every: int = 2000,
                   write_equilibrated: str = "equilibrated.data",
                   qm_lo: Optional[int] = None,
                   qm_hi: Optional[int] = None,
                   ) -> None:
    """
    Write a simple pre-relax LAMMPS input.

    If qm_lo/qm_hi are provided, QM atoms in that ID range are frozen and only
    the remaining atoms (group SOLVENT) are thermalized and time-integrated.
    """
    # Decide group/thermostat layout depending on whether QM range is known
    if qm_lo is not None and qm_hi is not None:
        freeze_block = f"""# ---- QM/MM groups: freeze QM slab, relax solvent ----
group QM id {qm_lo}:{qm_hi}
group SOLVENT subtract all QM

velocity QM set 0.0 0.0 0.0
fix freezeQM QM setforce 0.0 0.0 0.0

"""
        vel_group = "SOLVENT"
        nvt_group = "SOLVENT"
        unfix_freeze = "unfix freezeQM\n"
    else:
        freeze_block = ""
        vel_group = "all"
        nvt_group = "all"
        unfix_freeze = ""

    txt = f"""units real
atom_style full
boundary p p p

# ---- Force field styles must be declared before read_data ----
include {ff_dir}/bond_coeff.in
include {ff_dir}/angle_coeff.in
include {ff_dir}/pair_coeff.in
include {ff_dir}/kspace.in

read_data {data_file}

{freeze_block}

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# ---- Minimization ----
min_style cg
minimize {min_etol} {min_ftol} {min_maxiter} {min_maxeval}

# ---- Short NVT ----
timestep {timestep_fs}
velocity {vel_group} create {t_start} 12345 mom yes rot yes dist gaussian

thermo 500
thermo_style custom step temp pe ke etotal press vol

fix NVT {nvt_group} nvt temp {t_start} {t_stop} {tdamp_fs}
dump D1 all custom {dump_every} md.lammpstrj id type q x y z
dump_modify D1 sort id
run {nvt_steps}
unfix NVT
{unfix_freeze}undump D1

write_data {write_equilibrated}
"""
    path.write_text(txt, encoding="utf-8")

def write_sbatch(path: Path,
                job_name: str = "ces2_md_relax",
                partition: str = "compute",
                nodes: int = 1,
                ntasks: int = 32,
                time_hhmmss: str = "08:00:00",
                lmp_cmd: str = "lmp_mpi",
                in_file: str = "in.relax",
                out: str = "md.out",
                err: str = "md.err",
                module_lines: str = "module load lammps\n") -> None:
    path.write_text(f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH -N {nodes}
#SBATCH -n {ntasks}
#SBATCH -t {time_hhmmss}
#SBATCH -o {out}
#SBATCH -e {err}

set -euo pipefail

{module_lines}
mpirun {lmp_cmd} -in {in_file}
""", encoding="utf-8")

def write_pbs(path: Path,
             job_name: str = "ces2_md_relax",
             nodes: int = 1,
             ppn: int = 32,
             walltime: str = "08:00:00",
             lmp_cmd: str = "lmp_mpi",
             in_file: str = "in.relax",
             module_lines: str = "module load lammps\n") -> None:
    path.write_text(f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l walltime={walltime}
#PBS -j oe

set -euo pipefail
cd $PBS_O_WORKDIR

{module_lines}
mpirun {lmp_cmd} -in {in_file}
""", encoding="utf-8")

def write_run_sh(path: Path) -> None:
    path.write_text("""#!/bin/bash
set -euo pipefail
LMP=${1:-lmp_mpi}
${LMP} -in in.relax
""", encoding="utf-8")

def generate_md_bundle(export_dir: Path, md_cfg: Dict) -> MDOutputs:
    ff_dir = export_dir / md_cfg.get("ff_dir","ff")
    write_placeholder_ff(ff_dir)

    # Infer QM atom ID range from build_summary.json if available
    qm_lo: Optional[int] = None
    qm_hi: Optional[int] = None
    summary_path = export_dir / "build_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            n_mm = int(summary.get("n_mm", 0))
            n_qm = int(summary.get("n_qm", 0))
            if n_mm > 0 and n_qm > 0:
                qm_lo = n_mm + 1
                qm_hi = n_mm + n_qm
        except Exception:
            # If anything goes wrong, fall back to "no QM info" mode.
            qm_lo = None
            qm_hi = None

    write_in_relax(
        export_dir/"in.relax",
        data_file=md_cfg.get("data_file","data.file"),
        ff_dir=str(md_cfg.get("ff_dir","ff")),
        timestep_fs=float(md_cfg.get("timestep_fs",1.0)),
        min_etol=float(md_cfg.get("min_etol",1e-4)),
        min_ftol=float(md_cfg.get("min_ftol",1e-6)),
        min_maxiter=int(md_cfg.get("min_maxiter",5000)),
        min_maxeval=int(md_cfg.get("min_maxeval",10000)),
        t_start=float(md_cfg.get("t_start",10.0)),
        t_stop=float(md_cfg.get("t_stop",300.0)),
        tdamp_fs=float(md_cfg.get("tdamp_fs",100.0)),
        nvt_steps=int(md_cfg.get("nvt_steps",20000)),
        dump_every=int(md_cfg.get("dump_every",2000)),
        write_equilibrated=md_cfg.get("write_equilibrated","equilibrated.data"),
        qm_lo=qm_lo,
        qm_hi=qm_hi,
    )

    sl = md_cfg.get("slurm", {})
    write_sbatch(
        export_dir/"submit_md.sbatch",
        job_name=sl.get("job_name","ces2_md_relax"),
        partition=sl.get("partition","compute"),
        nodes=int(sl.get("nodes",1)),
        ntasks=int(sl.get("ntasks",32)),
        time_hhmmss=sl.get("time","08:00:00"),
        lmp_cmd=sl.get("lmp_cmd","lmp_mpi"),
        in_file="in.relax",
        out=sl.get("stdout","md.out"),
        err=sl.get("stderr","md.err"),
        module_lines=sl.get("module_lines","module load lammps\n"),
    )

    pb = md_cfg.get("pbs", {})
    write_pbs(
        export_dir/"submit_md.pbs",
        job_name=pb.get("job_name","ces2_md_relax"),
        nodes=int(pb.get("nodes",1)),
        ppn=int(pb.get("ppn",32)),
        walltime=pb.get("walltime","08:00:00"),
        lmp_cmd=pb.get("lmp_cmd","lmp_mpi"),
        in_file="in.relax",
        module_lines=pb.get("module_lines","module load lammps\n"),
    )

    write_run_sh(export_dir/"run_md.sh")
    try:
        (export_dir/"run_md.sh").chmod(0o755)
    except Exception:
        pass

    return MDOutputs(
        in_relax="export/in.relax",
        ff_dir=f"export/{md_cfg.get('ff_dir','ff')}/",
        sbatch="export/submit_md.sbatch",
        pbs="export/submit_md.pbs",
        run_sh="export/run_md.sh",
    )
