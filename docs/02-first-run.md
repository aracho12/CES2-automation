# 02 — Your First Run

<!-- TODO: fill in. Outline below; each step should show the exact commands and
     what "success" looks like on screen. -->

## 0. Prerequisites

- [ ] Access to a built DFT-CES2 stack (`dft-ces2` LAMMPS fork, patched QE `pw.x`/`pp.x`, cube tools) — ask for the module/paths used by your group
- [ ] This repo cloned, Python env set up ([main README §Quick Start](../README.md))
- [ ] A relaxed VASP `CONTCAR` of your slab

## 1. Build the run directory

<!-- TODO: config.yaml minimum example; run_builder.py invocation; what to check
     in export/ before submitting (build_summary.json, charged_system_params.json,
     eyeball data.file/combined.xyz in OVITO/VESTA). -->

## 2. Submit

<!-- TODO: submit_ces2.sh (SLURM sbatch / PBS qsub); how NP relates to the node
     request; typical wall time per QM/MM step. -->

## 3. Monitor

<!-- TODO: tail -f qmmm.out; the "######### Starting N QMMM step #########" markers;
     which qm_N/mm_N directories should appear when. Link to 05. -->

## 4. Resume after walltime

<!-- TODO: tools/ces2_resubmit.sh usage — run it inside export/; what it patches
     (QMMMINISTEP, initialqm, skipequil, LAMMPSRESTART) and the .bak / log files
     it leaves. Link to 05. -->

## 5. First analysis

<!-- TODO: workfunction.py, z_density.py on the last converged steps; checking
     step-to-step convergence. Link to 06. -->
