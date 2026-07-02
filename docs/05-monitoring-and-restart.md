# 05 — Monitoring & Restart

<!-- TODO: fill in. Outline below. -->

## Reading `qmmm.out`

<!-- TODO: the phase markers and what a healthy step looks like:
     "######### Starting N QMMM step #########"
     "### QM calculation done"
     "### Running LAMMPS(emxext) for equilibration ..."   ← equilibration
     "### Running LAMMPS(emdext) for averaging ..."        ← averaging
     "### ERROR: ..." lines and preflight_check failures. -->

## Where progress lives on disk

<!-- TODO: qm_N/mm_N appearance order; how to tell which phase a killed run
     died in (mm_N empty vs. populated, newest restart timestep vs. MDEQUIL/MDAVERAGE). -->

## Resuming a killed run: `tools/ces2_resubmit.sh`

<!-- TODO: run inside export/; what it detects; the four variables it patches
     (QMMMINISTEP, initialqm, skipequil, LAMMPSRESTART — see 03); .bak and log
     files; when NOT to trust it (hand-edited scripts, partial cube writes). -->

## Common failures

<!-- TODO:
     - PBS/SLURM walltime kill → resubmit (the standard case)
     - pw.x "pw_readschemafile" crash on resume of charged/neutral runs:
       stale solute.save with startingwfc='file' → set STARTINGWFC/STARTINGPOT
       to 'atomic' for the resumed step (or clear solute/)
     - LAMMPS "Non-numeric pressure" / lost atoms after bad packing → md_relax
     - preflight_check FAILED → wrong binary paths in ces2_script config
     - QM not converging (check pw.out electron_maxstep / mixing_beta) -->
