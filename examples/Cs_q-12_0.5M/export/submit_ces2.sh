#!/bin/bash
#SBATCH -J 4OH_Cs_q-12_0.5M
#SBATCH -A members
#SBATCH -p skylake_24c,broadwell_24c,haswell_24c_ib_sm61,haswell_24c
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --no-requeue
#SBATCH -t 00:00:00
#SBATCH -e %x.e%j
#SBATCH -o %x.o%j
#SBATCH --comment qmmm

# submit_ces2.sh — SLURM batch (tars); nurion->tars seed-fix migration
prefix="4OH_Cs_q-12_0.5M"
curr_dir=${SLURM_SUBMIT_DIR}
cd $curr_dir

./qmmm_dftces2_charging_pts.sh > $curr_dir/qmmm.out
