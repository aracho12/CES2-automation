#!/bin/bash

if [ "$#" -ne 1 ]; then
        echo 'Usage: Submit_vasp.sh <job_name>'
        exit 1
fi

echo '#!/bin/bash' > $1.pbs
echo '#SBATCH -J '$1 >> $1.pbs
echo '#SBATCH -A members' >> $1.pbs
echo '#SBATCH -p skylake_24c,broadwell_24c,haswell_24c,ivybridge_20c,sandybridge_16c' >> $1.pbs
echo '#SBATCH -N 1' >> $1.pbs
echo '#SBATCH -t 00:00:00' >> $1.pbs
echo '#SBATCH -e %x.e%j' >> $1.pbs
echo '#SBATCH -o %x.o%j' >> $1.pbs
echo '#SBATCH --comment vasp' >> $1.pbs
echo '' >> $1.pbs
echo 'PREFIX=${SLURM_JOB_NAME}' >> $1.pbs
echo 'CURR_DIR=${SLURM_SUBMIT_DIR}' >> $1.pbs
echo 'SCRATCH_DIR=/scratch/${USER}' >> $1.pbs
echo 'NODES=${SLURM_JOB_NODELIST}' >> $1.pbs
echo 'let NPROCS=${SLURM_JOB_NUM_NODES}*${SLURM_CPUS_ON_NODE}' >> $1.pbs
echo 'DO_PARALLEL="srun -n ${NPROCS}"' >> $1.pbs
echo 'EXECUTABLE="/apps/programs/vasp/vasp.5.4.4/bin/vasp_std"' >> $1.pbs
echo '' >> $1.pbs
echo 'mkdir ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'mkdir ${SCRATCH_DIR}/${PREFIX}' >> $1.pbs
echo '' >> $1.pbs
echo 'cp -p ${CURR_DIR}/${PREFIX}.in ${SCRATCH_DIR}/${PREFIX}/INCAR' >> $1.pbs
echo 'cp -p ${CURR_DIR}/${PREFIX}.pos ${SCRATCH_DIR}/${PREFIX}/POSCAR' >> $1.pbs
echo 'cp -p ${CURR_DIR}/${PREFIX}.pot ${SCRATCH_DIR}/${PREFIX}/POTCAR' >> $1.pbs
echo 'cp -p ${CURR_DIR}/${PREFIX}.k ${SCRATCH_DIR}/${PREFIX}/KPOINTS' >> $1.pbs
#echo 'cp -p ${CURR_DIR}/../mm_9/EXTCAR ${SCRATCH_DIR}/${PREFIX}/' >> $1.pbs
#echo 'cp -p ${CURR_DIR}/vdw_kernel.bindat ${SCRATCH_DIR}/${PREFIX}/' >> $1.pbs
#echo 'cp -p ${CURR_DIR}/${PREFIX}.w ${SCRATCH_DIR}/${PREFIX}/WAVECAR' >> $1.pbs
echo '' >> $1.pbs
echo 'cd ${SCRATCH_DIR}/${PREFIX}' >> $1.pbs
echo 'echo Running ${EXECUTABLE} on ${NODES} with ${NPROCS} processors, launched from ${SLURM_SUBMIT_HOST} > ${CURR_DIR}/${PREFIX}/${PREFIX}.log' >> $1.pbs
echo '' >> $1.pbs
echo '${DO_PARALLEL} ${EXECUTABLE} >> ${CURR_DIR}/${PREFIX}/${PREFIX}.log' >> $1.pbs
echo 'if [ $? -ne 0 ]; then' >> $1.pbs
echo ' echo ERROR OCCURRED: Check output file' >> $1.pbs
echo ' exit 1' >> $1.pbs
echo 'fi' >> $1.pbs
echo '' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/OUTCAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/CONTCAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/OSZICAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/POSCAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/DOSCAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/XDATCAR ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo 'cp ${SCRATCH_DIR}/${PREFIX}/vasprun.xml ${CURR_DIR}/${PREFIX}' >> $1.pbs
echo '' >> $1.pbs
echo 'echo No errors detected' >> $1.pbs

sbatch $1.pbs
