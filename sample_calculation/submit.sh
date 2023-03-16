#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=36
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=mash@lanl.gov
#SBATCH --job-name e9beta-C
#SBATCH --error myjob_%j.err
#SBATCH -A w22_imidas

module load intel-mkl/2021.2.0
module load intel/19.1.3
module laod intel-ccl/2021.2.0
module load intel-mpi/2019.9.304
module load intel-ipp/2021.2.0
module load openmpi/4.1.1
ulimit -s unlimited
rm slurm* myjob*

srun ../lammps-23Jun2022/src/lmp_mpi -in loading.in > loading.out
