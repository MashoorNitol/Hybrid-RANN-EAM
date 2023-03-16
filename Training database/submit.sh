module load intel-mkl/2021.2.0
module load intel/19.1.3
module load intel-ccl/2021.2.0
module load intel-mpi/2019.9.304
module load intel-ipp/2021.2.0
module load openmpi

ulimit -s unlimited

../calibration_software/nn_calib_icc_new -in Sn.input
