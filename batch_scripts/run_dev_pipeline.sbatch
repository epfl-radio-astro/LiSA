#!/bin/bash
#SBATCH --chdir /home/etolley/EPFL-SKA-SDC2/
#SBATCH --ntasks 1
#SBATCH --array=0-4%4
#SBATCH --cpus-per-task 1
#SBATCH --time 8:00:00 
#SBATCH --partition parallel
echo STARTING AT `date`
echo "Setting up enviroment to use MPI & wavelets..."

module load intel/18.0.2
module load python/3.6.5
module load intel-mpi
module load cmake/3.11.1 # for pysap
export I_MPI_FABRICS=shm:ofa
source venvs/sdc2-deneb1-mpi-pysap/bin/activate
export PYTHONPATH="$PWD" 
pip install tensorflow
srun python pipelines/pipeline_dev_test.py 144 $SLURM_ARRAY_TASK_ID
echo FINISHED at `date`