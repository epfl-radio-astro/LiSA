#!/bin/bash
#SBATCH --chdir /home/etolley/LiSA/
#SBATCH --ntasks 1
#SBATCH --array=0-3024%32
#SBATCH --cpus-per-task 1
#SBATCH --time 8:00:00 
#SBATCH --partition parallel
echo STARTING AT `date`

# load the appropriate modules for your system
module load intel/18.0.2
module load python/3.6.5
module load cmake/3.11.1 # for pysap
source venvs/lisa/bin/activate
export PYTHONPATH="$PWD" 

srun python pipelines/pipeline.py pipelines/SDC2_full.config 3025 $DOMAIN_INDEX
echo FINISHED at `date`
