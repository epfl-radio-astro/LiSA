#!/bin/bash
#SBATCH --chdir /home/etolley/LiSA/
#SBATCH --ntasks 1
#SBATCH --array=0-143%32
#SBATCH --time 8:00:00 
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem 100GB
echo STARTING AT `date`

# load the appropriate modules for your system
module load intel/19.0.5
module load python/3.7.7
module load cmake/3.16.5 # for pysap
source venvs/lisa/bin/activate
export PYTHONPATH="$PWD" 

srun python3 pipelines/pipeline.py pipelines/SDC2_ldev.config 144 $SLURM_ARRAY_TASK_ID
echo FINISHED at `date`