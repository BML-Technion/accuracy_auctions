#!/bin/bash
#SBATCH --job-name=bana
#SBATCH --account=bml
#SBATCH --partition=bml
#SBATCH --nodelist=plato2
#SBATCH --output=logs/bana-%j.out
#SBATCH --error=logs/bana-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=120

source /home/sadi.bana/miniconda3/etc/profile.d/conda.sh
conda activate myenv

mkdir -p logs

echo "Job started on node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

python exp_4.py 

echo "Job finished at $(date)"
