#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/latest.out logs/latest.err
ln -s "$SLURM_JOBID.out" "logs/latest.out"
ln -s "$SLURM_JOBID.err" "logs/latest.err"

module purge
module load tensorflow/2.2-hvd
source venv/bin/activate

echo "START $SLURM_JOBID: $(date)"

srun python "$@"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
