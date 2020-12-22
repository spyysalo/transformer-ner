#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# transformers cache directory
CACHE_DIR=transformers-models

# remove output and job marker on exit
function on_exit {
    rm -f out-$SLURM_JOBID.tsv
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

# check arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 model data_dir seq_len batch_size learning_rate epochs"
    exit 1
fi

model="$1"
data_dir="$2"
max_seq_length="$3"
batch_size="$4"
learning_rate="$5"
epochs="$6"

# symlink logs for this run as "latest"
rm -f logs/latest.out logs/latest.err
ln -s "$SLURM_JOBID.out" "logs/latest.out"
ln -s "$SLURM_JOBID.err" "logs/latest.err"

module purge
module load tensorflow/2.2-hvd
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun python train.py \
    --model_name "$model" \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --max_seq_length $max_seq_length \
    --batch_size $batch_size \
    --train_data "$data_dir/train.tsv" \
    --dev_data "$data_dir/dev.tsv" \
    --labels "$data_dir/labels.txt" \
    --output_file "out-$SLURM_JOBID.tsv" \
    --cache_dir $CACHE_DIR

result=$(python conlleval.py out-$SLURM_JOBID.tsv \
    | egrep '^accuracy' | perl -pe 's/.*FB1:\s+(\S+).*/$1/')

echo -n 'DEV-RESULT'$'\t'
echo -n 'model'$'\t'"$model"$'\t'
echo -n 'data_dir'$'\t'"$data_dir"$'\t'
echo -n 'max_seq_length'$'\t'"$max_seq_length"$'\t'
echo -n 'train_batch_size'$'\t'"$batch_size"$'\t'
echo -n 'learning_rate'$'\t'"$learning_rate"$'\t'
echo -n 'num_train_epochs'$'\t'"$epochs"$'\t'
echo 'FB1'$'\t'"$result"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
