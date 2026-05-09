#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --job-name=urban_final
#SBATCH --output=logs/urban_final_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kanaghasivakumar2027@u.northwestern.edu

module purge
module load anaconda3
module load cuda
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
source activate /projects/e32706/omb8654/conda/envs/dit-env

export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:.

/projects/e32706/omb8654/conda/envs/dit-env/bin/python src/train.py \
    --epochs 150 \
    --lr_batch 4e-4,128 \
    --patch_size 8 \
    --depth 12 \
    --num_heads 8 \
    --warmup_epochs 3 \
    --cfg_dropout 0.05
