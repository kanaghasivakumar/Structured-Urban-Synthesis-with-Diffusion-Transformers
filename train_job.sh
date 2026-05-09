#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=36:00:00
#SBATCH --job-name=urban_sweep
#SBATCH --output=logs/urban_sweep_%A_%a.log
#SBATCH --array=1-6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kanaghasivakumar2027@u.northwestern.edu

module purge
module load anaconda3
module load cuda

source /software/anaconda3/2018.12/etc/profile.d/conda.sh
source activate /projects/e32706/omb8654/conda/envs/dit-env

export WANDB_API_KEY=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
export PYTHONPATH=$PYTHONPATH:.
export PATH=/projects/e32706/omb8654/conda/envs/dit-env/bin:$PATH


/projects/e32706/omb8654/conda/envs/dit-env/bin/python -m wandb agent kanaghasivakumar-northwestern-university/structured-urban-synthesis/eihh8bbu
