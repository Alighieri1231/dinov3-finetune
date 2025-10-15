#!/usr/bin/env bash
#SBATCH --account=bcastane_lab
#SBATCH --partition=kuelap
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# (Optional) conda
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate lightning || true
fi

free -h
nvidia-smi
srun --mem=64G python main.py --exp_name test_dino --dataset liver --root /scratch/bcastane_lab/eochoaal --split_json ./data/splits_final.json --size large --dino_type dinov3 --batch_size 256 --epochs 10 --warmup_epochs 1 --fold 0 --debug --n_workers 18