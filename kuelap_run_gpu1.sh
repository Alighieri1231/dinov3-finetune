#!/usr/bin/env bash
#SBATCH --account=bcastane_lab
#SBATCH --partition=kuelap
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x-%j.out


set -euo pipefail

# (Optional) activar conda dentro del script
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate lightning || true
fi

# No fijes CUDA_VISIBLE_DEVICES manualmente; Slurm lo define.
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === Config base ===
PY=python
MAIN=main.py
DATASET=liver
ROOT=/scratch/bcastane_lab/eochoaal
SPLIT=./data/splits_final.json
SIZE=large
DINO=dinov3
BATCH=32
EPOCHS=100
WARMUP=20
N_WORKERS=18
FOLDS=(1 3)
LORA_RANKS=(4 8)

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p output/logs logs

run_job () {
  local exp="$1"; shift
  echo ">>> [$SLURM_JOB_ID] ${exp}"
  # Slurm 24.05 tip: pass --mem to srun to avoid env conflicts
  srun --exclusive --mem=64G ${PY} ${MAIN} --exp_name "${exp}" "$@" \
      2>&1 | tee "output/logs/${exp}.log"
}

for FOLD in "${FOLDS[@]}"; do
  # 1) HEAD (linear)
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_HEAD_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --debug \
    --n_workers "${N_WORKERS}" --wandb

  # 2) FPN
  EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_${STAMP}"
  run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
    --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" --use_fpn --debug \
    --n_workers "${N_WORKERS}" --wandb
done

for FOLD in "${FOLDS[@]}"; do
  # 3) LoRA (HEAD) r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_lora --r "${r}" --n_workers "${N_WORKERS}" --debug --wandb
  done

  # 4) FPN + LoRA r=4,8
  for r in "${LORA_RANKS[@]}"; do
    EXP="dino_${DATASET}_fold${FOLD}_bs${BATCH}_FPN_LoRA_r${r}_${STAMP}"
    run_job "${EXP}" --dataset "${DATASET}" --root "${ROOT}" --split_json "${SPLIT}" \
      --size "${SIZE}" --dino_type "${DINO}" --batch_size "${BATCH}" \
      --epochs "${EPOCHS}" --warmup_epochs "${WARMUP}" --fold "${FOLD}" \
      --use_fpn --use_lora --r "${r}" --n_workers "${N_WORKERS}" --debug --wandb
  done
done

echo "=== GPU1 listo. Logs en output/logs y logs/%x-%j.out ==="
