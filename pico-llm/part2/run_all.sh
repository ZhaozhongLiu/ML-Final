#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART2_DIR="${ROOT_DIR}/pico-llm/part2"

PY="${PYTHON_BIN:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  PY="python"
fi

DEVICE="${DEVICE:-cuda:0}"
PRETRAIN_SUBSET_SIZE="${PRETRAIN_SUBSET_SIZE:-2000}"
PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-200}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
BASE_CKPT_OVERRIDE="${BASE_CKPT_OVERRIDE:-}"

TS="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${PART2_DIR}/runs/${TS}"
DATA_DIR="${RUN_DIR}/data"
CKPT_DIR="${RUN_DIR}/checkpoints"
METRICS_DIR="${RUN_DIR}/metrics"
PLOTS_DIR="${RUN_DIR}/plots"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${METRICS_DIR}" "${PLOTS_DIR}"

"${PY}" -m pip install -r "${PART2_DIR}/requirements.txt"

echo "[part2] generating datasets -> ${DATA_DIR}"
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.make_datasets --out_dir "${DATA_DIR}" --seed 0 \
  --n_sft_train 256 --n_sft_val 64 --n_sft_test 64 \
  --n_dpo_train 256 --n_dpo_val 64 --n_dpo_test 64

echo "[part2] pretraining base transformer (TinyStories subset) -> ${CKPT_DIR}"
if [[ -n "${BASE_CKPT_OVERRIDE}" ]]; then
  echo "[part2] using BASE_CKPT_OVERRIDE=${BASE_CKPT_OVERRIDE}"
  cp -f "${BASE_CKPT_OVERRIDE}" "${CKPT_DIR}/transformer_final.pt"
else
  "${PY}" "${ROOT_DIR}/pico-llm/pico-llm.py" \
    --models transformer \
    --device_id "${DEVICE}" \
    --num_epochs 1 \
    --batch_size 8 \
    --learning_rate 3e-4 \
    --train_subset_size "${PRETRAIN_SUBSET_SIZE}" \
    --block_size 256 \
    --transformer_d_model 128 \
    --transformer_num_heads 4 \
    --transformer_num_layers 2 \
    --transformer_mlp_ratio 2.0 \
    --transformer_dropout 0.1 \
    --max_steps_per_epoch "${PRETRAIN_MAX_STEPS}" \
    --tinystories_weight 1.0 \
    --checkpoint_dir "${CKPT_DIR}" \
    --prompt "Once upon a"
fi

BASE_CKPT="${CKPT_DIR}/transformer_final.pt"
SFT_CKPT="${CKPT_DIR}/transformer_sft.pt"
DPO_CKPT="${CKPT_DIR}/transformer_dpo.pt"

echo "[part2] SFT -> ${SFT_CKPT}"
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.train_sft \
  --base_checkpoint "${BASE_CKPT}" \
  --train_jsonl "${DATA_DIR}/sft_train.jsonl" \
  --val_jsonl "${DATA_DIR}/sft_val.jsonl" \
  --out_checkpoint "${SFT_CKPT}" \
  --device "${DEVICE}" \
  --epochs "${SFT_EPOCHS}" \
  --batch_size 8 \
  --lr 5e-5 \
  --max_tokens 256 \
  --monitor_dpo_jsonl "${DATA_DIR}/dpo_val.jsonl" \
  --log_jsonl "${RUN_DIR}/logs_sft.jsonl"

echo "[part2] DPO -> ${DPO_CKPT}"
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.train_dpo \
  --policy_checkpoint "${SFT_CKPT}" \
  --ref_checkpoint "${SFT_CKPT}" \
  --train_jsonl "${DATA_DIR}/dpo_train.jsonl" \
  --val_jsonl "${DATA_DIR}/dpo_val.jsonl" \
  --out_checkpoint "${DPO_CKPT}" \
  --device "${DEVICE}" \
  --epochs "${DPO_EPOCHS}" \
  --batch_size 4 \
  --lr 2e-5 \
  --beta 0.1 \
  --label_smoothing 0.05 \
  --max_tokens 256 \
  --log_jsonl "${RUN_DIR}/logs_dpo.jsonl"

echo "[part2] evaluate -> ${METRICS_DIR}/metrics.json"
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.evaluate \
  --checkpoint "${DPO_CKPT}" \
  --sft_test_jsonl "${DATA_DIR}/sft_test.jsonl" \
  --dpo_test_jsonl "${DATA_DIR}/dpo_test.jsonl" \
  --out_json "${METRICS_DIR}/metrics.json" \
  --device "${DEVICE}" \
  --batch_size 8 \
  --max_tokens 256 \
  --n_samples 5 \
  --sample_new_tokens 160

echo "[part2] plot curves -> ${PLOTS_DIR}/curves.png"
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.plot_curves \
  --sft_log_jsonl "${RUN_DIR}/logs_sft.jsonl" \
  --dpo_log_jsonl "${RUN_DIR}/logs_dpo.jsonl" \
  --out_png "${PLOTS_DIR}/curves.png"

echo "[part2] done: ${RUN_DIR}"
