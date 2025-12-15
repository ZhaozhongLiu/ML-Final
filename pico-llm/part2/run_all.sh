#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PART2_DIR="${ROOT_DIR}/pico-llm/part2"

PY="${PYTHON_BIN:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  PY="python"
fi

DEVICE="${DEVICE:-cuda:0}"
PRETRAIN_SUBSET_SIZE="${PRETRAIN_SUBSET_SIZE:-50000}"
PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-2000}"
PRETRAIN_MAX_SECONDS="${PRETRAIN_MAX_SECONDS:-14400}"   # 4 hours
SFT_MAX_SECONDS="${SFT_MAX_SECONDS:-1800}"              # 30 minutes
DPO_MAX_SECONDS="${DPO_MAX_SECONDS:-1800}"              # 30 minutes
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-60}"
SFT_EPOCHS="${SFT_EPOCHS:-9999}"
DPO_EPOCHS="${DPO_EPOCHS:-9999}"
BASE_CKPT_OVERRIDE="${BASE_CKPT_OVERRIDE:-}"
DATA_PROVIDER="${DATA_PROVIDER:-deepseek}"   # template | deepseek | chatgpt

# DeepSeek (OpenAI-compatible) defaults
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"

# Generic generation knobs (used for both deepseek/chatgpt providers)
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.8}"
LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-1200}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-5}"
LLM_FALLBACK="${LLM_FALLBACK:-template}"          # template | stop
LLM_MAX_CONSEC_FAILS="${LLM_MAX_CONSEC_FAILS:-3}"
LLM_MAX_CALLS="${LLM_MAX_CALLS:-0}"               # 0 = unlimited
LLM_MAX_TOTAL_TOKENS="${LLM_MAX_TOTAL_TOKENS:-2500000}" # token budget target (~$5; adjust per pricing)
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"             # >=1; batch multiple specs per API call

# OpenAI ChatGPT provider (optional)
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# HuggingFace access (TinyStories pretrain)
# If huggingface.co is blocked (common in some regions), set HF_ENDPOINT_TINY to a mirror like https://hf-mirror.com
HF_ENDPOINT_TINY="${HF_ENDPOINT_TINY:-}"
HF_ENDPOINT_FALLBACK="${HF_ENDPOINT_FALLBACK:-https://hf-mirror.com}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

TS="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="${PART2_DIR}/runs/${TS}"
DATA_DIR="${RUN_DIR}/data"
CKPT_DIR="${RUN_DIR}/checkpoints"
METRICS_DIR="${RUN_DIR}/metrics"
PLOTS_DIR="${RUN_DIR}/plots"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${METRICS_DIR}" "${PLOTS_DIR}"

"${PY}" -m pip install -r "${PART2_DIR}/requirements.txt"

echo "[part2] generating datasets -> ${DATA_DIR}"
LLM_MODEL=""
LLM_BASE_URL=""
LLM_API_KEY_ENV=""
LLM_BASE_URL_ENV=""
LLM_USE_RESPONSE_FORMAT_JSON="1"

if [[ "${DATA_PROVIDER}" == "deepseek" ]]; then
  LLM_MODEL="${DEEPSEEK_MODEL}"
  LLM_BASE_URL="${DEEPSEEK_BASE_URL}"
  LLM_API_KEY_ENV="DEEPSEEK_API_KEY"
  LLM_BASE_URL_ENV="DEEPSEEK_BASE_URL"
  LLM_USE_RESPONSE_FORMAT_JSON="0"
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[part2] WARN: DATA_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set; falling back to template."
    DATA_PROVIDER="template"
  else
    echo "[part2] checking DeepSeek API connectivity..."
    if ! PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.check_openai \
      --model "${LLM_MODEL}" \
      --base_url "${LLM_BASE_URL}" \
      --api_key_env "${LLM_API_KEY_ENV}" \
      --base_url_env "${LLM_BASE_URL_ENV}" \
      --use_response_format_json "${LLM_USE_RESPONSE_FORMAT_JSON}" \
      --max_retries 2; then
      echo "[part2] WARN: DeepSeek API check failed; falling back to template."
      DATA_PROVIDER="template"
    fi
  fi
elif [[ "${DATA_PROVIDER}" == "chatgpt" ]]; then
  LLM_MODEL="${OPENAI_MODEL}"
  LLM_BASE_URL="${OPENAI_BASE_URL:-}"
  LLM_API_KEY_ENV="OPENAI_API_KEY"
  LLM_BASE_URL_ENV="OPENAI_BASE_URL"
  LLM_USE_RESPONSE_FORMAT_JSON="1"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[part2] WARN: DATA_PROVIDER=chatgpt but OPENAI_API_KEY is not set; falling back to template."
    DATA_PROVIDER="template"
  else
    echo "[part2] checking OpenAI API connectivity..."
    if ! PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.check_openai --model "${OPENAI_MODEL}" --max_retries 2; then
      echo "[part2] WARN: OpenAI API check failed; falling back to template."
      DATA_PROVIDER="template"
    fi
  fi
fi

set +e
PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.make_datasets \
  --out_dir "${DATA_DIR}" \
  --seed 0 \
  --provider "${DATA_PROVIDER}" \
  --openai_model "${LLM_MODEL:-${OPENAI_MODEL}}" \
  --openai_base_url "${LLM_BASE_URL}" \
  --openai_temperature "${LLM_TEMPERATURE}" \
  --openai_max_output_tokens "${LLM_MAX_OUTPUT_TOKENS}" \
  --openai_max_retries "${LLM_MAX_RETRIES}" \
  --openai_fallback "${LLM_FALLBACK}" \
  --openai_max_consecutive_failures "${LLM_MAX_CONSEC_FAILS}" \
  --openai_max_calls "${LLM_MAX_CALLS}" \
  --openai_max_total_tokens "${LLM_MAX_TOTAL_TOKENS}" \
  --openai_batch_size "${LLM_BATCH_SIZE}" \
  --n_sft_train 256 --n_sft_val 64 --n_sft_test 64 \
  --n_dpo_train 256 --n_dpo_val 64 --n_dpo_test 64
DATASET_RC=$?
set -e
if [[ "${DATASET_RC}" -ne 0 ]]; then
  echo "[part2] WARN: dataset generation failed (rc=${DATASET_RC}); continuing anyway."
fi

echo "[part2] pretraining base transformer (TinyStories subset) -> ${CKPT_DIR}"
if [[ -n "${BASE_CKPT_OVERRIDE}" ]]; then
  echo "[part2] using BASE_CKPT_OVERRIDE=${BASE_CKPT_OVERRIDE}"
  cp -f "${BASE_CKPT_OVERRIDE}" "${CKPT_DIR}/transformer_final.pt"
else
  pretrain_cmd() {
    local hf_endpoint="${1:-}"
    if [[ -n "${hf_endpoint}" ]]; then
      echo "[part2] pretrain using HF_ENDPOINT=${hf_endpoint}"
    fi
    HF_ENDPOINT="${hf_endpoint}" HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY}" \
      "${PY}" "${ROOT_DIR}/pico-llm/pico-llm.py" \
        --models transformer \
        --device_id "${DEVICE}" \
        --num_epochs 9999 \
        --max_train_seconds "${PRETRAIN_MAX_SECONDS}" \
        --progress_interval_seconds "${PROGRESS_INTERVAL_SECONDS}" \
        --batch_size 16 \
        --learning_rate 3e-4 \
        --train_subset_size "${PRETRAIN_SUBSET_SIZE}" \
        --block_size 512 \
        --transformer_d_model 256 \
        --transformer_num_heads 8 \
        --transformer_num_layers 4 \
        --transformer_mlp_ratio 4.0 \
        --transformer_dropout 0.1 \
        --max_steps_per_epoch "${PRETRAIN_MAX_STEPS}" \
        --val_split 0.0 \
        --log_interval_steps 200 \
        --sample_interval_seconds 600 \
        --tinystories_weight 1.0 \
        --checkpoint_dir "${CKPT_DIR}" \
        --prompt "Once upon a"
  }

  set +e
  if [[ -n "${HF_ENDPOINT_TINY}" ]]; then
    pretrain_cmd "${HF_ENDPOINT_TINY}"
    PRETRAIN_RC=$?
  else
    pretrain_cmd ""
    PRETRAIN_RC=$?
    if [[ "${PRETRAIN_RC}" -ne 0 ]]; then
      echo "[part2] WARN: pretrain failed; retrying with HF mirror: ${HF_ENDPOINT_FALLBACK}"
      pretrain_cmd "${HF_ENDPOINT_FALLBACK}"
      PRETRAIN_RC=$?
    fi
  fi
  set -e

  if [[ "${PRETRAIN_RC}" -ne 0 ]]; then
    echo "[part2] ERROR: pretrain failed after retries (rc=${PRETRAIN_RC})."
    echo "[part2] Tip: set HF_ENDPOINT_TINY=${HF_ENDPOINT_FALLBACK} or provide BASE_CKPT_OVERRIDE to skip pretrain."
    exit "${PRETRAIN_RC}"
  fi
fi

BASE_CKPT="${CKPT_DIR}/transformer_final.pt"
SFT_CKPT="${CKPT_DIR}/transformer_sft.pt"
DPO_CKPT="${CKPT_DIR}/transformer_dpo.pt"

echo "[part2] SFT -> ${SFT_CKPT}"
if [[ -s "${DATA_DIR}/sft_train.jsonl" && -s "${DATA_DIR}/sft_val.jsonl" ]]; then
  PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.train_sft \
    --base_checkpoint "${BASE_CKPT}" \
    --train_jsonl "${DATA_DIR}/sft_train.jsonl" \
    --val_jsonl "${DATA_DIR}/sft_val.jsonl" \
    --out_checkpoint "${SFT_CKPT}" \
    --device "${DEVICE}" \
    --epochs "${SFT_EPOCHS}" \
    --max_train_seconds "${SFT_MAX_SECONDS}" \
    --progress_interval_seconds "${PROGRESS_INTERVAL_SECONDS}" \
    --eval_every 5 \
    --batch_size 8 \
    --lr 5e-5 \
    --max_tokens 256 \
    --monitor_dpo_jsonl "${DATA_DIR}/dpo_val.jsonl" \
    --log_jsonl "${RUN_DIR}/logs_sft.jsonl"
else
  echo "[part2] WARN: missing/empty SFT JSONL; skipping SFT stage."
  cp -f "${BASE_CKPT}" "${SFT_CKPT}"
fi

echo "[part2] DPO -> ${DPO_CKPT}"
if [[ -s "${DATA_DIR}/dpo_train.jsonl" && -s "${DATA_DIR}/dpo_val.jsonl" ]]; then
  PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.train_dpo \
    --policy_checkpoint "${SFT_CKPT}" \
    --ref_checkpoint "${SFT_CKPT}" \
    --train_jsonl "${DATA_DIR}/dpo_train.jsonl" \
    --val_jsonl "${DATA_DIR}/dpo_val.jsonl" \
    --out_checkpoint "${DPO_CKPT}" \
    --device "${DEVICE}" \
    --epochs "${DPO_EPOCHS}" \
    --max_train_seconds "${DPO_MAX_SECONDS}" \
    --progress_interval_seconds "${PROGRESS_INTERVAL_SECONDS}" \
    --eval_every 5 \
    --batch_size 4 \
    --lr 2e-5 \
    --beta 0.1 \
    --label_smoothing 0.05 \
    --max_tokens 256 \
    --log_jsonl "${RUN_DIR}/logs_dpo.jsonl"
else
  echo "[part2] WARN: missing/empty DPO JSONL; skipping DPO stage."
  cp -f "${SFT_CKPT}" "${DPO_CKPT}"
fi

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
if [[ -s "${RUN_DIR}/logs_sft.jsonl" && -s "${RUN_DIR}/logs_dpo.jsonl" ]]; then
  PYTHONPATH="${ROOT_DIR}/pico-llm" "${PY}" -m part2.plot_curves \
    --sft_log_jsonl "${RUN_DIR}/logs_sft.jsonl" \
    --dpo_log_jsonl "${RUN_DIR}/logs_dpo.jsonl" \
    --out_png "${PLOTS_DIR}/curves.png"
else
  echo "[part2] WARN: missing logs JSONL; skipping plot."
fi

echo "[part2] done: ${RUN_DIR}"
