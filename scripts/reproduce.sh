#!/usr/bin/env bash
# reproduce.sh — End-to-end FAGER pipeline on a single prompt.
#
# Prerequisites:
#   pip install -r requirements.txt
#   cp .env.example .env  # fill in OPENAI_API_KEY (and optionally GOOGLE_API_KEY, HF_TOKEN)
#   python scripts/download_reference_images.py --dataset ABO
#
# Usage:
#   bash scripts/reproduce.sh
#
# All paths are relative to the repo root. Run from there:
#   cd /path/to/FAGER && bash scripts/reproduce.sh

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
DATASET="ABO"
PROMPTS_CSV="data/${DATASET}/prompts.csv"
EXCLUDE_CSV="data/${DATASET}/exclude_index.csv"
IMAGE_DIR="data/${DATASET}/selected_images"
OUT_DIR="runs/demo_${DATASET}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"   # override with a reasoning model for best results

# Load .env if present
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -o allexport && source .env && set +o allexport
fi

mkdir -p "${OUT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " FAGER demo — dataset: ${DATASET}"
echo " Output dir: ${OUT_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Fact Proposal ────────────────────────────────────────────────────
# Reads: prompts.csv (columns: index, prompt)
# Writes: step1_proposals.csv  (rubric_json per prompt)
echo ""
echo "▶ Step 1 — Fact Proposal"
python fager/step1_fact_proposal.py \
  --in_csv        "${PROMPTS_CSV}" \
  --out_csv       "${OUT_DIR}/step1_proposals.csv" \
  --exclude_csv   "${EXCLUDE_CSV}" \
  --model         "${OPENAI_MODEL}" \
  --reasoning_effort high \
  --sleep         0.2

# ── Step 2: VLM Fact Extraction (from reference images) ─────────────────────
# Reads: prompts.csv + selected_images/
# Writes: step2_vlm.jsonl  (visual_elements per image)
echo ""
echo "▶ Step 2 — VLM Fact Extraction  (requires GPU)"
python fager/step2_vlm_extraction.py \
  --in_csv        "${PROMPTS_CSV}" \
  --out_jsonl     "${OUT_DIR}/step2_vlm.jsonl" \
  --image_dir     "${IMAGE_DIR}" \
  --model_name    "Qwen/Qwen3-VL-8B-Instruct" \
  --max_new_tokens 512

# ── Step 3: Fact Verification ────────────────────────────────────────────────
# Reads: step1_proposals.csv + step2_vlm.jsonl
# Writes: step3_verified.jsonl  (verified_rubric per prompt)
echo ""
echo "▶ Step 3 — Fact Verification"
python fager/step3_verification.py \
  --in_agent1     "${OUT_DIR}/step1_proposals.csv" \
  --in_agent2     "${OUT_DIR}/step2_vlm.jsonl" \
  --out_jsonl     "${OUT_DIR}/step3_verified.jsonl" \
  --model         "${OPENAI_MODEL}" \
  --resume

# ── Step 4: QA Generation ────────────────────────────────────────────────────
# Reads: step3_verified.jsonl
# Writes: step4_QA.csv  (one row per atomic question)
echo ""
echo "▶ Step 4 — QA Generation"
python fager/step4_qa_generation.py \
  --in_jsonl          "${OUT_DIR}/step3_verified.jsonl" \
  --out_csv           "${OUT_DIR}/step4_QA.csv" \
  --model             "${OPENAI_MODEL}" \
  --importance_keep   "high,medium,low" \
  --reasoning_effort  none \
  --skip_if_exists

# ── Step 5: MLLM Evaluation ──────────────────────────────────────────────────
# Reads: step4_QA.csv + a directory of generated images to evaluate
# Writes: step5_eval/summary.csv, step5_eval/details.jsonl
#
# Replace --image_dir with the directory that contains your generated images.
# Each image filename must contain its prompt index (e.g. "1.png", "01_generated.jpg").
echo ""
echo "▶ Step 5 — MLLM Evaluation  (set --image_dir to your generated images)"
python fager/step5_eval_openai.py \
  --qa_csv        "${OUT_DIR}/step4_QA.csv" \
  --image_dir     "${IMAGE_DIR}" \
  --out_dir       "${OUT_DIR}/step5_eval" \
  --model         "${OPENAI_MODEL}" \
  --regenerate_threshold 20.0 \
  --keep_threshold       95.0

# ── Final Score ───────────────────────────────────────────────────────────────
echo ""
echo "▶ Final Score"
python fager/final_score.py --csv "${OUT_DIR}/step5_eval/summary.csv"

# ── Pairwise Accuracy (optional) ─────────────────────────────────────────────
# Compare real vs generated images using the pre-computed benchmark results.
# Uncomment and set paths once you have two summary.csv files to compare.
#
# python evaluate/pairwise_accuracy.py \
#   --real_csv  "data/${DATASET}/results/real/summary.csv" \
#   --fake_csv  "${OUT_DIR}/step5_eval/summary.csv" \
#   --scores    "overall_score,level_1_score,level_2_score,level_3_score"

echo ""
echo "✓ Done. Results in: ${OUT_DIR}"
