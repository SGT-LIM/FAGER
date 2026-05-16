#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py
QA-based MLLM evaluator for T2I factuality.

Inputs:
  1) QA CSV with columns:
     index,prompt,level,category,importance,question,gold_answer,answer_space,source_fact
  2) Image directory containing generated images.

Behavior:
  - Per-question answer: yes|no|unknown + one-line rationale
  - MUST use ONLY visual evidence in the image (NO prior knowledge).
  - Scoring: yes=1, no=0, unknown=0.5
  - Score (%) = (sum(scores) / #questions) * 100

Decision logic:
  1) Evaluate level_1 first.
  2) If level_1_score <= regenerate_threshold:
       -> decision = "regenerate"
       -> skip level_2/3
       -> generate regenerate_prompt based ONLY on QA yes/no/unknown results
       -> final format: original_prompt + " @ " + constraints
  3) Else:
       -> evaluate level_2/3
       -> if overall_score >= keep_threshold:
             decision = "keep"
          else:
             decision = "edit"
             generate edit_instruction based ONLY on QA yes/no/unknown results

Outputs:
  - summary.csv
  - details.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

# 변경
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError("Please install google-genai (pip install google-genai).") from e


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LEVELS = ["level_1", "level_2", "level_3"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_images(image_dir: str) -> List[str]:
    out = []
    for fn in sorted(os.listdir(image_dir)):
        p = os.path.join(image_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in IMAGE_EXTS:
            out.append(p)
    return out


def guess_mime(image_path: str) -> str:
    ext = os.path.splitext(image_path.lower())[1]
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "image/png"


def b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def safe_json_load(s: str) -> Any:
    s = strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def extract_index(filename: str, regex: str) -> Optional[int]:
    m = re.search(regex, filename)
    if not m:
        return None
    g = m.group(1) if (m.lastindex and m.lastindex >= 1) else m.group(0)
    try:
        return int(g)
    except Exception:
        return None


def score_answer(ans: str) -> float:
    a = (ans or "").strip().lower()
    if a == "yes":
        return 1.0
    if a == "no":
        return 0.0
    return 0.5


def percent(total: float, n: int) -> float:
    return 0.0 if n <= 0 else (total / n) * 100.0


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


@dataclass
class EvalConfig:
    model: str = "gemini-2.5-flash"
    reasoning_effort_answers: str = "none"
    reasoning_effort_rewrite: str = "none"
    max_retries: int = 2
    chunk_size: int = 12
    level_chunk_trigger: int = 12
    regenerate_threshold: float = 20.0
    keep_threshold: float = 95.0


# 변경
class MLLMEvaluator:
    def __init__(self, cfg: EvalConfig, api_key: Optional[str] = None):
        self.cfg = cfg
        self.client = genai.Client(api_key=api_key)

    def evaluate_one_image(
        self,
        image_path: str,
        qa_df_for_prompt: pd.DataFrame,
        index: int,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        base64_img = b64_image(image_path)
        mime = guess_mime(image_path)

        lvl1 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_1"].copy()
        lvl2 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_2"].copy()
        lvl3 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_3"].copy()

        original_prompt = ""
        if "prompt" in qa_df_for_prompt.columns and len(qa_df_for_prompt) > 0:
            original_prompt = str(qa_df_for_prompt.iloc[0]["prompt"]).strip()

        per_q_lvl1 = self._eval_one_level(base64_img, mime, lvl1)
        lvl1_score = self._aggregate_score(per_q_lvl1)

        if lvl1_score <= self.cfg.regenerate_threshold:
            decision = "regenerate"
            per_q_all = per_q_lvl1
            evaluated_levels = ["level_1"]
            skipped_levels = [lv for lv, df in [("level_2", lvl2), ("level_3", lvl3)] if len(df) > 0]
            level_scores = {"level_1": lvl1_score}
            overall = lvl1_score

            regenerate_prompt = self._generate_regenerate_prompt(
                original_prompt=original_prompt,
                per_question=per_q_lvl1,
                level1_score=lvl1_score,
            )
            edit_instruction = None

        else:
            per_q_all = list(per_q_lvl1)
            evaluated_levels = ["level_1"]
            skipped_levels = []
            level_scores = {"level_1": lvl1_score}

            if len(lvl2) > 0:
                per_q_lvl2 = self._eval_one_level(base64_img, mime, lvl2)
                per_q_all.extend(per_q_lvl2)
                evaluated_levels.append("level_2")
                level_scores["level_2"] = self._aggregate_score(per_q_lvl2)

            if len(lvl3) > 0:
                per_q_lvl3 = self._eval_one_level(base64_img, mime, lvl3)
                per_q_all.extend(per_q_lvl3)
                evaluated_levels.append("level_3")
                level_scores["level_3"] = self._aggregate_score(per_q_lvl3)

            overall = self._aggregate_score(per_q_all)

            if overall >= self.cfg.keep_threshold:
                decision = "keep"
                regenerate_prompt = None
                edit_instruction = None
            else:
                decision = "edit"
                regenerate_prompt = None
                edit_instruction = self._generate_edit_instruction(
                    original_prompt=original_prompt,
                    per_question=per_q_all,
                    level1_score=lvl1_score,
                    overall_score=overall,
                )

        counts = self._count_answers(per_q_all)

        return {
            "index": index,
            "image_id": image_id,
            "image_path": image_path,
            "original_prompt": original_prompt,
            "gating": {
                "evaluated_levels": evaluated_levels,
                "skipped_levels": skipped_levels,
                "level_1_score": round(lvl1_score, 2),
                "decision": decision,
                "regenerate_threshold": round(self.cfg.regenerate_threshold, 2),
                "keep_threshold": round(self.cfg.keep_threshold, 2),
            },
            "scores": {
                "overall_score": round(overall, 2),
                "level_scores": {k: round(v, 2) for k, v in level_scores.items()},
            },
            "answer_counts": counts,
            "per_question": per_q_all,
            "next_step": {
                "decision": decision,
                "regenerate_prompt": regenerate_prompt,
                "edit_instruction": edit_instruction,
            },
            "improvement_suggestions": (
                [regenerate_prompt] if regenerate_prompt else
                [edit_instruction] if edit_instruction else
                []
            ),
        }

    def _count_answers(self, per_question: List[Dict[str, Any]]) -> Dict[str, int]:
        c = {"yes": 0, "no": 0, "unknown": 0, "total": 0}
        for x in per_question:
            a = str(x.get("answer", "unknown")).strip().lower()
            if a not in ("yes", "no", "unknown"):
                a = "unknown"
            c[a] += 1
            c["total"] += 1
        return c

    def _aggregate_score(self, per_question: List[Dict[str, Any]]) -> float:
        if not per_question:
            return 0.0
        total = sum(float(x.get("score", 0.5)) for x in per_question)
        return percent(total, len(per_question))

    def _eval_one_level(self, base64_img: str, mime: str, df_level: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_level is None or len(df_level) == 0:
            return []

        n = len(df_level)
        if n < self.cfg.level_chunk_trigger:
            return self._eval_rows_chunk(base64_img, mime, df_level)

        rows = df_level.to_dict(orient="records")
        out: List[Dict[str, Any]] = []
        cs = max(1, int(self.cfg.chunk_size))

        for start in range(0, n, cs):
            chunk = rows[start:start + cs]
            chunk_df = pd.DataFrame(chunk)
            out.extend(self._eval_rows_chunk(base64_img, mime, chunk_df))
        return out

    def _eval_rows_chunk(self, base64_img: str, mime: str, df_rows: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_rows is None or len(df_rows) == 0:
            return []

        rows = df_rows.to_dict(orient="records")
        answers = self._call_mllm_answers(base64_img, mime, rows)

        if len(answers) != len(rows):
            pad = [{"answer": "unknown", "rationale": "Output length mismatch."}] * len(rows)
            answers = (answers + pad)[:len(rows)]

        out: List[Dict[str, Any]] = []
        for r, a in zip(rows, answers):
            ans = str(a.get("answer", "unknown")).strip().lower()
            if ans not in {"yes", "no", "unknown"}:
                ans = "unknown"
            rationale = str(a.get("rationale", "")).strip() or "No clear visual evidence to decide."

            out.append(
                {
                    "level": r.get("level"),
                    "category": r.get("category"),
                    "importance": r.get("importance"),
                    "question": r.get("question"),
                    "answer": ans,
                    "score": score_answer(ans),
                    "rationale": rationale,
                    "source_fact": r.get("source_fact"),
                }
            )
        return out

    def _call_mllm_answers(self, base64_img: str, mime: str, chunk_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = []
        for i, r in enumerate(chunk_rows):
            payload.append(
                {
                    "id": i,
                    "question": r.get("question", ""),
                    "level": r.get("level", ""),
                    "category": r.get("category", ""),
                    "answer_space": "yes|no|unknown",
                }
            )

        instructions = (
            "You are a STRICT visual fact evaluator for images.\n"
            "You MUST base judgments ONLY on what is visible in the provided image.\n"
            "Do NOT use any prior knowledge.\n"
            "If the needed evidence is not visually verifiable because of occlusion, crop, blur, small size, "
            "or viewpoint, answer 'unknown'.\n\n"
            "Return STRICT JSON ONLY in this schema:\n"
            "{ \"answers\": [ {\"id\":0,\"answer\":\"yes|no|unknown\",\"rationale\":\"...\"}, ... ] }\n\n"
            "Constraints:\n"
            "- rationale must be EXACTLY one short sentence.\n"
            "- rationale must cite visible evidence or lack of verifiability.\n"
            "- no markdown, no code fences, no extra keys.\n"
        )

        user_text = (
            "Evaluate the following questions against the image.\n"
            "Use ONLY visual evidence. If not visually verifiable, answer unknown.\n\n"
            f"QUESTIONS_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_err: Optional[Exception] = None
        # 변경 (_call_mllm_answers 내부)
        for _ in range(self.cfg.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model,
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(base64_img),
                            mime_type=mime,
                        ),
                        instructions + "\n\n" + user_text,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                )
                content = resp.text or ""
                data = safe_json_load(content)
                answers = data.get("answers", [])
                answers = sorted(answers, key=lambda x: int(x.get("id", 0)))
                return [
                    {"answer": x.get("answer", "unknown"), "rationale": x.get("rationale", "")}
                    for x in answers
                ]
            except Exception as e:
                last_err = e
                user_text += "\n\nPrevious output was invalid. Return ONLY valid JSON."

        return [{"answer": "unknown", "rationale": f"Evaluator error: {str(last_err)[:120]}"} for _ in chunk_rows]

    def _generate_regenerate_prompt(
        self,
        original_prompt: str,
        per_question: List[Dict[str, Any]],
        level1_score: float,
    ) -> str:
        failures = [x for x in per_question if str(x.get("answer", "")).lower() == "no"]
        unknowns = [x for x in per_question if str(x.get("answer", "")).lower() == "unknown"]

        def compact(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "category": x.get("category"),
                "question": x.get("question"),
                "answer": x.get("answer"),
                "rationale": x.get("rationale"),
                "source_fact": x.get("source_fact"),
            }

        instructions = (
            "You create SHORT factual constraints for text-to-image regeneration.\n"
            "The constraints MUST be based ONLY on the provided QA results (yes/no/unknown, rationale, source_fact).\n"
            "Do NOT use the image directly. Do NOT invent new facts.\n\n"
            "Goal:\n"
            "- Output only essential factual constraints that should be appended to the original prompt.\n\n"
            "Include only:\n"
            "- required objects\n"
            "- counts\n"
            "- visual attributes\n"
            "- relations\n"
            "- necessary viewpoint/composition constraints if unknown answers suggest visibility issues\n\n"
            "Exclude:\n"
            "- style words\n"
            "- aesthetic words\n"
            "- realism/quality adjectives\n"
            "- photorealistic, realistic, beautiful, cinematic, detailed, highly detailed, masterpiece, 4k, aesthetic\n\n"
            "Keep it short.\n"
            "Return STRICT JSON only:\n"
            '{ "constraints": "..." }\n'
        )

        user_obj = {
            "original_prompt": original_prompt,
            "level1_score": round(level1_score, 2),
            "failed_items": [compact(x) for x in failures][:20],
            "unknown_items": [compact(x) for x in unknowns][:10],
        }

        last_err: Optional[Exception] = None
        user_text = json.dumps(user_obj, ensure_ascii=False)

        # 변경 (_generate_regenerate_prompt 내부)
        for _ in range(self.cfg.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model,
                    contents=[instructions + "\n\n" + user_text],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                content = resp.text or ""
                data = safe_json_load(content)
                constraints = str(data.get("constraints", "")).strip()
                constraints = self._sanitize_short_text(constraints)
                if constraints:
                    if original_prompt.strip():
                        return f"{original_prompt.strip()} @ {constraints}"
                    return constraints
            except Exception as e:
                last_err = e
                user_text += "\n\nReturn ONLY valid JSON."

        facts = []
        for x in failures[:8]:
            sf = str(x.get("source_fact", "")).strip()
            if sf:
                facts.append(sf)
        for x in unknowns[:4]:
            sf = str(x.get("source_fact", "")).strip()
            if sf:
                facts.append(sf)

        facts = dedup_keep_order(facts)
        constraint_text = self._sanitize_short_text("; ".join(facts[:8]))

        if original_prompt.strip() and constraint_text:
            return f"{original_prompt.strip()} @ {constraint_text}"
        if original_prompt.strip():
            return original_prompt.strip()
        return constraint_text or f"Regeneration constraint generation error: {str(last_err)[:120]}"

    def _generate_edit_instruction(
        self,
        original_prompt: str,
        per_question: List[Dict[str, Any]],
        level1_score: float,
        overall_score: float,
    ) -> str:
        failures = [x for x in per_question if str(x.get("answer", "")).lower() == "no"]
        unknowns = [x for x in per_question if str(x.get("answer", "")).lower() == "unknown"]

        def compact(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "level": x.get("level"),
                "category": x.get("category"),
                "question": x.get("question"),
                "answer": x.get("answer"),
                "rationale": x.get("rationale"),
                "source_fact": x.get("source_fact"),
            }

        instructions = (
            "You create a SHORT image editing instruction.\n"
            "The instruction MUST be based ONLY on the provided QA results (yes/no/unknown, rationale, source_fact).\n"
            "Do NOT use the image directly. Do NOT invent new facts.\n\n"
            "Goal:\n"
            "- Specify only the factual edits needed.\n"
            "- Preserve existing correct content implicitly by editing only what is necessary.\n\n"
            "Include only:\n"
            "- required object changes\n"
            "- count fixes\n"
            "- attribute fixes\n"
            "- relation fixes\n"
            "- viewpoint/composition guidance only if needed due to unknown/visibility issues\n\n"
            "Exclude:\n"
            "- style words\n"
            "- aesthetic words\n"
            "- realism/quality adjectives\n"
            "- photorealistic, realistic, beautiful, cinematic, detailed, highly detailed, masterpiece, 4k, aesthetic\n"
            "- any mention of QA, scores, yes/no/unknown, rationale\n\n"
            "Keep it short.\n"
            "Return STRICT JSON only:\n"
            '{ "edit_instruction": "..." }\n'
        )

        user_obj = {
            "original_prompt": original_prompt,
            "level1_score": round(level1_score, 2),
            "overall_score": round(overall_score, 2),
            "failed_items": [compact(x) for x in failures][:20],
            "unknown_items": [compact(x) for x in unknowns][:10],
        }

        last_err: Optional[Exception] = None
        user_text = json.dumps(user_obj, ensure_ascii=False)

        # 변경 (_generate_edit_instruction 내부)
        for _ in range(self.cfg.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model,
                    contents=[instructions + "\n\n" + user_text],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                content = resp.text or ""
                data = safe_json_load(content)
                inst = str(data.get("edit_instruction", "")).strip()
                inst = self._sanitize_short_text(inst)
                if inst:
                    return inst
            except Exception as e:
                last_err = e
                user_text += "\n\nReturn ONLY valid JSON."

        facts = []
        for x in failures[:8]:
            sf = str(x.get("source_fact", "")).strip()
            if sf:
                facts.append(sf)
        for x in unknowns[:4]:
            sf = str(x.get("source_fact", "")).strip()
            if sf:
                facts.append(sf)

        facts = dedup_keep_order(facts)
        fallback = self._sanitize_short_text("; ".join(facts[:8]))
        return fallback or f"Edit instruction generation error: {str(last_err)[:120]}"

    def _sanitize_short_text(self, text: str) -> str:
        if not text:
            return ""

        t = str(text).strip()

        banned_patterns = [
            r"\brealistic\b",
            r"\bphotorealistic\b",
            r"\bbeautiful\b",
            r"\bhigh[- ]quality\b",
            r"\bcinematic\b",
            r"\bdetailed\b",
            r"\bhighly detailed\b",
            r"\bmasterpiece\b",
            r"\baesthetic\b",
            r"\bstunning\b",
            r"\baward[- ]winning\b",
            r"\b4k\b",
            r"\bultra[- ]detailed\b",
            r"\bultra detailed\b",
        ]
        for pat in banned_patterns:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)

        t = re.sub(r"\s*[,;]\s*[,;]\s*", "; ", t)
        t = re.sub(r"\s{2,}", " ", t)
        t = re.sub(r"\s*;\s*;\s*", "; ", t)
        t = re.sub(r"\s*,\s*,\s*", ", ", t)
        t = re.sub(r"\s*([,;])\s*", r"\1 ", t)
        t = re.sub(r"\s{2,}", " ", t)
        t = t.strip(" ,;.")

        return t

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="QA-based MLLM evaluator with regenerate/edit/keep logic."
    )
    ap.add_argument(
        "--qa_csv",
        type=str,
        required=True,
        help="Path to the QA CSV from step 4 (columns: index, prompt, level, category, importance, question, gold_answer, answer_space, source_fact).",
    )
    ap.add_argument("--image_dir", type=str, required=True, help="Directory of images to evaluate.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    # ap.add_argument("--model", type=str, default="gpt-5.4-mini", help="Vision-capable model name.")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash", help="Vision-capable model name.")

    ap.add_argument(
        "--regenerate_threshold",
        type=float,
        default=20.0,
        help="If level_1_score <= this, decision=regenerate and skip higher levels.",
    )
    ap.add_argument(
        "--keep_threshold",
        type=float,
        default=95.0,
        help="If level_1 passes and overall_score >= this, decision=keep; else edit.",
    )

    ap.add_argument("--max_retries", type=int, default=2, help="Retries on invalid JSON.")
    ap.add_argument("--chunk_size", type=int, default=12, help="Chunk size when a level has many questions.")
    ap.add_argument("--level_chunk_trigger", type=int, default=12, help="If a level has >= this many questions, split into chunks.")
    ap.add_argument(
        "--index_regex",
        type=str,
        default=r"(\d+)",
        help="Regex to extract index from filename.",
    )
    ap.add_argument("--api_key", type=str, default=None, help="OpenAI API key.")
    ap.add_argument("--reasoning_effort_answers", type=str, default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--reasoning_effort_rewrite", type=str, default="none", choices=["none", "low", "medium", "high"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)

    qa_df = pd.read_csv(args.qa_csv)

    required_cols = [
        "index",
        "prompt",
        "level",
        "category",
        "importance",
        "question",
        "gold_answer",
        "answer_space",
        "source_fact",
    ]
    for c in required_cols:
        if c not in qa_df.columns:
            print(f"[ERROR] QA CSV missing required column: {c}", file=sys.stderr)
            return 2

    qa_df["index"] = qa_df["index"].astype(int)
    qa_df["level"] = qa_df["level"].astype(str)

    grouped = {int(k): v.copy() for k, v in qa_df.groupby("index")}

    effective_regen_threshold = float(args.regenerate_threshold)

    cfg = EvalConfig(
        model=args.model,
        max_retries=args.max_retries,
        chunk_size=max(1, int(args.chunk_size)),
        level_chunk_trigger=max(1, int(args.level_chunk_trigger)),
        regenerate_threshold=effective_regen_threshold,
        keep_threshold=float(args.keep_threshold),
        reasoning_effort_answers=args.reasoning_effort_answers,
        reasoning_effort_rewrite=args.reasoning_effort_rewrite,
    )

    evaluator = MLLMEvaluator(cfg=cfg, api_key=args.api_key)

    images = list_images(args.image_dir)
    if not images:
        print(f"[ERROR] No images found in: {args.image_dir}", file=sys.stderr)
        return 2

    details_path = os.path.join(args.out_dir, "details.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.csv")

    done_files = set()
    existing_summary_rows: List[Dict[str, Any]] = []

    if os.path.exists(summary_path):
        try:
            prev = pd.read_csv(summary_path)
            existing_summary_rows = prev.to_dict(orient="records")
            if "index" in prev.columns:
                done_files = set(prev["index"].dropna().astype(int).tolist())
            print(f"[INFO] Resume: loaded {len(done_files)} done indices from {summary_path}")
        except Exception as e:
            print(f"[WARN] Could not read existing summary.csv for resume: {e}", file=sys.stderr)

    new_summary_rows: List[Dict[str, Any]] = []
    skipped = 0

    with open(details_path, "a", encoding="utf-8") as fjsonl:
        for img_path in images:
            fn = os.path.basename(img_path)
            pidx = extract_index(fn, args.index_regex)

            if pidx is None:
                print(f"[WARN] Skipping (cannot extract index): {fn}", file=sys.stderr)
                skipped += 1
                continue
            if pidx not in grouped:
                print(f"[WARN] Skipping (no QA rows for index={pidx}): {fn}", file=sys.stderr)
                skipped += 1
                continue
            if pidx in done_files:
                print(f"[SKIP] already evaluated: {fn}")
                continue

            qa_for_prompt = grouped[pidx]

            try:
                result = evaluator.evaluate_one_image(
                    image_path=img_path,
                    qa_df_for_prompt=qa_for_prompt,
                    index=pidx,
                    image_id=fn,
                )
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {fn}: {e}", file=sys.stderr)
                skipped += 1
                continue

            fjsonl.write(json.dumps(result, ensure_ascii=False) + "\n")

            counts = result.get("answer_counts", {})
            lvl_scores = result.get("scores", {}).get("level_scores", {})
            next_step = result.get("next_step", {})

            row = {
                "image_file": fn,
                "index": result.get("index"),
                "decision": result.get("gating", {}).get("decision"),
                "level_1_score": lvl_scores.get("level_1"),
                "level_2_score": lvl_scores.get("level_2", None),
                "level_3_score": lvl_scores.get("level_3", None),
                "overall_score": result.get("scores", {}).get("overall_score"),
                "n_total": counts.get("total", 0),
                "n_yes": counts.get("yes", 0),
                "n_no": counts.get("no", 0),
                "n_unknown": counts.get("unknown", 0),
                "regenerate_prompt": next_step.get("regenerate_prompt"),
                "edit_instruction": next_step.get("edit_instruction"),
            }
            new_summary_rows.append(row)
            done_files.add(pidx)

            print(
                f"[OK] {fn} | index={pidx} | level1={row['level_1_score']} | "
                f"overall={row['overall_score']} | decision={row['decision']}"
            )

    all_rows = existing_summary_rows + new_summary_rows
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        if "index" in df_out.columns:
            df_out = df_out.drop_duplicates(subset=["index"], keep="last")
        df_out.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=[
            "image_file", "index", "decision", "level_1_score", "level_2_score",
            "level_3_score", "overall_score", "n_total", "n_yes", "n_no", "n_unknown",
            "regenerate_prompt", "edit_instruction"
        ]).to_csv(summary_path, index=False)

    print(f"\n[DONE] Wrote:\n  - {summary_path}\n  - {details_path}\n[INFO] Skipped images: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())