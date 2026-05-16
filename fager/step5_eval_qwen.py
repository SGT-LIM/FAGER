#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

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


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def remove_think_blocks(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL | re.IGNORECASE).strip()


def safe_json_load(s: str) -> Any:
    s = remove_think_blocks(strip_code_fences(s))
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


def resolve_dtype(name: str) -> torch.dtype:
    x = str(name).lower()
    if x == "float16":
        return torch.float16
    if x == "float32":
        return torch.float32
    return torch.bfloat16

def load_done_indices_from_jsonl(path: str) -> set[int]:
    done = set()
    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("index", None)
                if idx is not None:
                    done.add(int(idx))
            except Exception:
                continue
    return done

@dataclass
class EvalConfig:
    vl_model_name: str = "Qwen3-VL-8B-Instruct"
    text_model_name: str = "Qwen/Qwen3.5-9B"
    max_retries: int = 2
    chunk_size: int = 12
    level_chunk_trigger: int = 12
    regenerate_threshold: float = 20.0
    keep_threshold: float = 95.0
    vl_max_new_tokens: int = 256
    text_max_new_tokens: int = 256
    temperature_vl: float = 0.2
    top_p_vl: float = 0.8
    top_k_vl: int = 20
    temperature_text: float = 0.2
    top_p_text: float = 0.8
    top_k_text: int = 20
    attn_implementation: str = "flash_attention_2"
    dtype: str = "bfloat16"
    device_map: str = "auto"


class QwenBackends:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.dtype = resolve_dtype(cfg.dtype)
        self.vl_model = None
        self.vl_processor = None
        self.text_model = None
        self.text_tokenizer = None

    def load_vl(self) -> None:
        if self.vl_model is not None:
            return
        kwargs = {"torch_dtype": self.dtype, "device_map": self.cfg.device_map}
        if self.cfg.attn_implementation:
            kwargs["attn_implementation"] = self.cfg.attn_implementation

        self.vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.vl_model_name,
            trust_remote_code=True,
            **kwargs,
        )
        self.vl_processor = AutoProcessor.from_pretrained(
            self.cfg.vl_model_name,
            trust_remote_code=True,
        )

        assert self.vl_processor is not None, "vl_processor is None"

    def load_text(self) -> None:
        if self.text_model is not None:
            return
        kwargs = {"torch_dtype": self.dtype, "device_map": self.cfg.device_map}
        if self.cfg.attn_implementation:
            kwargs["attn_implementation"] = self.cfg.attn_implementation
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.text_model_name, **kwargs
        )

    @torch.inference_mode()
    def generate_vl_json(self, image_path: str, user_text: str) -> str:
        self.load_vl()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }]
        assert self.vl_processor is not None, "self.vl_processor is None"
        # 1) 먼저 prompt text만 생성
        prompt_text = self.vl_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2) 그 다음 processor로 text+image를 별도로 인코딩
        inputs = self.vl_processor(
            text=[prompt_text],
            images=[image_path],
            return_tensors="pt",
            padding=True,
        )

        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.vl_model.device) for k, v in inputs.items()}

        generated_ids = self.vl_model.generate(
            **inputs,
            max_new_tokens=self.cfg.vl_max_new_tokens,
            do_sample=False,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, prompt_len:]

        output_text = self.vl_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""

    @torch.inference_mode()
    def generate_text_json(self, system_text: str, user_text: str) -> str:
        self.load_text()
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        text = self.text_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.text_tokenizer([text], return_tensors="pt").to(self.text_model.device)
        generated_ids = self.text_model.generate(
            **inputs,
            max_new_tokens=self.cfg.text_max_new_tokens,
            do_sample=False,
            temperature=self.cfg.temperature_text,
            top_p=self.cfg.top_p_text,
            top_k=self.cfg.top_k_text,
        )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""


class MLLMEvaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.backends = QwenBackends(cfg)

    def evaluate_one_image(
        self,
        image_path: str,
        qa_df_for_prompt: pd.DataFrame,
        index: int,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        lvl1 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_1"].copy()
        lvl2 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_2"].copy()
        lvl3 = qa_df_for_prompt[qa_df_for_prompt["level"] == "level_3"].copy()

        original_prompt = ""
        if "prompt" in qa_df_for_prompt.columns and len(qa_df_for_prompt) > 0:
            original_prompt = str(qa_df_for_prompt.iloc[0]["prompt"]).strip()

        # 1) level 1만 먼저 평가
        per_q_lvl1 = self._eval_one_level(image_path, lvl1)
        lvl1_score = self._aggregate_score(per_q_lvl1)

        # 2) level 1이 너무 낮으면 바로 regenerate
        if lvl1_score <= self.cfg.regenerate_threshold:
            decision = "regenerate"
            per_q_all = per_q_lvl1
            evaluated_levels = ["level_1"]
            skipped_levels = [
                lv for lv, df in [("level_2", lvl2), ("level_3", lvl3)] if len(df) > 0
            ]
            level_scores = {"level_1": lvl1_score}
            overall = lvl1_score

            regenerate_prompt = self._generate_regenerate_prompt(
                original_prompt=original_prompt,
                per_question=per_q_lvl1,
                level1_score=lvl1_score,
            )
            edit_instruction = None

        else:
            # 3) level 2 + level 3를 합쳐서 한 번만 평가
            per_q_all = list(per_q_lvl1)
            evaluated_levels = ["level_1"]
            skipped_levels = []
            level_scores = {"level_1": lvl1_score}

            higher_levels = pd.concat([lvl2, lvl3], ignore_index=True)
            per_q_higher = []

            if len(higher_levels) > 0:
                per_q_higher = self._eval_one_level(image_path, higher_levels)
                per_q_all.extend(per_q_higher)

                if len(lvl2) > 0:
                    evaluated_levels.append("level_2")
                if len(lvl3) > 0:
                    evaluated_levels.append("level_3")

                # level별 score는 합쳐서 받은 결과를 다시 분리해서 계산
                per_q_lvl2 = [q for q in per_q_higher if q.get("level") == "level_2"]
                per_q_lvl3 = [q for q in per_q_higher if q.get("level") == "level_3"]

                if per_q_lvl2:
                    level_scores["level_2"] = self._aggregate_score(per_q_lvl2)
                if per_q_lvl3:
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

    def _eval_one_level(self, image_path: str, df_level: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_level is None or len(df_level) == 0:
            return []
        n = len(df_level)
        if n < self.cfg.level_chunk_trigger:
            return self._eval_rows_chunk(image_path, df_level)
        rows = df_level.to_dict(orient="records")
        out: List[Dict[str, Any]] = []
        cs = max(1, int(self.cfg.chunk_size))
        for start in range(0, n, cs):
            chunk = rows[start:start + cs]
            chunk_df = pd.DataFrame(chunk)
            out.extend(self._eval_rows_chunk(image_path, chunk_df))
        return out

    def _eval_rows_chunk(self, image_path: str, df_rows: pd.DataFrame) -> List[Dict[str, Any]]:
        if df_rows is None or len(df_rows) == 0:
            return []
        rows = df_rows.to_dict(orient="records")
        answers = self._call_mllm_answers(image_path, rows)
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

    def _call_mllm_answers(self, image_path: str, chunk_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        user_text = (
            "You are a STRICT visual fact evaluator for images.\n"
            "You MUST base judgments ONLY on what is visible in the provided image.\n"
            "Do NOT use any prior knowledge.\n"
            "If the needed evidence is not visually verifiable because of occlusion, crop, blur, small size, "
            "or viewpoint, answer 'unknown'.\n\n"
            "Return STRICT JSON ONLY in this schema:\n"
            '{ "answers": [ {"id":0,"answer":"yes|no|unknown","rationale":"..."}, ... ] }\n\n'
            "Constraints:\n"
            "- rationale must be EXACTLY one short sentence.\n"
            "- rationale must cite visible evidence or lack of verifiability.\n"
            "- no markdown, no code fences, no extra keys.\n\n"
            f"QUESTIONS_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        last_err: Optional[Exception] = None
        for _ in range(self.cfg.max_retries + 1):
            try:
                content = self.backends.generate_vl_json(image_path, user_text)
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

        system_text = (
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
            '{ "constraints": "..." }'
        )

        user_obj = {
            "original_prompt": original_prompt,
            "level1_score": round(level1_score, 2),
            "failed_items": [compact(x) for x in failures][:20],
            "unknown_items": [compact(x) for x in unknowns][:10],
        }

        last_err: Optional[Exception] = None
        user_text = json.dumps(user_obj, ensure_ascii=False)
        for _ in range(self.cfg.max_retries + 1):
            try:
                content = self.backends.generate_text_json(system_text, user_text)
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

        system_text = (
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
            '{ "edit_instruction": "..." }'
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
        for _ in range(self.cfg.max_retries + 1):
            try:
                content = self.backends.generate_text_json(system_text, user_text)
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
    ap = argparse.ArgumentParser(description="Qwen-based QA evaluator with regenerate/edit/keep logic.")
    ap.add_argument("--qa_csv", type=str, required=True, help="Path to the QA CSV from step 4 (columns: index, prompt, level, category, importance, question, gold_answer, answer_space, source_fact).")
    ap.add_argument("--image_dir", type=str, required=True, help="Directory of images to evaluate.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--vl_model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--text_model_name", type=str, default="Qwen/Qwen3.5-9B")
    ap.add_argument("--regenerate_threshold", type=float, default=20.0, help="If level_1_score <= this, decision=regenerate and skip higher levels.")
    ap.add_argument("--keep_threshold", type=float, default=95.0, help="If level_1 passes and overall_score >= this, decision=keep; else edit.")
    ap.add_argument("--max_retries", type=int, default=1, help="Retries on invalid JSON.")
    ap.add_argument("--chunk_size", type=int, default=20, help="Chunk size when a level has many questions.")
    ap.add_argument("--level_chunk_trigger", type=int, default=20, help="If a level has >= this many questions, split into chunks.")
    ap.add_argument("--index_regex", type=str, default=r"(\d+)", help="Regex to extract index from filename.")
    ap.add_argument("--vl_max_new_tokens", type=int, default=256)
    ap.add_argument("--text_max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature_vl", type=float, default=0.2)
    ap.add_argument("--top_p_vl", type=float, default=0.8)
    ap.add_argument("--top_k_vl", type=int, default=20)
    ap.add_argument("--temperature_text", type=float, default=0.2)
    ap.add_argument("--top_p_text", type=float, default=0.8)
    ap.add_argument("--top_k_text", type=int, default=20)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", type=str, default="auto")
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

    cfg = EvalConfig(
        vl_model_name=args.vl_model_name,
        text_model_name=args.text_model_name,
        max_retries=args.max_retries,
        chunk_size=max(1, int(args.chunk_size)),
        level_chunk_trigger=max(1, int(args.level_chunk_trigger)),
        regenerate_threshold=float(args.regenerate_threshold),
        keep_threshold=float(args.keep_threshold),
        vl_max_new_tokens=int(args.vl_max_new_tokens),
        text_max_new_tokens=int(args.text_max_new_tokens),
        temperature_vl=float(args.temperature_vl),
        top_p_vl=float(args.top_p_vl),
        top_k_vl=int(args.top_k_vl),
        temperature_text=float(args.temperature_text),
        top_p_text=float(args.top_p_text),
        top_k_text=int(args.top_k_text),
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    evaluator = MLLMEvaluator(cfg=cfg)
    images = list_images(args.image_dir)
    if not images:
        print(f"[ERROR] No images found in: {args.image_dir}", file=sys.stderr)
        return 2

    details_path = os.path.join(args.out_dir, "details.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.csv")

    done_files = set()
    existing_summary_rows: List[Dict[str, Any]] = []

    # 1) summary.csv 기반
    if os.path.exists(summary_path):
        try:
            prev = pd.read_csv(summary_path)
            existing_summary_rows = prev.to_dict(orient="records")
            if "index" in prev.columns:
                done_files |= set(prev["index"].dropna().astype(int).tolist())
            print(f"[INFO] Loaded {len(done_files)} indices from summary.csv")
        except Exception as e:
            print(f"[WARN] summary.csv read failed: {e}", file=sys.stderr)

    # 2) details.jsonl 기반 (추가!)
    jsonl_done = load_done_indices_from_jsonl(details_path)
    done_files |= jsonl_done
    print(f"[INFO] Loaded {len(jsonl_done)} indices from details.jsonl")

    print(f"[INFO] Total done indices: {len(done_files)}")

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
