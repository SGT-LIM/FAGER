#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def safe_json_loads(s: str, fallback: Any) -> Any:
    try:
        return json.loads(strip_code_fences(s))
    except Exception:
        return fallback


def load_exclude_ids(exclude_csv: Optional[str], id_col: str = "index") -> Set[str]:
    exclude_ids: Set[str] = set()

    if not exclude_csv or str(exclude_csv).lower() == "none":
        return exclude_ids

    exclude_path = Path(exclude_csv)
    if not exclude_path.exists():
        raise FileNotFoundError(f"exclude_csv not found: {exclude_csv}")

    with exclude_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or id_col not in reader.fieldnames:
            raise ValueError(
                f"exclude_csv에 '{id_col}' 컬럼이 없습니다. 현재 컬럼: {reader.fieldnames}"
            )

        for row in reader:
            rid = str((row.get(id_col) or "")).strip()
            if rid:
                exclude_ids.add(rid)

    return exclude_ids


def load_done_ids(out_jsonl: Path) -> Set[str]:
    done: Set[str] = set()
    if not out_jsonl.exists():
        return done

    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("id") is not None:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def find_image_by_index(image_dir: str, sample_id: str) -> Optional[str]:
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    sid = str(sample_id).strip()
    if not sid:
        return None

    candidates: List[str] = []

    # 1 -> "1", "01", "001"
    try:
        n = int(sid)
        candidates.extend([str(n), f"{n:02d}", f"{n:03d}", f"{n:04d}", f"{n:05d}"])
    except Exception:
        candidates.append(sid)

    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    files = list(img_dir.iterdir())
    files = [p for p in files if p.is_file() and p.suffix.lower() in exts]

    # prefix match: 01_, 1_, 001_ ...
    for cand in candidates:
        for p in files:
            name = p.stem
            if name == cand or name.startswith(cand + "_") or name.startswith(cand + "-"):
                return str(p)

    # fallback: stem 첫 토큰 숫자 비교
    for p in files:
        stem = p.stem
        first_token = re.split(r"[_\- ]+", stem)[0]
        try:
            if int(first_token) == int(sid):
                return str(p)
        except Exception:
            continue

    return None


class QwenVisualFactExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        model_kwargs = {
            "device_map": "auto",
        }

        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _build_prompt(self, prompt: Optional[str]) -> str:
        prompt_text = (prompt or "").strip()

        schema = {
            "status": "ok",
            "image_summary": "short visual summary",
            "visual_elements": [
                "fact 1",
                "fact 2"
            ],
            "confidence": 0.0,
            "notes": "brief note"
        }

        instruction = (
            "You are a strict visual fact extractor.\n"
            "Look only at the image.\n"
            "Do NOT use external knowledge or product priors.\n"
            "Extract only directly visible, visually verifiable facts.\n"
            "Be conservative and concise.\n"
        )

        if prompt_text:
            instruction += (
                f"\nReference prompt:\n{prompt_text}\n"
                "This prompt is only context. Output only facts visually supported by the image.\n"
            )

        instruction += (
            "\nReturn STRICT JSON only.\n"
            "Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
            "\nRules:\n"
            "- status must be one of: ok, partial, none\n"
            "- image_summary must be one short sentence\n"
            "- visual_elements must be a list of short atomic visible facts\n"
            "- confidence must be a number between 0 and 1\n"
            "- notes must be brief\n"
            "- no markdown fences\n"
            "- keep visual_elements typically between 5 and 12 items\n"
        )
        return instruction

    @torch.inference_mode()
    def extract_from_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_path = str(image_path)
        img_path = Path(image_path)

        if not img_path.exists():
            return {
                "status": "none",
                "image_summary": "",
                "visual_elements": [],
                "confidence": 0.0,
                "notes": f"image file not found: {image_path}",
            }

        image = Image.open(img_path).convert("RGB")
        user_text = self._build_prompt(prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        fallback = {
            "status": "partial",
            "image_summary": "",
            "visual_elements": [],
            "confidence": 0.0,
            "notes": "model output was not valid JSON",
            "raw_output": output_text,
        }

        out = safe_json_loads(output_text, fallback=fallback)
        if not isinstance(out, dict):
            out = fallback

        if out.get("status") not in {"ok", "partial", "none"}:
            out["status"] = "partial"

        if not isinstance(out.get("visual_elements"), list):
            out["visual_elements"] = []

        if not isinstance(out.get("image_summary"), str):
            out["image_summary"] = ""

        try:
            out["confidence"] = float(out.get("confidence", 0.0))
        except Exception:
            out["confidence"] = 0.0

        if not isinstance(out.get("notes"), str):
            out["notes"] = ""

        out["image_path"] = image_path
        return out


def run_single(
    extractor: QwenVisualFactExtractor,
    sample_id: Optional[str],
    prompt: Optional[str],
    image_path: str,
) -> Dict[str, Any]:
    result = extractor.extract_from_image(image_path=image_path, prompt=prompt)
    return {
        "id": sample_id,
        "prompt": prompt,
        "planner": None,
        "searches": [],
        "image_path": image_path,
        "visual_fact_extraction": result,
    }


def run_batch(
    in_csv: str,
    out_jsonl: str,
    image_dir: str,
    prompt_col: Optional[str] = "prompt",
    id_col: Optional[str] = "index",
    exclude_csv: Optional[str] = None,
    limit: Optional[int] = None,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_new_tokens: int = 256,
) -> None:
    in_path = Path(in_csv)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(out_path)
    print(f"[resume] already have {len(done_ids)} ids in {out_jsonl}")

    exclude_ids = set()
    if id_col:
        exclude_ids = load_exclude_ids(exclude_csv, id_col=id_col) if exclude_csv else set()
    print(f"[exclude] loaded {len(exclude_ids)} ids from exclude csv")

    extractor = QwenVisualFactExtractor(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )

    processed = 0
    skipped_done = 0
    skipped_exclude = 0
    skipped_no_image = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)

        if not reader.fieldnames:
            raise ValueError("입력 CSV 헤더를 읽지 못했습니다.")

        if prompt_col and prompt_col not in reader.fieldnames:
            raise ValueError(
                f"입력 CSV에 prompt_col='{prompt_col}' 컬럼이 없습니다. 현재 컬럼: {reader.fieldnames}"
            )

        if id_col and id_col not in reader.fieldnames:
            raise ValueError(
                f"입력 CSV에 id_col='{id_col}' 컬럼이 없습니다. 현재 컬럼: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=1):
            if limit is not None and processed >= limit:
                break

            prompt = None
            if prompt_col:
                prompt = str((row.get(prompt_col) or "")).strip() or None

            sample_id = None
            if id_col:
                sample_id = str((row.get(id_col) or "")).strip() or str(row_idx)
            else:
                sample_id = str(row_idx)

            if sample_id in done_ids:
                skipped_done += 1
                print(f"[skip done] row={row_idx}, id={sample_id}")
                continue

            if sample_id in exclude_ids:
                skipped_exclude += 1
                print(f"[skip exclude] row={row_idx}, id={sample_id}")
                continue

            image_path = find_image_by_index(image_dir, sample_id)
            if not image_path:
                skipped_no_image += 1
                print(f"[skip no image] row={row_idx}, id={sample_id}")
                continue

            print(f"[run] row={row_idx}, id={sample_id}, image={image_path}")

            try:
                record = run_single(
                    extractor=extractor,
                    sample_id=sample_id,
                    prompt=prompt,
                    image_path=image_path,
                )
            except Exception as e:
                record = {
                    "id": sample_id,
                    "prompt": prompt,
                    "planner": None,
                    "searches": [],
                    "image_path": image_path,
                    "error": str(e),
                    "visual_fact_extraction": {},
                }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            done_ids.add(sample_id)
            processed += 1

    print(
        f"[done] processed={processed}, skipped_done={skipped_done}, "
        f"skipped_exclude={skipped_exclude}, skipped_no_image={skipped_no_image}, "
        f"out={out_jsonl}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True, help="input csv path")
    parser.add_argument("--out_jsonl", required=True, help="output jsonl path")
    parser.add_argument("--image_dir", required=True, help="directory containing images")
    parser.add_argument("--prompt_col", default="prompt", help="prompt column name")
    parser.add_argument("--id_col", default="index", help="id column name")
    parser.add_argument("--exclude_csv", default=None, help="csv path containing ids to skip")
    parser.add_argument("--limit", type=int, default=None, help="process only first N rows")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", help="HF model name")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max generation tokens")
    args = parser.parse_args()

    run_batch(
        in_csv=args.in_csv,
        out_jsonl=args.out_jsonl,
        image_dir=args.image_dir,
        prompt_col=args.prompt_col,
        id_col=args.id_col,
        exclude_csv=args.exclude_csv,
        limit=args.limit,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()