#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_flux_next_step.py

Reads summary.csv from the evaluator and executes:
- regenerate -> FLUX.1-dev with regenerate_prompt
- edit       -> FLUX.1-Kontext-dev with edit_instruction
- keep       -> copy original image (optional)

Expected input summary.csv columns:
  image_file,index,decision,regenerate_prompt,edit_instruction,...

Example:
python run_flux_next_step.py \
  --summary_csv /path/to/out/summary.csv \
  --image_dir /path/to/images \
  --out_dir /path/to/next_step \
  --copy_keep

Notes:
- Requires access to the gated FLUX model repos on Hugging Face.
- Make sure you ran: huggingface-cli login
- Recommended packages:
    pip install -U diffusers transformers accelerate sentencepiece safetensors pillow pandas torch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from diffusers import FluxPipeline, FluxKontextPipeline


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_.")
    return text[:max_len] if len(text) > max_len else text


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_hw_from_image(image_path: str, fallback_hw: Tuple[int, int]) -> Tuple[int, int]:
    try:
        img = Image.open(image_path)
        return img.size  # (width, height)
    except Exception:
        return fallback_hw


def round_to_flux_size(width: int, height: int, multiple: int = 16) -> Tuple[int, int]:
    width = max(multiple, int(round(width / multiple) * multiple))
    height = max(multiple, int(round(height / multiple) * multiple))
    return width, height


def maybe_truncate_prompt(prompt: str, max_chars: int) -> str:
    prompt = str(prompt or "").strip()
    if max_chars <= 0:
        return prompt
    return prompt[:max_chars].strip()


class FluxRunner:
    def __init__(
        self,
        regen_model_id: str,
        edit_model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cpu_offload: bool = False,
    ):
        self.regen_model_id = regen_model_id
        self.edit_model_id = edit_model_id
        self.device = device
        self.torch_dtype = self._resolve_dtype(dtype)
        self.cpu_offload = cpu_offload

        self._regen_pipe: Optional[FluxPipeline] = None
        self._edit_pipe: Optional[FluxKontextPipeline] = None

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        dtype = dtype.lower()
        if dtype == "float16":
            return torch.float16
        if dtype == "float32":
            return torch.float32
        return torch.bfloat16

    def _load_regen_pipe(self) -> FluxPipeline:
        if self._regen_pipe is None:
            pipe = FluxPipeline.from_pretrained(
                self.regen_model_id,
                torch_dtype=self.torch_dtype,
            )
            if self.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            self._regen_pipe = pipe
        return self._regen_pipe

    def _load_edit_pipe(self) -> FluxKontextPipeline:
        if self._edit_pipe is None:
            pipe = FluxKontextPipeline.from_pretrained(
                self.edit_model_id,
                torch_dtype=self.torch_dtype,
            )
            if self.cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)
            self._edit_pipe = pipe
        return self._edit_pipe

    @torch.inference_mode()
    def regenerate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
        seed: int,
    ) -> Image.Image:
        pipe = self._load_regen_pipe()
        generator = torch.Generator("cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return result.images[0]

    @torch.inference_mode()
    def edit(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
        seed: int,
    ) -> Image.Image:
        pipe = self._load_edit_pipe()
        generator = torch.Generator("cpu").manual_seed(seed)
        result = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        )
        return result.images[0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run FLUX.1-dev / FLUX.1-Kontext-dev from evaluator summary.csv")
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True, help="Directory containing the original input images.")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--regen_model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--edit_model_id", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--cpu_offload", action="store_true")

    ap.add_argument("--regen_steps", type=int, default=40)
    ap.add_argument("--edit_steps", type=int, default=28)
    ap.add_argument("--regen_guidance", type=float, default=3.5)
    ap.add_argument("--edit_guidance", type=float, default=2.5)

    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--max_prompt_chars", type=int, default=600, help="Hard truncate prompt/instruction by characters if needed.")
    ap.add_argument("--default_width", type=int, default=1024)
    ap.add_argument("--default_height", type=int, default=1024)

    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--copy_keep", action="store_true", help="Copy original image for keep decisions.")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)

    final_dir = os.path.join(args.out_dir, "final_images")
    meta_dir = os.path.join(args.out_dir, "meta")
    ensure_dir(final_dir)
    ensure_dir(meta_dir)

    df = pd.read_csv(args.summary_csv)
    required = ["image_file", "index", "decision"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"summary.csv missing required column: {c}")

    runner = FluxRunner(
        regen_model_id=args.regen_model_id,
        edit_model_id=args.edit_model_id,
        device=args.device,
        dtype=args.dtype,
        cpu_offload=args.cpu_offload,
    )

    rows_done = 0
    rows_skipped = 0

    for _, row in df.iterrows():
        image_file = str(row["image_file"])
        idx = int(row["index"])
        decision = str(row["decision"]).strip().lower()

        src_image_path = os.path.join(args.image_dir, image_file)
        if not os.path.exists(src_image_path):
            print(f"[WARN] missing original image: {src_image_path}")
            rows_skipped += 1
            continue

        width, height = infer_hw_from_image(
            src_image_path,
            fallback_hw=(args.default_width, args.default_height),
        )
        width, height = round_to_flux_size(width, height, multiple=16)
        seed = int(args.base_seed + idx)

        out_path = os.path.join(final_dir, f"{idx}.png")

        if decision == "regenerate":
            prompt = maybe_truncate_prompt(row.get("regenerate_prompt", ""), args.max_prompt_chars)
            if not prompt:
                print(f"[WARN] regenerate_prompt missing for index={idx}")
                rows_skipped += 1
                continue

            meta_path = os.path.join(meta_dir, f"{idx}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[SKIP] exists: {out_path}")
                rows_skipped += 1
                continue

            print(f"[RUN] regenerate | index={idx} | file={image_file}")
            print(f"      prompt={prompt}")

            out_img = runner.regenerate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=args.regen_steps,
                guidance_scale=args.regen_guidance,
                max_sequence_length=args.max_sequence_length,
                seed=seed,
            )
            out_img.save(out_path)

            save_json(meta_path, {
                "index": idx,
                "image_file": image_file,
                "decision": decision,
                "source_image": src_image_path,
                "output_image": out_path,
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": args.regen_steps,
                "guidance_scale": args.regen_guidance,
                "max_sequence_length": args.max_sequence_length,
                "model_id": args.regen_model_id,
            })
            rows_done += 1

        elif decision == "edit":
            instruction = maybe_truncate_prompt(row.get("edit_instruction", ""), args.max_prompt_chars)
            if not instruction:
                print(f"[WARN] edit_instruction missing for index={idx}")
                rows_skipped += 1
                continue

            meta_path = os.path.join(meta_dir, f"{idx}.json")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[SKIP] exists: {out_path}")
                rows_skipped += 1
                continue

            print(f"[RUN] edit | index={idx} | file={image_file}")
            print(f"      instruction={instruction}")

            src_img = load_image_rgb(src_image_path)
            out_img = runner.edit(
                image=src_img,
                prompt=instruction,
                num_inference_steps=args.edit_steps,
                guidance_scale=args.edit_guidance,
                max_sequence_length=args.max_sequence_length,
                seed=seed,
            )
            out_img.save(out_path)

            save_json(meta_path, {
                "index": idx,
                "image_file": image_file,
                "decision": decision,
                "source_image": src_image_path,
                "output_image": out_path,
                "instruction": instruction,
                "seed": seed,
                "num_inference_steps": args.edit_steps,
                "guidance_scale": args.edit_guidance,
                "max_sequence_length": args.max_sequence_length,
                "model_id": args.edit_model_id,
            })
            rows_done += 1

        elif decision == "keep":
            if args.copy_keep:
                meta_path = os.path.join(meta_dir, f"{idx}.json")
                if os.path.exists(out_path) and not args.overwrite:
                    print(f"[SKIP] exists: {out_path}")
                    rows_skipped += 1
                    continue

                shutil.copy2(src_image_path, out_path)
                save_json(meta_path, {
                    "index": idx,
                    "image_file": image_file,
                    "decision": decision,
                    "source_image": src_image_path,
                    "output_image": out_path,
                    "action": "copied_original",
                })
                print(f"[COPY] keep | index={idx} | file={image_file}")
                rows_done += 1
            else:
                print(f"[KEEP] index={idx} | file={image_file}")
                rows_skipped += 1

        else:
            print(f"[WARN] unknown decision={decision} for index={idx}")
            rows_skipped += 1

    print("\n[DONE]")
    print(f"processed: {rows_done}")
    print(f"skipped:   {rows_skipped}")
    print(f"outputs:   {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())