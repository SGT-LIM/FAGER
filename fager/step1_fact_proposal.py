#!/usr/bin/env python3
import os
import time
import json
import re
import argparse
from collections import OrderedDict

import pandas as pd
from openai import OpenAI

# =========================
# Schema order (STRICT)
# =========================
LEVEL_KEYS = ["level_1", "level_2", "level_3"]
CATEGORIES = [
    "existence",
    "counting",
    "relation",
    "shape",
    "size",
    "color",
    "posture",
    "scene",
    "others",
]
IMPORTANCE_SET = {"high", "medium", "low"}

# =========================
# Instructions
# =========================
FACT_PROPOSAL_INSTRUCTIONS = (
    "You are a STRICT visual factual-requirements extraction agent.\n\n"
    "Input: a single text prompt.\n"
    "Task: Propose visually verifiable factual requirements implied by the prompt.\n"
    "Your task is to EXTRACT and ENUMERATE ONLY the visually checkable facts\n"
    "that are NECESSARY for an image to correctly satisfy the prompt.\n\n"
    "Constraints:\n"
    "- Only include facts that can be visually checked from pixels by a human observer, in principle.\n"
    "- Do NOT interpret symbolism, meaning, or aesthetics.\n"
    "- Do NOT assume artistic intent or stylistic preference.\n"
    "- Describe in as much detail as possible any visually observable things that you consider to be important factual information.\n"
    "- Facts must be written as direct visual requirements (e.g., 'X holds Y in right hand').\n\n"
    "NECESSITY PRINCIPLE (MANDATORY):\n"
    "- Include a fact ONLY IF the image would be considered INCORRECT if that fact were violated.\n"
    "- If the condition is not explicitly mentioned, or if it does not necessarily need to be visually represented, do not require visual measurement tools or indicators to show it.\n"
    "- If multiple visually distinct depictions could still satisfy the prompt, do NOT include the differing aspect as a fact.\n\n"
    "IMPLICIT CONDITION RULE (MANDATORY):\n"
    "- If the prompt specifies a condition (time, place, temperature, age, season, lighting, etc.), treat it ONLY as a constraint on the subject/scene’s appearance.\n"
    "- DO NOT introduce or require any measurement/display objects unless explicitly mentioned in the prompt.\n"
    "  * Forbidden unless explicitly mentioned: thermometers, gauges, clocks, calendars, date labels, captions, text overlays, HUD/UI, scales/rulers, measurement readouts.\n"
    "\n"
    "Levels (MUST use 1/2/3):\n"
    "- Level 1: coarse/canonical gist (main entity existence + very large structures + overall scene).\n"
    "- Level 2: key components (salient parts/attributes typically visible).\n"
    "- Level 3: fine-grained details (exact counts, tiny parts, inscriptions/text, very sensitive details).\n\n"
    "Categories (within each level):\n"
    "  existence, counting, relation, shape, size, color, posture, scene, others.\n"
    "- others: visually checkable constraints that do not fit any category above.\n\n"
    "Importance per category:\n"
    "- high: required to satisfy the prompt.\n"
    "- medium: helpful/expected but not strictly required.\n"
    "- low: optional or stress-test.\n\n"
    "Output format (STRICT JSON ONLY):\n"
    "- Return ONLY valid JSON.\n"
    "- Top-level keys MUST be exactly: level_1, level_2, level_3 (in this order).\n"
    "- Each level key maps to an object with EXACTLY these 9 category keys (in this order):\n"
    "  existence, counting, relation, shape, size, color, posture, scene, others.\n"
    "- Each category maps to: {\"importance\": \"high|medium|low\", \"facts\": [ ... ] }.\n"
    "- facts MUST be a list of strings.\n"
    "- No extra keys. No markdown. No code fences.\n"
)

# =========================
# ICL Examples (2)
# =========================
ICL_EX1_USER = {"prompt": "A water molecule"}
ICL_EX1_ASSISTANT = OrderedDict([
    ("level_1", OrderedDict([
        ("existence", {"importance": "high", "facts": ["A single water molecule is depicted."]}),
        ("counting", {"importance": "high", "facts": ["There is one visible molecule."]}),
        ("relation", {"importance": "low", "facts": []}),
        ("shape", {"importance": "low", "facts": []}),
        ("size", {"importance": "low", "facts": []}),
        ("color", {"importance": "low", "facts": []}),
        ("posture", {"importance": "low", "facts": []}),
        ("scene", {"importance": "low", "facts": []}),
        ("others", {"importance": "low", "facts": []}),
    ])),
    ("level_2", OrderedDict([
        ("existence", {"importance": "high", "facts": ["The molecule contains one oxygen atom and two hydrogen atoms."]}),
        ("counting", {"importance": "high", "facts": ["Exactly one oxygen atom is present.", "Exactly two hydrogen atoms are present."]}),
        ("relation", {"importance": "high", "facts": ["The oxygen atom is bonded/connected to both hydrogen atoms (two O–H connections)."]}),
        ("shape", {"importance": "high", "facts": ["The molecule has a bent (V-shaped) geometry rather than a straight line."]}),
        ("size", {"importance": "high", "facts": ["Hydrogen atoms are shown smaller than the central oxygen atom."]}),
        ("color", {"importance": "low", "facts": []}),
        ("posture", {"importance": "low", "facts": []}),
        ("scene", {"importance": "low", "facts": []}),
        ("others", {"importance": "low", "facts": []}),
    ])),
    ("level_3", OrderedDict([
        ("existence", {"importance": "low", "facts": []}),
        ("counting", {"importance": "low", "facts": []}),
        ("relation", {"importance": "low", "facts": []}),
        ("shape", {"importance": "low", "facts": []}),
        ("size", {"importance": "low", "facts": []}),
        ("color", {"importance": "low", "facts": []}),
        ("posture", {"importance": "low", "facts": []}),
        ("scene", {"importance": "low", "facts": []}),
        ("others", {"importance": "low", "facts": []}),
    ])),
])

ICL_EX2_USER = {"prompt": "A color image of the Statue of Liberty in the 1890s"}
ICL_EX2_ASSISTANT = OrderedDict([
    ("level_1", OrderedDict([
        ("existence", {"importance": "high", "facts": ["A Statue of Liberty-like large female statue is present."]}),
        ("counting", {"importance": "low", "facts": []}),
        ("relation", {"importance": "low", "facts": []}),
        ("shape", {"importance": "medium", "facts": ["The statue is a large sculptural figure standing on a pedestal/base."]}),
        ("size", {"importance": "medium", "facts": ["The statue appears as a large, dominant subject in the image rather than a tiny distant object."]}),
        ("color", {"importance": "high", "facts": ["The image is presented in color (not purely grayscale)."]}),
        ("posture", {"importance": "low", "facts": []}),
        ("scene", {"importance": "medium", "facts": ["The scene is outdoors (e.g., sky or city visible in the background)."]}),
        ("others", {"importance": "low", "facts": []}),
    ])),
    ("level_2", OrderedDict([
        ("existence", {"importance": "high", "facts": ["A torch is present and held by the statue.", "A tablet/book-like object is present and held by the statue.", "A crown is present on the statue’s head."]}),
        ("counting", {"importance": "low", "facts": []}),
        ("relation", {"importance": "high", "facts": ["The statue holds a torch in its right hand with the right arm raised.", "The statue holds a tablet in its left hand."]}),
        ("shape", {"importance": "low", "facts": ["The torch has a flame-like element at the top.", "The crown consists of a band with multiple outward spikes."]}),
        ("size", {"importance": "low", "facts": []}),
        ("color", {"importance": "high", "facts": ["The statue’s surface shows a copper brown color."]}),
        ("posture", {"importance": "high", "facts": ["The statue is standing upright in a stable pose, with its right arm raised."]}),
        ("scene", {"importance": "low", "facts": []}),
        ("others", {"importance": "low", "facts": []}),
    ])),
    ("level_3", OrderedDict([
        ("existence", {"importance": "high", "facts": ["There is a broken chain on the pedestal."]}),
        ("counting", {"importance": "high", "facts": ["The crown has exactly seven spikes."]}),
        ("relation", {"importance": "high", "facts": ["The statue is stepping on a broken chain."]}),
        ("shape", {"importance": "low", "facts": []}),
        ("size", {"importance": "low", "facts": []}),
        ("color", {"importance": "low", "facts": []}),
        ("posture", {"importance": "low", "facts": []}),
        ("scene", {"importance": "low", "facts": []}),
        ("others", {"importance": "high", "facts": ["Any inscriptions on the tablet or pedestal, if visible, appear as 'JULY IV MDCCLXXVI'."]}),
    ])),
])

# =========================
# Utils
# =========================
def read_csv_first_comma_only(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        col1, col2 = header.split(",", 1)
        if col2 != "prompt":
            raise ValueError("CSV header must be: index,prompt")

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx, prompt = line.split(",", 1)
            rows.append({"index": idx, "prompt": prompt})

    return pd.DataFrame(rows)

def load_exclude_indices(exclude_csv_path: str):
    """
    exclude_csv 안의 index 칼럼을 읽어 str set으로 반환
    """
    if not exclude_csv_path:
        return set()

    if not os.path.exists(exclude_csv_path):
        raise FileNotFoundError(f"exclude_csv not found: {exclude_csv_path}")

    exclude_df = pd.read_csv(exclude_csv_path)

    if "index" not in exclude_df.columns:
        raise ValueError(
            f"exclude_csv must contain an 'index' column. Columns: {list(exclude_df.columns)}"
        )

    exclude_indices = set(
        exclude_df["index"].dropna().astype(str).str.strip().tolist()
    )
    return exclude_indices

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _empty_schema_ordered():
    out = OrderedDict()
    for lk in LEVEL_KEYS:
        lvl = OrderedDict()
        for cat in CATEGORIES:
            lvl[cat] = {"importance": "low", "facts": []}
        out[lk] = lvl
    return out

def _normalize_importance(x):
    x = (x or "").strip().lower()
    return x if x in IMPORTANCE_SET else "low"

def _ensure_list_of_strings(x):
    if not isinstance(x, list):
        return []
    out = []
    for it in x:
        if it is None:
            continue
        s = str(it).strip()
        if s:
            out.append(s)
    return out

def _normalize_output(obj):
    """
    Force output into OrderedDict with strict key order:
    level_1/2/3 then categories in CATEGORIES.
    """
    fixed = _empty_schema_ordered()
    if not isinstance(obj, dict):
        return fixed

    for lk in LEVEL_KEYS:
        lvl_obj = obj.get(lk, {})
        if not isinstance(lvl_obj, dict):
            continue

        for cat in CATEGORIES:
            cat_obj = lvl_obj.get(cat, {})
            if not isinstance(cat_obj, dict):
                continue

            imp = _normalize_importance(cat_obj.get("importance", "low"))
            facts = _ensure_list_of_strings(cat_obj.get("facts", []))

            fixed[lk][cat] = {"importance": imp, "facts": facts}

    return fixed

def _facts_to_pipe(facts):
    # join for easy csv column
    return " | ".join([f for f in facts if isinstance(f, str) and f.strip()])

# =========================
# Core: propose
# =========================
def propose_facts(client: OpenAI, model: str, prompt: str, reasoning_effort: str = "high"):
    input_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": json.dumps(ICL_EX1_USER, ensure_ascii=False)}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(ICL_EX1_ASSISTANT, ensure_ascii=False)}]},
        {"role": "user", "content": [{"type": "input_text", "text": json.dumps(ICL_EX2_USER, ensure_ascii=False)}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(ICL_EX2_ASSISTANT, ensure_ascii=False)}]},
        {"role": "user", "content": [{"type": "input_text", "text": json.dumps({"prompt": prompt}, ensure_ascii=False)}]},
    ]

    resp = client.responses.create(
        model=model,
        instructions=FACT_PROPOSAL_INSTRUCTIONS,
        input=input_messages,
        reasoning={"effort": reasoning_effort},
    )

    raw = _strip_code_fences((resp.output_text or "").strip())

    # Parse with fallbacks
    try:
        obj = json.loads(raw)
        fixed = _normalize_output(obj)
        return fixed, raw
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                fixed = _normalize_output(obj)
                return fixed, raw
            except Exception:
                pass
        return _empty_schema_ordered(), raw

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV path (must include 'prompt' column).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument(
        "--exclude_csv",
        default=None,
        help="Optional CSV path containing an 'index' column. Those indices will be excluded."
    )
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5.2"), help="Model name.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests (seconds).")
    ap.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high"])
    args = ap.parse_args()


    df = pd.read_csv(args.in_csv)
    # df = read_csv_first_comma_only(args.in_csv)

    if "prompt" not in df.columns:
        raise ValueError("Input CSV must have a 'prompt' column.")

    # =========================
    # Resume : skip index already done
    # =========================
    processed_indices = set()
    out_exists = os.path.exists(args.out_csv)
    if out_exists:
        print(f"🔁 Existing output detected: {args.out_csv}")
        existing_df = pd.read_csv(args.out_csv)
        if "index" in existing_df.columns:
            processed_indices = set(existing_df["index"].astype(str).tolist())
            print(f"Found {len(processed_indices)} already processed indices.")
        else:
            print("⚠ Warning: existing out_csv has no 'index' column. Proceeding without skip.")
    else:
        print("🆕 No existing output file. Starting fresh.")

    exclude_indices = load_exclude_indices(args.exclude_csv)
    if exclude_indices:
        print(f"🚫 Loaded {len(exclude_indices)} indices from exclude_csv.")

    skip_indices = processed_indices | exclude_indices
    print(f"⏭ Total indices to skip: {len(skip_indices)}")

    wrote_header = out_exists

    client = OpenAI()

    # =========================
    # Main loop
    # =========================
    for i, row in df.iterrows():
        idx = str(row["index"])
        prompt = str(row["prompt"])

        if idx in skip_indices:
            print(f"[{i+1}/{len(df)}] ⏭ Skipping already processed index {idx}")
            continue

        print(f"[{i+1}/{len(df)}] Fact proposal: {prompt}")

        try:
            rubric, raw = propose_facts(
                client=client,
                model=args.model,
                prompt=prompt,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as e:
            print(f"  ⚠ Error row {i} (index={idx}): {e}")
            rubric, raw = _empty_schema_ordered(), ""

        row_out = {
            "index": idx,
            "prompt": prompt,
            "rubric_json": json.dumps(rubric, ensure_ascii=False),
            "rubric_raw_output": raw,
        }

        for lk in LEVEL_KEYS:
            for cat in CATEGORIES:
                key_imp = f"{lk}_{cat}_importance"
                key_facts = f"{lk}_{cat}_facts"
                row_out[key_imp] = rubric[lk][cat]["importance"]
                row_out[key_facts] = _facts_to_pipe(rubric[lk][cat]["facts"])

        row_df = pd.DataFrame([row_out])

        mode = "a" if wrote_header else "w"
        header = not wrote_header
        row_df.to_csv(args.out_csv, mode=mode, header=header, index=False)
        wrote_header = True

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\n✅ Incremental save complete: {args.out_csv}")

if __name__ == "__main__":
    main()
