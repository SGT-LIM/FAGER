#!/usr/bin/env python3
import os
import time
import json
import re
import argparse
import pandas as pd
from openai import OpenAI

MODEL_NAME_DEFAULT = "gpt-5.4-mini" #"gpt-5.2"

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

# -------------------------
# Utils
# -------------------------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _safe_json_loads(s: str, default=None):
    try:
        return json.loads(s)
    except Exception:
        if not isinstance(s, str):
            return default
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return default
        return default

def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield ln, obj
            except Exception:
                # skip malformed line
                continue

# -------------------------
# Single fact -> atomic questions
# -------------------------
FACT_QA_INSTRUCTIONS = (
    "You generate ONE atomic QA question from a single visual factual requirement.\n\n"
    "Given:\n"
    "- prompt\n"
    "- level (level_1|level_2|level_3)\n"
    "- category\n"
    "- fact: a single factual requirement that must hold in any correct image.\n"
    "- category_hint: optional hint about how to phrase the question.\n\n"
    "Goal:\n"
    "- Create a question that evaluates this fact as independently as possible.\n"
    "- Avoid repeated penalization: if object identity is uncertain, other questions should still be answerable when possible.\n"
    "- However, the question must remain faithful to the source fact and must not become overly generic.\n\n"
    "Rules (MUST FOLLOW):\n"
    "1) The evaluator answers using ONLY: yes / no / unknown.\n"
    "2) For a correct image, the intended answer MUST be YES.\n"
    "3) Output exactly ONE question per fact.\n"
    "4) Do NOT produce complementary, contrastive, or negative questions.\n"
    "   - Forbidden patterns include: 'more than', 'less than', 'at least', 'at most', 'not', 'without',\n"
    "     'different from', 'isn't', 'aren't', 'never', 'instead of'.\n"
    "5) NO visibility/assumption language.\n"
    "   - Do NOT use: 'if', 'when', 'assuming', 'clearly', 'can be determined', 'if shown', 'appears to'.\n"
    "6) Preserve the semantic target of the fact.\n"
    "   - The question must test the same visual requirement as the fact.\n"
    "   - Do NOT weaken a specific fact into a vague one.\n"
    "   - Do NOT replace a relation fact with a generic scene question unless the relation itself is still directly tested.\n"
    "7) Reduce repeated identity dependence whenever possible.\n"
    "   - Do NOT make the question depend on first confirming the target object's identity unless necessary.\n"
    "   - A failure on object recognition should NOT automatically cause failure on many later questions.\n"
    "   - Avoid repeating the exact target-object name unless:\n"
    "     (a) the category is 'existence' in 'level_1', or\n"
    "     (b) using a generic anchor would make the question ambiguous or less faithful to the fact.\n"
    "8) Use generic anchors only when safe.\n"
    "   - Prefer anchors like 'the main object', 'the main subject', 'the central item', or 'the flag/map/sign' ONLY when\n"
    "     the image is expected to contain one clear primary subject and the fact remains unambiguous.\n"
    "   - If a generic anchor would make the question ambiguous, use the explicit target entity instead.\n"
    "9) If the fact contains a conditional clause (e.g. starts with 'If ...'), drop the condition and convert it into an unconditional requirement question.\n"
    "10) Keep the wording short, concrete, and visually answerable.\n"
    "11) Prefer direct positive formulations that align tightly with the fact.\n"
    "12) Do NOT ask about metadata, evaluation procedure, or non-visual background knowledge.\n"
    "13) When a fact is highly similar to common identity facts already likely covered elsewhere, phrase the question to isolate only the additional visual attribute introduced by this fact.\n\n"
    "Category-specific guidance:\n"
    "- existence:\n"
    "  Ask directly whether the target entity/object is present. Direct identity wording is allowed and preferred for level_1 existence.\n"
    "- counting:\n"
    "  Ask about the count required by the fact. Use a generic anchor only if the intended counted subject is obvious and unique.\n"
    "- relation:\n"
    "  Ask the relation itself directly. Avoid wording that first requires identity confirmation if a direct spatial/structural relation can be asked.\n"
    "  But do NOT collapse a specific relation into a looser scene-presence question.\n"
    "- shape/color/size/posture:\n"
    "  Prefer a generic anchor only when the main target is visually obvious. Otherwise keep the explicit target so the question stays faithful.\n"
    "- scene:\n"
    "  Ask about the environment/scene content directly. Do not route the question through the target object unless the fact itself does so.\n"
    "- others:\n"
    "  Ask the direct visual property itself, staying as close as possible to the fact.\n\n"
    "Output STRICT JSON only:\n"
    "{ \"question\": \".\" }\n"
)

def _category_hint(category: str) -> str:
    cat = (category or "").strip().lower()

    if cat == "existence":
        return (
            "Ask directly whether the target entity or object is present. "
            "Direct identity wording is appropriate here, especially for level_1 existence."
        )

    if cat == "counting":
        return (
            "Ask about the required count. "
            "Use a generic anchor only if the counted subject is the single obvious main subject; "
            "otherwise keep the explicit target."
        )

    if cat == "relation":
        return (
            "Ask the specific relation directly and keep it faithful to the fact. "
            "Avoid making the question depend on identity confirmation when possible, "
            "but do not weaken the fact into a broad scene question."
        )

    if cat == "scene":
        return (
            "Ask about the scene or environment content directly. "
            "Do not phrase it through the target object unless the fact itself is object-centered."
        )

    if cat in {"shape", "color", "size", "posture"}:
        return (
            "Use a generic anchor like 'the main object' or 'the main subject' only when the primary target is visually obvious and unique. "
            "If that would be ambiguous, keep the explicit target entity so the question stays precise."
        )

    return (
        "Ask a short, direct, visually answerable question that stays very close to the fact."
    )

def split_fact_into_questions(
    client: OpenAI,
    model: str,
    prompt: str,
    level: str,
    category: str,
    fact: str,
    reasoning_effort: str = "none",
) -> list[str]:
    # payload = {"prompt": prompt, "level": level, "category": category, "fact": fact}
    payload = {
        "prompt": prompt,
        "level": level,
        "category": category,
        "fact": fact,
        "category_hint": _category_hint(category),
    }
    response = client.responses.create(
        model=model,
        instructions=FACT_QA_INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
        }],
        reasoning={"effort": reasoning_effort},
    )
    raw = _strip_code_fences(response.output_text or "")
    obj = _safe_json_loads(raw, default=None)
    if not isinstance(obj, dict):
        return []

    # Accept both formats:
    # 1) {"questions": ["...", ...]}
    qlist = obj.get("questions", None)
    if isinstance(qlist, list):
        return [q.strip() for q in qlist if isinstance(q, str) and q.strip()]

    # 2) {"question": "..."}
    q = obj.get("question", None)
    if isinstance(q, str) and q.strip():
        return [q.strip()]

    return []

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="/home/coder/passenger/data/image_hallucination/2_Fact_Verification/fact_verification.jsonl",
                    help="verification jsonl (each line must contain: id, prompt, verified_rubric)")
    ap.add_argument("--out_csv", default="/home/coder/passenger/data/image_hallucination/3_QA_Generation/QA.csv",
                    help="qa csv output path")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", MODEL_NAME_DEFAULT))
    ap.add_argument("--reasoning_effort", default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--importance_keep", default="high,medium,low", help="comma list among high,medium,low (default high,medium)")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--skip_if_exists", action="store_true",
                    help="if set, skip prompts whose questions already exist in out_csv")
    args = ap.parse_args()

    keep_set = {x.strip().lower() for x in args.importance_keep.split(",") if x.strip()}
    client = OpenAI()

    # Load existing indices from out_csv to skip
    existing_ids = set()
    wrote_header = False
    if args.skip_if_exists and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0:
        try:
            existing = pd.read_csv(args.out_csv)
            if "index" in existing.columns:
                col = existing["index"].astype(str).str.strip()
                # accept numeric ids only
                col = col[col.str.match(r"^\d+$", na=False)]
                existing_ids = set(col.astype(int).tolist())
            wrote_header = True  # append without header
            print(f"[skip_if_exists] loaded existing QA indices: {len(existing_ids)}")
        except Exception as e:
            print(f"[skip_if_exists] ⚠ failed to read out_csv, will not skip. err={e}")
            existing_ids = set()

    saved_count = 0

    def append_row(row_out: dict):
        nonlocal wrote_header
        pd.DataFrame([row_out]).to_csv(args.out_csv, index=False, mode="a", header=(not wrote_header))
        wrote_header = True

    # Iterate jsonl
    for n, (ln, obj) in enumerate(_iter_jsonl(args.in_jsonl), start=1):
        # id
        pid_raw = obj.get("id", None)
        try:
            prompt_id = int(str(pid_raw).strip())
        except Exception:
            # fallback: use line number
            prompt_id = ln

        if args.skip_if_exists and (prompt_id in existing_ids):
            print(f"[{n}] skip prompt_id={prompt_id} (already has QA)")
            continue

        prompt = str(obj.get("prompt", "")).strip()
        rubric = obj.get("verified_rubric", None)

        if not prompt:
            print(f"[{n}] ⚠ missing prompt (id={prompt_id}), skip")
            continue
        if not isinstance(rubric, dict):
            # sometimes rubric stored as string
            if isinstance(rubric, str):
                rubric = _safe_json_loads(rubric, default=None)
            if not isinstance(rubric, dict):
                print(f"[{n}] ⚠ verified_rubric missing/invalid (id={prompt_id}), skip")
                continue

        print(f"[{n}] QA from verified rubric (id={prompt_id}):")
        print(f"   {prompt}")

        for level in LEVEL_KEYS:
            level_obj = rubric.get(level, {})
            if not isinstance(level_obj, dict):
                continue

            for cat in CATEGORIES:
                cat_info = level_obj.get(cat, {})
                if not isinstance(cat_info, dict):
                    continue

                importance = str(cat_info.get("importance", "low")).strip().lower()
                if importance not in keep_set:
                    continue

                facts = cat_info.get("facts", [])
                if not isinstance(facts, list):
                    continue

                for fact in facts:
                    fact = str(fact).strip()
                    if not fact:
                        continue

                    print(f"  - [{level} / {cat} / {importance}] {fact[:90]}")
                    try:
                        questions = split_fact_into_questions(
                            client, args.model, prompt, level, cat, fact,
                            reasoning_effort=args.reasoning_effort
                        )
                    except Exception as e:
                        print(f"    ⚠ split_fact_into_questions error: {e}")
                        questions = []

                    for q in questions:
                        row_out = {
                            # IMPORTANT: use prompt_id (jsonl id), not row number
                            "index": prompt_id,
                            "prompt": prompt,
                            "level": level,
                            "category": cat,
                            "importance": importance,
                            "question": q,
                            "gold_answer": "yes",
                            "answer_space": "yes|no|unknown",
                            "source_fact": fact,
                        }
                        append_row(row_out)
                        saved_count += 1

                    time.sleep(args.sleep)

        if args.skip_if_exists and saved_count > 0:
            existing_ids.add(prompt_id)

    if saved_count == 0:
        print("⚠ No QA generated. Check parsing or importance_keep.")
    else:
        print(f"\n✅ Saved {saved_count} QA rows to:\n  {args.out_csv}")


if __name__ == "__main__":
    main()