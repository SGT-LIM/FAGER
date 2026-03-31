#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI

# -----------------------------
# Constants (same spirit as your wiki verifier)
# -----------------------------
LEVEL_KEYS = ["level_1", "level_2", "level_3"]
CATEGORIES = ["existence", "counting", "relation", "shape", "size", "color", "posture", "scene", "others"]
IMPORTANCE_SET = {"high", "medium", "low"}

# -----------------------------
# Utils (ported from your code style)
# -----------------------------
def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def safe_json_loads_maybe_double_escaped(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    s = str(x).strip()
    if not s:
        return {}
    # 1) direct
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                obj2 = json.loads(obj)
                return obj2 if isinstance(obj2, dict) else {}
            except Exception:
                return {}
        return {}
    except Exception:
        pass
    # 2) CSV double quotes style
    s2 = s.replace('""', '"')
    if len(s2) >= 2 and s2[0] == '"' and s2[-1] == '"':
        s2 = s2[1:-1]
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                obj2 = json.loads(obj)
                return obj2 if isinstance(obj2, dict) else {}
            except Exception:
                return {}
        return {}
    except Exception:
        return {}

def normalize_proposal_rubric(proposal: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for lk in LEVEL_KEYS:
        out[lk] = {}
        lvl = proposal.get(lk, {}) if isinstance(proposal.get(lk, {}), dict) else {}
        for cat in CATEGORIES:
            cell = lvl.get(cat, {}) if isinstance(lvl.get(cat, {}), dict) else {}
            imp = str(cell.get("importance", "")).strip().lower()
            if imp not in IMPORTANCE_SET:
                imp = "low"
            facts = cell.get("facts", [])
            if isinstance(facts, str):
                facts = [facts]
            if not isinstance(facts, list):
                facts = []
            facts = [str(f).strip() for f in facts if str(f).strip()]
            out[lk][cat] = {"importance": imp, "facts": facts}
    return out

def apply_diffs(proposal: Dict[str, Any],
                dropped: List[Dict[str, Any]],
                added: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified = normalize_proposal_rubric(proposal)

    # drop
    for d in dropped or []:
        lk = d.get("level")
        cat = d.get("category")
        fact = str(d.get("fact", "")).strip()
        if lk in verified and cat in verified[lk] and fact:
            verified[lk][cat]["facts"] = [f for f in verified[lk][cat]["facts"] if f != fact]

    # add
    for a in added or []:
        lk = a.get("level")
        cat = a.get("category")
        fact = str(a.get("fact", "")).strip()
        if not (lk in verified and cat in verified[lk] and fact):
            continue
        if fact not in verified[lk][cat]["facts"]:
            verified[lk][cat]["facts"].append(fact)

    return verified

# -----------------------------
# JSONL loader for agent2 outputs
# -----------------------------
def load_agent2_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    """
    returns: dict keyed by id (string) => full record
    """
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("id", "")).strip()
            if rid:
                out[rid] = obj
    return out

def collect_image_elements(agent2_rec: Dict[str, Any],
                           max_images: int = 5,
                           max_elements_per_image: int = 25) -> Dict[str, Any]:
    """
    Pull visual_elements from agent2 results.

    Supports both schemas:
    1) searches -> web_search -> images[*].visual_elements
    2) visual_fact_extraction.visual_elements (single local/reference image)
    """
    if not agent2_rec:
        return {"status": "missing", "images_used": 0, "elements": [], "by_image": []}

    elements: List[str] = []
    by_image: List[Dict[str, Any]] = []

    def _normalize_visual_elements(ve: Any) -> List[str]:
        if isinstance(ve, str):
            ve = [ve]
        if not isinstance(ve, list):
            return []
        return [str(x).strip() for x in ve if str(x).strip()][:max_elements_per_image]

    # Schema A: top-level visual_fact_extraction
    vfe = agent2_rec.get("visual_fact_extraction") or {}
    if isinstance(vfe, dict):
        ve = _normalize_visual_elements(vfe.get("visual_elements"))
        if ve:
            by_image.append({
                "image_url": None,
                "page_url": None,
                "image_path": vfe.get("image_path") or agent2_rec.get("image_path"),
                "confidence": vfe.get("confidence"),
                "notes": vfe.get("notes", ""),
                "visual_elements": ve,
            })
            elements.extend(ve)

    # Schema B: searches -> web_search -> images[*]
    searches = agent2_rec.get("searches", [])
    all_imgs = []
    for s in searches:
        ws = (s.get("web_search") or {})
        imgs = ws.get("images") or []
        for im in imgs:
            all_imgs.append(im)

    remaining_slots = max(0, max_images - len(by_image))
    picked = all_imgs[:remaining_slots]
    for im in picked:
        ve = _normalize_visual_elements(im.get("visual_elements"))
        if not ve:
            continue
        by_image.append({
            "image_url": im.get("image_url"),
            "page_url": im.get("page_url"),
            "image_path": im.get("image_path"),
            "confidence": im.get("confidence", None),
            "notes": im.get("notes", ""),
            "visual_elements": ve,
        })
        elements.extend(ve)

    # de-dup while keeping order
    seen = set()
    dedup = []
    for e in elements:
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)

    if dedup:
        return {"status": "ok", "images_used": len(by_image), "elements": dedup, "by_image": by_image}

    # fallback status reporting
    status = None
    if isinstance(vfe, dict):
        status = vfe.get("status")
    if not status:
        try:
            status = searches[0].get("web_search", {}).get("status")
        except Exception:
            pass
    return {"status": status or "no_images", "images_used": 0, "elements": [], "by_image": []}

# -----------------------------
# LLM instructions (prompt-anchored, image-assisted)
# -----------------------------
FACT_CHECK_INSTRUCTIONS = (
    "You are a STRICT verification agent for prompt-based visual factuality evaluation.\n\n"
    "You will be given:\n"
    "1) prompt\n"
    "2) proposal_rubric (levels->categories->facts) extracted from the prompt by agent-1\n"
    "3) retrieved_image_elements extracted from real images by agent-2\n\n"

    "Core principle:\n"
    "- The prompt is the ground-truth anchor for what is REQUIRED.\n"
    "- Retrieved image elements are evidence for how the prompt-referred entity appears visually.\n"
    "- For generic concepts/classes/scientific entities, be conservative.\n"
    "- For specific named real-world instances, include conservative appearance facts that materially help visual identification.\n\n"

    "IMPORTANT DISTINCTION:\n"
    "- Do NOT add depiction conventions as required facts.\n"
    "- A depiction convention is a visual representation choice that is common in diagrams, textbooks, icons, or illustrations, but is not itself a factual property of the target.\n"
    "- Examples: oxygen shown in red, hydrogen shown in white, schematic colors, illustrative shading, textbook diagram layouts.\n"
    "- These must NOT be treated as required facts unless the prompt itself explicitly requires them.\n\n"

    "Specific-instance rule:\n"
    "- If the prompt refers to a specific named instance (brand/product/model/place/official design/object variant),\n"
    "  you MAY add appearance facts from retrieved_image_elements when they are identity-defining and visually verifiable.\n"
    "- These may include dominant object color, packaging color, cap color, label layout, logo presence, standard markings, distinctive shape, or other stable visual identifiers.\n"
    "- Do NOT reject such facts merely because they are not class-invariant.\n"
    "- But do NOT add incidental background details, lighting, camera angle, or temporary presentation artifacts.\n\n"

    "Decision rule for image-based additions:\n"
    "- Add the fact if it is either:\n"
    "  1) explicitly stated in the prompt,\n"
    "  2) semantically implied by the prompt, or\n"
    "  3) an identity-defining visual property of a specific named instance.\n"
    "- Do NOT add the fact if it is only a depiction convention or rendering habit.\n\n"

    "Level guidance:\n"
    "- level_1: coarse identity, object type, overall form.\n"
    "- level_2: brand, product name, dosage/count/model, major appearance cues needed for recognition.\n"
    "- level_3: secondary but still useful visible details.\n\n"
    "Output STRICT JSON only with NO extra keys:\n"
    "{\n"
    "  \"verified_rubric\": {\"level_1\":{...},\"level_2\":{...},\"level_3\":{...}},\n"
    "  \"dropped\": [{\"level\":\"level_1|level_2|level_3\",\"category\":\"...\",\"fact\":\"...\","
    "               \"reason\":\"not_necessary|not_visually_verifiable|malformed\","
    "               \"short_rationale\":[\"...\"]}],\n"
    "  \"added\":   [{\"level\":\"level_1|level_2|level_3\",\"category\":\"...\",\"fact\":\"...\","
    "               \"reason\":\"missing_necessary_fact\","
    "               \"short_rationale\":[\"...\"]}],\n"
    "}\n"
)

# -----------------------------
# Agent
# -----------------------------
class PromptImageVerifier:
    def __init__(self, model: str = "gpt-5.4-mini", temperature: float = 0.0):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def llm_json(self, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning={"effort": "high"},
        )
        txt = strip_code_fences(resp.output_text or "")
        return json.loads(txt)

    def verify_one(self, prompt: str, proposal: Dict[str, Any], img_pack: Dict[str, Any]) -> Dict[str, Any]:
        # Let the model decide drops/adds; then re-apply diffs for robustness & normalization.
        result = self.llm_json(
            FACT_CHECK_INSTRUCTIONS,
            {
                "prompt": prompt,
                "proposal_rubric": proposal,
                "retrieved_image_elements": img_pack,  # includes elements + by_image + status
            },
        )
        dropped = result.get("dropped", []) or []
        added = result.get("added", []) or []
        verified = apply_diffs(proposal, dropped, added)

        # overwrite verified_rubric with our normalized+diff-applied version
        result["verified_rubric"] = verified
        return result

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_agent1", default="/mnt/data/results.csv")
    ap.add_argument("--in_agent2", default="/mnt/data/results.jsonl", help="agent2 jsonl path, or 'none' to disable")
    ap.add_argument("--out_jsonl", default="verified_results.jsonl")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5.4-mini"))
    ap.add_argument("--max_rows", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.in_agent1)

    # agent2 = load_agent2_jsonl(args.in_agent2)
    agent2 = {}
    use_agent2 = True

    if args.in_agent2 is None:
        use_agent2 = False
    else:
        s = str(args.in_agent2).strip().lower()
        if s in {"none", "null", ""}:
            use_agent2 = False
        elif not os.path.exists(args.in_agent2):
            print(f"[WARN] agent2 file not found: {args.in_agent2} -> disable agent2")
            use_agent2 = False

    if use_agent2:
        agent2 = load_agent2_jsonl(args.in_agent2)
    else:
        agent2 = {}

    # optional resume
    done_ids = set()
    if args.resume and os.path.exists(args.out_jsonl):
        with open(args.out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_ids.add(str(obj.get("id", "")))
                except Exception:
                    pass

    verifier = PromptImageVerifier(model=args.model)

    n = len(df) if args.max_rows < 0 else min(len(df), args.max_rows)
    with open(args.out_jsonl, "a", encoding="utf-8") as wf:
        for i in range(n):
            rid = str(df.loc[i, "index"])  # csv has "index" column = 1..N
            if use_agent2 and rid not in agent2:
                print(f"[{i+1}/{n}] ⏭ skip id={rid} (missing in agent2)")
                continue

            if args.resume and rid in done_ids:
                print(f"[{i+1}/{n}] ⏭ skip id={rid} (already done)")
                continue

            prompt = str(df.loc[i, "prompt"])
            proposal = safe_json_loads_maybe_double_escaped(df.loc[i, "rubric_json"])
            proposal = normalize_proposal_rubric(proposal)

            # attach agent2 pack (may be missing)
            img_pack = collect_image_elements(agent2.get(rid, {}))

            out = verifier.verify_one(prompt, proposal, img_pack)
            record = {
                "id": rid,
                "prompt": prompt,
                "agent2_status": img_pack.get("status"),
                "images_used": img_pack.get("images_used"),
                "verified_rubric": out.get("verified_rubric", {}),
                "dropped": out.get("dropped", []),
                "added": out.get("added", []),
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            print(f"[{i+1}/{n}] ✅ wrote id={rid}")

    print(f"\n✅ Done. Saved: {args.out_jsonl}")

if __name__ == "__main__":
    main()