#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, requests
from typing import Any, Dict, List, Optional
from openai import OpenAI
import pandas as pd

# =========================
# Config
# =========================
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "FactVerificationAgent/1.0 (research; contact: none)",
    "Accept-Language": "en-US,en;q=0.9",
}

LEVEL_KEYS = ["level_1", "level_2", "level_3"]
CATEGORIES = ["existence", "counting", "relation", "shape", "size", "color", "posture", "scene", "others"]
IMPORTANCE_SET = {"high", "medium", "low"}

# =========================
# Utils
# =========================
def slugify(text: str, max_len: int = 80) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "empty")[:max_len]

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

def ensure_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pages_dir = os.path.join(out_dir, "wiki_pages")
    imgs_dir = os.path.join(out_dir, "wiki_images")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    return {"pages": pages_dir, "images": imgs_dir}

# =========================
# Wikipedia functions
# =========================
def wiki_search_topk(query: str, k: int = 5, timeout: int = 15) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": k,
        "format": "json",
        "utf8": 1,
        "origin": "*",
    }
    r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    results = data.get("query", {}).get("search", [])
    return [{"title": it.get("title"), "pageid": it.get("pageid"), "snippet": it.get("snippet", "")} for it in results]

def wiki_fetch_page_extract_and_image(pageid: int, timeout: int = 20, max_chars: int = 35000) -> Dict[str, Any]:
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "extracts|pageimages|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "piprop": "thumbnail|original",
        "pithumbsize": 512,
        "format": "json",
        "utf8": 1,
        "origin": "*",
    }
    r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {}) if isinstance(pages, dict) else {}
    extract = (page.get("extract") or "")[:max_chars]
    return {
        "title": page.get("title"),
        "pageid": pageid,
        "url": page.get("fullurl"),
        "extract": extract,
        "thumbnail": (page.get("thumbnail") or {}).get("source"),
        "original_image": (page.get("original") or {}).get("source"),
    }

def download_image(url: str, out_path: str, timeout: int = 20) -> bool:
    try:
        r = requests.get(url, headers=WIKI_HEADERS, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False

# =========================
# LLM prompts
# =========================
PAGE_SELECTION_INSTRUCTIONS = (
    "You are a careful Wikipedia page selector.\n"
    "Input: an image prompt and up to 5 Wikipedia search candidates (title + snippet).\n"
    "Task:\n"
    "- Decide whether ANY candidate is relevant enough to ground canonical facts.\n"
    "- If none are relevant, use_wikipedia=false and chosen_index=-1.\n"
    "- If relevant, choose exactly one best candidate.\n"
    "Output STRICT JSON only: {\"use_wikipedia\":true/false,\"chosen_index\":0-4 or -1,\"reason\":\"...\"}\n"
)

FACT_VERIFICATION_INSTRUCTIONS = (
    "You are a STRICT fact verification and rubric-auditing agent for visual factual evaluation.\n\n"
    "Given:\n"
    "1) prompt\n"
    "2) proposal_rubric (levels->categories->facts)\n"
    "3) OPTIONAL Wikipedia evidence (may be missing or irrelevant): title/url/extract and optional image urls.\n\n"
    "Output a corrected rubric that is the FINAL usable rubric (proposal after correction), containing ONLY facts that are:\n"
    "(A) visually verifiable from pixels in principle, AND\n"
    "(B) NECESSARY for the prompt.\n\n"
    "You must also audit for COMPLETENESS:\n"
    "- If the proposal missed an identity-defining necessary visual attribute, ADD it.\n\n"
    "Wikipedia usage:\n"
    "- If Wikipedia is used and relevant, drop facts that contradict it.\n"
    "- If Wikipedia is used, you may drop facts that are unsupported *only if* they are not canonical identity-defining.\n"
    "- Wikipedia does not need to explicitly mention every obvious canonical visual attribute; keep canonical identity-defining parts "
    "even if not explicitly stated, as long as they do not contradict Wikipedia.\n"
    "- If Wikipedia is missing/unusable, rely on general knowledge conservatively.\n\n"
    "CRITICAL: No meta-visibility statements. Facts must be direct visual requirements.\n"
    "CHEMISTRY COMPLETENESS RULE:\n"
    "- If the prompt is about a specific molecule (e.g., ethanol, water), ensure the rubric includes atom-count facts implied by the chemical formula.\n"
    "Visual information about the state of a substance and any changes in that state must also be included.\n\n"
    "Output STRICT JSON only:\n"
    "{\n"
    "  \"selected_wikipedia_page\": {\"used\": true/false, \"title\": ..., \"pageid\": ..., \"url\": ...},\n"
    "  \"verified_rubric\": {\"level_1\":{...},\"level_2\":{...},\"level_3\":{...}},\n"
    "  \"dropped\": [{\"level\":\"level_1|level_2|level_3\",\"category\":\"...\",\"fact\":\"...\","
    "\"reason\":\"contradicts_wikipedia|not_supported_by_wikipedia|not_visually_verifiable|not_necessary\","
    "\"evidence_snippets\":[\"...\"]}],\n"
    "  \"added\": [{\"level\":\"level_1|level_2|level_3\",\"category\":\"...\",\"fact\":\"...\","
    "\"reason\":\"missing_necessary_fact\",\"evidence_snippets\":[\"...\"]}]\n"
    "}\n"
    "No markdown. No code fences. No extra keys.\n"
)

# =========================
# Optional post-fix: add H count if formula implies it and missing
# (simple heuristic just for ethanol & water; extend later)
# =========================
KNOWN_FORMULA = {
    "ethanol": {"H": 6, "C": 2, "O": 1},
    "water": {"H": 2, "O": 1},
}

def add_missing_hydrogen_fact(prompt: str, verified: Dict[str, Any]) -> Dict[str, Any]:
    p = (prompt or "").lower()
    key = None
    for name in KNOWN_FORMULA:
        if name in p:
            key = name
            break
    if not key:
        return verified

    H = KNOWN_FORMULA[key].get("H")
    if not H:
        return verified

    # check if any fact already mentions "Exactly 6 hydrogen" etc.
    def has_h_fact() -> bool:
        for lk in LEVEL_KEYS:
            lvl = verified.get(lk, {})
            for cat in CATEGORIES:
                facts = (lvl.get(cat, {}) or {}).get("facts", [])
                if isinstance(facts, list):
                    for f in facts:
                        if re.search(r"\bhydrogen\b", str(f), re.IGNORECASE) and re.search(rf"\b{H}\b", str(f)):
                            return True
        return False

    if has_h_fact():
        return verified

    # add to level_2 counting (most consistent for molecules)
    lvl2 = verified.setdefault("level_2", {})
    cell = lvl2.setdefault("counting", {"importance": "high", "facts": []})
    if "importance" not in cell:
        cell["importance"] = "high"
    if cell["importance"] == "low":
        cell["importance"] = "high"

    cell["facts"].append(f"Exactly {H} hydrogen atoms are part of the molecule overall (explicit or implicit).")
    return verified

# =========================
# Agent
# =========================
class Verifier:
    def __init__(self, model: str = "gpt-5.2", temperature: float = 0.0):
        self.client = OpenAI()
        self.model = model

    def llm_json(self, system: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            reasoning={"effort": "high"},
        )
        return json.loads(strip_code_fences(resp.output_text or ""))

    def select_wiki_page(self, prompt: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        sel = self.llm_json(PAGE_SELECTION_INSTRUCTIONS, {"prompt": prompt, "candidates": candidates})
        used = bool(sel.get("use_wikipedia", False))
        idx = int(sel.get("chosen_index", -1))
        if not used:
            return {"used": False, "idx": -1, "reason": sel.get("reason", "")}
        idx = max(0, min(idx, len(candidates) - 1))
        return {"used": True, "idx": idx, "reason": sel.get("reason", "")}

    # def verify_one(self, prompt: str, proposal: Dict[str, Any], save_pages_dir: str, save_imgs_dir: str) -> Dict[str, Any]:
    def verify_one(self, idx: int, prompt: str, proposal: Dict[str, Any], save_pages_dir: str, save_imgs_dir: str) -> Dict[str, Any]:

        # slug = slugify(prompt)
        slug = f"{idx:03d}_{slugify(prompt)}" 

        # Wikipedia: search + choose one page, but DON'T save candidates
        wiki_used = False
        wiki_page = {"title": None, "pageid": None, "url": None, "extract": "", "thumbnail": None, "original_image": None}
        selected_meta = {"used": False, "title": None, "pageid": None, "url": None}

        try:
            candidates = wiki_search_topk(prompt, k=5)
        except Exception:
            candidates = []

        if candidates:
            sel = self.select_wiki_page(prompt, candidates)
            if sel["used"]:
                wiki_used = True
                pageid = candidates[sel["idx"]]["pageid"]
                try:
                    wiki_page = wiki_fetch_page_extract_and_image(pageid)
                    selected_meta = {"used": True, "title": wiki_page["title"], "pageid": wiki_page["pageid"], "url": wiki_page["url"]}
                except Exception:
                    wiki_used = False

        # Save ONLY chosen wiki page (if any)
        # (Even if not used, still write a small json so you can audit)
        page_json_path = os.path.join(save_pages_dir, f"{slug}.json")
        page_txt_path = os.path.join(save_pages_dir, f"{slug}.txt")
        with open(page_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_wikipedia_page": selected_meta,
                    "thumbnail_url": wiki_page.get("thumbnail"),
                    "original_image_url": wiki_page.get("original_image"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(page_txt_path, "w", encoding="utf-8") as f:
            f.write(wiki_page.get("extract") or "")

        # Save ONLY chosen image (prefer original > thumbnail)
        img_url = wiki_page.get("original_image") or wiki_page.get("thumbnail")
        img_saved = None
        if img_url:
            img_saved = os.path.join(save_imgs_dir, f"{slug}.jpg")
            ok = download_image(img_url, img_saved)
            if not ok:
                img_saved = None

        # LLM verification
        result = self.llm_json(
            FACT_VERIFICATION_INSTRUCTIONS,
            {
                "prompt": prompt,
                "proposal_rubric": proposal,
                "wikipedia_used": wiki_used,
                "wikipedia_evidence": wiki_page,
            },
        )

        # verified = result.get("verified_rubric", {})
        dropped = result.get("dropped", [])
        added = result.get("added", [])

        verified = apply_diffs(proposal, dropped, added)

        return {
            "selected_wikipedia_page": result.get("selected_wikipedia_page", selected_meta),
            "verified_rubric": verified,
            "dropped": dropped, #result.get("dropped", []),
            "added": added, #result.get("added", []),
            "wiki_page_json": page_json_path,
            "wiki_page_txt": page_txt_path,
            "wiki_image_path": img_saved,
        }

def normalize_proposal_rubric(proposal: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
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

        # importance: (1) LLM 제공 -> (2) 해당 카테고리 기존 importance -> (3) low
        imp = str(a.get("importance", "")).strip().lower()
        if imp not in IMPORTANCE_SET:
            imp = verified[lk][cat].get("importance", "low")
            if imp not in IMPORTANCE_SET:
                imp = "low"

        # 카테고리 importance는 proposal 그대로 유지하는 게 원칙이라면 아래 두 줄은 주석 처리
        # 다만 "added fact는 high인데 category importance가 low" 같은 게 싫으면 max로 올릴 수도 있음.
        # verified[lk][cat]["importance"] = max_importance(verified[lk][cat]["importance"], imp)

        if fact not in verified[lk][cat]["facts"]:
            verified[lk][cat]["facts"].append(fact)

    return verified

# =========================
# Main: CSV → CSV (single file)
# =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_csv", required=True, help="Path to the fact-proposal CSV from step 1.")
#     ap.add_argument("--out_csv", required=True, help="Path where the verified-rubric CSV will be written.")
#     ap.add_argument("--out_dir", required=True, help="Directory for wiki page cache and downloaded images.")
#     ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5.2"))
#     ap.add_argument("--max_rows", type=int, default=-1)  # debug
#     args = ap.parse_args()

#     dirs = ensure_dirs(args.out_dir)
#     df = pd.read_csv(args.in_csv)

#     # Your columns
#     PROMPT_COL = "prompt"
#     PROPOSAL_COL = "rubric_json"

#     if PROMPT_COL not in df.columns or PROPOSAL_COL not in df.columns:
#         raise ValueError("CSV must contain columns: prompt, rubric_json")

#     v = Verifier(model=args.model)

#     verified_json = []
#     dropped_json = []
#     added_json = []
#     wiki_used = []
#     wiki_title = []
#     wiki_url = []
#     wiki_page_json_path = []
#     wiki_img_path = []

#     n = len(df) if args.max_rows < 0 else min(len(df), args.max_rows)

#     for i in range(len(df)):
#         if i >= n:
#             # pad
#             verified_json.append("")
#             dropped_json.append("")
#             added_json.append("")
#             wiki_used.append(False)
#             wiki_title.append(None)
#             wiki_url.append(None)
#             wiki_page_json_path.append(None)
#             wiki_img_path.append(None)
#             continue

#         prompt = str(df.loc[i, PROMPT_COL])
#         proposal = safe_json_loads_maybe_double_escaped(df.loc[i, PROPOSAL_COL])

#         row_idx = int(df.loc[i, "index"])  # CSV에 index 컬럼이 있을 때
#         # out = v.verify_one(prompt, proposal, dirs["pages"], dirs["images"])
#         out = v.verify_one(row_idx, prompt, proposal, dirs["pages"], dirs["images"])

#         verified_json.append(json.dumps(out["verified_rubric"], ensure_ascii=False))
#         dropped_json.append(json.dumps(out.get("dropped", []), ensure_ascii=False))
#         added_json.append(json.dumps(out.get("added", []), ensure_ascii=False))

#         sp = out.get("selected_wikipedia_page", {}) or {}
#         wiki_used.append(bool(sp.get("used", False)))
#         wiki_title.append(sp.get("title"))
#         wiki_url.append(sp.get("url"))

#         wiki_page_json_path.append(out.get("wiki_page_json"))
#         wiki_img_path.append(out.get("wiki_image_path"))

#     # Append columns (single output file like proposal CSV)
#     df["verified_rubric_json"] = verified_json
#     df["dropped_json"] = dropped_json
#     df["added_json"] = added_json
#     df["wiki_used"] = wiki_used
#     df["wiki_title"] = wiki_title
#     df["wiki_url"] = wiki_url
#     df["wiki_page_json_path"] = wiki_page_json_path
#     df["wiki_image_path"] = wiki_img_path

#     df.to_csv(args.out_csv, index=False)
#     print(f"✅ Saved CSV: {args.out_csv}")
#     print(f"✅ Wiki pages dir: {dirs['pages']}")
#     print(f"✅ Wiki images dir: {dirs['images']}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Path to the fact-proposal CSV from step 1.")
    ap.add_argument("--out_csv", required=True, help="Path where the verified-rubric CSV will be written.")
    ap.add_argument("--out_dir", required=True, help="Directory for wiki page cache and downloaded images.")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5.2"))
    ap.add_argument("--max_rows", type=int, default=-1)
    args = ap.parse_args()

    dirs = ensure_dirs(args.out_dir)

    # 💾 1) out_csv가 있으면 거기서 이어서, 없으면 in_csv에서 시작
    if os.path.exists(args.out_csv):
        print(f"🔁 Existing output detected: {args.out_csv}")
        df = pd.read_csv(args.out_csv)
    else:
        print(f"🆕 No existing output. Loading from in_csv: {args.in_csv}")
        df = pd.read_csv(args.in_csv)

    PROMPT_COL = "prompt"
    PROPOSAL_COL = "rubric_json"

    if PROMPT_COL not in df.columns or PROPOSAL_COL not in df.columns:
        raise ValueError("CSV must contain columns: prompt, rubric_json")

    v = Verifier(model=args.model)

    # 출력 컬럼 미리 생성 (없으면 생성)
    new_cols = [
        "verified_rubric_json",
        "dropped_json",
        "added_json",
        "wiki_used",
        "wiki_title",
        "wiki_url",
        "wiki_page_json_path",
        "wiki_image_path",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    total = len(df)
    n = total if args.max_rows < 0 else min(total, args.max_rows)

    for i in range(n):
        # 이미 검증된 row는 스킵 (resume 지원)
        vr = df.loc[i, "verified_rubric_json"]
        if isinstance(vr, str) and vr.strip():
            print(f"[{i+1}/{n}] ⏭ already verified, skipping", flush=True)
            continue

        print(f"[{i+1}/{n}] start", flush=True)

        prompt = str(df.loc[i, PROMPT_COL])
        proposal = safe_json_loads_maybe_double_escaped(df.loc[i, PROPOSAL_COL])
        row_idx = int(df.loc[i, "index"])

        try:
            out = v.verify_one(row_idx, prompt, proposal, dirs["pages"], dirs["images"])
        except Exception as e:
            print(f"❌ Error at row {i}: {e}", flush=True)
            # 실패한 row는 그냥 비워둔다 (다음 실행 때 다시 시도하게)
            continue

        df.loc[i, "verified_rubric_json"] = json.dumps(out["verified_rubric"], ensure_ascii=False)
        df.loc[i, "dropped_json"] = json.dumps(out.get("dropped", []), ensure_ascii=False)
        df.loc[i, "added_json"] = json.dumps(out.get("added", []), ensure_ascii=False)

        sp = out.get("selected_wikipedia_page", {}) or {}
        df.loc[i, "wiki_used"] = bool(sp.get("used", False))
        df.loc[i, "wiki_title"] = sp.get("title")
        df.loc[i, "wiki_url"] = sp.get("url")
        df.loc[i, "wiki_page_json_path"] = out.get("wiki_page_json")
        df.loc[i, "wiki_image_path"] = out.get("wiki_image_path")

        # 🔥 매 row마다 현재까지의 상태를 전체 저장 (덮어쓰기)
        df.to_csv(args.out_csv, index=False)
        print(f"[{i+1}/{n}] saved", flush=True)

    print(f"\n✅ Finished. Output saved to: {args.out_csv}")
    print(f"📂 Wiki pages dir: {dirs['pages']}")
    print(f"📂 Wiki images dir: {dirs['images']}")

if __name__ == "__main__":
    main()