#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict
import csv

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_exclude_indices(csv_path):
    exclude = set()
    if csv_path is None:
        return exclude

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = row[0].strip()
            if idx.isdigit():
                exclude.add(int(idx))
            else:
                exclude.add(idx)
    return exclude

def get_val(item, key="score"):
    ev = item.get("llm_evaluation", {}) or {}
    v = ev.get(key, None)
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="all_metadata_with_evaluations.json", help="all_metadata_with_evaluations.json path")
    ap.add_argument("--metric", choices=["score", "boolean"], default="score",
                    help="pairwise comparator: score or boolean")
    ap.add_argument("--tie", choices=["zero", "half", "skip"], default="zero",
                    help="how to treat ties (fake == real)") #zero means, tie is wrong in pairwise accuracy.
    ap.add_argument("--real_key", default=None,
                    help="optional: exact key name for real set (e.g., real_science). If None, auto-detect source=='real' or 'fake'")
    ap.add_argument("--exclude_csv", default=None, help="CSV file containing indices to exclude")
    args = ap.parse_args()

    data = load_json(args.in_path)

    exclude_indices = load_exclude_indices(args.exclude_csv)
    print(f"[INFO] Loaded {len(exclude_indices)} excluded indices")

    # Flatten all items by dataset-key (e.g., real_science, dalle3_science, sdxl_science...)
    all_sets = {}
    for set_name, items in data.items():
        if not isinstance(items, list):
            continue
        all_sets[set_name] = items

    # Build mapping: for each set_name -> dict(pair_index -> item)
    idx_map = {}
    for set_name, items in all_sets.items():
        m = {}
        for it in items:
            pi = it.get("pair_index", None)

            if pi is None:
                pi = it.get("index", None)  # fallback
            if pi is None:
                pi = it.get("id", None)     # fallback2
            if pi is None:
                continue
            if isinstance(pi, str) and pi.isdigit():
                pi = int(pi)

            # only keep items that have evaluation value
            v = get_val(it, args.metric)
            if v is None:
                continue
            m[pi] = it
        idx_map[set_name] = m

    # Identify "real" set
    real_set_name = args.real_key
    if real_set_name is None:
        # auto-detect: choose set with most items having source == 'real'
        best = None
        best_cnt = -1
        for set_name, items in all_sets.items():
            cnt = sum(1 for it in items if it.get("source") == "real" and get_val(it, args.metric) is not None)
            if cnt > best_cnt:
                best_cnt = cnt
                best = set_name
        real_set_name = best

    if real_set_name is None or real_set_name not in idx_map:
        raise RuntimeError("Could not detect real set. Pass --real_key real_science (or correct key).")

    real_by_pi = idx_map[real_set_name]

    # Compare each other set against real
    results = {}
    for set_name, fake_by_pi in idx_map.items():
        if set_name == real_set_name:
            continue

        total = 0
        correct = 0.0
        skipped = 0
        ties = 0

        common_pis = sorted(set(real_by_pi.keys()) & set(fake_by_pi.keys()))
        for pi in common_pis:
            if pi in exclude_indices:
                skipped += 1
                continue
                
            real_item = real_by_pi[pi]
            fake_item = fake_by_pi[pi]

            rv = get_val(real_item, args.metric)
            fv = get_val(fake_item, args.metric)
            if rv is None or fv is None:
                skipped += 1
                continue

            total += 1
            if fv > rv:
                correct += 1.0
            elif fv == rv:
                ties += 1
                if args.tie == "half":
                    correct += 0.5
                elif args.tie == "skip":
                    total -= 1
                    skipped += 1
                # tie == "zero" -> add nothing
            else:
                # fv < rv => wrong
                pass

        acc = (correct / total) if total > 0 else None
        results[set_name] = {
            "pairwise_accuracy": acc,
            "total_pairs_used": total,
            "ties": ties,
            "skipped": skipped,
            "real_set": real_set_name,
            "metric": args.metric,
            "tie_policy": args.tie,
        }

    # Print summary
    print(f"[REAL SET] {real_set_name}")
    for set_name, r in sorted(results.items(), key=lambda x: (x[1]["pairwise_accuracy"] is None, -(x[1]["pairwise_accuracy"] or 0))):
        print(
            f"{set_name}: acc={r['pairwise_accuracy']:.4f}" if r["pairwise_accuracy"] is not None else f"{set_name}: acc=None"
        )
        print(f"  pairs={r['total_pairs_used']}, ties={r['ties']}, skipped={r['skipped']}, metric={r['metric']}, tie={r['tie_policy']}")

if __name__ == "__main__":
    main()
