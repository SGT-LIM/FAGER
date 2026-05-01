#!/usr/bin/env python3
import argparse
import pandas as pd

def compute_pairwise(real_df, fake_df, key, score_col, tie_mode, lower_is_better=False):
    # keep only needed columns
    r = real_df[[key, score_col]].copy()
    f = fake_df[[key, score_col]].copy()

    # merge
    m = r.merge(f, on=key, how="inner", suffixes=("_real", "_fake"))
    m = m.dropna(subset=[f"{score_col}_real", f"{score_col}_fake"])

    # comparisons
    real_s = m[f"{score_col}_real"].astype(float)
    fake_s = m[f"{score_col}_fake"].astype(float)

    # if lower_is_better, flip sign so "higher is better" after transform
    if lower_is_better:
        real_s = -real_s
        fake_s = -fake_s

    gt = (real_s > fake_s)   # correct: real better
    lt = (real_s < fake_s)   # wrong
    eq = (real_s == fake_s)  # tie

    pairs = len(m)
    ties = int(eq.sum())
    wrong = int(lt.sum())
    correct = int(gt.sum())

    skipped = 0
    if tie_mode == "skip":
        denom = pairs - ties
        skipped = ties
        acc = (correct / denom) if denom > 0 else float("nan")
    elif tie_mode == "half":
        denom = pairs
        acc = (correct + 0.5 * ties) / denom if denom > 0 else float("nan")
    elif tie_mode == "zero":
        denom = pairs
        acc = correct / denom if denom > 0 else float("nan")
    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}")

    return {
        "pairs": pairs,
        "correct": correct,
        "wrong": wrong,
        "ties": ties,
        "skipped": skipped,
        "acc": acc,
        "score_col": score_col,
        "tie_mode": tie_mode,
        "lower_is_better": lower_is_better,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", default="/home/coder/passenger/data/culture/results/our-metric/real/summary.csv")
    ap.add_argument("--fake_csv", default="/home/coder/passenger/data/culture/results/our-metric/flux2/summary.csv", help="e.g., dalle3 csv")
    ap.add_argument("--key", default="index", choices=["index", "image_file"])
    ap.add_argument("--tie", default="zero", choices=["zero", "half", "skip"])
    ap.add_argument("--lower_is_better", action="store_true",
                    help="Use if lower score means better quality")
    ap.add_argument("--scores", default="overall_score",
                    help="Comma-separated list, e.g. overall_score,level_1_score,level_2_score,level_3_score")
    args = ap.parse_args()

    real_df = pd.read_csv(args.real_csv)
    fake_df = pd.read_csv(args.fake_csv)

    # basic cleanup: ensure key exists
    for df, name in [(real_df, "real"), (fake_df, "fake")]:
        if args.key not in df.columns:
            raise KeyError(f"{name} csv missing key column: {args.key}")

    score_cols = [s.strip() for s in args.scores.split(",") if s.strip()]
    for sc in score_cols:
        if sc not in real_df.columns:
            raise KeyError(f"real csv missing score column: {sc}")
        if sc not in fake_df.columns:
            raise KeyError(f"fake csv missing score column: {sc}")

    print(f"[PAIRWISE] key={args.key} tie={args.tie} lower_is_better={args.lower_is_better}")
    print(f"real_csv={args.real_csv}")
    print(f"fake_csv={args.fake_csv}\n")

    for sc in score_cols:
        res = compute_pairwise(
            real_df, fake_df,
            key=args.key,
            score_col=sc,
            tie_mode=args.tie,
            lower_is_better=args.lower_is_better,
        )
        print(f"{sc}: acc={res['acc']:.4f}  pairs={res['pairs']}  correct={res['correct']}  wrong={res['wrong']}  ties={res['ties']}  skipped={res['skipped']}")

if __name__ == "__main__":
    main()
