#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def safe_mean(x):
    return float(x.mean()) if len(x) > 0 else float("nan")

def safe_std(x):
    return float(x.std(ddof=0)) if len(x) > 0 else float("nan")

def compute_pairwise(real_df, fake_df, key, score_col, tie_mode, lower_is_better=False):
    # keep only needed columns
    r = real_df[[key, score_col]].copy()
    f = fake_df[[key, score_col]].copy()

    # merge
    m = r.merge(f, on=key, how="inner", suffixes=("_real", "_fake"))
    m = m.dropna(subset=[f"{score_col}_real", f"{score_col}_fake"]).copy()

    # original scores
    m[f"{score_col}_real"] = m[f"{score_col}_real"].astype(float)
    m[f"{score_col}_fake"] = m[f"{score_col}_fake"].astype(float)

    real_orig = m[f"{score_col}_real"]
    fake_orig = m[f"{score_col}_fake"]

    # comparison scores
    if lower_is_better:
        real_cmp = -real_orig
        fake_cmp = -fake_orig
    else:
        real_cmp = real_orig
        fake_cmp = fake_orig

    # comparisons
    gt = (real_cmp > fake_cmp)   # correct: real better
    lt = (real_cmp < fake_cmp)   # wrong
    eq = (real_cmp == fake_cmp)  # tie

    # per-index details
    m["winner"] = gt.map({True: "real"}).where(~lt, "fake")
    m.loc[eq, "winner"] = "tie"

    # signed difference in the "better direction"
    # positive => real better, negative => fake better
    m["score_diff"] = real_cmp - fake_cmp
    m["abs_score_diff"] = m["score_diff"].abs()

    pairs = len(m)
    ties = int(eq.sum())
    wrong = int(lt.sum())
    correct = int(gt.sum())

    skipped = 0
    if tie_mode == "skip":
        denom = pairs - ties
        skipped = ties
        acc = (correct / denom) if denom > 0 else float("nan")
    elif tie_mode == "full":
        denom = pairs
        acc = (correct + ties) / denom if denom > 0 else float("nan")
    elif tie_mode == "zero":
        denom = pairs
        acc = correct / denom if denom > 0 else float("nan")
    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}")

    # margin stats
    all_margin = m["score_diff"]
    correct_margin = m.loc[gt, "score_diff"]
    wrong_margin = m.loc[lt, "score_diff"]

    summary = {
        "pairs": pairs,
        "correct": correct,
        "wrong": wrong,
        "ties": ties,
        "skipped": skipped,
        "acc": acc,
        "score_col": score_col,
        "tie_mode": tie_mode,
        "lower_is_better": lower_is_better,

        "margin_mean_all": safe_mean(all_margin),
        "margin_std_all": safe_std(all_margin),

        "margin_mean_correct": safe_mean(correct_margin),
        "margin_std_correct": safe_std(correct_margin),

        "margin_mean_wrong": safe_mean(wrong_margin),
        "margin_std_wrong": safe_std(wrong_margin),
    }

    details = m[[key,
                 f"{score_col}_real",
                 f"{score_col}_fake",
                 "winner",
                 "score_diff",
                 "abs_score_diff"]].copy()

    return summary, details, real_orig, fake_orig, all_margin, correct_margin, wrong_margin

def save_margin_distribution_plot(all_margin, correct_margin, wrong_margin, score_col, out_path, bins=30):
    plt.figure(figsize=(8, 5))

    if len(all_margin) > 0:
        plt.hist(all_margin, bins=bins, alpha=0.5, label=f"all (n={len(all_margin)})", density=False)
    if len(correct_margin) > 0:
        plt.hist(correct_margin, bins=bins, alpha=0.5, label=f"correct (n={len(correct_margin)})", density=False)
    if len(wrong_margin) > 0:
        plt.hist(wrong_margin, bins=bins, alpha=0.5, label=f"wrong (n={len(wrong_margin)})", density=False)

    plt.axvline(0, linestyle="--")
    plt.xlabel("Margin (real - fake, after direction adjustment)")
    plt.ylabel("Count")
    plt.title(f"Margin distribution: {score_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_margin_values(details, key, score_col, out_path):
    margin_df = pd.DataFrame({
        key: details[key],
        "winner": details["winner"],
        "margin_all": details["score_diff"],
        "margin_correct": details["score_diff"].where(details["winner"] == "real"),
        "margin_wrong": details["score_diff"].where(details["winner"] == "fake"),
    })
    margin_df.to_csv(out_path, index=False)


def plot_score_distribution(real_scores, fake_scores, score_col, out_prefix, bins=30):

    # Histogram
    plt.figure(figsize=(7,5))

    plt.hist(real_scores, bins=bins, alpha=0.5, label=f"real (n={len(real_scores)})")
    plt.hist(fake_scores, bins=bins, alpha=0.5, label=f"fake (n={len(fake_scores)})")

    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Score distribution: {score_col}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{out_prefix}_{score_col}_hist.png", dpi=200)
    plt.close()

    # Scatter
    plt.figure(figsize=(5,5))

    plt.scatter(real_scores, fake_scores, alpha=0.6)

    min_v = min(real_scores.min(), fake_scores.min())
    max_v = max(real_scores.max(), fake_scores.max())

    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")  # y=x line

    plt.xlabel("Real score")
    plt.ylabel("Fake score")
    plt.title(f"Real vs Fake: {score_col}")

    plt.tight_layout()

    plt.savefig(f"{out_prefix}_{score_col}_scatter.png", dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_csv", default="path/to/real.csv")
    ap.add_argument("--fake_csv", default="path/to/fake.csv")
    ap.add_argument("--key", default="index", choices=["index", "image_file"])
    ap.add_argument("--tie", default="full", choices=["zero", "full", "skip"])
    ap.add_argument("--lower_is_better", action="store_true", help="Use if lower score means better quality")
    ap.add_argument("--scores", default="overall_score", help="Comma-separated list, e.g. overall_score,level_1_score,level_2_score,level_3_score")
    ap.add_argument("--save_details", action="store_true", help="Save per-index comparison results as CSV")
    ap.add_argument("--details_prefix", default="pairwise_details", help="Prefix for saved detail CSV files")
    ap.add_argument("--exclude_indices", type=str, default="", help="Comma-separated indices to exclude, e.g. 3,7,10")

    # new args
    ap.add_argument("--save_margin_csv", action="store_true", help="Save per-index margin values as CSV")
    ap.add_argument("--margin_csv_prefix", default="margin_values", help="Prefix for saved margin CSV files")
    ap.add_argument("--plot_margin_dist", action="store_true", help="Save histogram plot for three margin distributions")
    ap.add_argument("--bins", type=int, default=100, help="Number of bins for histogram")
    ap.add_argument("--plot_score_dist", action="store_true")
    ap.add_argument("--plot_prefix", default="score_distribution")

    args = ap.parse_args()

    real_df = pd.read_csv(args.real_csv)
    fake_df = pd.read_csv(args.fake_csv)

    # basic cleanup: ensure key exists
    for df, name in [(real_df, "real"), (fake_df, "fake")]:
        if args.key not in df.columns:
            raise KeyError(f"{name} csv missing key column: {args.key}")

    exclude_set = set()
    if args.exclude_indices:
        exclude_set.update(int(x.strip()) for x in args.exclude_indices.split(",") if x.strip())

    if exclude_set:
        print(f"[INFO] Excluding indices: {sorted(exclude_set)}")
        real_df = real_df[~real_df[args.key].isin(exclude_set)]
        fake_df = fake_df[~fake_df[args.key].isin(exclude_set)]

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
        res, details, real_scores, fake_scores, all_margin, correct_margin, wrong_margin = compute_pairwise(
            real_df, fake_df,
            key=args.key,
            score_col=sc,
            tie_mode=args.tie,
            lower_is_better=args.lower_is_better,
        )

        print(
            f"{sc}: acc={res['acc']:.4f}  "
            f"pairs={res['pairs']}  correct={res['correct']}  wrong={res['wrong']}  "
            f"ties={res['ties']}  skipped={res['skipped']}"
        )

        print(f"  margin(all):     mean ± std = {res['margin_mean_all']:.4f} ± {res['margin_std_all']:.4f}")
        print(f"  margin(correct): mean ± std = {res['margin_mean_correct']:.4f} ± {res['margin_std_correct']:.4f}")
        print(f"  margin(wrong):   mean ± std = {res['margin_mean_wrong']:.4f} ± {res['margin_std_wrong']:.4f}")

        print(details.to_string(index=False))
        print()

        if args.save_details:
            out_path = f"{args.details_prefix}_{sc}.csv"
            details.to_csv(out_path, index=False)
            print(f"saved details: {out_path}")

        if args.save_margin_csv:
            out_path = f"{args.margin_csv_prefix}_{sc}.csv"
            save_margin_values(details, args.key, sc, out_path)
            print(f"saved margin csv: {out_path}")

        if args.plot_margin_dist:
            out_path = f"{args.plot_prefix}_{sc}.png"
            save_margin_distribution_plot(
                all_margin=all_margin,
                correct_margin=correct_margin,
                wrong_margin=wrong_margin,
                score_col=sc,
                out_path=out_path,
                bins=args.bins,
            )
            print(f"saved margin plot: {out_path}")

        if args.save_details or args.save_margin_csv or args.plot_margin_dist:
            print()

    if args.plot_score_dist:
        plot_score_distribution(
            real_scores,
            fake_scores,
            sc,
            args.plot_prefix,
            bins=args.bins
        )

if __name__ == "__main__":
    main()