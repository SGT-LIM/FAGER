#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compute simple mean of overall_score")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()

    # CSV 읽기
    df = pd.read_csv(args.csv)

    if "overall_score" not in df.columns:
        raise ValueError("CSV must contain 'overall_score' column")

    # 숫자로 변환 + NaN 제거
    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    df = df.dropna(subset=["overall_score"])

    # 평균 계산
    mean_score = df["overall_score"].mean()

    print(f"CSV: {args.csv}")
    print(f"Num images: {len(df)}")
    print(f"Simple mean overall_score: {mean_score:.4f}")


if __name__ == "__main__":
    main()