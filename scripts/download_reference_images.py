#!/usr/bin/env python3
"""
Download reference images for the FAGER benchmark.

ABO mode:
  Downloads product images from the Amazon Berkeley Objects S3 bucket.
  Verifies sha256 checksums; skips files that are already present and correct.

Culture mode (interactive):
  Culture images cannot be redistributed. This script prints a Google Images
  search URL for each prompt, then asks you to paste a direct image URL.
  It downloads and saves the image to the expected filename.

Usage:
    python scripts/download_reference_images.py --dataset ABO
    python scripts/download_reference_images.py --dataset culture
    python scripts/download_reference_images.py --dataset ABO --dry-run
    python scripts/download_reference_images.py --dataset ABO --output-dir /tmp/abo_imgs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download url → dest.  Returns True on success."""
    if dry_run:
        print(f"  [dry-run] would download: {url} → {dest}")
        return True

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    if HAS_TQDM:
        bar = tqdm(total=total, unit="B", unit_scale=True,
                   desc=dest.name, leave=False)
    else:
        bar = None

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            if bar:
                bar.update(len(chunk))

    if bar:
        bar.close()
    return True


def guess_ext_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else ".jpg"


# ---------------------------------------------------------------------------
# ABO
# ---------------------------------------------------------------------------

def run_abo(manifest_path: Path, output_dir: Path, dry_run: bool) -> None:
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"ABO manifest: {len(rows)} images  →  {output_dir}")
    skipped = downloaded = failed = 0

    for row in rows:
        filename   = row["filename"]
        source_url = row["source_url"]
        expected   = row["sha256"]
        dest       = output_dir / filename

        if dest.exists() and expected:
            actual = sha256_file(dest)
            if actual == expected:
                print(f"  [skip] {filename} (already present, sha256 OK)")
                skipped += 1
                continue
            else:
                print(f"  [redownload] {filename} (sha256 mismatch)")

        print(f"  [download] {filename}")
        try:
            download_file(source_url, dest, dry_run=dry_run)
            if not dry_run and expected:
                actual = sha256_file(dest)
                if actual != expected:
                    print(f"  [WARN] sha256 mismatch after download: {filename}")
                    print(f"         expected: {expected}")
                    print(f"         got:      {actual}")
                    failed += 1
                    continue
            downloaded += 1
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed.")


# ---------------------------------------------------------------------------
# Culture (interactive)
# ---------------------------------------------------------------------------

def run_culture(manifest_path: Path, output_dir: Path, dry_run: bool) -> None:
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("Culture images cannot be redistributed.")
    print("For each prompt you will be shown a Google Images search URL.")
    print("Find an appropriate image, copy its direct URL, and paste it here.\n")

    downloaded = skipped = failed = 0

    for row in rows:
        prompt       = row["prompt"]
        filename     = row["filename"]
        expected_sha = row["sha256"]
        query        = row["search_query"]
        dest         = output_dir / filename

        if dest.exists() and expected_sha:
            actual = sha256_file(dest)
            if actual == expected_sha:
                print(f"[skip] {filename} — already present and sha256 matches")
                skipped += 1
                continue

        encoded = urllib.parse.quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded}&tbm=isch"

        print("─" * 70)
        print(f"Prompt:  {prompt}")
        print(f"Save as: {filename}")
        print(f"Search:  {search_url}")

        if dry_run:
            print("  [dry-run] skipping download")
            downloaded += 1
            continue

        while True:
            try:
                image_url = input("Paste direct image URL (or 'skip'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(0)

            if image_url.lower() == "skip":
                print("  Skipped.")
                skipped += 1
                break

            if not image_url.startswith("http"):
                print("  URL must start with http. Try again.")
                continue

            # derive extension from the pasted URL; keep manifest filename stem
            stem = Path(filename).stem
            ext = guess_ext_from_url(image_url)
            # always save with the manifest filename (ignore pasted extension)
            actual_dest = output_dir / filename

            try:
                download_file(image_url, actual_dest, dry_run=False)
                print(f"  Saved → {actual_dest}")
                downloaded += 1
                break
            except Exception as e:
                print(f"  Download failed: {e}  Try again or type 'skip'.")

    print(f"\nDone: {downloaded} downloaded/selected, {skipped} skipped, {failed} failed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download FAGER reference images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["ABO", "culture"],
        help="Which dataset's reference images to download.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Where to save images. Defaults to data/<dataset>/selected_images/.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without actually downloading.",
    )
    args = ap.parse_args()

    dataset    = args.dataset
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "data" / dataset / "selected_images"
    )
    manifest   = REPO_ROOT / "data" / dataset / "selected_images_manifest.csv"

    if dataset == "ABO":
        run_abo(manifest, output_dir, dry_run=args.dry_run)
    else:
        run_culture(manifest, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
