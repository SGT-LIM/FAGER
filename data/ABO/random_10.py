import gzip
import json
import pandas as pd
import os
import glob
import random

# =========================
# 1. 이미지 메타데이터 로드
# =========================
img_meta = pd.read_csv("images/metadata/images.csv.gz")
id2path = dict(zip(img_meta["image_id"], img_meta["path"]))

# =========================
# 2. 헬퍼 함수들
# =========================
def pick_en(d, field, key="value"):
    vals = d.get(field, [])
    if not isinstance(vals, list):
        return None
    for obj in vals:
        if obj.get("language_tag", "").startswith("en"):
            return obj.get(key)
    return None

def pick_en_name(d):
    return pick_en(d, "item_name", key="value")

def pick_en_brand(d):
    return pick_en(d, "brand", key="value")

# =========================
# 3. listings에서 후보 수집
# =========================
items = []

for fpath in sorted(glob.glob("listings/metadata/listings_*.json.gz")):
    print("Reading:", fpath)
    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            d = json.loads(line)

            image_id = d.get("main_image_id")
            if not image_id or image_id not in id2path:
                continue

            name = pick_en_name(d)
            if not name:
                continue

            brand = pick_en_brand(d)

            # ---- Solimo 제품 필터링 ----
            name_lower = name.lower()
            brand_lower = brand.lower() if brand else ""

            if "solimo" in name_lower or "solimo" in brand_lower:
                continue
            # --------------------------

            image_path = os.path.join("images/small", id2path[image_id])

            items.append(
                {
                    "prompt": name,
                    "brand": brand,
                    "image_id": image_id,
                    "image_path": image_path,
                }
            )

    # 어느 정도 모이면 멈춰도 됨 (원하면 더 크게 늘려도 됨)
    if len(items) >= 2000:
        break

print("Collected non-Solimo items:", len(items))

if len(items) < 10:
    raise ValueError("Non-Solimo items < 10. Relax the filter or scan more files.")

# =========================
# 4. 랜덤 10개 선택
# =========================
random.seed(42)
sample = random.sample(items, 100)

# =========================
# 5. CSV 저장
# =========================
df = pd.DataFrame(sample)
df.index = df.index + 1

output_path = "abo_random_10_nosolimo.csv"
df.to_csv(output_path, index=True, index_label="index")

print(f"\nSaved to {output_path}")
print(df[["prompt", "brand", "image_path"]])