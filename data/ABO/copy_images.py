import pandas as pd
import shutil
import os

# =========================
# 1. CSV 로드
# =========================
csv_path = "abo_random_100.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. 저장 폴더 생성
# =========================
output_dir = "selected_images"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 3. 이미지 복사
# =========================
for idx, row in df.iterrows():
    src_path = row["image_path"]  # ex: images/small/e4/e429aab6.jpg
    
    # 파일명만 추출
    filename = os.path.basename(src_path)
    
    # 새 파일명: index_원본파일명.jpg
    new_filename = f"{idx+1:02d}_{filename}"
    dst_path = os.path.join(output_dir, new_filename)
    
    # 복사
    shutil.copy(src_path, dst_path)

    print(f"Copied: {src_path} → {dst_path}")

print("\nDone.")