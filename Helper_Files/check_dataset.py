import os

# ========== MODIFY THESE PATHS AS NEEDED ==========
IMAGE_DIRS = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/images/test"
]
LABEL_DIR = "dataset/labels"
# ==================================================

def get_image_files(dir_path):
    exts = [".jpg", ".jpeg", ".png"]
    return [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts]

missing = []
empty = []
bad_format = []

for sub in IMAGE_DIRS:
    img_dir = os.path.join(os.getcwd(), sub)
    lab_dir = os.path.join(os.getcwd(), LABEL_DIR, sub.split("/")[-1])

    if not os.path.isdir(img_dir):
        print(f"[WARNING] Missing image directory: {img_dir}")
        continue
    if not os.path.isdir(lab_dir):
        print(f"[WARNING] Missing label directory: {lab_dir}")

    images = get_image_files(img_dir)
    for img in images:
        img_name = os.path.splitext(img)[0]
        txt_file = os.path.join(lab_dir, img_name + ".txt")

        # Missing label file?
        if not os.path.isfile(txt_file):
            missing.append(os.path.join(sub, img))
            continue

        # Empty label?
        size = os.path.getsize(txt_file)
        if size == 0:
            empty.append(os.path.join(sub, img))
            continue

        # Check YOLO format (simple)
        with open(txt_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            empty.append(os.path.join(sub, img))
            continue

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad_format.append(os.path.join(sub, img))
                break
            # class index check
            if not parts[0].isdigit():
                bad_format.append(os.path.join(sub, img))
                break

print("\n========== Dataset Check Results ==========\n")
print(f"Missing label files: {len(missing)}")
for x in missing:
    print("  -", x)

print(f"\nEmpty label files: {len(empty)}")
for x in empty:
    print("  -", x)

print(f"\nBad label format (not 5 values): {len(bad_format)}")
for x in bad_format:
    print("  -", x)

if not missing and not empty and not bad_format:
    print("\nAll labels exist, non‑empty, and look correctly formatted!")
