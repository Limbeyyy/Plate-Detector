import os

images_dir = "dataset/images/val"
labels_dir = "dataset/labels/val"

os.makedirs(labels_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Get all images
images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(image_extensions)
]

images.sort()  # keep order stable

start_index = 1  # change if needed

for idx, img_name in enumerate(images, start=start_index):
    old_img_path = os.path.join(images_dir, img_name)
    base_name, ext = os.path.splitext(img_name)

    new_base = f"plate_{idx:04d}"
    new_img_name = new_base + ext.lower()
    new_img_path = os.path.join(images_dir, new_img_name)

    # Rename image
    os.rename(old_img_path, new_img_path)

    # Rename label if exists
    old_label_path = os.path.join(labels_dir, base_name + ".txt")
    new_label_path = os.path.join(labels_dir, new_base + ".txt")

    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)
    else:
        print(f"Label missing for {img_name}")

print(" Images and labels renamed successfully")
