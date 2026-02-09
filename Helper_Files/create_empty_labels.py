import os

images_dir = "dataset/images/val"
labels_dir = "dataset/labels/val"

os.makedirs(labels_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")

for img_name in os.listdir(images_dir):
    if img_name.lower().endswith(image_extensions):
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            open(label_path, "w").close()

print("Empty label files created successfully")
