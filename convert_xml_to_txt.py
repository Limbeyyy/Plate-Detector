import os
import xml.etree.ElementTree as ET

# Paths
xml_dir = r"C:\Users\A C E R\OneDrive\Desktop\OCR Plate Detector\dataset\images\xml"
txt_dir = r"C:\Users\A C E R\OneDrive\Desktop\OCR Plate Detector\dataset\labels\train"

# Define your classes (0: plate)
classes = ["Plate"]

os.makedirs(txt_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    txt_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        txt_lines.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    txt_file = os.path.join(txt_dir, xml_file.replace(".xml", ".txt"))
    with open(txt_file, "w") as f:
        f.write("\n".join(txt_lines))

    print("Plates Converted Successfully!")
