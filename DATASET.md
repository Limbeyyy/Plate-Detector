# Dataset Documentation

Comprehensive guide for preparing, organizing, and managing datasets for the License Plate Detector system.

## 📋 Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Directory Structure](#directory-structure)
3. [Data Format](#data-format)
4. [Dataset Creation](#dataset-creation)
5. [Labeling Guidelines](#labeling-guidelines)
6. [Data Augmentation](#data-augmentation)
7. [Dataset Statistics](#dataset-statistics)
8. [Best Practices](#best-practices)

---

## 📊 Dataset Overview

### Purpose

The dataset is used to train a YOLO11 object detection model to identify and localize license plates ("Kataho_Plate") in images.

### Dataset Split

| Split | Purpose | Recommended Size |
|-------|---------|------------------|
| **Training** | Model learning | 70-80% of total data |
| **Validation** | Hyperparameter tuning | 20-30% of total data |
| **Test** | Final evaluation | Optional (can use validation set) |

### Minimum Dataset Size

- **Minimum**: 100-200 images (may underfit)
- **Recommended**: 500-1000 images (good performance)
- **Optimal**: 2000+ images (best performance)

**Rule of thumb**: More diverse data = better generalization

---

## 📁 Directory Structure

### Standard Layout

```
dataset/
├── data.yaml                 # Dataset configuration
├── images/
│   ├── train/               # Training images (70-80%)
│   │   ├── plate_0001.jpg
│   │   ├── plate_0002.jpg
│   │   └── ...
│   ├── val/                 # Validation images (20-30%)
│   │   ├── plate_0201.jpg
│   │   ├── plate_0202.jpg
│   │   └── ...
│   └── test/                # Optional test images
│       └── ...
└── labels/
    ├── train/               # Training labels (YOLO format)
    │   ├── plate_0001.txt
    │   ├── plate_0002.txt
    │   └── ...
    ├── val/                 # Validation labels
    │   ├── plate_0201.txt
    │   ├── plate_0202.txt
    │   └── ...
    └── test/                # Optional test labels
        └── ...
```

### Rules

1. **One label per image**: Each `.jpg`/`.jpeg`/`.png` must have a corresponding `.txt` file
2. **Same filename**: `plate_0001.jpg` ↔ `plate_0001.txt`
3. **Same structure**: Mirror structure in `images/` and `labels/`
4. **Empty labels**: Images without plates need empty `.txt` files (for negative samples)

---

## 🏷️ Data Format

### Image Format

**Supported Formats**:
- `.jpg`, `.jpeg` (recommended)
- `.png`
- `.bmp`

**Image Requirements**:
- **Resolution**: 640×640 minimum (higher is better)
- **Quality**: Good lighting, minimal blur
- **Color space**: RGB (3 channels)

**Example Images**:
```
plate_0001.jpg  (1920×1080, RGB)
plate_0002.jpg  (1280×720, RGB)
plate_0003.jpg  (640×480, RGB)
```

### Label Format (YOLO)

Each label file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Parameters**:
- `class_id`: Class index (0 for "Kataho_Plate")
- `x_center`: Center X coordinate (normalized to 0-1)
- `y_center`: Center Y coordinate (normalized to 0-1)
- `width`: Bounding box width (normalized to 0-1)
- `height`: Bounding box height (normalized to 0-1)

**Example** (`plate_0001.txt`):
```
0 0.512 0.347 0.284 0.156
```

This represents:
- Class: 0 (Kataho_Plate)
- Center: (51.2% of image width, 34.7% of image height)
- Size: (28.4% of image width, 15.6% of image height)

### Multiple Objects

If an image has multiple license plates:

```
0 0.312 0.247 0.184 0.126
0 0.712 0.547 0.224 0.156
0 0.512 0.847 0.204 0.136
```

---

## 🎨 Dataset Creation

### Step 1: Collect Images

**Sources**:
1. **Custom photography**: Take photos of vehicles with visible plates
2. **Public datasets**: Download and relabel existing datasets
3. **Web scraping**: Collect images from the internet (respect copyright)
4. **Synthetic data**: Generate synthetic license plates

**Diversity Requirements**:
- **Angles**: Front, rear, tilted (0-45°)
- **Distances**: Close-up, medium, far
- **Lighting**: Day, night, shadows, reflections
- **Weather**: Sunny, cloudy, rainy
- **Backgrounds**: Road, parking lot, urban, rural
- **Plate conditions**: Clean, dirty, damaged, partially occluded

### Step 2: Organize Images

Split images into train/val sets:

**Option A: Manual Split**
```bash
# Place 70% in train, 30% in val
cp image_001.jpg dataset/images/train/
cp image_002.jpg dataset/images/val/
```

**Option B: Automated Split** (Python)
```python
import os
import shutil
import random

images = [f for f in os.listdir("all_images/") if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

split_idx = int(len(images) * 0.7)
train_images = images[:split_idx]
val_images = images[split_idx:]

for img in train_images:
    shutil.copy(f"all_images/{img}", "dataset/images/train/")

for img in val_images:
    shutil.copy(f"all_images/{img}", "dataset/images/val/")
```

### Step 3: Annotate Images

Use annotation tools to create bounding boxes.

**Recommended Tools**:

#### 1. LabelImg (Recommended)
- **Format**: YOLO (direct export)
- **Install**: `pip install labelImg`
- **Run**: `labelImg`
- **Steps**:
  1. Open directory: `dataset/images/train`
  2. Change save format to "YOLO"
  3. Draw bounding boxes around plates
  4. Press Ctrl+S to save

#### 2. CVAT (Advanced)
- **Format**: YOLO (export)
- **URL**: https://cvat.org/
- **Features**: Collaborative annotation, auto-labeling

#### 3. Roboflow (Cloud)
- **Format**: YOLO (export)
- **URL**: https://roboflow.com/
- **Features**: Auto-annotation, augmentation, hosting

### Step 4: Validate Dataset

Run validation script:

```bash
python check_dataset.py
```

Fix any issues before training.

---

## 📏 Labeling Guidelines

### Bounding Box Rules

1. **Tight Fit**: Box should closely fit the license plate edges
   ```
   ✓ GOOD: [  PLATE  ]
   ✗ BAD:  [    PLATE    ]  (too much padding)
   ```

2. **Include All Characters**: Entire plate text must be inside the box
   ```
   ✓ GOOD: [ ABC 1234 ]
   ✗ BAD:  [ ABC 12 ]34  (text cut off)
   ```

3. **Exclude Extras**: Don't include surrounding vehicle parts
   ```
   ✓ GOOD: Just the plate
   ✗ BAD:  Plate + bumper + lights
   ```

4. **Partial Occlusion**: If >50% of plate is visible, label it
   ```
   ✓ GOOD: 70% visible → Label
   ✗ BAD:  <30% visible → Don't label
   ```

5. **Blurry/Unreadable**: Still label if plate shape is distinguishable
   ```
   ✓ GOOD: Blurred but plate-shaped → Label
   ✗ BAD:  Cannot distinguish → Don't label
   ```

### Edge Cases

| Scenario | Action |
|----------|--------|
| Multiple plates in image | Label all plates |
| Reflections/glare | Label if >50% visible |
| Severely damaged plate | Label if recognizable |
| Toy/fake plate | Don't label (unless collecting toys) |
| Plate in photo/poster | Depends on use case (usually don't label) |
| Very small plate (<1% of image) | Optional (may hurt performance) |

---

## 🔄 Data Augmentation

YOLO11 automatically applies augmentations during training. You can customize in `train.py`:

### Default Augmentations

YOLO applies these by default:
- **Mosaic**: Combine 4 images into 1
- **Mixup**: Blend two images
- **Random flip**: Horizontal flip
- **Random scale**: Scale ±50%
- **Random crop**: Crop and resize
- **Color jitter**: HSV adjustments

### Custom Augmentation

```python
model.train(
    data="dataset/data.yaml",
    epochs=100,
    
    # Augmentation settings
    hsv_h=0.015,        # Hue augmentation
    hsv_s=0.7,          # Saturation augmentation
    hsv_v=0.4,          # Value augmentation
    degrees=0.0,        # Rotation (±degrees)
    translate=0.1,      # Translation (±fraction)
    scale=0.5,          # Scale (±fraction)
    shear=0.0,          # Shear (±degrees)
    perspective=0.0,    # Perspective distortion
    flipud=0.0,         # Vertical flip probability
    fliplr=0.5,         # Horizontal flip probability
    mosaic=1.0,         # Mosaic probability
    mixup=0.0,          # Mixup probability
)
```

### Manual Pre-Augmentation

Generate augmented images beforehand:

```python
import albumentations as A
import cv2

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
    A.RandomRain(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

image = cv2.imread("plate_0001.jpg")
# Read YOLO labels, apply transform, save augmented image/label
```

---

## 📊 Dataset Statistics

### Example Dataset

```
Total Images: 500
├── Train: 350 (70%)
└── Val: 150 (30%)

Class Distribution:
├── Kataho_Plate: 500 instances

Average Instances per Image: 1.0
Median Instances per Image: 1

Bounding Box Size Statistics:
├── Mean Width: 0.245 (24.5% of image)
├── Mean Height: 0.142 (14.2% of image)
├── Min Area: 0.012 (1.2% of image)
└── Max Area: 0.156 (15.6% of image)

Image Resolution Distribution:
├── 1920×1080: 320 images
├── 1280×720: 150 images
└── 640×480: 30 images
```

### Generate Statistics (Python)

```python
import os
import numpy as np

label_dir = "dataset/labels/train"
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

widths = []
heights = []
instances_per_image = []

for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        instances_per_image.append(len(lines))
        
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                cls_id, x, y, w, h = map(float, parts)
                widths.append(w)
                heights.append(h)

print(f"Total images: {len(label_files)}")
print(f"Total instances: {sum(instances_per_image)}")
print(f"Mean instances per image: {np.mean(instances_per_image):.2f}")
print(f"Mean width: {np.mean(widths):.3f}")
print(f"Mean height: {np.mean(heights):.3f}")
```

---

## ✅ Best Practices

### Data Quality

1. ✅ **High resolution**: ≥640 pixels on smallest dimension
2. ✅ **Good lighting**: Avoid extreme darkness or overexposure
3. ✅ **Sharp focus**: Minimize motion blur
4. ✅ **Diverse angles**: Include various viewpoints
5. ✅ **Balanced split**: 70-80% train, 20-30% val

### Annotation Quality

1. ✅ **Accurate boxes**: Tight fit around plates
2. ✅ **Consistent labels**: Same labeling standards across dataset
3. ✅ **Double-check**: Review annotations for errors
4. ✅ **Handle edge cases**: Decide rules for occlusions, reflections

### Dataset Size

| Dataset Size | Expected Performance |
|--------------|---------------------|
| 100-200 | Basic detection, may miss edge cases |
| 500-1000 | Good detection, handles most scenarios |
| 2000-5000 | Excellent detection, robust to variations |
| 5000+ | State-of-the-art, handles rare cases |

### Iteration Strategy

1. **Start small**: Train on 100-200 images to validate pipeline
2. **Analyze errors**: Identify failure cases
3. **Collect targeted data**: Gather images for failure scenarios
4. **Retrain**: Incrementally improve dataset
5. **Repeat**: Continue until performance is acceptable

---

## 🔍 Troubleshooting

### Issue: Model Not Learning

**Possible Causes**:
- Incorrect label format
- Mismatched image/label filenames
- All labels in wrong split (e.g., all in val, none in train)

**Solution**: Run `python check_dataset.py`

### Issue: Low Validation mAP

**Possible Causes**:
- Insufficient data diversity
- Train/val distribution mismatch
- Annotation errors

**Solution**:
- Collect more diverse data
- Re-split dataset randomly
- Review and fix annotations

### Issue: Overfitting

**Symptoms**: High train mAP, low val mAP

**Solution**:
- Increase data augmentation
- Collect more training data
- Reduce model complexity (yolo11n instead of yolo11m)

---

## 📚 Resources

- [labelImg GitHub](https://github.com/tzutalin/labelImg)
- [CVAT Documentation](https://opencv.github.io/cvat/)
- [Roboflow Tutorials](https://docs.roboflow.com/)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)

---

For training with your dataset, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
