# API Reference

Complete reference for all modules, scripts, and functions in the License Plate Detector system.

## 📑 Table of Contents

1. [Training Module](#training-module)
2. [Inference Module](#inference-module)
3. [Data Preprocessing Utilities](#data-preprocessing-utilities)
4. [Configuration](#configuration)
5. [YOLO Model API](#yolo-model-api)

---

## 🏋️ Training Module

### train.py

Main training script for initial model training.

**Location**: `training/train.py`

**Function**: `main()`

Initializes and trains a YOLO11 model on custom license plate dataset.

**Usage**:
```bash
python training/train.py
```

**Configuration**:
```python
from ultralytics import YOLO

def main():
    # Initialize model
    model = YOLO("yolo11n.pt")
    
    # Train
    model.train(
        data: str,              # Path to data.yaml
        epochs: int,            # Number of training epochs
        imgsz: int,             # Input image size
        batch: int,             # Batch size
        device: int | str,      # Device (0 for GPU, 'cpu' for CPU)
        project: str,           # Output directory
        name: str               # Experiment name
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | str | Required | Path to dataset YAML file |
| `epochs` | int | 100 | Number of training epochs |
| `imgsz` | int | 640 | Input image size (pixels) |
| `batch` | int | 16 | Batch size for training |
| `device` | int/str | 0 | GPU device ID or 'cpu' |
| `project` | str | "runs" | Output directory name |
| `name` | str | "plate_detector_yolo11" | Experiment name |

**Optional Parameters**:

```python
model.train(
    # ... required params ...
    
    # Optimization
    lr0=0.01,                    # Initial learning rate
    lrf=0.01,                    # Final learning rate factor
    momentum=0.937,              # SGD momentum
    weight_decay=0.0005,         # L2 regularization
    warmup_epochs=3,             # Warmup epochs
    
    # Data
    augment=True,                # Enable data augmentation
    cache=False,                 # Cache images ('ram', 'disk', False)
    workers=8,                   # DataLoader workers
    
    # Training control
    patience=50,                 # Early stopping patience
    save_period=10,              # Save checkpoint every N epochs
    val=True,                    # Validate during training
    
    # Advanced
    amp=True,                    # Automatic Mixed Precision
    pretrained=True,             # Use pretrained weights
    optimizer='auto',            # Optimizer ('SGD', 'Adam', 'auto')
)
```

**Output**:
```
runs/plate_detector_yolo11/
├── weights/
│   ├── best.pt              # Best model weights
│   └── last.pt              # Last epoch weights
├── results.csv              # Training metrics
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
└── args.yaml                # Training arguments
```

**Returns**: None (saves weights to disk)

---

### resume_train.py

Resume interrupted training from last checkpoint.

**Location**: `training/resume_train.py`

**Usage**:
```bash
python training/resume_train.py
```

**Code**:
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/last.pt")
model.train(resume=True)
```

**Parameters**:
- `resume=True`: Continue training from checkpoint

**Requirements**: `last.pt` must exist in the specified path

---

## 🔍 Inference Module

### detect_image.py

Detect license plates in a single image.

**Location**: `inference/detect_image.py`

**Usage**:
```bash
python inference/detect_image.py
```

**Code**:
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
results = model("test.jpg", conf=0.4)
results[0].show()
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str | Required | Path to image file |
| `conf` | float | 0.4 | Confidence threshold (0-1) |
| `iou` | float | 0.7 | NMS IoU threshold |
| `imgsz` | int | 640 | Input image size |
| `half` | bool | False | Use FP16 precision |
| `device` | int/str | 0 | GPU device or 'cpu' |

**Returns**: `Results` object

**Results Object Attributes**:
```python
results[0].boxes          # Bounding boxes (xyxy, conf, cls)
results[0].boxes.xyxy     # Box coordinates [x1, y1, x2, y2]
results[0].boxes.conf     # Confidence scores
results[0].boxes.cls      # Class IDs
results[0].orig_img       # Original image (numpy array)
results[0].show()         # Display annotated image
results[0].save()         # Save annotated image
```

**Example**:
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
results = model("car.jpg", conf=0.5)

# Access detections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    class_id = box.cls[0]
    print(f"Box: ({x1}, {y1}, {x2}, {y2}), Conf: {confidence:.2f}")
```

---

### detect_video.py

Detect license plates in video files.

**Location**: `inference/detect_video.py`

**Usage**:
```bash
python inference/detect_video.py
```

**Code**:
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
model.predict(source="video.mp4", conf=0.4, save=True)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str | Required | Path to video file |
| `conf` | float | 0.4 | Confidence threshold |
| `save` | bool | True | Save output video |
| `stream` | bool | False | Stream results (memory efficient) |
| `save_txt` | bool | False | Save detections as text |

**Output**: Annotated video saved to `runs/detect/predict/`

**Streaming Example** (memory efficient):
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")

for result in model.predict(source="video.mp4", stream=True):
    # Process each frame
    boxes = result.boxes
    # Your custom processing here
```

---

### detect_webcam.py

Real-time license plate detection from webcam.

**Location**: `inference/detect_webcam.py`

**Usage**:
```bash
python inference/detect_webcam.py
```

**Code**:
```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
model.predict(source=0, conf=0.4, show=True)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | int | 0 | Camera index (0 for default webcam) |
| `conf` | float | 0.4 | Confidence threshold |
| `show` | bool | True | Display real-time results |

**Controls**:
- Press **`q`** to quit
- Press **`p`** to pause/resume

---

### run_model.py

Advanced real-time license plate detection with interactive UI and controls.

**Location**: `run_model.py` (root directory)

**Usage**:
```bash
python run_model.py
```

**Architecture**:

The script implements a `LicensePlateDetector` class that provides production-ready webcam detection with real-time monitoring and control.

#### Class: `LicensePlateDetector`

**Initialization**:
```python
detector = LicensePlateDetector(
    model_path: str,
    conf_threshold: float = 0.4,
    camera_id: int = 0
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to YOLO model weights (`best.pt`) |
| `conf_threshold` | float | 0.4 | Initial confidence threshold (0.0-1.0) |
| `camera_id` | int | 0 | Camera device ID (0=default, 1=external, etc.) |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | YOLO | Loaded YOLO model instance |
| `conf_threshold` | float | Current confidence threshold |
| `default_conf` | float | Default confidence for reset |
| `frame_count` | int | Total frames processed |
| `saved_count` | int | Number of frames saved |
| `output_dir` | Path | Directory for saved frames (`runs/webcam_detections/`) |

**Methods**:

##### `run()`

Start the webcam detection loop with real-time display and keyboard controls.

```python
detector.run()
```

**Behavior**:
- Opens webcam stream
- Displays live detection window
- Processes keyboard input for interactive control
- Shows FPS, confidence, detection count, and status
- Handles errors and cleanup automatically

**Returns**: None

**Raises**: 
- `SystemExit`: If model or camera fails to load

##### `_draw_info_overlay(frame, fps, num_detections)`

Draw information overlay on the detection frame (internal method).

**Parameters**:
- `frame` (np.ndarray): Video frame to annotate
- `fps` (float): Current frames per second
- `num_detections` (int): Number of plates detected in frame

**Overlay Content**:
- FPS counter
- Confidence threshold
- Detection count (green if >0)
- Frame number
- Status message ("PLATE DETECTED!" or "Scanning...")

##### `_handle_key_press(key, frame)`

Handle keyboard input for interactive controls (internal method).

**Parameters**:
- `key` (int): OpenCV key code
- `frame` (np.ndarray): Current frame

**Returns**: 
- `bool`: True to continue, False to exit

##### `_print_controls()`

Print keyboard control reference to console (internal method).

**Keyboard Controls**:

| Key | Action | Behavior |
|-----|--------|----------|
| **Q** or **ESC** | Quit | Exit application |
| **+** or **=** | Increase confidence | Threshold += 0.05 (max 1.0) |
| **-** | Decrease confidence | Threshold -= 0.05 (min 0.1) |
| **S** | Save frame | Save current frame to `runs/webcam_detections/` |
| **R** | Reset | Reset confidence to default (0.4) |

**Configuration**:

Edit constants in the `main()` function:

```python
def main():
    # Configuration
    MODEL_PATH = "training/runs/plate_detector_yolo113/weights/best.pt"
    CONFIDENCE_THRESHOLD = 0.4  # Initial threshold
    CAMERA_ID = 0               # 0=default, 1=external
    
    detector = LicensePlateDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        camera_id=CAMERA_ID
    )
    
    detector.run()
```

**Usage Examples**:

**Example 1: Default Configuration**
```python
from run_model import LicensePlateDetector

# Use default settings
detector = LicensePlateDetector(
    model_path="training/runs/plate_detector_yolo113/weights/best.pt"
)
detector.run()
```

**Example 2: Custom Configuration**
```python
# External webcam with lower confidence
detector = LicensePlateDetector(
    model_path="training/runs/plate_detector_yolo113/weights/best.pt",
    conf_threshold=0.3,  # More sensitive
    camera_id=1          # External camera
)
detector.run()
```

**Example 3: Different Model**
```python
# Use last.pt instead of best.pt
detector = LicensePlateDetector(
    model_path="training/runs/plate_detector_yolo113/weights/last.pt",
    conf_threshold=0.5,  # Higher confidence
    camera_id=0
)
detector.run()
```

**Output**:

**Console Output**:
```
Loading YOLO model from: training/runs/plate_detector_yolo113/weights/best.pt
✓ Model loaded successfully!

Opening camera (ID: 0)...
✓ Camera opened successfully!

============================================================
LIVE DETECTION STARTED
============================================================

 KEYBOARD CONTROLS:
  Q or ESC  : Quit
  + or =    : Increase confidence threshold
  - or _    : Decrease confidence threshold
  S         : Save current frame
  R         : Reset confidence to default

============================================================
```

**Saved Frames**:
- Directory: `runs/webcam_detections/`
- Format: `detection_0001.jpg`, `detection_0002.jpg`, ...
- Content: Annotated frame with bounding boxes

**Exit Summary**:
```
============================================================
Total frames processed: 1523
Frames saved: 5
============================================================
✓ Cleanup complete. Exiting...
```

**Features**:
- ✅ Real-time FPS monitoring
- ✅ Interactive confidence adjustment
- ✅ Frame capture with annotations
- ✅ Professional on-screen overlay
- ✅ Automatic error handling
- ✅ Multi-camera support
- ✅ Graceful shutdown (Ctrl+C safe)

**Performance**:
- **FPS**: 30-45 on GPU, 10-15 on CPU
- **Latency**: ~22-33ms per frame (GPU)
- **Memory**: ~500MB (includes model + OpenCV)

**Error Handling**:
- Model not found → Exits with helpful message
- Camera unavailable → Suggests trying different IDs
- Keyboard interrupt → Clean shutdown with statistics

---

## 🛠️ Data Preprocessing Utilities

### convert_xml_to_txt.py

Convert Pascal VOC XML annotations to YOLO format.

**Location**: `convert_xml_to_txt.py`

**Usage**:
```bash
python convert_xml_to_txt.py
```

**Configuration**:
```python
import os
import xml.etree.ElementTree as ET

# Paths to configure
xml_dir = r"C:\path\to\xml\annotations"      # Input XML directory
txt_dir = r"C:\path\to\yolo\labels\train"    # Output TXT directory
classes = ["Plate"]                           # Class names

# Conversion runs automatically
```

**Input Format** (XML):
```xml
<annotation>
  <size>
    <width>1920</width>
    <height>1080</height>
  </size>
  <object>
    <name>Plate</name>
    <bndbox>
      <xmin>450</xmin>
      <ymin>300</ymin>
      <xmax>650</xmax>
      <ymax>380</ymax>
    </bndbox>
  </object>
</annotation>
```

**Output Format** (YOLO TXT):
```
0 0.2865 0.3148 0.1042 0.0741
```

**Conversion Formula**:
```python
x_center = (xmin + xmax) / 2 / image_width
y_center = (ymin + ymax) / 2 / image_height
width = (xmax - xmin) / image_width
height = (ymax - ymin) / image_height
```

---

### check_dataset.py

Validate dataset for missing, empty, or malformed labels.

**Location**: `check_dataset.py`

**Usage**:
```bash
python check_dataset.py
```

**Configuration**:
```python
# Edit these paths
IMAGE_DIRS = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/images/test"
]
LABEL_DIR = "dataset/labels"
```

**Checks Performed**:
1. **Missing labels**: Images without corresponding `.txt` files
2. **Empty labels**: Label files with 0 bytes or no content
3. **Bad format**: Labels not following YOLO format (5 values per line)

**Output**:
```
========== Dataset Check Results ==========

Missing label files: 2
  - dataset/images/train/plate_0050.jpeg
  - dataset/images/val/plate_0012.jpeg

Empty label files: 1
  - dataset/images/train/plate_0023.jpeg

Bad label format (not 5 values): 0

All labels exist, non-empty, and look correctly formatted!
```

**Return Code**: 0 if all checks pass

---

### create_empty_labels.py

Create empty label files for images without annotations.

**Location**: `create_empty_labels.py`

**Usage**:
```bash
python create_empty_labels.py
```

**Configuration**:
```python
images_dir = "dataset/images/val"    # Image directory
labels_dir = "dataset/labels/val"    # Label directory
```

**Behavior**:
- Scans `images_dir` for `.jpg`, `.jpeg`, `.png` files
- Creates empty `.txt` file in `labels_dir` if not exists
- Skips if label already exists

**Use Case**: Background images without license plates (negative samples)

---

### rename_img_labels.py

Batch rename images and corresponding labels to standardized format.

**Location**: `rename_img_labels.py`

**Usage**:
```bash
python rename_img_labels.py
```

**Configuration**:
```python
images_dir = "dataset/images/val"    # Image directory
labels_dir = "dataset/labels/val"    # Label directory
start_index = 1                       # Starting number
```

**Naming Convention**:
```
plate_0001.jpg → plate_0001.txt
plate_0002.jpg → plate_0002.txt
...
plate_9999.jpg → plate_9999.txt
```

**Behavior**:
- Renames images in sorted order
- Renames corresponding labels
- Warns if label is missing

**Example Output**:
```
Label missing for old_image_name.jpg
Images and labels renamed successfully
```

---

## ⚙️ Configuration

### data.yaml

Dataset configuration file for YOLO training.

**Location**: `dataset/data.yaml`

**Format**:
```yaml
# Paths (absolute recommended)
path: C:/Users/YourName/Desktop/Plate_Detector/dataset
train: C:/Users/YourName/Desktop/Plate_Detector/dataset/images/train
val: C:/Users/YourName/Desktop/Plate_Detector/dataset/images/val
test: C:/Users/YourName/Desktop/Plate_Detector/dataset/images/val

# Dataset metadata
nc: 1                    # Number of classes
names:
  - Kataho_Plate         # Class names (index 0)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | Dataset root directory (absolute path) |
| `train` | str | Training images directory |
| `val` | str | Validation images directory |
| `test` | str | Test images directory (optional) |
| `nc` | int | Number of classes |
| `names` | list | List of class names (ordered by index) |

**Multi-Class Example**:
```yaml
nc: 3
names:
  - Plate        # Class 0
  - Car          # Class 1
  - Motorcycle   # Class 2
```

---

## 🧠 YOLO Model API

### Model Initialization

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO("yolo11n.pt")

# Load custom trained model
model = YOLO("runs/plate_detector/weights/best.pt")

# Load from checkpoint
model = YOLO("runs/plate_detector/weights/last.pt")
```

### Training

```python
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

### Prediction

```python
# Single image
results = model("image.jpg")

# Multiple images
results = model(["img1.jpg", "img2.jpg"])

# Video
results = model("video.mp4", stream=True)

# Webcam
results = model(0)  # 0 = default camera
```

### Validation

```python
metrics = model.val()
print(metrics.box.map)     # mAP50-95
print(metrics.box.map50)   # mAP50
print(metrics.box.map75)   # mAP75
```

### Export

```python
# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to CoreML
model.export(format='coreml')
```

### Results API

```python
results = model("image.jpg")

# Access first result
result = results[0]

# Bounding boxes
boxes = result.boxes
boxes.xyxy      # Tensor [N, 4] (x1, y1, x2, y2)
boxes.xywh      # Tensor [N, 4] (x_center, y_center, width, height)
boxes.conf      # Tensor [N] (confidence scores)
boxes.cls       # Tensor [N] (class IDs)

# Original image
orig_img = result.orig_img  # numpy array (H, W, C)

# Display
result.show()

# Save
result.save(filename="output.jpg")

# Plot
import matplotlib.pyplot as plt
plt.imshow(result.plot())
plt.show()
```

---

## 📊 Complete Example

### End-to-End Pipeline

```python
from ultralytics import YOLO
import cv2

# 1. Train model
model = YOLO("yolo11n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

# 2. Validate
metrics = model.val()
print(f"mAP50-95: {metrics.box.map:.3f}")

# 3. Inference
trained_model = YOLO("runs/plate_detector_yolo11/weights/best.pt")
results = trained_model("test_image.jpg", conf=0.4)

# 4. Process results
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = box.conf[0].item()
    
    print(f"Plate detected at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    print(f"Confidence: {confidence:.2f}")
    
    # Crop plate region
    img = results[0].orig_img
    plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite("plate_crop.jpg", plate_crop)

# 5. Export for deployment
trained_model.export(format='onnx')
```

---

## 🔗 External Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

For high-level usage, see [USAGE_GUIDE.md](USAGE_GUIDE.md). For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).
