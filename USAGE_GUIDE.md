# Usage Guide

Comprehensive guide for using the License Plate Detector system, from dataset preparation to model deployment.

## 📚 Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Training Workflow](#training-workflow)
3. [Inference Workflow](#inference-workflow)
4. [Configuration Customization](#configuration-customization)
5. [Performance Optimization](#performance-optimization)
6. [Common Issues](#common-issues)

---

## 📊 Dataset Preparation

### Step 1: Organize Your Images

Create the directory structure:

```bash
mkdir -p dataset/images/train
mkdir -p dataset/images/val
mkdir -p dataset/labels/train
mkdir -p dataset/labels/val
```

Place your images:
- **Training images** (70-80% of data): `dataset/images/train/`
- **Validation images** (20-30% of data): `dataset/images/val/`

Supported formats: `.jpg`, `.jpeg`, `.png`

### Step 2: Create Annotations

#### Option A: Manual Annotation (Recommended)

Use labeling tools:
- [LabelImg](https://github.com/tzutalin/labelImg) - Save in YOLO format
- [CVAT](https://github.com/opencv/cvat) - Export as YOLO
- [Roboflow](https://roboflow.com/) - Online annotation + export

**YOLO Format**:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example annotation (`plate_0001.txt`):
```
0 0.512 0.347 0.284 0.156
```

#### Option B: Convert from XML (Pascal VOC)

If you have XML annotations:

1. **Edit `convert_xml_to_txt.py`** - Update paths:
   ```python
   xml_dir = r"C:\path\to\your\xml\annotations"
   txt_dir = r"C:\path\to\your\dataset\labels\train"
   classes = ["Plate"]  # Update if different
   ```

2. **Run conversion**:
   ```bash
   python convert_xml_to_txt.py
   ```

### Step 3: Validate Dataset

Run the dataset checker:

```bash
python check_dataset.py
```

**Output Analysis**:
- ✅ **All clear**: Proceed to training
- ❌ **Missing labels**: Use `create_empty_labels.py` or annotate missing images
- ❌ **Bad format**: Check label files, ensure 5 values per line

**Fix missing labels** (for images without plates):
```bash
# Edit create_empty_labels.py to specify directory
python create_empty_labels.py
```

### Step 4: Configure Dataset YAML

Edit `dataset/data.yaml`:

```yaml
# Update to your absolute paths
path: C:/Users/YourName/Desktop/Plate_Detector/Plate-Detector/dataset
train: C:/Users/YourName/Desktop/Plate_Detector/Plate-Detector/dataset/images/train
val: C:/Users/YourName/Desktop/Plate_Detector/Plate-Detector/dataset/images/val
test: C:/Users/YourName/Desktop/Plate_Detector/Plate-Detector/dataset/images/val

nc: 1  # Number of classes
names:
  - Kataho_Plate  # Your class name
```

**Important**: Use absolute paths, not relative paths.

### Optional: Batch Rename Files

To rename images/labels to a consistent format (`plate_0001.jpg`, etc.):

1. **Edit `rename_img_labels.py`**:
   ```python
   images_dir = "dataset/images/train"  # or val
   labels_dir = "dataset/labels/train"  # or val
   start_index = 1
   ```

2. **Run**:
   ```bash
   python rename_img_labels.py
   ```

---

## 🏋️ Training Workflow

### Initial Training

#### Step 1: Configure Training Parameters

Edit `training/train.py`:

```python
from ultralytics import YOLO

def main():
    # Load pre-trained model (auto-downloads on first run)
    model = YOLO("yolo11n.pt")  # Options: yolo11n, yolo11s, yolo11m

    model.train(
        data=r"C:\path\to\your\dataset\data.yaml",  # Update path
        epochs=100,        # Training iterations
        imgsz=640,        # Image size (640 recommended)
        batch=16,         # Batch size (adjust for GPU memory)
        device=0,         # GPU device (0=first GPU, 'cpu' for CPU)
        project="runs",   # Output directory
        name="plate_detector_yolo11"  # Experiment name
    )

if __name__ == "__main__":
    main()
```

**Key Parameters**:
- `epochs`: More epochs = better accuracy (100-200 recommended)
- `batch`: Higher = faster training (reduce if GPU memory errors)
- `device`: `0` for GPU, `'cpu'` for CPU (much slower)
- `imgsz`: 640 is standard, 1280 for higher accuracy (slower)

#### Step 2: Start Training

```bash
cd training
python train.py
```

**Training Progress**:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      4.5G      1.234      0.567      1.123         45        640
  2/100      4.5G      1.012      0.456      0.987         48        640
...
```

**Monitor**:
- `box_loss`: Bounding box accuracy (should decrease)
- `cls_loss`: Classification loss (should decrease)
- Training completes when losses plateau

#### Step 3: View Results

After training:
```
runs/plate_detector_yolo11/
├── weights/
│   ├── best.pt              # Best model
│   └── last.pt              # Last epoch
├── results.png              # Loss/metrics curves
├── confusion_matrix.png     # Classification matrix
└── results.csv              # Epoch data
```

**Check `results.png`** to verify training convergence.

### Resume Training

If training is interrupted:

```bash
cd training
python resume_train.py
```

This loads `last.pt` and continues from the last epoch.

### Transfer Learning (Advanced)

To train on a different model variant:

```python
# In train.py
model = YOLO("yolo11s.pt")  # Small model (better accuracy)
# or
model = YOLO("yolo11m.pt")  # Medium model (even better)
```

---

## 🔍 Inference Workflow

### Detect in Images

#### Method 1: Using detect_image.py

1. **Edit `inference/detect_image.py`**:
   ```python
   from ultralytics import YOLO

   # Load trained model
   model = YOLO("runs/plate_detector_yolo11/weights/best.pt")
   
   # Run inference
   results = model("test.jpg", conf=0.4)  # Update image path
   results[0].show()  # Display result
   ```

2. **Run**:
   ```bash
   python inference/detect_image.py
   ```

#### Method 2: Command Line (Ultralytics CLI)

```bash
yolo detect predict model=runs/plate_detector_yolo11/weights/best.pt source=test.jpg conf=0.4
```

### Detect in Videos

1. **Edit `inference/detect_video.py`**:
   ```python
   from ultralytics import YOLO

   model = YOLO("runs/plate_detector_yolo11/weights/best.pt")
   model.predict(source="video.mp4", conf=0.4, save=True)
   ```

2. **Run**:
   ```bash
   python inference/detect_video.py
   ```

**Output**: Results saved to `runs/detect/predict/`

### Real-time Webcam Detection

#### Advanced Webcam Detection (run_model.py)

The `run_model.py` script provides a production-ready webcam detection interface with advanced features:

**Run the script:**
```bash
python run_model.py
```

**Features:**
- ✅ Live bounding box visualization
- ✅ Real-time FPS counter
- ✅ Interactive confidence threshold adjustment
- ✅ Frame capture and saving
- ✅ Professional on-screen overlay with detection statistics
- ✅ Multi-camera support
- ✅ Automatic error handling and recovery

**On-Screen Information Display:**
- Current FPS (frames per second)
- Active confidence threshold
- Number of detections in current frame
- Total frames processed
- Detection status ("PLATE DETECTED!" or "Scanning...")

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| **Q** or **ESC** | Quit application |
| **+** or **=** | Increase confidence threshold (+0.05) |
| **-** | Decrease confidence threshold (-0.05) |
| **S** | Save current frame with detections |
| **R** | Reset confidence to default (0.4) |

**Configuration Options:**

Edit the constants in `run_model.py` to customize:

```python
# At the bottom of run_model.py, in main() function
MODEL_PATH = "training/runs/plate_detector_yolo113/weights/best.pt"  # Model path
CONFIDENCE_THRESHOLD = 0.4   # Initial confidence (0.0-1.0)
CAMERA_ID = 0                # Camera device (0 for default, 1, 2, etc.)
```

**Saved Frames:**
- Location: `runs/webcam_detections/`
- Format: `detection_0001.jpg`, `detection_0002.jpg`, etc.
- Frames saved with bounding boxes and annotations

**Example Usage:**

1. **Default webcam with standard settings:**
   ```bash
   python run_model.py
   ```

2. **Use external webcam (camera ID 1):**
   ```python
   # Edit run_model.py, change CAMERA_ID
   CAMERA_ID = 1
   ```

3. **Start with lower confidence for more detections:**
   ```python
   # Edit run_model.py
   CONFIDENCE_THRESHOLD = 0.3
   ```

**Troubleshooting:**

- **Error: Could not open camera**
  - Try different camera IDs (0, 1, 2)
  - Check if another application is using webcam
  - Verify webcam permissions

- **Low FPS**
  - Close other applications using GPU
  - Reduce input resolution in the script
  - Use a smaller YOLO model variant

#### Basic Webcam Detection (detect_webcam.py)

For simple, minimal webcam detection:

```bash
python inference/detect_webcam.py
```

- Press **`q`** to quit
- Adjust `conf=0.4` in the script to change sensitivity
- No UI overlay or interactive controls

### Batch Processing (Multiple Images)

```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector_yolo11/weights/best.pt")

# Process folder of images
results = model("path/to/image/folder/", conf=0.4, save=True)
```

### Save Detection Results

```python
from ultralytics import YOLO

model = YOLO("runs/plate_detector_yolo11/weights/best.pt")
results = model("test.jpg", conf=0.4, save=True, save_txt=True)

# Outputs:
# - Annotated image: runs/detect/predict/test.jpg
# - Coordinates: runs/detect/predict/labels/test.txt
```

---

## ⚙️ Configuration Customization

### Training Configuration

**Hyperparameters** (advanced users):

```python
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    
    # Advanced options
    lr0=0.01,              # Initial learning rate
    lrf=0.01,              # Final learning rate (lr0 * lrf)
    momentum=0.937,        # SGD momentum
    weight_decay=0.0005,   # L2 regularization
    warmup_epochs=3,       # Learning rate warmup
    augment=True,          # Data augmentation
    cache=False,           # Cache images in RAM (faster, more memory)
    workers=8,             # DataLoader workers
    patience=50,           # Early stopping patience
    save_period=10,        # Save checkpoint every N epochs
    val=True,              # Validate during training
)
```

### Inference Configuration

**Confidence Threshold**:
```python
# Lower = more detections (may include false positives)
results = model("image.jpg", conf=0.3)  # 30% confidence

# Higher = fewer detections (may miss some plates)
results = model("image.jpg", conf=0.6)  # 60% confidence
```

**NMS (Non-Maximum Suppression) Threshold**:
```python
results = model("image.jpg", conf=0.4, iou=0.5)
# iou: IoU threshold for NMS (default 0.7)
```

**Image Size**:
```python
results = model("image.jpg", imgsz=1280)  # Higher resolution (slower, more accurate)
```

---

## 🚀 Performance Optimization

### Training Optimization

#### 1. GPU Utilization

**Check GPU usage**:
```bash
nvidia-smi
```

**Maximize GPU usage**:
- Increase `batch` size (16 → 32)
- Enable mixed precision: `amp=True` in `model.train()`

#### 2. Faster Iteration

- **Cache images**: `cache='ram'` (uses ~2-4 GB RAM)
- **Reduce workers**: `workers=4` if CPU bottleneck
- **Smaller validation**: Use fewer validation images

#### 3. Better Accuracy

- **More epochs**: 100 → 200
- **Larger model**: `yolo11n` → `yolo11s`
- **Higher resolution**: `imgsz=1280`
- **More data**: Collect 2-3x more training images

### Inference Optimization

#### 1. Faster Detection

```python
# Use smaller input size
results = model("image.jpg", imgsz=416, half=True)  # FP16 precision
```

#### 2. Batch Inference (Videos)

```python
# Process multiple frames at once
for result in model.predict(source="video.mp4", stream=True, batch=8):
    # Process result
    pass
```

#### 3. Export for Deployment

**Export to ONNX** (faster inference):
```python
from ultralytics import YOLO
model = YOLO("runs/plate_detector_yolo11/weights/best.pt")
model.export(format='onnx')  # Creates best.onnx
```

**Use exported model**:
```python
model = YOLO("best.onnx")
results = model("image.jpg")
```

---

## 🐛 Common Issues

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```python
   batch=8  # or batch=4
   ```
2. Use smaller model: `yolo11n` instead of `yolo11s`
3. Reduce image size: `imgsz=416`
4. Enable gradient accumulation (coming soon in Ultralytics)

### Issue 2: Training is Slow

**Symptoms**: <10 images/sec

**Solutions**:
1. **Verify GPU usage**:
   ```bash
   nvidia-smi  # Check if GPU is active
   ```
2. **Check device setting**:
   ```python
   device=0  # Ensure using GPU, not 'cpu'
   ```
3. **Reduce data loading overhead**:
   ```python
   workers=4  # Reduce if CPU bottleneck
   ```

### Issue 3: Low Detection Accuracy

**Symptoms**: Missing plates, false positives

**Solutions**:
1. **More training data**: Collect 500+ images (more is better)
2. **Train longer**: `epochs=200`
3. **Check data quality**:
   ```bash
   python check_dataset.py
   ```
4. **Lower confidence threshold**:
   ```python
   results = model("image.jpg", conf=0.3)
   ```
5. **Use larger model**: `yolo11s` or `yolo11m`

### Issue 4: Missing Label Files

**Error**: "Label file not found"

**Solution**:
```bash
python check_dataset.py  # Identify missing labels
python create_empty_labels.py  # Create empty labels if needed
```

### Issue 5: Bad Label Format

**Error**: "Line has X values, expected 5"

**Solution**:
- Each label line must have exactly 5 values:
  ```
  0 0.512 0.347 0.284 0.156
  ```
- Check for spaces, tabs, or malformed lines
- Regenerate labels using annotation tool

### Issue 6: Model Not Detecting Anything

**Symptoms**: No bounding boxes shown

**Solutions**:
1. **Check model path**:
   ```python
   model = YOLO("runs/plate_detector_yolo11/weights/best.pt")  # Verify path
   ```
2. **Lower confidence**:
   ```python
   results = model("image.jpg", conf=0.1)  # Very low threshold for testing
   ```
3. **Verify training completed**: Check `results.png` for convergence

### Issue 7: Path Errors on Windows

**Error**: `FileNotFoundError`

**Solution**: Use raw strings for Windows paths:
```python
data=r"C:\Users\YourName\Desktop\dataset\data.yaml"
```

Or use forward slashes:
```python
data="C:/Users/YourName/Desktop/dataset/data.yaml"
```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check logs**: Review terminal output for specific errors
2. **Consult documentation**:
   - [Ultralytics Docs](https://docs.ultralytics.com/)
   - [ARCHITECTURE.md](ARCHITECTURE.md)
   - [API_REFERENCE.md](API_REFERENCE.md)
3. **Open an issue**: Provide error logs, system info, and steps to reproduce

---

**Next Steps**: See [API_REFERENCE.md](API_REFERENCE.md) for programmatic usage and customization options.
