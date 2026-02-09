# Kataho Plate Detector using YOLOv11

A custom kataho plate detection system built with Ultralytics YOLO11 for detecting "Kataho_Plate" (Nepali license plates) in images, videos, and real-time webcam streams.

## 🚀 Features

- **Custom YOLOv11 Training**: Train on your own kataho plate dataset
- **Multi-Source Inference**: Detect plates in images, videos, and webcam feeds
- **Data Preprocessing Tools**: XML to YOLO format conversion, dataset validation
- **Resume Training**: Continue training from checkpoints
- **High Performance**: Leverages GPU acceleration for fast inference

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and inference)
- CUDA toolkit and cuDNN (for GPU support)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Plate-Detector/Plate-Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO11 pretrained weights** (optional, will auto-download on first run)
   ```bash
   # The train.py script will automatically download yolo11n.pt
   ```

## 📁 Project Structure

```
Plate-Detector/
├── dataset/
│   ├── data.yaml              # Dataset configuration
│   ├── images/
│   │   ├── train/            # Training images
│   │   └── val/              # Validation images
│   └── labels/
│       ├── train/            # Training labels (YOLO format)
│       └── val/              # Validation labels
├── training/
│   ├── train.py              # Initial training script
│   ├── resume_train.py       # Resume training from checkpoint
│   └── runs/                 # Training outputs and weights
├── inference/
│   ├── detect_image.py       # Detect plates in images
│   ├── detect_video.py       # Detect plates in videos
│   └── detect_webcam.py      # Real-time webcam detection
├── Helper_Files/
│   ├── check_dataset.py      # Validate dataset integrity
│   ├── convert_xml_to_txt.py # Convert XML annotations to YOLO
│   ├── create_empty_labels.py # Create empty label files
│   └── rename_img_labels.py  # Batch rename images and labels
├── run_model.py              # Advanced webcam detection with UI
└── requirements.txt          # Python dependencies
```

## 🎯 Quick Start

### 1. Prepare Your Dataset

Ensure your dataset follows this structure:
```
dataset/
├── images/
│   ├── train/   # Training images (.jpg, .jpeg, .png)
│   └── val/     # Validation images
└── labels/
    ├── train/   # Training labels (.txt, YOLO format)
    └── val/     # Validation labels
```

**Update the dataset paths** in `dataset/data.yaml`:
```yaml
path: /path/to/your/dataset
train: /path/to/your/dataset/images/train
val: /path/to/your/dataset/images/val
test: /path/to/your/dataset/images/val

nc: 1  # Number of classes
names:
  - Kataho_Plate
```

### 2. Validate Dataset

Check for missing or malformed labels:
```bash
python check_dataset.py
```

### 3. Train the Model

**Initial Training:**
```bash
python training/train.py
```

Training parameters in `train.py`:
- `epochs=100` - Number of training epochs
- `imgsz=640` - Input image size
- `batch=16` - Batch size
- `device=0` - GPU device (0 for first GPU, 'cpu' for CPU)

**Resume Training:**
```bash
python training/resume_train.py
```

### 4. Run Inference

**Detect in Image:**
```bash
python inference/detect_image.py
```
- Update `test.jpg` path in the script
- Results displayed in a window

**Detect in Video:**
```bash
python inference/detect_video.py
```
- Update `video.mp4` path in the script
- Results saved to `runs/detect/`

**Real-time Webcam Detection (Advanced):**
```bash
python run_model.py
```

**Features**:
- Live detection with bounding boxes and FPS counter
- Interactive confidence adjustment
- Frame saving capability
- Professional on-screen overlay with detection stats

**Keyboard Controls**:
- **Q** or **ESC**: Quit application
- **+** or **=**: Increase confidence threshold
- **-**: Decrease confidence threshold
- **S**: Save current frame
- **R**: Reset confidence to default

**Basic Webcam Detection:**
```bash
python inference/detect_webcam.py
```
- Simple real-time detection

## 🛠️ Data Preprocessing Utilities

### Convert XML Annotations to YOLO Format
```bash
python convert_xml_to_txt.py
```
Converts Pascal VOC XML annotations to YOLO format (class x_center y_center width height).

### Create Empty Label Files
```bash
python create_empty_labels.py
```
Creates empty `.txt` label files for images without annotations (useful for negative samples).

### Batch Rename Images and Labels
```bash
python rename_img_labels.py
```
Renames images and corresponding labels to a standardized format: `plate_0001.jpg`, `plate_0002.jpg`, etc.

## ⚙️ Configuration

### Training Configuration

Edit `training/train.py` to customize training:
```python
model.train(
    data="path/to/data.yaml",
    epochs=100,           # Training epochs
    imgsz=640,           # Image size
    batch=16,            # Batch size
    device=0,            # GPU device
    project="runs",      # Output directory
    name="plate_detector_yolo11"
)
```

### Inference Configuration

Adjust confidence threshold in inference scripts:
```python
# In detect_image.py, detect_video.py, detect_webcam.py
results = model(source, conf=0.4)  # 0.4 = 40% confidence
```

## 📊 Model Performance

The trained model weights are saved in:
```
training/runs/plate_detector_yolo11/weights/
├── best.pt    # Best model checkpoint
└── last.pt    # Last epoch checkpoint
```

Use `best.pt` for inference to get the best performance.

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `train.py`: `batch=8` or `batch=4`

### Issue: Training is slow
**Solution**: Ensure GPU is being used. Check `device=0` in `train.py`. Verify CUDA installation with `torch.cuda.is_available()`

### Issue: Missing labels error
**Solution**: Run `python check_dataset.py` to identify missing labels, then use `create_empty_labels.py` if needed

### Issue: Low detection accuracy
**Solutions**:
- Increase training epochs
- Collect more diverse training data
- Adjust augmentation settings
- Lower confidence threshold during inference

## 📚 Additional Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and technical details
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Comprehensive usage guide
- [API_REFERENCE.md](API_REFERENCE.md) - API and module reference
- [DATASET.md](DATASET.md) - Dataset preparation guidelines

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - YOLO implementation
- Built for Nepali license plate detection

## 📧 Contact

For questions or support, please open an issue in this repository.

---

**Note**: Before running training or inference, ensure all file paths in the scripts match your local directory structure.
