from ultralytics import YOLO

def main():
    # YOLO11 nano model (fast & lightweight)
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"C:\Users\A C E R\OneDrive\Desktop\OCR Plate Detector\dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      # 0 = GPU, 'cpu' for CPU
        project="runs",
        name="plate_detector_yolo11"
    )

if __name__ == "__main__":
    main()
