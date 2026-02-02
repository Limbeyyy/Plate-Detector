from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
model.predict(source=0, conf=0.4, show=True)
