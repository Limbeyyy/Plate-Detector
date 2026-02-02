from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
model.predict(source="video.mp4", conf=0.4, save=True)
