from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/last.pt")
model.train(resume=True)
