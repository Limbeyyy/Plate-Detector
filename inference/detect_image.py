from ultralytics import YOLO

model = YOLO("runs/plate_detector/weights/best.pt")
results = model("test.jpg", conf=0.4)
results[0].show()
