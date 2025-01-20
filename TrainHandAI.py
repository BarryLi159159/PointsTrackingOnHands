from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a pretrained model 

# Train the model
results = model.train(data="hand-keypoints.yaml", epochs=10, imgsz=640)
