from ultralytics import YOLO
import os

# Load a model
model = YOLO("pt/catdog/11_n.pt")

# Export the model
# model.export(format="onnx",imgsz=640,half=False,device="cpu",nms=True)
model.export(format="onnx",imgsz=640,simplify=True,batch=1,device="0")