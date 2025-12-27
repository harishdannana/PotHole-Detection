
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Get the directory of the current script
script_dir = Path(__file__).parent

# Add yolov5 directory to Python path
yolo_path = script_dir / 'yolov5'
if str(yolo_path) not in sys.path:
    sys.path.append(str(yolo_path))

from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Initialize FastAPI app
app = FastAPI()

# Load the ONNX model
onnx_model_path = yolo_path / 'runs/train/exp3/weights/best.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Define the prediction endpoint
@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Pre-process image
    img0 = img.copy()
    img = letterbox(img, new_shape=640, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)

    # ONNX inference
    pred = session.run(output_names, {input_name: img})[0]
    pred = torch.from_numpy(pred)

    # Post-process
    pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=False)

    results = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                results.append({
                    'box': [int(c) for c in xyxy],
                    'confidence': float(conf),
                    'class': int(cls)
                })

    return JSONResponse(content={'results': results})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
