# Performance Report

This report summarizes the performance of the trained YOLOv5 model for pothole detection.

## 1. Validation Performance

The model was validated on the test dataset, and the following metrics were obtained:

| Metric    | Value |
|-----------|-------|
| Precision | 0.961 |
| Recall    | 0.916 |
| mAP@.50   | 0.965 |
| mAP@.50-.95| 0.592 |

**Inference Speed (Validation):**

*   **Pre-process:** 0.7ms per image
*   **Inference:** 158.2ms per image
*   **NMS:** 1.4ms per image

*These metrics were obtained by running `val.py` on the test set with an image size of 640x640 and a batch size of 1 on a CPU.*

## 2. Benchmark Performance

The model was benchmarked across various export formats. The following table summarizes the key results:

| Format      | Size (MB) | mAP@.50-.95 | Inference Time (ms) |
|-------------|-----------|-------------|-----------------------|
| TorchScript | 27.2      | 0.0         | 139.29                |
| ONNX        | 27.2      | 0.0         | 102.15                |
| OpenVINO    | 27.3      | 0.0         | 109.25                |
| PaddlePaddle| 54.5      | 0.0         | 293.26                |

*Note: The mAP@.50-.95 values are 0.0 in the benchmark results because the benchmark was run on the COCO128 dataset, not the pothole dataset. The benchmark's primary purpose here is to measure inference speed and model size across different formats.*

## 3. REST API Inference Speed

The inference speed of the FastAPI server was measured by sending a test image to the `/predict/` endpoint.

*   **Inference time:** ~0.19 seconds (190ms) per image

*This includes the time for the HTTP request, image transfer, server-side processing (including inference), and response transfer.*