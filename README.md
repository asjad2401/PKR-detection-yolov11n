# Pakistani Rupee (PKR) Currency Detection using YOLOv11n

[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-blue.svg)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An object detection model built using the lightweight **YOLOv11 Nano (`yolov11n`)** architecture to detect and classify Pakistani Rupee (PKR) currency banknotes in real-time. 

Designed for high efficiency and edge computing, this model is ideal for integration into mobile applications, assistive devices for visually impaired individuals, and automated cash management systems.

---

##  Supported Banknote Denominations

The model is trained to identify standard Pakistani Rupee currency notes:
*  **10 PKR**
*  **20 PKR**
*  **50 PKR**
*  **100 PKR**
*  **500 PKR**
*  **1000 PKR**
*  **5000 PKR**

---

##  Repository Structure

```text
PKR-detection-yolov11n/
│
├── my_trained_model/
│   └── runs/pk_currency_yolo11n/
│       ├── weights/
│       │   ├── best.pt              # Best checkpoint (100 epochs, imgsz 416, mAP50 0.978)
│       │   ├── best_float16.tflite  # Mobile export (see "Mobile Model")
│       │   └── best.mlpackage       # Core ML export
│       └── results.csv, *.png       # Training curves and validation samples
├── dataset/                  # Dataset directory (images & labels)
│   ├── train/
│   ├── val/
│   └── data.yaml             # YOLO dataset configuration file
├── predict.py                # Python script for running inference
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

##  Quick Start

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/asjad2401/PKR-detection-yolov11n.git
cd PKR-detection-yolov11n

# Set up virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install ultralytics opencv-python torch torchvision
```

---

##  Running Inference

### Using Python API

```python
from ultralytics import YOLO
import cv2

# Load your custom trained YOLOv11 model
model = YOLO('my_trained_model/runs/pk_currency_yolo11n/weights/best.pt')

# Run inference on an image or video source (0 for webcam)
results = model.predict(source='path/to/currency_image.jpg', conf=0.5, show=True)

# Process results
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"Detected: {label} ({confidence:.2f})")

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Using YOLO CLI

```bash
# Run prediction on an image
yolo task=detect mode=predict model=my_trained_model/runs/pk_currency_yolo11n/weights/best.pt source='path/to/image.jpg' conf=0.5

# Run real-time detection on webcam
yolo task=detect mode=predict model=my_trained_model/runs/pk_currency_yolo11n/weights/best.pt source=0 conf=0.5
```

---

##  Training the Model

If you wish to retrain or fine-tune the model on a custom dataset:

1. Ensure your `dataset/data.yaml` is configured:
   ```yaml
   path: ./dataset
   train: train/images
   val: val/images

   nc: 7
   names: ['10_PKR', '100_PKR', '1000_PKR', '20_PKR', '50_PKR', '500_PKR', '5000_PKR']
   ```

2. Start training:
   ```bash
   yolo detect train data=dataset/data.yaml model=yolov11n.pt epochs=50 imgsz=640 batch=16 name=pkr_yolov11n
   ```

---

##  Mobile Model (TFLite)

A ready-to-use TensorFlow Lite export is included:
`my_trained_model/runs/pk_currency_yolo11n/weights/best_float16.tflite` (5.3 MB, float16 weights).
Its detections were checked against `best.pt` and match.

| | |
|---|---|
| Input | `1 x 416 x 416 x 3`, float32, RGB scaled to 0..1, letterboxed with gray (114) padding |
| Output | `1 x 11 x 3549`, float32: rows 0-3 are `cx, cy, w, h` in input pixels, rows 4-10 are class scores |
| Post-processing | keep anchors whose best score is above ~0.5, then class-agnostic NMS (IoU 0.45) |

**Class order matters.** The checkpoint stores classes alphabetically, not by value:

```
0: 10_PKR   1: 100_PKR   2: 1000_PKR   3: 20_PKR   4: 50_PKR   5: 500_PKR   6: 5000_PKR
```

This model is used by the Currency mode of the Vision Assist app
(`asjad2401/voice-assisted-calling`, `lib/modules/currency_module.dart`).

### Re-exporting

Recent Ultralytics versions route `format='tflite'` through a LiteRT path that can fail
(the previous `best_w8a32.tflite` export came out as a 0-byte file and was removed). Exporting via ONNX works:

```bash
pip install ultralytics onnx onnxslim onnx2tf tensorflow tf_keras onnx_graphsurgeon sng4onnx ai-edge-litert onnxruntime psutil
yolo export model=my_trained_model/runs/pk_currency_yolo11n/weights/best.pt format=onnx imgsz=416 simplify=True opset=17
onnx2tf -i my_trained_model/runs/pk_currency_yolo11n/weights/best.onnx -o tflite_out -nuo -osd
# -> tflite_out/best_float16.tflite and best_float32.tflite
```

onnx2tf downloads a calibration sample from GitHub. If you are offline, save any
`calibration_image_sample_data_20x128x128x3_float32.npy` array of shape (20,128,128,3) in the working directory.

---

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/asjad2401/PKR-detection-yolov11n/issues).

---

##  License

This project is open-source under the [MIT License](LICENSE).
