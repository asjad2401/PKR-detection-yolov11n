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
├── my_trained_model/         # Trained weights (.pt / export formats)
│   └── weights/
│       ├── best.pt            # Best performing model checkpoint
│       └── last.pt            # Latest epoch checkpoint
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
model = YOLO('my_trained_model/weights/best.pt')

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
yolo task=detect mode=predict model=my_trained_model/weights/best.pt source='path/to/image.jpg' conf=0.5

# Run real-time detection on webcam
yolo task=detect mode=predict model=my_trained_model/weights/best.pt source=0 conf=0.5
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
   names: ['10_PKR', '20_PKR', '50_PKR', '100_PKR', '500_PKR', '1000_PKR', '5000_PKR']
   ```

2. Start training:
   ```bash
   yolo detect train data=dataset/data.yaml model=yolov11n.pt epochs=50 imgsz=640 batch=16 name=pkr_yolov11n
   ```

---

##  Exporting for Mobile & Edge Devices

To integrate this model into Android (Flutter / Kotlin), iOS, or ONNX runtimes:

```python
from ultralytics import YOLO

model = YOLO('my_trained_model/weights/best.pt')

# Export to ONNX format
model.export(format='onnx')

# Export to TensorFlow Lite (TFLite) for mobile devices
model.export(format='tflite')
```

---

##  Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/asjad2401/PKR-detection-yolov11n/issues).

---

##  License

This project is open-source under the [MIT License](LICENSE).
