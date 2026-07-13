# 🧠 Deep Learning Projects

A collection of deep learning projects built during my MSc in Artificial Intelligence at the University of Salford. Each project demonstrates hands-on application of transfer learning, object detection, and fine-tuning using TensorFlow/Keras and PyTorch.

---

## Projects

### 1. 🚶 Image Classification with Transfer Learning (MobileNetV2)
**File:** `DL_1.ipynb`

Comparative study of transfer learning architectures applied to a **pedestrian safety dataset** (4 classes). The project demonstrates fine-tuning pre-trained ImageNet models for a custom classification task.

**Key techniques:**
- Transfer learning with **MobileNetV2** (pre-trained on ImageNet)
- Progressive fine-tuning: unfreezing last 50 layers, then last 30 layers
- Class imbalance handling via **computed class weights** (sklearn)
- Data augmentation: rotation, width/height shift, shear, zoom, horizontal flip
- Early stopping with best weight restoration
- Evaluation: classification report, confusion matrix, misclassification analysis

**Tech stack:** Python · TensorFlow/Keras · scikit-learn · Matplotlib · Google Colab

---

### 2. ♟️ Chess Piece Detection with YOLOv8
**File:** `YOLOv8.ipynb`

Real-time object detection system trained to identify **13 chess piece classes** (black/white: bishop, king, knight, pawn, queen, rook) from images.

**Key techniques:**
- YOLOv8 fine-tuned on a custom 13-class chess dataset
- YOLO-format dataset preparation and YAML configuration
- 50-epoch training with image size 640×640, batch size 16
- Confusion matrix and prediction visualisation
- Model export (`.pt` weights) for deployment

**Tech stack:** Python · Ultralytics YOLOv8 · PyTorch · Google Colab

**Classes (13):** bishop, black-bishop, black-king, black-knight, black-pawn, black-queen, black-rook, white-bishop, white-king, white-knight, white-pawn, white-queen, white-rook

---

### 3. 🎯 Object Detection with YOLOv9
**File:** `YOLOv9.ipynb`

Object detection pipeline using **YOLOv9** on a custom dataset, exploring the latest advances in the YOLO architecture family with GPU-accelerated training on Google Colab.

**Key techniques:**
- YOLOv9 training pipeline with Ultralytics framework
- Custom dataset integration via YAML configuration
- UTF-8 encoding fixes for Colab compatibility
- GPU-accelerated training (CUDA)
- Model weight export and evaluation

**Tech stack:** Python · Ultralytics YOLOv9 · PyTorch · TorchVision · Google Colab

---

## Repository Structure

```
Deep-Learning-Projects/
│
├── DL_1.ipynb                  # Transfer learning: MobileNetV2 pedestrian classification
├── YOLOv8.ipynb                # Chess piece detection with YOLOv8 (13 classes)
├── YOLOv9.ipynb                # Object detection with YOLOv9
├── best.pt                     # Best YOLOv8 model weights
├── best_weights.pt             # Best YOLOv9 model weights
├── YoloV8.txt                  # YOLOv8 training notes
├── YOLOv9 ReadMe.txt           # YOLOv9 setup notes
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## How to Run

### Transfer Learning (DL_1.ipynb)
1. Open in **Google Colab**
2. Mount Google Drive and set your dataset path
3. Run cells sequentially — class weights are auto-computed
4. Model trains for up to 20 epochs with early stopping

### Chess Piece Detection (YOLOv8.ipynb)
1. Open in **Google Colab** with GPU enabled
2. Upload your chess piece dataset zip
3. Run cells to extract dataset, fix YAML paths, and train
4. Results saved to `runs/detect/chess_yolov8_model/`

### YOLOv9 Detection (YOLOv9.ipynb)
1. Open in **Google Colab** with GPU enabled (CUDA 11.8)
2. Upload your dataset zip
3. Update `data.yaml` paths and run training cells
4. Best weights saved to `runs/detect/train/weights/best.pt`

---

## Skills Demonstrated

| Skill | Projects |
|-------|----------|
| Transfer learning & fine-tuning | DL_1 |
| Class imbalance handling | DL_1 |
| Real-time object detection | YOLOv8, YOLOv9 |
| Custom dataset preparation | YOLOv8, YOLOv9 |
| Model evaluation & visualisation | All |
| GPU-accelerated training | YOLOv8, YOLOv9 |
| Model export & deployment | YOLOv8, YOLOv9 |

---

## Author

**Abdullah Al Kafi**
MSc Artificial Intelligence, University of Salford (2025)
[LinkedIn](https://www.linkedin.com/in/abdullah-al-kafi-173b4325b) | [GitHub](https://github.com/Kafi001)

> See also: [Weapon Detection — YOLOv8 vs YOLOv11](https://github.com/Kafi001/weapon-detection-yolov8-vs-yolov11) (MSc Dissertation, Distinction 70%)
