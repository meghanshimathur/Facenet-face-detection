# Face Detection using MTCNN for FaceNet

## Overview

This project demonstrates **face detection and preprocessing for FaceNet-based face recognition systems**.

The implementation uses **MTCNN (Multi-task Cascaded Convolutional Networks)** to detect faces in images and extract cropped face regions. These cropped faces are normalized and resized to match the **FaceNet input format (160x160)**.

The project helps understand how images are prepared before being used in **face recognition models**.

---

## Key Features

- Detect faces using **MTCNN**
- Draw bounding boxes around detected faces
- Crop detected faces
- Resize faces to **160×160 pixels**
- Normalize images for FaceNet preprocessing
- Visualize detection results

---

## Technologies Used

- Python
- OpenCV
- MTCNN
- NumPy
- Matplotlib

---

## How the System Works

The pipeline used in this project:

```
Input Image
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
Bounding Box Extraction
      │
      ▼
Face Cropping
      │
      ▼
Resize to 160×160
      │
      ▼
Normalization
      │
      ▼
FaceNet Ready Image
```

---

## How to Run

### Install Dependencies

```bash
pip install mtcnn opencv-python matplotlib pillow-heif
```

### Run the Script

```bash
python facenet_mtcnn_face_detection.py
```

Upload an image when prompted and the script will:

1. Detect faces
2. Draw bounding boxes
3. Crop and normalize the faces
4. Display the results

---

## Real World Applications

Face detection and recognition systems are widely used in:

- Smartphone face unlock systems
- Security and surveillance systems
- Attendance systems
- Airport identity verification
- Social media photo tagging
- Smart access control systems

---

## What I Learned

Through this project I learned:

- Face detection using MTCNN
- Image preprocessing for FaceNet
- Bounding box extraction
- Image normalization techniques
- Visualizing results using Matplotlib
- Building preprocessing pipelines for face recognition systems

---

## Future Improvements

Possible improvements include:

- Integrating a full **FaceNet embedding model**
- Implementing **face recognition (identity matching)**
- Using **real-time webcam detection**
- Deploying as a **face recognition application**

---

