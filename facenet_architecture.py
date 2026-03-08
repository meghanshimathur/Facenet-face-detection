

!pip -q install mtcnn opencv-python matplotlib pillow-heif
# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from google.colab import files

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
print("image:/content/another_image2.jpg")

import matplotlib.pyplot as plt
# -------------------------------------------------------
# Function: detect_and_crop
# Detects faces using MTCNN and crops them for FaceNet
# -------------------------------------------------------
def detect_and_crop(image_path, required_size=(160,160)):
 # Initialize MTCNN face detector
    detector = MTCNN()
    # Read image using OpenCV
    img_bgr = cv2.imread(image_path)
    # Convert image from BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Detect faces in the image
    results = detector.detect_faces(img_rgb)
# If no faces detected
    if len(results) == 0:
        print("No face detected in:", image_path)
        return None

    x, y, w, h = results[0]['box']
    x, y = max(0,x), max(0,y)

    # GREEN BOX
    cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0,255,0), 3)

    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)

    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / (std + 1e-6)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Detected Face")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow((face - face.min())/(face.max()-face.min()))
    plt.title("Cropped Face")
    plt.axis('off')
    plt.show()

    return face

face = detect_and_crop(image_path)

if face is not None:
    print("Face detected!")
    print("Face shape:", face.shape)
else:
    print("No face detected.")
#your file path
image_path="/content/a4776e47-3261-4375-86c1-f06b5a8f47a7.jpeg"
def detect_and_crop(image_path, required_size=(160,160)):

    detector = MTCNN()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img_rgb)
    # If no faces detected
    if len(results) == 0:
        print("No face detected in:", image_path)
        return []

    faces = []

    for result in results:

        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)

        # Draw green box for each face
        cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0,255,0), 3)

        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, required_size)

        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / (std + 1e-6)

        faces.append(face)

    # Show result
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb)
    plt.title(f"{len(faces)} Faces Detected")
    plt.axis('off')
    plt.show()

    return faces

faces = detect_and_crop(image_path)

if len(faces) > 0:
    print("Faces detected:", len(faces))

    for i, face in enumerate(faces):
        print(f"Face {i+1} shape:", face.shape)
else:
    print("No face detected.")

import matplotlib.pyplot as plt
#your file path
image_path="/content/a4776e47-3261-4375-86c1-f06b5a8f47a7.jpeg"
def detect_and_crop(image_path, required_size=(160,160)):

    detector = MTCNN()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img_rgb)

    if len(results) == 0:
        print("No face detected in:", image_path)
        return []

    cropped_faces = []

    # LOOP over all detected faces
    for result in results:
        x, y, w, h = result['box']
        x, y = max(0,x), max(0,y)

        # Draw green rectangle
        cv2.rectangle(img_rgb, (x,y), (x+w, y+h), (0,255,0), 3)

        # Crop face
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, required_size)

        # Normalize (FaceNet)
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / (std + 1e-6)

        cropped_faces.append(face)

    # --------- DISPLAY ----------
    plt.figure(figsize=(12,5))

    # Show detection image
    plt.subplot(1, len(cropped_faces)+1, 1)
    plt.imshow(img_rgb)
    plt.title("Detected Faces")
    plt.axis('off')

    # Show all cropped faces
    for i, face in enumerate(cropped_faces):
        plt.subplot(1, len(cropped_faces)+1, i+2)
        plt.imshow((face - face.min())/(face.max()-face.min()))
        plt.title(f"Face {i+1}")
        plt.axis('off')

    plt.show()

    return cropped_faces

faces = detect_and_crop(image_path)

