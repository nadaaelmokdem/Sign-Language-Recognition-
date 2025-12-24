# 🖐️ Sign Language Recognition System

**Computer Vision University Project**

## 📌 Overview

This project implements a **Sign Language Recognition System** using **Computer Vision and Deep Learning**.
The system recognizes static sign language gestures in real time using a webcam and translates them into textual labels.

The project was developed as part of a **Computer Vision course university project**, focusing on building a complete and structured machine learning pipeline using a **dataset-based approach**.

---

## 🎯 Project Objectives

* Build a sign language recognition system using a **pre-collected dataset**
* Extract meaningful hand features using **MediaPipe**
* Train a neural network for gesture classification
* Perform **real-time recognition** using a webcam
* Ensure consistent preprocessing between training and inference

---

## 🧠 System Pipeline

1. Load labeled sign language dataset
2. Extract hand landmarks from images
3. Convert landmarks into numerical feature vectors
4. Train a neural network classifier
5. Perform real-time prediction using webcam input

---

## 📂 Dataset

* The project uses a **publicly available sign language dataset**
* Data is organized in a folder-based structure
* Each folder represents one sign label
* Images contain static hand gestures

📊 **Dataset Link:**
👉 *Add dataset link here*

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **MediaPipe**
* **TensorFlow / Keras**
* **NumPy**
* **Scikit-learn**

---

## 🧩 Feature Extraction

* MediaPipe Hands is used to detect one hand per image/frame
* 21 hand landmarks are extracted
* Each landmark contains (x, y, z) coordinates
* Total features per sample: **63 values**

This approach improves speed and robustness compared to using raw images.

---

## 🏗️ Model Architecture

* Fully Connected Neural Network (MLP)
* Input layer: 63 features
* Hidden layers:

  * 128 neurons (ReLU)
  * 64 neurons (ReLU)
* Output layer: Softmax activation
* Optimizer: Adam
* Loss function: Categorical Cross-Entropy

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn
```

---

### 2️⃣ Project Structure

```
New Project/
│
├── extract_landmarks.py
├── train_model.py
├── realtime_predict.py
├── X.npy
├── y.npy
│
└── dataset/
    ├── A/
    ├── B/
    ├── C/
```

---

### 3️⃣ Extract Hand Landmarks

```bash
python extract_landmarks.py
```

This step generates:

* `X.npy` → feature vectors
* `y.npy` → labels

---

### 4️⃣ Train the Model

```bash
python train_model.py
```

The trained model is saved for later use.

---

### 5️⃣ Run Real-Time Recognition

```bash
python realtime_predict.py
```

The webcam will open and display the predicted sign in real time.

---

## 📊 Results

* Accurate recognition of static hand signs
* Fast real-time performance
* Efficient landmark-based representation
* Works best with clear hand visibility and good lighting

---

## ⚠️ Limitations

* Supports **static signs only**
* Single-hand recognition
* No sentence-level translation
* Performance depends on dataset quality and lighting conditions

---

## 🚀 Future Improvements

* Dynamic sign recognition using LSTM
* Word and sentence-level translation
* Arabic Sign Language expansion
* Mobile or web deployment
* Multi-hand recognition

---


## 🔗 Links



* **Dataset:**
  👉 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## 📌 Conclusion

This project demonstrates how computer vision and deep learning can be combined to build an efficient sign language recognition system.
It serves as a strong foundation for more advanced research and real-world applications in assistive technologies.

---
