import cv2
import os
import numpy as np
import mediapipe as mp

DATASET_PATH = "dataset"
OUTPUT_X = []
OUTPUT_Y = []

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

labels = os.listdir(DATASET_PATH)
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            OUTPUT_X.append(landmarks)
            OUTPUT_Y.append(label_map[label])

X = np.array(OUTPUT_X)
y = np.array(OUTPUT_Y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved:", X.shape, y.shape)
