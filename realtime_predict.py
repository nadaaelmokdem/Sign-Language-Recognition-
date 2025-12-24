import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

model = load_model("sign_model.h5")

labels = os.listdir("dataset")

mp_hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)
        sign = labels[np.argmax(prediction)]

        cv2.putText(
            frame,
            sign,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
