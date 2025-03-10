import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# Load the trained model
model = keras.models.load_model(r"D:\video_classification_mannually\video_classification_without_any_model\video_classification_model_using_resnet.h5")

IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 20

def extract_frames(video_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame / 255.0  # Normalize
            frames.append(frame)

        frame_count += 1

    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        print("❌ Not enough frames for classification!")
        return None

    return np.array(frames[:SEQUENCE_LENGTH])

def classify_video(video_path):
    frames = extract_frames(video_path, "temp_frames")
    if frames is None:
        return "❌ Error: Not enough frames"

    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)[0]

    return "🎬 Movie Scene" if prediction[0] < prediction[1] else "📹 Real-Life Scene"

# Test Classification
video_path = r"D:\video_classification_mannually\video_classification_without_any_model\real_video.mp4"
result = classify_video(video_path)
print(f"Final Classification: {result}")
