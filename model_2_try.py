import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the Saved Model
model = keras.models.load_model("video_classification_model_two.h5")
print("✅ Model Loaded Successfully!")

IMG_SIZE = (128, 128)
SEQUENCE_LENGTH = 25
FRAME_INTERVAL = 10

# Function to Extract Frames from a Video
def extract_frames(video_path, save_folder="temp_frames"):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame_filename = os.path.join(save_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_list.append(frame_filename)
        frame_count += 1

    cap.release()
    return frame_list if frame_list else None

# Function to Classify a Video
def classify_video(video_path):
    frames = extract_frames(video_path)

    if frames is None or len(frames) < SEQUENCE_LENGTH:
        print("❌ Not enough frames extracted!")
        return "❌ Error"

    # Load and Normalize Frames
    frames = [img_to_array(load_img(f)) / 255.0 for f in frames[:SEQUENCE_LENGTH]]
    frames = np.array(frames).reshape((1, SEQUENCE_LENGTH, 128, 128, 3))  # Ensure Correct Shape

    # Make Prediction
    prediction = model.predict(frames)[0]
    print(f"🔍 Prediction Scores: {prediction}")  # Debugging Output

    return "🎬 Movie Scene" if prediction[0] < prediction[1] else "📹 Real-Life Scene"

# Test Classification
video_path = r"D:\video_classification_mannually\video_classification_without_any_model\real_video.mp4"
result = classify_video(video_path)
print(f"Final Classification: {result}")
