import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load trained model
model = keras.models.load_model("scene_classifier3.h5")

# Image size should match training size
IMG_SIZE = (128, 128)

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:  # Extract every 10th frame
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return np.array(frames) / 255.0  # Normalize frames

# Function to classify video using the trained model
def classify_video(video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        return "Error: No frames extracted"
    
    predictions = model.predict(frames)
    avg_prediction = np.mean(predictions)
    return "Movie Scene" if avg_prediction > 0.50 else "Real-Life Scene"

# If running this script directly, classify an example video
if __name__ == "__main__":
    video_path = r"D:\video_classification_mannually\video_classification_without_any_model\fake4.mp4"  # Change this to your test video
    result = classify_video(video_path)
    print(f"Final Classification: {result}")
