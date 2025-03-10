import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import random

# Constants
FRAME_SIZE = (224, 224)
MAX_FRAMES = 20  # Fixed number of frames per video
FEATURE_DIM = 512  # Adjust based on the CNN model used

# Load Feature Extractor
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_frames(video_path, max_frames=MAX_FRAMES):
    """ Extract evenly spaced frames from a video. """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = frame / 255.0  # Normalize
            frames.append(frame)
    
    cap.release()
    return np.array(frames) if frames else None

def extract_features(frames):
    """ Extract features from frames using pre-trained CNN """
    if frames is None or len(frames) == 0:
        return None
    features = feature_extractor.predict(frames, verbose=0)
    return np.mean(features, axis=0)  # Aggregate features

def prepare_dataset(video_paths, labels):
    """ Extract features and create dataset. """
    X, Y = [], []
    for video_path, label in zip(video_paths, labels):
        frames = extract_frames(video_path)
        features = extract_features(frames)
        if features is not None:
            X.append(features)
            Y.append(label)
    return np.array(X), np.array(Y)

# Load dataset
movie_videos = ["path_to_movie_clip1.mp4", "path_to_movie_clip2.mp4"]
real_videos = ["path_to_real_life_clip1.mp4", "path_to_real_life_clip2.mp4"]

dataset_videos = movie_videos + real_videos
dataset_labels = [1] * len(movie_videos) + [0] * len(real_videos)

# Prepare data
X, Y = prepare_dataset(dataset_videos, dataset_labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(FEATURE_DIM, 1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
X_train = X_train.reshape((X_train.shape[0], FEATURE_DIM, 1))
X_test = X_test.reshape((X_test.shape[0], FEATURE_DIM, 1))

model.fit(X_train, Y_train, epochs=10, batch_size=8, validation_data=(X_test, Y_test))

# Prediction Function
def predict_scene(video_path):
    frames = extract_frames(video_path)
    features = extract_features(frames)
    if features is not None:
        features = features.reshape(1, FEATURE_DIM, 1)
        prediction = model.predict(features)
        return "Movie Scene" if prediction > 0.5 else "Real-Life Scene"
    return "Error processing video"

# Example Prediction
print(predict_scene("test_video.mp4"))
