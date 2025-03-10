import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Model Parameters
IMG_SIZE = (128, 128)
IMG_SHAPE = (128, 128, 3)
SEQUENCE_LENGTH = 20  # Number of frames per sequence
FRAME_INTERVAL = 5  # Extract every 5th frame

# Function to Extract Frames
def extract_frames(video_path, save_folder):
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

# Extract Frames for Movie & Real-Life Videos
for video in os.listdir("dataset/movie_videos"):
    extract_frames(f"dataset/movie_videos/{video}", "dataset/movie_frames")

for video in os.listdir("dataset/real_life_videos"):
    extract_frames(f"dataset/real_life_videos/{video}", "dataset/real_life_frames")


# CNN Model for Feature Extraction
cnn = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SHAPE),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten()
])

# LSTM Model for Sequence Processing
model = keras.Sequential([
    layers.TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH, 128, 128, 3)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 Classes: Movie vs Real-Life
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Save Model Correctly
model.save("video_classification_model_two.h5", save_format="h5")
print("✅ Model saved successfully!")

IMG_SIZE = (64, 64)
IMG_SHAPE = (64, 64, 3)
SEQUENCE_LENGTH = 20
FRAME_INTERVAL = 5

# Function to Load Video Sequences for Training
def load_video_sequences(folder, label):
    sequences, labels = [], []
    video_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    for vid_folder in video_folders:
        frames = sorted(os.listdir(os.path.join(folder, vid_folder)))
        frames = [img_to_array(load_img(os.path.join(folder, vid_folder, f))) / 255.0 for f in frames]

        if len(frames) >= SEQUENCE_LENGTH:
            for i in range(0, len(frames) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
                sequences.append(np.array(frames[i:i+SEQUENCE_LENGTH]))
                labels.append(label)

    return np.array(sequences), np.array(labels)

# Load Data
movie_x, movie_y = load_video_sequences("dataset/movie_frames", [1, 0])
real_x, real_y = load_video_sequences("dataset/real_life_frames", [0, 1])

# Combine & Shuffle
X = np.concatenate((movie_x, real_x))
Y = np.concatenate((movie_y, real_y))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

print(f"Loaded {X.shape[0]} samples")
if X.shape[0] == 0:
    raise ValueError("❌ No training data found!")

# Train the Model
model.fit(X, Y, epochs=15, batch_size=8, validation_split=0.2)
