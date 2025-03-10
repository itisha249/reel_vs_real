# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import random
# import os
# import cv2


# IMG_SIZE = (128, 128)
# FRAME_INTERVAL = 5 

# def extract_frames(video_path, save_folder):
#     os.makedirs(save_folder, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     frame_list = []  # Store frame paths

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % FRAME_INTERVAL == 0:
#             frame = cv2.resize(frame, IMG_SIZE)
#             frame_filename = os.path.join(save_folder, f"frame_{frame_count}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             frame_list.append(frame_filename)  # Store extracted frame paths

#         frame_count += 1

#     cap.release()

#     if len(frame_list) == 0:
#         return None  # Return None if no frames were extracted

#     print(f"✅ Extracted frames from {video_path} to {save_folder}")
#     return frame_list  # Return the list of extracted frames

# # Extract frames for movie & real-life videos
# for video in os.listdir("dataset/movie_videos"):
#     extract_frames(f"dataset/movie_videos/{video}", "dataset/movie_frames")

# for video in os.listdir("dataset/real_life_videos"):
#     extract_frames(f"dataset/real_life_videos/{video}", "dataset/real_life_frames")


# # **Model Parameters**
# IMG_SHAPE = (128, 128, 3)
# SEQUENCE_LENGTH = 20  # Number of frames per sequence

# # **Define CNN for Feature Extraction**
# cnn = keras.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SHAPE),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Flatten()
# ])

# # **LSTM Model for Sequence Processing**
# model = keras.Sequential([
#     layers.TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH, 128, 128, 3)),
#     layers.LSTM(128, return_sequences=True),
#     layers.LSTM(64),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(2, activation='softmax')  # 2 Classes: Movie vs Real-Life
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# model.save("video_classification_model_using_resnet.h5")
# print("✅ Model saved successfully!")




# # Function to create frame sequences
# def load_video_sequences(folder, label):
#     sequences = []
#     labels = []
    
#     # video_folders = os.listdir(folder)
#     video_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]


#     for vid_folder in video_folders:
#         frames = sorted(os.listdir(os.path.join(folder, vid_folder)))
#         frames = [img_to_array(load_img(os.path.join(folder, vid_folder, f))) / 255.0 for f in frames]

#         # Ensure sequence length
#         if len(frames) >= SEQUENCE_LENGTH:
#             for i in range(0, len(frames) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
#                 sequences.append(np.array(frames[i:i+SEQUENCE_LENGTH]))
#                 labels.append(label)

#     return np.array(sequences), np.array(labels)

# # Load data
# movie_x, movie_y = load_video_sequences("dataset/movie_frames", [1, 0])
# real_x, real_y = load_video_sequences("dataset/real_life_frames", [0, 1])

# # Combine & Shuffle
# X = np.concatenate((movie_x, real_x))
# Y = np.concatenate((movie_y, real_y))

# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# X, Y = X[indices], Y[indices]

# print(f"Loaded {X.shape[0]} samples")  # Debugging statement
# if X.shape[0] == 0:
#     raise ValueError("❌ No training data found. Check frame extraction and directory structure!")


# # Train the model
# model.fit(X, Y, epochs=15, batch_size=8, validation_split=0.2)


# def classify_video(video_path):
#     frames = extract_frames(video_path, "temp_frames")
    
#     if len(frames) < SEQUENCE_LENGTH:
#         print("❌ Not enough frames!")
#         return "❌ Error"

#     frames = np.array(frames[:SEQUENCE_LENGTH])
#     frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    
#     prediction = model.predict(frames)[0]
    
#     return "🎬 Movie Scene" if prediction[0] > prediction[1] else "📹 Real-Life Scene"

# # Test Classification
# result = classify_video(r"D:\video_classification_mannually\video_classification_without_any_model\fake.mp4")
# print(f"Final Classification: {result}")

# this is code that how can i save model of video_classification_model_using_resnet.h5



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random
import os
import cv2

IMG_SIZE = (128, 128)
FRAME_INTERVAL = 5 

def extract_frames(video_path, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_list = []  # Store frame paths

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame_filename = os.path.join(save_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_list.append(frame_filename)  # Store extracted frame paths

        frame_count += 1

    cap.release()

    if len(frame_list) == 0:
        return None  # Return None if no frames were extracted

    print(f"✅ Extracted frames from {video_path} to {save_folder}")
    return frame_list  # Return the list of extracted frames

# Extract frames for movie & real-life videos
for video in os.listdir("dataset/movie_videos"):
    extract_frames(f"dataset/movie_videos/{video}", "dataset/movie_frames")

for video in os.listdir("dataset/real_life_videos"):
    extract_frames(f"dataset/real_life_videos/{video}", "dataset/real_life_frames")


# **Model Parameters**
IMG_SHAPE = (128, 128, 3)
SEQUENCE_LENGTH = 20  # Number of frames per sequence

# **Define CNN for Feature Extraction**
cnn = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SHAPE),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten()
])

# **LSTM Model for Sequence Processing**
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

# ✅ Save the model using `.keras` format
model.save("video_classification_model.keras", save_format="keras")
print("✅ Model saved successfully!")


# Function to create frame sequences
def load_video_sequences(folder, label):
    sequences = []
    labels = []
    
    video_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    for vid_folder in video_folders:
        frames = sorted(os.listdir(os.path.join(folder, vid_folder)))
        frames = [img_to_array(load_img(os.path.join(folder, vid_folder, f))) / 255.0 for f in frames]

        # Ensure sequence length
        if len(frames) >= SEQUENCE_LENGTH:
            for i in range(0, len(frames) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
                sequences.append(np.array(frames[i:i+SEQUENCE_LENGTH]))
                labels.append(label)

    return np.array(sequences), np.array(labels)

# Load data
movie_x, movie_y = load_video_sequences("dataset/movie_frames", [1, 0])
real_x, real_y = load_video_sequences("dataset/real_life_frames", [0, 1])

# Combine & Shuffle
X = np.concatenate((movie_x, real_x))
Y = np.concatenate((movie_y, real_y))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

print(f"Loaded {X.shape[0]} samples")  # Debugging statement
if X.shape[0] == 0:
    raise ValueError("❌ No training data found. Check frame extraction and directory structure!")

# Train the model
model.fit(X, Y, epochs=15, batch_size=8, validation_split=0.2)

# ✅ Load the model safely
model_path = "video_classification_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")


def classify_video(video_path):
    frames = extract_frames(video_path, "temp_frames")
    
    if len(frames) < SEQUENCE_LENGTH:
        print("❌ Not enough frames!")
        return "❌ Error"

    frames = [img_to_array(load_img(f)) / 255.0 for f in frames[:SEQUENCE_LENGTH]]  # Load images properly
    frames = np.array(frames).reshape(1, SEQUENCE_LENGTH, 128, 128, 3)  # Reshape for LSTM model
    
    prediction = model.predict(frames)[0]
    
    return "🎬 Movie Scene" if prediction[0]  prediction[1] else "📹 Real-Life Scene"

# Test Classification
result = classify_video(r"D:\video_classification_mannually\video_classification_without_any_model\fake.mp4")
print(f"Final Classification: {result}")
