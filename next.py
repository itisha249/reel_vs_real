# import streamlit as st
# import cv2
# import numpy as np
# import os
# import soundfile as sf
# from scipy.signal import find_peaks
# from moviepy.editor import VideoFileClip
# from skimage.metrics import structural_similarity as ssim
# from tensorflow.keras.models import load_model

# # Load pre-trained model
# MODEL_PATH = "h5_models/scene_classifier3.h5"
# model = load_model(MODEL_PATH)

# # Function to extract frames
# def extract_frames(video_path, num_frames=10):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = []
    
#     for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.resize(frame, (224, 224)) / 255.0
#             frames.append(frame)
#     cap.release()
#     return np.array(frames)



# def preprocess_frames(frames, target_size=(128, 128)):
#     processed_frames = []
#     for frame in frames:
#         resized_frame = cv2.resize(frame, target_size)
#         normalized_frame = resized_frame / 255.0
#         processed_frames.append(normalized_frame)

#     processed_frames = np.array(processed_frames)

#     # FIX: Reshape correctly based on model expectations
#     if len(processed_frames.shape) == 4:  # (10, 128, 128, 3)
#         processed_frames = processed_frames[0:1]  # Take only 1 frame, reshape to (1, 128, 128, 3)

#     return processed_frames


# # Function to get FPS
# def get_fps_opencv(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     print(fps)
#     return fps

# # Function to extract and analyze audio
# def extract_audio(video_path, output_audio_path):
#     try:
#         video = VideoFileClip(video_path)
#         video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")
#     except Exception as e:
#         st.error(f"Error extracting audio: {e}")

# def analyze_audio_peaks(video_path):
#     audio_path = "temp_audio.wav"
#     extract_audio(video_path, audio_path)
#     if not os.path.exists(audio_path):
#         return 0.0
    
#     y, sr = sf.read(audio_path)
#     os.remove(audio_path)
    
#     if len(y.shape) > 1:
#         y = np.mean(y, axis=1)
    
#     energy = np.abs(y)
#     peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.5)
#     peak_density = len(peaks) / len(y)
    
#     return min(1.0, peak_density * 1000)

# # Function to analyze scene changes
# def analyze_scene_changes(video_path):
#     cap = cv2.VideoCapture(video_path)
#     prev_frame = None
#     scene_changes = 0
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         if prev_frame is not None:
#             score, _ = ssim(prev_frame, gray, full=True)
#             if score < 0.6:
#                 scene_changes += 1
        
#         prev_frame = gray
#         frame_count += 1
#         if frame_count > 150:
#             break
    
#     cap.release()
#     return min(1.0, scene_changes / 10)

# # Classification function
# def classify_scene(video_path):
#     fps =  get_fps_opencv(video_path)
#     frames = extract_frames(video_path)
    
#     # # Model prediction
#     # frames_expanded = np.expand_dims(frames, axis=0)
#     # model_prediction = model.predict(frames_expanded)[0]
#     # model_classification = "Movie Scene" if model_prediction > 0.5 else "Real-Life Scene"

#     frames = extract_frames(video_path)  # Extract frames from video
#     processed_frames = preprocess_frames(frames)  # Resize & Normalize

#     # Ensure the shape matches model input
#     print("Processed Frames Shape:", processed_frames.shape)  # Debugging

#     # Predict using the trained model
#     model_prediction = model.predict(processed_frames)[0]
#     model_classification = "Movie Scene" if np.argmax(model_prediction) == 1 else "Real-Life Scene"

    
#     # Rule-based classification
#     audio_prob = analyze_audio_peaks(video_path)
#     scene_change_prob = analyze_scene_changes(video_path)
#     final_probability = (audio_prob * 0.5) + (scene_change_prob * 0.5)

#     if 22 < fps < 26:
#         fps_based_classification = "Movie Scene"
#     elif fps==0:
#         fps_based_classification = "Movie Scene"
#     else:
#         fps_based_classification = "Real-Life Scene"
#     # fps_based_classification = "Movie Scene" if 22 < fps < 26 or fps == 0 else "Real-Life Scene"
#     rule_based_classification = "Movie Scene" if final_probability > 0.55 else "Real-Life Scene"
    
#     return {
#         "CNN + LSTM + VGG16 Model Classification": model_classification,
#         "FPS-Based Classification": fps_based_classification,
#         "Rule-Based Classification": rule_based_classification,
#         "Audio-Based Probability": audio_prob,
#         "Scene Change Probability": scene_change_prob,
#         "Final Movie Scene Probability": final_probability
#     }

# # Streamlit UI
# st.title("Video Scene Classification")

# uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

# if uploaded_file is not None:
#     with open("temp_video.mp4", "wb") as f:
#         f.write(uploaded_file.read())

#     st.video("temp_video.mp4")  # Streamlit may keep the file open

#     if st.button("Classify Scene"):
#         result = classify_scene("temp_video.mp4")
        
#         st.write("## Classification Results")
#         st.json(result)

#     # Add a try-except block to handle file deletion errors
#     import time

#     time.sleep(1)  # Give some time for OS to release the file

#     try:
#         os.remove("temp_video.mp4")
#     except PermissionError:
#         st.warning("Could not delete temp_video.mp4. Try restarting the app.")

import streamlit as st
import cv2
import numpy as np
import os
import soundfile as sf
import time
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model

# Load pre-trained model
MODEL_PATH = "h5_models/scene_classifier3.h5"
model = load_model(MODEL_PATH)

# Extract frames for rule-based classification
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224)) / 255.0
            frames.append(frame)
    cap.release()
    return np.array(frames)

# Extract frames for model prediction
def extract_frames_for_model(video_path, num_frames=10, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0
            frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted.")

    selected_frame = np.array(frames[0])  # Select the first frame
    return np.expand_dims(selected_frame, axis=0)  # Shape → (1, 128, 128, 3)



# Model prediction
def predict_with_model(video_path):
    frames = extract_frames_for_model(video_path)
    prediction = model.predict(frames)[0][0]
    return "Movie Scene" if prediction > 0.5 else "Real-Life Scene"

# Get FPS
def get_fps_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# Extract & analyze audio
def extract_audio(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")  # WAV format
    except Exception as e:
        print(f"Error extracting audio: {e}")

def analyze_audio_peaks(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    if not os.path.exists(audio_path):
        return 0.0
    y, sr = sf.read(audio_path)
    os.remove(audio_path)
    
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)  # Convert stereo to mono
    
    energy = np.abs(y)
    peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.5)
    peak_density = len(peaks) / len(y)
    
    return min(1.0, peak_density * 1000)

# Analyze scene changes
def analyze_scene_changes(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    scene_changes = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            score, _ = ssim(prev_frame, gray, full=True)
            if score < 0.6:
                scene_changes += 1
        
        prev_frame = gray
        frame_count += 1
        if frame_count > 150:
            break
    
    cap.release()
    return min(1.0, scene_changes / 10)

# Classification function
def classify_scene(video_path):
    fps = get_fps_opencv(video_path)
    
    audio_prob = analyze_audio_peaks(video_path)
    scene_change_prob = analyze_scene_changes(video_path)
    
    final_probability = (
        (audio_prob * 0.25) + 
        (scene_change_prob * 0.3)
    )
    
    if 22 < fps < 26 or fps == 0:
        fps_based_classification = "Movie Scene"
    else:
        fps_based_classification = "Real-Life Scene"
    
    model_classification = predict_with_model(video_path)
    
    final_classification = "Movie Scene" if final_probability > 0.55 else "Real-Life Scene"
    
    return {
        "CNN + LSTM + VGG16 Model Classification": model_classification,
        "FPS-Based Classification": fps_based_classification,
        "Audio-Based Probability": audio_prob,
        "Scene Change Probability": scene_change_prob,
        "Final Movie Scene Probability": final_probability,
        "Final Classification": final_classification
    }

# Streamlit UI
st.title("Video Scene Classification")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    if st.button("Classify Scene"):
        result = classify_scene(video_path)

        st.write("## Classification Results")
        st.json(result)

        time.sleep(1)  # Ensure OS releases the file
        try:
            os.remove(video_path)
        except PermissionError:
            st.warning("Could not delete temp_video.mp4. Try restarting the app.")
