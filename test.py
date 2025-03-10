import cv2
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

def extract_audio(video_path, output_audio_path):
    os.system(f"ffmpeg -i \"{video_path}\" -ac 1 -ar 44100 -vn \"{output_audio_path}\" -y")

def analyze_audio(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    if not os.path.exists(audio_path):
        return 0.0
    
    y, sr = librosa.load(audio_path, sr=None)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    os.remove(audio_path)
    
    return min(1.0, (spectral_contrast / 30 + zero_crossing_rate * 10 + mfccs / 100))

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
            if score < 0.6:  # Lower SSIM means a significant scene change
                scene_changes += 1
        
        prev_frame = gray
        frame_count += 1
        if frame_count > 150:
            break
    
    cap.release()
    return min(1.0, scene_changes / 10)

def analyze_edge_density(video_path):
    cap = cv2.VideoCapture(video_path)
    edge_scores = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_scores.append(np.mean(edges))
        
        frame_count += 1
        if frame_count > 150:
            break
    
    cap.release()
    return min(1.0, np.mean(edge_scores) / 50)

def analyze_brightness_variance(video_path):
    cap = cv2.VideoCapture(video_path)
    brightness_scores = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_scores.append(np.var(gray))
        
        frame_count += 1
        if frame_count > 150:
            break
    
    cap.release()
    return min(1.0, np.mean(brightness_scores) / 5000)

def analyze_motion_smoothness(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motion_scores = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)
            motion_scores.append(np.mean(np.linalg.norm(flow, axis=2)))
        
        prev_gray = gray
        frame_count += 1
        if frame_count > 150:
            break
    
    cap.release()
    return min(1.0, np.mean(motion_scores) / 6)

def classify_scene(video_path):
    audio_prob = analyze_audio(video_path)
    scene_change_prob = analyze_scene_changes(video_path)
    edge_prob = analyze_edge_density(video_path)
    brightness_prob = analyze_brightness_variance(video_path)
    motion_prob = analyze_motion_smoothness(video_path)
    
    final_probability = (
        (audio_prob * 0.25) + 
        (scene_change_prob * 0.3) + 
        (edge_prob * 0.15) + 
        (brightness_prob * 0.15) + 
        (motion_prob * 0.15)
    )
    
    result = {
        "Audio-based Probability": audio_prob,
        "Scene Change Probability": scene_change_prob,
        "Edge Density Probability": edge_prob,
        "Brightness Variance Probability": brightness_prob,
        "Motion Smoothness Probability": motion_prob,
        "Final Movie Scene Probability": final_probability,
        "Classification": "Movie Scene" if final_probability > 0.51 else "Real-Life Scene"
    }
    return result

# Example usage:
video_path = r"D:\video_classification_mannually\video_classification_without_any_model\fake3.mp4"
result = classify_scene(video_path)
print(result)


