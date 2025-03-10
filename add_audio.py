#type: ignore
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import moviepy
from moviepy.editor import VideoFileClip
from skimage.metrics import structural_similarity as ssim
import soundfile as sf
from scipy.signal import find_peaks

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
    
    # Ensure audio is 1-D
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)  # Convert stereo to mono
    
    # Compute energy envelope
    energy = np.abs(y)
    peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.5)
    peak_density = len(peaks) / len(y)
    
    return min(1.0, peak_density * 1000)  # Normalize peak density

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
    audio_prob = analyze_audio_peaks(video_path)
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
        "Classification": "Movie Scene" if final_probability > 0.55 else "Real-Life Scene"
    }
    return result

# Example usage:
video_path = r"D:\video_classification_mannually\nls\fake.mp4"
result = classify_scene(video_path)
print(result)
