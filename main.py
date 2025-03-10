import cv2
import numpy as np
import librosa
import librosa.display
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import entropy

def extract_audio(video_path, output_audio_path):
    try:
        command = ["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "44100", "-vn", output_audio_path, "-y"]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"Error extracting audio: {e}")

def analyze_audio(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    
    if not os.path.exists(audio_path):
        print("Error: Audio file was not extracted correctly.")
        return 0.0
    
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(spectral_contrast)
    
    # Compute energy entropy
    frame_length = 2048
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    entropy_value = entropy(energy)
    
    os.remove(audio_path)  # Cleanup
    
    # High spectral contrast and low entropy indicate a structured audio track (movie scene)
    audio_probability = min(1.0, contrast_mean / 50) * (1 - min(1.0, entropy_value / 10))
    return audio_probability

def analyze_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    blur_scores = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(laplacian_var)
        frame_count += 1
        
        if frame_count > 100:  # Limit analysis to 100 frames for performance
            break
    
    cap.release()
    avg_blur = np.mean(blur_scores)
    
    # High variance indicates sharp image (movie), low indicates blur (CCTV/real scene)
    clarity_probability = min(1.0, avg_blur / 100)
    return clarity_probability

def analyze_camera_motion(video_path):
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
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_magnitude = np.mean(np.linalg.norm(flow, axis=2))
            motion_scores.append(motion_magnitude)
        
        prev_gray = gray
        frame_count += 1
        if frame_count > 100:  # Limit analysis to 100 frames
            break
    
    cap.release()
    avg_motion = np.mean(motion_scores)
    
    # High motion variance indicates cinematic movement (movie), low is more static (real scene)
    motion_probability = min(1.0, avg_motion / 10)
    return motion_probability

def classify_scene(video_path):
    audio_prob = analyze_audio(video_path)
    clarity_prob = analyze_video_frames(video_path)
    motion_prob = analyze_camera_motion(video_path)
    
    final_probability = (audio_prob + clarity_prob + motion_prob) / 3
    
    result = {
        "Audio-based Probability": audio_prob,
        "Clarity-based Probability": clarity_prob,
        "Motion-based Probability": motion_prob,
        "Final Movie Scene Probability": final_probability
    }
    return result

# Example usage:
video_path = r"D:\video_classification_mannually\video_classification_without_any_model\videos\real_videos\real_video4.mp4"
result = classify_scene(video_path)
print(result)
