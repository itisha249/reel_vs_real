import cv2
import numpy as np
import os
import librosa
import librosa.display
import soundfile as sf
from moviepy.editor import VideoFileClip
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew
import soundfile as sf
import soundfile as sf


# Step 1: Extract Audio from Video
def extract_audio(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)  # Downsample to 16kHz to reduce memory usage
    
    # Use shorter frame sizes to reduce memory consumption
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=512)
    
    pitch_mean = np.mean(librosa.yin(y, 50, 300, frame_length=2048, hop_length=512))  # Lower frame length
    energy = np.mean(np.abs(y))

    os.remove(audio_path)  # Cleanup

    return np.concatenate([
        np.mean(mfccs, axis=1),
        np.mean(spectral_contrast, axis=1),
        [pitch_mean, energy]
    ])

# Step 3: Extract Visual Features (Texture, Color, Motion, Scene Changes)
def extract_frame_features(frame, prev_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Texture Features (GLCM & LBP)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist = np.histogram(lbp.ravel(), bins=26, range=(0, 26))[0]
    lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()
    
    # Color Features (Histogram)
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Motion Features (Optical Flow)
    motion = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.5, 0)
        motion = np.mean(np.linalg.norm(flow, axis=2))
    
    return np.concatenate([
        [contrast, homogeneity],  # Texture
        lbp_hist,  # Texture (LBP)
        hist,  # Color
        [motion]  # Motion
    ]), gray

# Step 4: Process Video and Extract Features
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    prev_gray = None
    scene_changes = 0
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (320, 180))  # Resize for efficiency
        frame_features, prev_gray = extract_frame_features(frame, prev_gray)
        features.append(frame_features)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            scene_changes += np.mean(diff) > 30  # Count major scene changes
        prev_frame = gray
        
        frame_count += 1
        if frame_count > 50:
            break
    
    cap.release()
    return np.mean(features, axis=0), min(1.0, scene_changes / 10)

# Step 5: Classify a New Video Based on Rule-Based Logic
def classify_scene(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    
    video_features, scene_change_prob = extract_video_features(video_path)
    audio_features = extract_audio_features(audio_path)
    
    final_features = np.concatenate([video_features, audio_features, [scene_change_prob]])
    
    # Rule-based classification
    texture_score = final_features[0]  # Contrast
    motion_score = final_features[-2]  # Motion intensity
    energy_score = final_features[-3]  # Audio loudness
    scene_changes = final_features[-1]  # Scene change probability
    
    prob = (0.3 * texture_score + 0.2 * motion_score + 0.2 * energy_score + 0.3 * scene_changes)
    
    return {
        "Final Movie Scene Probability": prob,
        "Classification": "Movie Scene" if prob > 0.55 else "Real-Life Scene"
    }

# Example Usage
result = classify_scene(r"D:\video_classification_mannually\video_classification_without_any_model\fake3.mp4")
print(result)
