import cv2

def get_fps_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(fps)
    return fps

video_path = "D:\video_classification_mannually\nls\dataset\real_life_videos\real_video2.mp4"
fps = get_fps_opencv(video_path)
if fps>22 and fps<26:
    print("Give video is a movie clip")
elif fps==0:
     print("Give video is a movie clip")
else:
    print("Give video is not a movie clip")