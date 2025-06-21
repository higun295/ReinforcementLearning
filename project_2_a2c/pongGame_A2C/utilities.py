import cv2
import numpy as np
import os

def save_video(frames, path, fps=30):
    """List of RGB frames (np.array) -> mp4 영상으로 저장"""
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR 형식 사용
    out.release()
    print(f"[✔] 영상 저장 완료: {path}")
