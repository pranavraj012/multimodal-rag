import cv2
import subprocess
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import shutil
import glob
import os
from config import SSIM_THRESHOLD, FRAMES_PATH

def get_ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path: return path
    
    # Fallback: Search for winget installation
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "**", "ffmpeg.exe")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return "ffmpeg"

def extract_audio(video_path: str, output_wav: str):
    """Extract audio as 16kHz mono WAV — required format for faster-whisper"""
    abs_video = Path(video_path).resolve()
    abs_audio = Path(output_wav).resolve()
    ffmpeg_exe = get_ffmpeg_path()
    
    try:
        subprocess.run(
            f'"{ffmpeg_exe}" -i "{abs_video}" -ar 16000 -ac 1 -y "{abs_audio}"',
            shell=True, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[FFmpeg Error] stderr: {e.stderr}")
        raise e

def extract_keyframes(video_path: str, lecture_id: str) -> list[dict]:
    """
    SSIM-based keyframe extraction.
    """
    out_dir = Path(FRAMES_PATH) / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    keyframes = []
    prev_gray_small = None
    prev_frame = None
    prev_ts = 0.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (320, 180))  # Small for cheap SSIM

        if prev_gray_small is not None:
            # win_size is required for ssim when image is small
            score = ssim(prev_gray_small, gray_small, data_range=255, win_size=3)

            if score < SSIM_THRESHOLD:
                # Scene change detected — save the PREVIOUS fully-rendered frame
                save_path = str(out_dir / f"kf_{frame_idx:06d}_{prev_ts:.1f}s.jpg")
                cv2.imwrite(save_path, prev_frame)
                keyframes.append({
                    "frame_path": save_path,
                    "timestamp": prev_ts,
                    "frame_idx": frame_idx - 1
                })

        prev_gray_small = gray_small
        prev_frame = frame.copy()
        prev_ts = timestamp
        frame_idx += 1

    # Always save the final frame
    if prev_frame is not None:
        save_path = str(out_dir / f"kf_final_{prev_ts:.1f}s.jpg")
        cv2.imwrite(save_path, prev_frame)
        keyframes.append({
            "frame_path": save_path,
            "timestamp": prev_ts,
            "frame_idx": frame_idx - 1
        })

    cap.release()
    print(f"[Stage 1] Extracted {len(keyframes)} keyframes from {frame_idx} total frames")
    return keyframes
