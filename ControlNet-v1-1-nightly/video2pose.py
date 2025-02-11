import os
import sys
sys.path.append('.')
from annotator.dwpose import DWposeDetector
from pathlib import Path
import cv2

from utils import get_fps, read_frames, save_videos_from_pil
import numpy as np
from annotator.util import resize_image, HWC3
from PIL import Image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--folder", type=str, help="Folder containing the video")
    args = parser.parse_args()

    video_path = os.path.join(args.folder, "video.mp4")

    # if not os.path.exists(args.video_path):
    if not os.path.exists(video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    # dir_path, video_name = (
    #     os.path.dirname(args.video_path),
    #     os.path.splitext(os.path.basename(args.video_path))[0],
    # )
    # out_path = os.path.join(dir_path, video_name + "_kps.mp4")
    out_path = os.path.join(args.folder, "pose.mp4")

    detector = DWposeDetector()

    # fps = get_fps(args.video_path)
    # frames = read_frames(args.video_path)
    fps = get_fps(video_path)
    frames = read_frames(video_path)
    print(fps)
    slice_frame = int(fps*4)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        frame_pil = np.array(frame_pil, dtype=np.uint8)
        frame_pil = HWC3(frame_pil)
        frame_pil = resize_image(frame_pil, 512)
        result= detector(frame_pil)
        result = HWC3(result)
        img = resize_image(frame_pil, 512)
        H, W, C = img.shape
        result = cv2.resize(
            result, (W, H), interpolation=cv2.INTER_LINEAR
        )
        result = Image.fromarray(result)
        kps_results.append(result)

    print(out_path)
    save_videos_from_pil(kps_results, out_path, fps=fps)
