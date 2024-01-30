import os
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--folder", type=str, help="Folder containing the video")
    args = parser.parse_args()

    video_path = os.path.join(args.folder, "pose.mp4")

    # if not os.path.exists(args.video_path):
    if not os.path.exists(video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    # dir_path, video_name = (
    #     os.path.dirname(args.video_path),
    #     os.path.splitext(os.path.basename(args.video_path))[0],
    # )
    # out_path = os.path.join(dir_path, video_name + "_kps.mp4")
    out_path = os.path.join(args.folder, "pose_fps15.mp4")

    # Load your video
    clip = VideoFileClip(video_path)

    # Set the new FPS (for example, 15 FPS)
    new_fps = 15

    # Reduce the video's FPS and save the result
    new_clip = clip.set_fps(new_fps)
    new_clip.write_videofile(out_path)
