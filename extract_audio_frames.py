"""
extract_audio_frames.py
Usage:
    python extract_audio_frames.py video1.mp4 --fps 3
    python extract_audio_frames.py video2.mp4 --fps 5 --start 2 --duration 10
"""

import os
import sys
import argparse
from moviepy.editor import VideoFileClip
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def extract(video_path: str, outdir: str, fps: int = 3, start: float = None, duration: float = None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Prepare output directories
    audio_dir = Path(outdir) / "audio"
    frames_dir = Path(outdir) / "frames"
    ensure_dir(audio_dir)
    ensure_dir(frames_dir)
    
    # Load video
    print(f"Loading video: {video_path}")
    clip = VideoFileClip(str(video_path))
    # Optionally trim
    if start is not None or duration is not None:
        s = start if start is not None else 0
        d = duration if duration is not None else clip.duration - s
        end_t = min(s + d, clip.duration)
        clip = clip.subclip(s, end_t)
        print(f"Using subclip: start={s}s duration={d}s (trimmed to {clip.duration}s)")
    
    # Extract audio
    audio_out = audio_dir / f"{video_path.stem}.wav"
    print(f"Writing audio to: {audio_out}")
    # write_audiofile parameters: fps (sample rate), nbytes etc can be tuned
    clip.audio.write_audiofile(str(audio_out), fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
    
    # Write frames: choose fps (frames per second)
    print(f"Writing frames to: {frames_dir} with fps={fps}")
    # moviepy's write_images_sequence names files using the pattern - choose leading zeros
    frames_pattern = str(frames_dir / f"{video_path.stem}_%05d.jpg")
    clip.write_images_sequence(frames_pattern, fps=fps, verbose=False, logger=None)
    
    print("Done.")
    clip.close()

def main():
    parser = argparse.ArgumentParser(description="Extract audio (wav) and frames from a video file.")
    parser.add_argument("video", help="Path to video file (mp4/mov/etc.)")
    parser.add_argument("--outdir", default="output", help="Output folder")
    parser.add_argument("--fps", type=int, default=3, help="Frames per second to export")
    parser.add_argument("--start", type=float, default=None, help="Start time (seconds) to trim from")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds) to keep after start")
    args = parser.parse_args()

    extract(args.video, args.outdir, fps=args.fps, start=args.start, duration=args.duration)

if __name__ == "__main__":
    main()
