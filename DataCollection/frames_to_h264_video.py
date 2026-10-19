#!/usr/bin/env python3
"""Encode frame_*.jpg sequences to H.264 MP4s that Cursor can play.

Writes <trajectory>_h264.mp4 next to the frames. Existing <trajectory>.mp4
files (mp4v) are left untouched. An existing _h264.mp4 is skipped.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import imageio_ffmpeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_PATTERN = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)
TRAJECTORY_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")
DEFAULT_FPS = 10


def sort_frames(paths):
    def frame_number(path):
        match = FRAME_PATTERN.search(os.path.basename(path))
        if match is None:
            return float("inf")
        return int(match.group(1))

    return sorted(paths, key=frame_number)


def iter_trajectory_dirs(base_dir):
    for name in sorted(os.listdir(base_dir)):
        if not TRAJECTORY_DIR_PATTERN.match(name):
            continue
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            yield path, name


def output_path_for(traj_dir, traj_name):
    return os.path.join(traj_dir, f"{traj_name}_h264.mp4")


def write_concat_list(frame_paths, fps, list_path):
    duration = 1.0 / fps
    with open(list_path, "w", encoding="utf-8", newline="\n") as handle:
        for frame_path in frame_paths:
            posix_path = os.path.abspath(frame_path).replace("\\", "/")
            escaped = posix_path.replace("'", r"'\''")
            handle.write(f"file '{escaped}'\n")
            handle.write(f"duration {duration:.8f}\n")
        # Repeat the last file so the concat demuxer keeps its duration.
        last_path = os.path.abspath(frame_paths[-1]).replace("\\", "/")
        escaped_last = last_path.replace("'", r"'\''")
        handle.write(f"file '{escaped_last}'\n")


def encode_trajectory(traj_dir, traj_name, fps, ffmpeg_exe):
    frame_paths = [
        os.path.join(traj_dir, name)
        for name in os.listdir(traj_dir)
        if FRAME_PATTERN.search(name)
    ]
    if not frame_paths:
        return "skipped_no_frames", 0

    destination = output_path_for(traj_dir, traj_name)
    if os.path.exists(destination):
        return "skipped_exists", len(frame_paths)

    frame_paths = sort_frames(frame_paths)
    list_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix=f"{traj_name}_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            list_path = handle.name

        write_concat_list(frame_paths, fps, list_path)

        command = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            destination,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if os.path.exists(destination):
                os.remove(destination)
            message = (result.stderr or result.stdout or "").strip()
            print(f"[FAIL] {traj_name}: {message}")
            return "failed_encode", len(frame_paths)
    finally:
        if list_path and os.path.exists(list_path):
            os.remove(list_path)

    return "created", len(frame_paths)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one H.264 MP4 per trajectory for Cursor playback. "
            "Does not overwrite existing <trajectory>.mp4 files."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=BASE_DIR,
        help="Path to the DataCollection directory (default: this script's folder)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Output video frame rate (default: {DEFAULT_FPS})",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        print("Error: fps must be positive", file=sys.stderr)
        return 1

    if not os.path.isdir(args.base_dir):
        print(f"Error: base directory not found: {args.base_dir}", file=sys.stderr)
        return 1

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    stats = {
        "created": 0,
        "skipped_no_frames": 0,
        "skipped_exists": 0,
        "failed_encode": 0,
    }
    total_frames_written = 0
    start_time = time.time()

    for traj_dir, traj_name in iter_trajectory_dirs(args.base_dir):
        status, frame_count = encode_trajectory(
            traj_dir, traj_name, args.fps, ffmpeg_exe
        )
        stats[status] += 1

        if status == "created":
            total_frames_written += frame_count
            print(
                f"[OK]   {traj_name}: {frame_count} frames -> {traj_name}_h264.mp4"
            )
        elif status == "skipped_exists":
            print(f"[SKIP] {traj_name}: {traj_name}_h264.mp4 already exists")
        elif status == "skipped_no_frames":
            print(f"[SKIP] {traj_name}: no frame_*.jpg files")

    elapsed = time.time() - start_time
    print("\n=== Summary ===")
    print(f"Created:              {stats['created']}")
    print(f"Skipped (no frames):  {stats['skipped_no_frames']}")
    print(f"Skipped (exists):     {stats['skipped_exists']}")
    print(f"Failed:               {stats['failed_encode']}")
    print(f"Total frames written: {total_frames_written}")
    print(f"Elapsed:              {elapsed:.1f}s")
    print("Old <trajectory>.mp4 files were not modified.")

    return 1 if stats["failed_encode"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
