#!/usr/bin/env python3
"""Convert frame_*.jpg sequences in each trajectory folder to an MP4 video."""

import argparse
import glob
import os
import re
import sys
import time

import cv2

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


def convert_trajectory(traj_dir, traj_name, fps, overwrite):
    frame_paths = glob.glob(os.path.join(traj_dir, "frame_*.jpg"))
    if not frame_paths:
        return "skipped_no_frames", 0

    output_path = os.path.join(traj_dir, f"{traj_name}.mp4")
    if os.path.exists(output_path) and not overwrite:
        return "skipped_exists", len(frame_paths)

    frame_paths = sort_frames(frame_paths)
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return "failed_read", len(frame_paths)

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        return "failed_writer", len(frame_paths)

    writer.write(first_frame)
    for frame_path in frame_paths[1:]:
        frame = cv2.imread(frame_path)
        if frame is None:
            writer.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            return "failed_read", len(frame_paths)

        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return "created", len(frame_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Create one MP4 video per trajectory from frame_*.jpg files."
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate videos even if an MP4 already exists",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: base directory not found: {args.base_dir}", file=sys.stderr)
        return 1

    stats = {
        "created": 0,
        "skipped_no_frames": 0,
        "skipped_exists": 0,
        "failed_read": 0,
        "failed_writer": 0,
    }
    total_frames_written = 0
    start_time = time.time()

    for traj_dir, traj_name in iter_trajectory_dirs(args.base_dir):
        status, frame_count = convert_trajectory(
            traj_dir, traj_name, args.fps, args.overwrite
        )
        stats[status] += 1

        if status == "created":
            total_frames_written += frame_count
            print(f"[OK]   {traj_name}: {frame_count} frames -> {traj_name}.mp4")
        elif status == "skipped_exists":
            print(f"[SKIP] {traj_name}: video already exists")
        elif status == "skipped_no_frames":
            print(f"[SKIP] {traj_name}: no frame_*.jpg files")
        elif status == "failed_read":
            print(f"[FAIL] {traj_name}: could not read one or more frames")
        else:
            print(f"[FAIL] {traj_name}: could not open video writer")

    elapsed = time.time() - start_time
    failed = stats["failed_read"] + stats["failed_writer"]

    print("\n=== Summary ===")
    print(f"Created:              {stats['created']}")
    print(f"Skipped (no frames):  {stats['skipped_no_frames']}")
    print(f"Skipped (exists):     {stats['skipped_exists']}")
    print(f"Failed:               {failed}")
    print(f"Total frames written: {total_frames_written}")
    print(f"Elapsed:              {elapsed:.1f}s")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
