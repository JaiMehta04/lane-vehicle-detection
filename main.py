"""
Lane & Car Detection — Main Entry Point

Process dash-cam videos or images with lane detection, vehicle detection, or both.

Usage:
    python main.py --mode both --input data/project_video.mp4 --output output/result.mp4
    python main.py --mode lane --input data/solidWhiteRight.mp4
    python main.py --mode lane --input data/project_video.mp4 --lane-mode advanced
    python main.py --mode car  --input data/project_video.mp4
    python main.py --train
"""

import argparse
import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from src.lane_detection import detect_lanes
from src.advanced_lane_detection import AdvancedLaneDetector
from src.car_detection import VehicleDetector
from config import MODEL_PATH, OUTPUT_DIR


def _process_image(image, mode, detector, lane_detector=None):
    """Apply the selected pipeline(s) to a single frame."""
    if mode == "lane":
        if lane_detector is not None:
            return lane_detector.detect(image)
        return detect_lanes(image)
    elif mode == "car":
        return detector.detect(image)
    else:  # both
        if lane_detector is not None:
            lane_result = lane_detector.detect(image)
        else:
            lane_result = detect_lanes(image)
        return detector.detect(lane_result)


def process_video(input_path, output_path, mode, detector, lane_detector=None):
    """Process a video file frame-by-frame and write the result."""
    if detector:
        detector.reset()
    if lane_detector:
        lane_detector.reset()

    clip = VideoFileClip(input_path)
    result = clip.fl_image(lambda frame: _process_image(frame, mode, detector, lane_detector))
    result.write_videofile(output_path, audio=False, logger="bar")
    print(f"Output saved to {output_path}")


def process_single_image(input_path, output_path, mode, detector, lane_detector=None):
    """Process a single image and save the result."""
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _process_image(image, mode, detector, lane_detector)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result)
    print(f"Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Lane & Car Detection on dash-cam footage",
    )
    parser.add_argument(
        "--mode", choices=["lane", "car", "both"], default="both",
        help="Detection mode: lane lines only, cars only, or both (default: both)",
    )
    parser.add_argument("--input", type=str, help="Path to input video or image file")
    parser.add_argument("--output", type=str, help="Path for output file (auto-generated if omitted)")
    parser.add_argument(
        "--lane-mode", choices=["basic", "advanced"], default="advanced",
        help="Lane detection method: basic (Hough) or advanced (polynomial fit) (default: advanced)",
    )
    parser.add_argument("--train", action="store_true", help="Train the vehicle detection model")
    args = parser.parse_args()

    # Training mode
    if args.train:
        from src.train import train_model
        train_model()
        return

    if not args.input:
        parser.error("--input is required when not training")

    # Build output path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        ext = os.path.splitext(args.input)[1]
        output_path = os.path.join(OUTPUT_DIR, f"{base}_{args.mode}{ext}")

    # Load detector if needed
    detector = None
    if args.mode in ("car", "both"):
        if not os.path.isfile(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}. Training first ...")
            from src.train import train_model
            train_model()
        detector = VehicleDetector(MODEL_PATH)

    # Build advanced lane detector if requested
    lane_detector = None
    if args.lane_mode == "advanced" and args.mode in ("lane", "both"):
        lane_detector = AdvancedLaneDetector()
        print("Using advanced lane detection (polynomial fit + bird's-eye view)")

    # Dispatch based on file type
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        process_video(args.input, output_path, args.mode, detector, lane_detector)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp"):
        process_single_image(args.input, output_path, args.mode, detector, lane_detector)
    else:
        print(f"Unsupported file format: {ext}")


if __name__ == "__main__":
    main()
