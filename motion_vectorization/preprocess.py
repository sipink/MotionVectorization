# Extract frames and preprocess with AI engines (SAM2, FlowSeek, CoTracker3)
import cv2
import os
import sys
import numpy as np
import argparse
import torch
from pathlib import Path
from typing import Optional, List

np.random.seed(0)


def create_argument_parser():
    """Create and return the argument parser for preprocess module"""
    parser = argparse.ArgumentParser()
    # Video and directory information.
    parser.add_argument(
        "--video_file", type=str, required=True, help="Name of the video to process."
    )
    parser.add_argument(
        "--video_dir", default="videos", help="Directory containing videos."
    )
    parser.add_argument(
        "--thresh", default=1e-4, type=float, help="RGB difference threshold."
    )
    parser.add_argument(
        "--min_dim", default=1024, type=int, help="Minimum frame dimension."
    )
    parser.add_argument(
        "--max_frames",
        default=-1,
        type=int,
        help="Maximum number of frames to process.",
    )

    # Add AI engine processing arguments
    parser.add_argument(
        "--use_ai_engines",
        action="store_true",
        default=True,
        help="Use SAM2, CoTracker3, and FlowSeek for preprocessing",
    )
    parser.add_argument(
        "--generate_labels",
        action="store_true",
        default=True,
        help="Generate segmentation labels using SAM2",
    )
    parser.add_argument(
        "--generate_flow",
        action="store_true",
        default=True,
        help="Generate optical flow using FlowSeek",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for AI processing (auto, cuda, cpu)",
    )

    return parser


def extract_frames_only(args):
    """Extract frames without AI processing"""
    video_name = os.path.splitext(args.video_file.split("/")[-1])[0]
    video_folder = os.path.join(args.video_dir, video_name)
    rgb_folder = os.path.join(video_folder, "rgb")
    if not os.path.exists(rgb_folder):
        os.makedirs(rgb_folder)
    cap = cv2.VideoCapture(os.path.join(args.video_dir, args.video_file))
    prev_frame = None

    frame_idx = 0
    frames_saved = []
    min_dim = args.min_dim  # Use local copy to avoid modifying args
    while True:
        if args.max_frames >= 0:
            if frame_idx >= args.max_frames:
                break

        _, frame = cap.read()
        if frame is None:
            break
        frame_height, frame_width, _ = frame.shape
        if min_dim < 0:
            min_dim = max(frame_height, frame_width)
        if frame_height > min_dim or frame_width > min_dim:
            resize_ratio = min(min_dim / frame_height, min_dim / frame_width)
            frame = cv2.resize(
                frame,
                (int(resize_ratio * frame_width), int(resize_ratio * frame_height)),
            )
        save = True
        if frame_idx > 0 and prev_frame is not None:
            if np.mean(np.abs(frame / 255.0 - prev_frame / 255.0)) < args.thresh:
                save = False
        if save:
            frame_path = os.path.join(rgb_folder, f"{frame_idx + 1:03d}.png")
            cv2.imwrite(frame_path, frame)
            frames_saved.append(frame_path)
        frame_idx += 1
        prev_frame = frame.copy()

    cap.release()
    return video_name, video_folder, frames_saved


def process_with_ai_engines(args, video_name, video_folder, frames_saved):
    """Process frames with SAM2, CoTracker3, and FlowSeek"""
    print("\n🚀 Processing with AI engines...")

    # Import process_video module
    try:
        from .process_video import MotionVectorizationProcessor, VideoProcessorConfig
    except ImportError:
        # If relative import fails, try absolute import
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from motion_vectorization.process_video import (
            MotionVectorizationProcessor,
            VideoProcessorConfig,
        )

    # Configure processor
    config = VideoProcessorConfig(
        video_file=args.video_file,
        video_dir=args.video_dir,
        output_dir="motion_vectorization/outputs",
        suffix=None,
        max_frames=args.max_frames if args.max_frames > 0 else 200,
        device=(
            args.device
            if args.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        use_sam2=args.generate_labels,
        use_cotracker3=True,
        use_flowseek=args.generate_flow,
    )

    # Create processor and run
    processor = MotionVectorizationProcessor(config)

    # Process video to generate all required files
    results = processor.process_video()

    # The processor already saves everything we need:
    # - labels/*.npy (segmentation masks)
    # - fgbg/*.png (foreground/background masks)
    # - comps/*.png (composite visualizations)
    # - flow/forward/*.npy (forward optical flow)
    # - flow/backward/*.npy (backward optical flow)
    # - flow/viz/*.png (flow visualizations)

    print(f"✅ AI preprocessing complete!")
    print(f"📊 Generated files in {video_folder}:")
    print(f"   - RGB frames: {len(frames_saved)} files")

    labels_folder = Path(video_folder) / "labels"
    if labels_folder.exists():
        print(
            f"   - Segmentation labels: {len(list(labels_folder.glob('*.npy')))} files"
        )

    fgbg_folder = Path(video_folder) / "fgbg"
    if fgbg_folder.exists():
        print(f"   - Foreground masks: {len(list(fgbg_folder.glob('*.png')))} files")

    flow_folder = Path(video_folder) / "flow" / "forward"
    if flow_folder.exists():
        print(
            f"   - Optical flow: {len(list(flow_folder.glob('*.npy')))} forward, {len(list((Path(video_folder) / 'flow' / 'backward').glob('*.npy')))} backward"
        )

    return results


def main(args: Optional[List[str]] = None):
    """Main preprocessing pipeline with AI engines"""
    # Parse arguments if not provided
    if args is None:
        # We're being called from command line, parse sys.argv
        parser = create_argument_parser()
        parsed_args = parser.parse_args()
    else:
        # We're being called programmatically with custom args
        parser = create_argument_parser()
        parsed_args = parser.parse_args(args)

    print(f"\n{'='*80}")
    print(f"🎬 MOTION VECTORIZATION PREPROCESSING")
    print(f"{'='*80}")
    print(f"📹 Video: {parsed_args.video_file}")
    print(f"🤖 AI Engines: {'ENABLED' if parsed_args.use_ai_engines else 'DISABLED'}")

    # Step 1: Extract frames
    print("\n📸 Step 1: Extracting frames...")
    video_name, video_folder, frames_saved = extract_frames_only(parsed_args)
    print(f"✅ Extracted {len(frames_saved)} frames to {video_folder}/rgb/")

    # Step 2: Process with AI engines if enabled
    if parsed_args.use_ai_engines and len(frames_saved) > 0:
        print("\n🧠 Step 2: Processing with AI engines...")
        print(
            f"   - SAM2 Segmentation: {'ENABLED' if parsed_args.generate_labels else 'DISABLED'}"
        )
        print(
            f"   - FlowSeek Optical Flow: {'ENABLED' if parsed_args.generate_flow else 'DISABLED'}"
        )
        print(f"   - CoTracker3 Tracking: ENABLED")
        print(f"   - Device: {parsed_args.device}")

        results = process_with_ai_engines(
            parsed_args, video_name, video_folder, frames_saved
        )
    else:
        print("\n⚠️ Skipping AI processing (disabled or no frames)")
        results = None

    print(f"\n{'='*80}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"📂 Output folder: {video_folder}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    main()
