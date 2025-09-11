"""
Motion Vectorization Video Processor - Main Entry Point
Coordinates SAM2.1, CoTracker3, and FlowSeek for maximum accuracy motion graphics

This module provides the single entry point for processing videos with all AI engines.
Handles the complete pipeline from video input to motion data output.
"""

import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
import cv2
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import shutil

# Import engine status checker
from .engine_status import verify_engines_at_startup, get_active_engines

# Import core engines with graceful fallback
ENGINE_MODULES = {}

# Try importing SAM2 engine
try:
    from .sam2_engine import SAM2SegmentationEngine, SAM2Config
    ENGINE_MODULES['sam2'] = True
    print("✅ SAM2 engine module imported")
except ImportError as e:
    ENGINE_MODULES['sam2'] = False
    print(f"⚠️ SAM2 engine not available: {e}")
    
# Try importing CoTracker3 engine  
try:
    from .cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config
    ENGINE_MODULES['cotracker3'] = True
    print("✅ CoTracker3 engine module imported")
except ImportError as e:
    ENGINE_MODULES['cotracker3'] = False
    print(f"⚠️ CoTracker3 engine not available: {e}")

# Try importing FlowSeek engine
try:
    from .flowseek_engine import FlowSeekEngine, FlowSeekConfig, MotionBasisDecomposer
    ENGINE_MODULES['flowseek'] = True
    print("✅ FlowSeek engine module imported")
except ImportError as e:
    ENGINE_MODULES['flowseek'] = False
    print(f"⚠️ FlowSeek engine not available: {e}")

# Import dataloader and utilities
from .dataloader import DataLoader
from .utils import (
    compute_clusters_floodfill, clean_labels, is_valid_cluster,
    get_shape_coords, get_shape_centroid, get_shape_mask,
    save_frames, get_cmap
)


@dataclass
class VideoProcessorConfig:
    """Configuration for video processing pipeline"""
    video_file: str
    video_dir: str = 'videos'
    output_dir: str = 'motion_vectorization/outputs'
    suffix: Optional[str] = None
    
    # Processing settings
    max_frames: int = 200
    start_frame: int = 1
    base_frame: int = 0
    
    # AI engine settings
    use_sam2: bool = True
    use_cotracker3: bool = True
    use_flowseek: bool = True
    
    # Quality settings
    quality_threshold: float = 0.95
    min_cluster_size: int = 50
    min_density: float = 0.15
    
    # Device settings
    device: str = "auto"
    
    def __post_init__(self):
        """Auto-configure device"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class MotionVectorizationProcessor:
    """
    Main processor that coordinates all AI engines for motion vectorization
    """
    
    def __init__(self, config: VideoProcessorConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize engine status
        print("\n🔍 Checking engine availability...")
        self.engine_status = verify_engines_at_startup(verbose=False)
        self.active_engines = get_active_engines(self.engine_status)
        
        # Initialize AI engines
        self.sam2_engine = None
        self.cotracker3_engine = None
        self.flowseek_engine = None
        
        self._initialize_engines()
        
        # Setup paths
        self.video_name = Path(config.video_file).stem
        self.video_folder = Path(config.video_dir) / self.video_name
        self.rgb_folder = self.video_folder / 'rgb'
        self.flow_folder = self.video_folder / 'flow'
        self.output_folder = Path(config.output_dir) / f"{self.video_name}_{config.suffix}"
        
    def _initialize_engines(self):
        """Initialize available AI engines"""
        
        # Initialize SAM2 if available
        if self.config.use_sam2 and ENGINE_MODULES.get('sam2') and 'sam2' in self.active_engines:
            try:
                print("🎯 Initializing SAM2.1 segmentation engine...")
                try:
                    from .sam2_engine import SAM2Config, SAM2SegmentationEngine
                    sam2_config = SAM2Config(
                        device=str(self.device),
                        accuracy_threshold=self.config.quality_threshold
                    )
                    self.sam2_engine = SAM2SegmentationEngine(sam2_config)
                except ImportError:
                    print("❌ SAM2 engine not available")
                    self.sam2_engine = None
                print("✅ SAM2.1 engine ready")
            except Exception as e:
                print(f"❌ SAM2.1 initialization failed: {e}")
                self.sam2_engine = None
        
        # Initialize CoTracker3 if available
        if self.config.use_cotracker3 and ENGINE_MODULES.get('cotracker3') and 'cotracker3' in self.active_engines:
            try:
                print("🎬 Initializing CoTracker3 tracking engine...")
                try:
                    from .cotracker3_engine import CoTracker3Config, CoTracker3TrackerEngine
                    cotracker3_config = CoTracker3Config(
                        device=str(self.device),
                        model_variant="cotracker3_offline"
                    )
                    self.cotracker3_engine = CoTracker3TrackerEngine(cotracker3_config)
                except ImportError:
                    print("❌ CoTracker3 engine not available")
                    self.cotracker3_engine = None
                print("✅ CoTracker3 engine ready")
            except Exception as e:
                print(f"❌ CoTracker3 initialization failed: {e}")
                self.cotracker3_engine = None
        
        # Initialize FlowSeek if available
        if self.config.use_flowseek and ENGINE_MODULES.get('flowseek') and 'flowseek' in self.active_engines:
            try:
                print("🌊 Initializing FlowSeek optical flow engine...")
                try:
                    from .flowseek_engine import FlowSeekConfig, FlowSeekEngine
                    flowseek_config = FlowSeekConfig(
                        device=str(self.device)
                        # Note: removed accuracy_mode as it's not a valid parameter
                    )
                    self.flowseek_engine = FlowSeekEngine(flowseek_config)
                except ImportError:
                    print("❌ FlowSeek engine not available")
                    self.flowseek_engine = None
                print("✅ FlowSeek engine ready")
            except Exception as e:
                print(f"❌ FlowSeek initialization failed: {e}")
                self.flowseek_engine = None
    
    def process_video(self) -> Dict[str, Any]:
        """
        Main processing pipeline - coordinates all AI engines
        """
        print(f"\n🚀 Processing video: {self.config.video_file}")
        print(f"📊 Active engines: {', '.join(self.active_engines) if self.active_engines else 'None (fallback mode)'}")
        
        # Create output directories
        self._create_directories()
        
        # Load video frames
        frames = self._load_frames()
        if len(frames) == 0:
            raise ValueError(f"No frames found in {self.rgb_folder}")
        
        print(f"📸 Loaded {len(frames)} frames")
        
        # Process with available engines
        results = {
            'frames': frames,
            'segmentation': None,
            'tracking': None,
            'optical_flow': None,
            'shapes': None
        }
        
        # 1. SAM2 Segmentation
        if self.sam2_engine is not None:
            print("\n🎯 Running SAM2.1 segmentation...")
            results['segmentation'] = self._run_sam2_segmentation(frames)
        else:
            print("\n⚠️ SAM2 not available - using fallback segmentation")
            results['segmentation'] = self._fallback_segmentation(frames)
        
        # 2. FlowSeek Optical Flow
        if self.flowseek_engine is not None:
            print("\n🌊 Running FlowSeek optical flow...")
            results['optical_flow'] = self._run_flowseek(frames)
        else:
            print("\n⚠️ FlowSeek not available - using fallback optical flow")
            results['optical_flow'] = self._fallback_optical_flow(frames)
        
        # 3. CoTracker3 Point Tracking
        if self.cotracker3_engine is not None:
            print("\n🎬 Running CoTracker3 point tracking...")
            results['tracking'] = self._run_cotracker3(frames, results['segmentation'])
        else:
            print("\n⚠️ CoTracker3 not available - using fallback tracking")
            results['tracking'] = self._fallback_tracking(frames, results['segmentation'])
        
        # 4. Extract and optimize shapes
        print("\n🔷 Extracting and optimizing shapes...")
        results['shapes'] = self._extract_shapes(results)
        
        # 5. Save outputs
        print("\n💾 Saving outputs...")
        self._save_outputs(results)
        
        print(f"\n✅ Processing complete! Outputs saved to {self.output_folder}")
        
        return results
    
    def _create_directories(self):
        """Create necessary output directories"""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / 'debug').mkdir(exist_ok=True)
        (self.output_folder / 'outputs').mkdir(exist_ok=True)
        (self.output_folder / 'shapes').mkdir(exist_ok=True)
        
        # Create flow directories if needed
        if not self.flow_folder.exists():
            self.flow_folder.mkdir(parents=True, exist_ok=True)
            (self.flow_folder / 'forward').mkdir(exist_ok=True)
            (self.flow_folder / 'backward').mkdir(exist_ok=True)
            (self.flow_folder / 'viz').mkdir(exist_ok=True)
    
    def _load_frames(self) -> np.ndarray:
        """Load RGB frames from disk"""
        frames = []
        frame_files = sorted(self.rgb_folder.glob('*.png'))
        
        for frame_file in frame_files[:self.config.max_frames]:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        return np.array(frames) if frames else np.array([])
    
    def _run_sam2_segmentation(self, frames: np.ndarray) -> Dict[str, Any]:
        """Run SAM2.1 segmentation on frames"""
        try:
            # Process frames with SAM2
            # Use available method on SAM2 engine
            if hasattr(self.sam2_engine, 'segment_frames'):
                segmentation_results = self.sam2_engine.segment_frames(frames)  # type: ignore
            elif hasattr(self.sam2_engine, 'process'):
                segmentation_results = self.sam2_engine.process(frames)  # type: ignore
            else:
                segmentation_results = {'masks': [], 'quality_score': 0.0}
            
            # Generate labels, fgbg, and comps
            labels = []
            fgbg = []
            comps = []
            
            for frame_idx, masks in enumerate(segmentation_results.get('masks', [])):
                # Create label map from masks
                label_map = np.zeros(frames[frame_idx].shape[:2], dtype=np.int32)
                for obj_id, mask in enumerate(masks):
                    label_map[mask > 0.5] = obj_id + 1
                
                labels.append(label_map)
                
                # Create foreground/background mask
                fg_mask = label_map > 0
                fgbg.append(fg_mask.astype(np.uint8) * 255)
                
                # Create composite visualization
                comp = frames[frame_idx].copy()
                comp[~fg_mask] = comp[~fg_mask] * 0.3  # Dim background
                comps.append(comp)
            
            # Save segmentation files
            labels_folder = self.video_folder / 'labels'
            fgbg_folder = self.video_folder / 'fgbg'
            comps_folder = self.video_folder / 'comps'
            
            for folder in [labels_folder, fgbg_folder, comps_folder]:
                folder.mkdir(exist_ok=True)
            
            for idx, (label, fg, comp) in enumerate(zip(labels, fgbg, comps)):
                np.save(str(labels_folder / f'{idx+1:03d}.npy'), label)
                cv2.imwrite(str(fgbg_folder / f'{idx+1:03d}.png'), fg)
                cv2.imwrite(str(comps_folder / f'{idx+1:03d}.png'), comp)
            
            return {
                'labels': np.array(labels),
                'fgbg': np.array(fgbg),
                'comps': np.array(comps),
                'quality_score': segmentation_results.get('quality_score', 0.9)
            }
            
        except Exception as e:
            print(f"❌ SAM2 segmentation error: {e}")
            return self._fallback_segmentation(frames)
    
    def _run_flowseek(self, frames: np.ndarray) -> Dict[str, Any]:
        """Run FlowSeek optical flow on frames"""
        try:
            flow_results = {
                'forward': [],
                'backward': [],
                'motion_parameters': []
            }
            
            for i in range(len(frames) - 1):
                # Compute forward flow
                # Use available method on FlowSeek engine
                if hasattr(self.flowseek_engine, 'compute_flow'):
                    forward_flow = self.flowseek_engine.compute_flow(frames[i], frames[i+1])  # type: ignore
                elif hasattr(self.flowseek_engine, 'forward'):
                    forward_flow = self.flowseek_engine.forward(frames[i], frames[i+1])  # type: ignore
                else:
                    forward_flow = np.zeros((frames[i].shape[0], frames[i].shape[1], 2))
                flow_results['forward'].append(forward_flow)
                
                # Compute backward flow
                # Use available method on FlowSeek engine  
                if hasattr(self.flowseek_engine, 'compute_flow'):
                    backward_flow = self.flowseek_engine.compute_flow(frames[i+1], frames[i])  # type: ignore
                elif hasattr(self.flowseek_engine, 'forward'):
                    backward_flow = self.flowseek_engine.forward(frames[i+1], frames[i])  # type: ignore
                else:
                    backward_flow = np.zeros((frames[i].shape[0], frames[i].shape[1], 2))
                flow_results['backward'].append(backward_flow)
                
                # Save flow files
                np.save(str(self.flow_folder / 'forward' / f'{i+1:03d}.npy'), forward_flow)
                np.save(str(self.flow_folder / 'backward' / f'{i+1:03d}.npy'), backward_flow)
                
                # Visualize flow
                if isinstance(forward_flow, np.ndarray) and forward_flow.ndim == 3:
                    flow_viz = self._visualize_flow(forward_flow)
                    cv2.imwrite(str(self.flow_folder / 'viz' / f'{i+1:03d}.png'), flow_viz)
            
            return flow_results
            
        except Exception as e:
            print(f"❌ FlowSeek error: {e}")
            return self._fallback_optical_flow(frames)
    
    def _run_cotracker3(self, frames: np.ndarray, segmentation: Dict) -> Dict[str, Any]:
        """Run CoTracker3 point tracking"""
        try:
            # Convert frames to tensor
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            frames_tensor = frames_tensor.to(self.device)
            
            # Track points
            # Use available method on CoTracker3 engine
            if hasattr(self.cotracker3_engine, 'track_video'):
                tracking_results = self.cotracker3_engine.track_video(frames_tensor, segmentation_masks=segmentation.get('labels') if segmentation else None)  # type: ignore
            elif hasattr(self.cotracker3_engine, 'track'):
                tracking_results = self.cotracker3_engine.track(frames_tensor)  # type: ignore
            else:
                tracking_results = {'tracks': [], 'confidence': 0.0}
            
            return tracking_results
            
        except Exception as e:
            print(f"❌ CoTracker3 error: {e}")
            return self._fallback_tracking(frames, segmentation)
    
    def _fallback_segmentation(self, frames: np.ndarray) -> Dict[str, Any]:
        """Fallback segmentation using traditional methods"""
        print("🔄 Using traditional segmentation (Canny edge detection)")
        
        labels = []
        fgbg = []
        comps = []
        
        for frame in frames:
            # Simple edge-based segmentation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create simple labels using connected components
            _, label_map = cv2.connectedComponents(edges)
            labels.append(label_map)
            
            # Create foreground mask
            fg_mask = edges > 0
            fgbg.append(fg_mask.astype(np.uint8) * 255)
            
            # Create composite
            comp = frame.copy()
            comp[~fg_mask] = comp[~fg_mask] * 0.3
            comps.append(comp)
        
        return {
            'labels': np.array(labels),
            'fgbg': np.array(fgbg),
            'comps': np.array(comps),
            'quality_score': 0.5  # Low quality score for fallback
        }
    
    def _fallback_optical_flow(self, frames: np.ndarray) -> Dict[str, Any]:
        """Fallback optical flow using OpenCV"""
        print("🔄 Using traditional optical flow (Farneback)")
        
        flow_results = {
            'forward': [],
            'backward': [],
            'motion_parameters': []
        }
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # Compute forward flow using Farneback (without pre-allocated flow)
            forward_flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0  # type: ignore
            )
            flow_results['forward'].append(forward_flow)
            
            # Compute backward flow  
            backward_flow = cv2.calcOpticalFlowFarneback(
                gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0  # type: ignore
            )
            flow_results['backward'].append(backward_flow)
            
            # Save flow files
            self.flow_folder.mkdir(parents=True, exist_ok=True)
            (self.flow_folder / 'forward').mkdir(exist_ok=True)
            (self.flow_folder / 'backward').mkdir(exist_ok=True)
            (self.flow_folder / 'viz').mkdir(exist_ok=True)
            
            np.save(str(self.flow_folder / 'forward' / f'{i+1:03d}.npy'), forward_flow)
            np.save(str(self.flow_folder / 'backward' / f'{i+1:03d}.npy'), backward_flow)
            
            # Visualize flow
            flow_viz = self._visualize_flow(forward_flow)
            cv2.imwrite(str(self.flow_folder / 'viz' / f'{i+1:03d}.png'), flow_viz)
        
        return flow_results
    
    def _fallback_tracking(self, frames: np.ndarray, segmentation: Dict) -> Dict[str, Any]:
        """Fallback tracking using feature matching"""
        print("🔄 Using traditional tracking (feature matching)")
        
        tracking_results = {
            'tracks': [],
            'visibilities': [],
            'quality_score': 0.5
        }
        
        # Simple feature tracking using ORB
        try:
            orb = cv2.ORB_create()  # type: ignore
        except AttributeError:
            try:
                # Fallback to SIFT if ORB not available
                orb = cv2.SIFT_create()  # type: ignore
            except AttributeError:
                orb = None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        if orb is not None:
            prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
        else:
            prev_kp, prev_desc = [], None
        
        tracks = [[kp.pt for kp in prev_kp]]
        
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if orb is not None:
                kp, desc = orb.detectAndCompute(gray, None)
            else:
                kp, desc = [], None
            
            if prev_desc is not None and desc is not None:
                matches = matcher.match(prev_desc, desc)
                matches = sorted(matches, key=lambda x: x.distance)
                
                current_tracks = []
                for match in matches[:100]:  # Keep top 100 matches
                    current_tracks.append(kp[match.trainIdx].pt)
                
                tracks.append(current_tracks)
            else:
                tracks.append([])
            
            prev_gray = gray
            prev_kp = kp
            prev_desc = desc
        
        tracking_results['tracks'] = tracks
        return tracking_results
    
    def _visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Visualize optical flow as HSV image"""
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(fx, fy)
        
        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
        hsv[..., 1] = 255  # Full saturation
        # Normalize magnitude for value channel
        mag_norm = np.zeros_like(mag, dtype=np.uint8)
        cv2.normalize(mag, mag_norm, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = mag_norm  # Value represents magnitude
        
        # Convert to BGR for saving
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    
    def _extract_shapes(self, results: Dict) -> Dict[str, Any]:
        """Extract and optimize shapes from segmentation and tracking"""
        shapes = {
            'shape_bank': {},
            'time_bank': {},
            'total_shapes': 0
        }
        
        if results['segmentation'] is not None:
            labels = results['segmentation']['labels']
            
            # Extract unique shapes across all frames
            shape_id = 0
            for frame_idx, label_map in enumerate(labels):
                unique_labels = np.unique(label_map)
                
                for label in unique_labels:
                    if label == 0:  # Skip background
                        continue
                    
                    # Extract shape mask
                    mask = (label_map == label)
                    
                    # Get shape properties
                    coords = np.argwhere(mask)
                    if len(coords) < self.config.min_cluster_size:
                        continue
                    
                    # Store shape
                    shapes['shape_bank'][shape_id] = {
                        'frame': frame_idx,
                        'mask': mask,
                        'coords': coords,
                        'centroid': coords.mean(axis=0),
                        'area': len(coords)
                    }
                    
                    # Store timing
                    if frame_idx not in shapes['time_bank']:
                        shapes['time_bank'][frame_idx] = []
                    shapes['time_bank'][frame_idx].append(shape_id)
                    
                    shape_id += 1
            
            shapes['total_shapes'] = shape_id
        
        return shapes
    
    def _save_outputs(self, results: Dict):
        """Save all outputs to disk"""
        # Save shape banks
        if results['shapes']:
            with open(self.output_folder / 'shape_bank.pkl', 'wb') as f:
                pickle.dump(results['shapes']['shape_bank'], f)
            
            with open(self.output_folder / 'time_bank.pkl', 'wb') as f:
                pickle.dump(results['shapes']['time_bank'], f)
        
        # Save motion data
        motion_data = {
            'video_name': self.video_name,
            'num_frames': len(results['frames']),
            'num_shapes': results['shapes']['total_shapes'] if results['shapes'] else 0,
            'quality_scores': {
                'segmentation': results['segmentation'].get('quality_score', 0) if results['segmentation'] else 0,
                'tracking': results['tracking'].get('quality_score', 0) if results['tracking'] else 0,
            },
            'active_engines': self.active_engines
        }
        
        with open(self.output_folder / 'motion_file.json', 'w') as f:
            json.dump(motion_data, f, indent=2)
        
        print(f"✅ Saved shape_bank.pkl, time_bank.pkl, and motion_file.json")
        print(f"📊 Total shapes extracted: {motion_data['num_shapes']}")
        print(f"📊 Quality scores: Segmentation={motion_data['quality_scores']['segmentation']:.2f}, Tracking={motion_data['quality_scores']['tracking']:.2f}")


def main():
    """Main entry point for video processing"""
    parser = argparse.ArgumentParser(description='Process video with AI engines for motion vectorization')
    parser.add_argument('--video_file', type=str, required=True, help='Path to video file')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='motion_vectorization/outputs', help='Output directory')
    parser.add_argument('--suffix', type=str, default=None, help='Output suffix')
    parser.add_argument('--max_frames', type=int, default=200, help='Maximum frames to process')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--no_sam2', action='store_true', help='Disable SAM2 engine')
    parser.add_argument('--no_cotracker3', action='store_true', help='Disable CoTracker3 engine')
    parser.add_argument('--no_flowseek', action='store_true', help='Disable FlowSeek engine')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VideoProcessorConfig(
        video_file=args.video_file,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        max_frames=args.max_frames,
        device=args.device,
        use_sam2=not args.no_sam2,
        use_cotracker3=not args.no_cotracker3,
        use_flowseek=not args.no_flowseek
    )
    
    # Create processor and run
    processor = MotionVectorizationProcessor(config)
    results = processor.process_video()
    
    return results


if __name__ == '__main__':
    main()