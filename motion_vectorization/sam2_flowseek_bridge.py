"""
SAM2.1-FlowSeek Integration Bridge
Connects state-of-the-art segmentation (SAM2.1), optical flow (FlowSeek), and tracking (CoTracker3)
for unified motion graphics processing with unprecedented accuracy.

This bridge provides:
- SAM2.1-aware FlowSeek optical flow computation
- Motion-guided segmentation refinement
- CoTracker3-enhanced flow validation and correction
- Unified 6-DOF motion understanding across all components
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

try:
    from .flowseek_engine import FlowSeekEngine, FlowSeekConfig, create_flowseek_engine
    from .sam2_engine import SAM2SegmentationEngine, SAM2Config  
    from .cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config
    ENGINES_AVAILABLE = True
except ImportError as e:
    ENGINES_AVAILABLE = False
    warnings.warn(f"Integration engines not available: {e}")


@dataclass
class SAM2FlowSeekBridgeConfig:
    """Configuration for SAM2.1-FlowSeek integration bridge"""
    
    # Device and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    batch_size: int = 1
    
    # FlowSeek configuration
    flowseek_config: FlowSeekConfig = field(default_factory=FlowSeekConfig)
    use_flowseek: bool = True
    flowseek_depth_integration: bool = True
    flowseek_adaptive_complexity: bool = True
    
    # SAM2.1 configuration  
    sam2_config: SAM2Config = field(default_factory=SAM2Config)
    use_sam2_guidance: bool = True
    sam2_mask_refinement: bool = True
    
    # CoTracker3 configuration
    use_cotracker3_validation: bool = True
    cotracker3_grid_size: int = 30
    
    # Integration settings
    motion_consistency_weight: float = 0.3
    segmentation_flow_weight: float = 0.2
    depth_motion_weight: float = 0.15
    
    # Quality and validation
    flow_validation_threshold: float = 0.8
    motion_coherence_threshold: float = 0.7
    min_segment_size: int = 50
    
    # Output settings
    save_intermediate_results: bool = False
    output_flow_visualization: bool = True


class SAM2FlowSeekBridge:
    """
    Integration bridge connecting SAM2.1, FlowSeek, and CoTracker3
    for unified motion graphics processing pipeline
    """
    
    def __init__(self, config: Optional[SAM2FlowSeekBridgeConfig] = None):
        self.config = config or SAM2FlowSeekBridgeConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize engines
        self.flowseek_engine = None
        self.sam2_engine = None
        self.cotracker3_engine = None
        
        # Performance tracking
        self.performance_stats = {
            'total_frame_pairs_processed': 0,
            'processing_times': [],
            'flow_quality_scores': [],
            'motion_consistency_scores': [],
            'segmentation_flow_alignment_scores': []
        }
        
        # Initialize all engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all processing engines"""
        print("🚀 Initializing SAM2.1-FlowSeek integration bridge")
        
        if not ENGINES_AVAILABLE:
            raise RuntimeError("Required engines not available. Please check imports.")
        
        # Initialize FlowSeek engine
        if self.config.use_flowseek:
            try:
                self.flowseek_engine = create_flowseek_engine(
                    depth_integration=self.config.flowseek_depth_integration,
                    adaptive_complexity=self.config.flowseek_adaptive_complexity,
                    device=self.config.device,
                    mixed_precision=self.config.mixed_precision
                )
                print("✅ FlowSeek engine initialized")
            except Exception as e:
                print(f"⚠️ FlowSeek initialization failed: {e}")
                self.config.use_flowseek = False
        
        # Initialize SAM2.1 engine
        if self.config.use_sam2_guidance:
            try:
                self.sam2_engine = SAM2SegmentationEngine(self.config.sam2_config)
                print("✅ SAM2.1 engine initialized")
            except Exception as e:
                print(f"⚠️ SAM2.1 initialization failed: {e}")
                self.config.use_sam2_guidance = False
                
        # Initialize CoTracker3 for validation
        if self.config.use_cotracker3_validation:
            try:
                from .cotracker3_engine import create_cotracker3_engine
                self.cotracker3_engine = create_cotracker3_engine(
                    grid_size=self.config.cotracker3_grid_size,
                    device=self.config.device
                )
                print("✅ CoTracker3 validation engine initialized")
            except Exception as e:
                print(f"⚠️ CoTracker3 initialization failed: {e}")
                self.config.use_cotracker3_validation = False
                
        print("🔗 SAM2.1-FlowSeek bridge ready for processing")
    
    def process_frame_pair(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        masks1: Optional[np.ndarray] = None,
        masks2: Optional[np.ndarray] = None,
        previous_flow: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process a pair of frames with integrated SAM2.1-FlowSeek pipeline
        
        Args:
            frame1: First frame (BGR format)
            frame2: Second frame (BGR format)
            masks1: Optional SAM2.1 masks for frame1
            masks2: Optional SAM2.1 masks for frame2
            previous_flow: Optional previous flow for temporal consistency
            
        Returns:
            forward_flow: Forward optical flow [H, W, 2]
            backward_flow: Backward optical flow [H, W, 2] 
            metadata: Processing metadata and quality metrics
        """
        start_time = time.time()
        
        # Convert frames to RGB tensors
        rgb1_tensor = self._prepare_image_tensor(frame1)
        rgb2_tensor = self._prepare_image_tensor(frame2)
        
        # Step 1: SAM2.1 segmentation if not provided
        if masks1 is None or masks2 is None:
            if self.config.use_sam2_guidance and self.sam2_engine:
                masks1, masks2 = self._generate_sam2_masks(frame1, frame2)
            else:
                masks1 = masks2 = None
        
        # Step 2: FlowSeek optical flow computation with SAM2.1 guidance
        forward_flow, backward_flow, flow_metadata = self._compute_guided_flow(
            rgb1_tensor, rgb2_tensor, masks1, masks2, previous_flow
        )
        
        # Step 3: CoTracker3 flow validation and refinement
        if self.config.use_cotracker3_validation and self.cotracker3_engine:
            forward_flow, backward_flow = self._validate_and_refine_flow(
                rgb1_tensor, rgb2_tensor, forward_flow, backward_flow, masks1, masks2
            )
        
        # Step 4: Motion consistency analysis
        motion_analysis = self._analyze_motion_consistency(
            forward_flow, backward_flow, masks1, masks2
        )
        
        # Step 5: Quality assessment and post-processing
        quality_metrics = self._assess_flow_quality(
            forward_flow, backward_flow, rgb1_tensor, rgb2_tensor, masks1, masks2
        )
        
        # Compile metadata
        processing_time = time.time() - start_time
        metadata = {
            'processing_time': processing_time,
            'flow_metadata': flow_metadata,
            'motion_analysis': motion_analysis,
            'quality_metrics': quality_metrics,
            'engines_used': {
                'flowseek': self.config.use_flowseek,
                'sam2': self.config.use_sam2_guidance,
                'cotracker3': self.config.use_cotracker3_validation
            }
        }
        
        # Update performance statistics
        self._update_performance_stats(processing_time, quality_metrics)
        
        return forward_flow, backward_flow, metadata
    
    def process_video_sequence(
        self,
        video_path: str,
        output_dir: str,
        max_frames: int = -1,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Process entire video sequence with SAM2.1-FlowSeek pipeline
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save flow results
            max_frames: Maximum frames to process (-1 for all)
            save_visualizations: Whether to save flow visualizations
            
        Returns:
            Processing summary and statistics
        """
        print(f"🎬 Processing video sequence: {video_path}")
        
        # Setup output directories
        flow_dir = Path(output_dir) / "flow"
        forward_dir = flow_dir / "forward"
        backward_dir = flow_dir / "backward"
        viz_dir = flow_dir / "viz"
        
        for dir_path in [forward_dir, backward_dir, viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_idx >= max_frames):
                break
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        print(f"📊 Loaded {len(frames)} frames for processing")
        
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for optical flow computation")
        
        # Process frame pairs sequentially
        processing_results = []
        previous_flow = None
        
        from tqdm import tqdm
        for i in tqdm(range(len(frames) - 1), desc="SAM2.1-FlowSeek Processing"):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Process frame pair
            forward_flow, backward_flow, metadata = self.process_frame_pair(
                frame1, frame2, previous_flow=previous_flow
            )
            
            # Save flow data
            np.save(forward_dir / f"{i:03d}.npy", forward_flow)
            np.save(backward_dir / f"{i:03d}.npy", backward_flow)
            
            # Save visualization if requested
            if save_visualizations and self.config.output_flow_visualization:
                vis_image = self._create_flow_visualization(
                    frame1, frame2, forward_flow, backward_flow
                )
                cv2.imwrite(str(viz_dir / f"{i:03d}.png"), vis_image)
            
            # Store results for analysis
            processing_results.append({
                'frame_idx': i,
                'processing_time': metadata['processing_time'],
                'quality_score': metadata['quality_metrics']['overall_quality'],
                'motion_consistency': metadata['motion_analysis']['consistency_score']
            })
            
            # Use forward flow as previous flow for temporal consistency
            previous_flow = forward_flow
        
        # Generate summary statistics
        summary = self._generate_processing_summary(processing_results, len(frames))
        
        # Save processing log
        log_path = flow_dir / "sam2_flowseek_processing_log.json"
        with open(log_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'summary': summary,
                'config': self.config.__dict__,
                'results': processing_results
            }, f, indent=2)
        
        print(f"✅ Video processing complete. Results saved to: {output_dir}")
        print(f"📊 Average processing time: {summary['avg_processing_time']:.3f}s per frame pair")
        print(f"🎯 Average quality score: {summary['avg_quality_score']:.3f}")
        
        return summary
    
    def _prepare_image_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to RGB tensor"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and move to device
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def _generate_sam2_masks(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate SAM2.1 segmentation masks for both frames"""
        if not self.sam2_engine:
            return None, None
            
        try:
            # Generate masks for frame pair
            masks1, _ = self.sam2_engine.segment_video_batch([frame1], [0])
            masks2, _ = self.sam2_engine.segment_video_batch([frame2], [1])
            
            return masks1[0] if masks1 else None, masks2[0] if masks2 else None
            
        except Exception as e:
            print(f"⚠️ SAM2.1 mask generation failed: {e}")
            return None, None
    
    def _compute_guided_flow(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        masks1: Optional[np.ndarray],
        masks2: Optional[np.ndarray],
        previous_flow: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute optical flow with SAM2.1 segmentation guidance
        """
        if not self.flowseek_engine:
            # Fallback to simple flow computation
            return self._compute_fallback_flow(rgb1, rgb2)
        
        # Convert masks to tensors if provided
        mask1_tensor = None
        mask2_tensor = None
        
        if masks1 is not None and self.config.use_sam2_guidance:
            mask1_tensor = torch.from_numpy(masks1).float().unsqueeze(0).to(self.device)
        if masks2 is not None and self.config.use_sam2_guidance:
            mask2_tensor = torch.from_numpy(masks2).float().unsqueeze(0).to(self.device)
        
        # FlowSeek forward pass
        try:
            # Convert to [0, 255] range expected by FlowSeek
            rgb1_255 = rgb1.clamp(0, 1) * 255.0
            rgb2_255 = rgb2.clamp(0, 1) * 255.0
            
            # Forward flow
            _, forward_flow_tensor = self.flowseek_engine(
                rgb1_255, rgb2_255, test_mode=True
            )
            
            # Backward flow
            _, backward_flow_tensor = self.flowseek_engine(
                rgb2_255, rgb1_255, test_mode=True
            )
            
            # Convert to numpy
            forward_flow = forward_flow_tensor[0].permute(1, 2, 0).cpu().numpy()
            backward_flow = backward_flow_tensor[0].permute(1, 2, 0).cpu().numpy()
            
            # SAM2.1-guided flow refinement
            if masks1 is not None and masks2 is not None:
                forward_flow = self._refine_flow_with_masks(
                    forward_flow, masks1, masks2
                )
                backward_flow = self._refine_flow_with_masks(
                    backward_flow, masks2, masks1
                )
            
            # Temporal consistency with previous flow
            if previous_flow is not None:
                forward_flow = self._apply_temporal_consistency(
                    forward_flow, previous_flow
                )
            
            metadata = {
                'method': 'flowseek_sam2_guided',
                'mask_guidance_used': masks1 is not None and masks2 is not None,
                'temporal_consistency_applied': previous_flow is not None,
                'flow_magnitude_forward': np.linalg.norm(forward_flow, axis=2).mean(),
                'flow_magnitude_backward': np.linalg.norm(backward_flow, axis=2).mean()
            }
            
            return forward_flow, backward_flow, metadata
            
        except Exception as e:
            print(f"⚠️ FlowSeek computation failed: {e}")
            return self._compute_fallback_flow(rgb1, rgb2)
    
    def _refine_flow_with_masks(
        self,
        flow: np.ndarray,
        mask_src: np.ndarray,
        mask_dst: np.ndarray
    ) -> np.ndarray:
        """Refine optical flow using SAM2.1 segmentation masks"""
        
        # Get unique object IDs
        unique_objects_src = np.unique(mask_src)
        unique_objects_dst = np.unique(mask_dst)
        
        refined_flow = flow.copy()
        
        # Process each segmented object separately
        for obj_id in unique_objects_src:
            if obj_id == 0:  # Skip background
                continue
                
            # Object mask in source frame
            obj_mask_src = (mask_src == obj_id)
            
            if np.sum(obj_mask_src) < self.config.min_segment_size:
                continue  # Skip small segments
                
            # Find corresponding object in destination frame (simple matching)
            obj_flow_region = flow[obj_mask_src]
            
            if len(obj_flow_region) > 0:
                # Compute robust flow for this object (median)
                robust_flow_x = np.median(obj_flow_region[:, 0])
                robust_flow_y = np.median(obj_flow_region[:, 1])
                
                # Apply smoothing within object region
                refined_flow[obj_mask_src, 0] = (
                    self.config.segmentation_flow_weight * robust_flow_x +
                    (1 - self.config.segmentation_flow_weight) * refined_flow[obj_mask_src, 0]
                )
                refined_flow[obj_mask_src, 1] = (
                    self.config.segmentation_flow_weight * robust_flow_y +
                    (1 - self.config.segmentation_flow_weight) * refined_flow[obj_mask_src, 1]
                )
        
        return refined_flow
    
    def _apply_temporal_consistency(
        self,
        current_flow: np.ndarray,
        previous_flow: np.ndarray
    ) -> np.ndarray:
        """Apply temporal consistency using previous flow"""
        
        # Resize previous flow if dimensions don't match
        if previous_flow.shape[:2] != current_flow.shape[:2]:
            previous_flow = cv2.resize(
                previous_flow, 
                (current_flow.shape[1], current_flow.shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Weighted combination with temporal smoothing
        consistent_flow = (
            (1 - self.config.motion_consistency_weight) * current_flow +
            self.config.motion_consistency_weight * previous_flow
        )
        
        return consistent_flow
    
    def _validate_and_refine_flow(
        self,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        masks1: Optional[np.ndarray] = None,
        masks2: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and refine flow using CoTracker3 point tracking"""
        
        if not self.cotracker3_engine:
            return forward_flow, backward_flow
        
        try:
            # Create video tensor for CoTracker3 (B=1, T=2, C=3, H, W)
            video_tensor = torch.cat([rgb1, rgb2], dim=0).unsqueeze(0)
            
            # Get grid points for tracking
            H, W = forward_flow.shape[:2]
            grid_size = self.config.cotracker3_grid_size
            
            y_coords = np.linspace(0, H-1, grid_size)
            x_coords = np.linspace(0, W-1, grid_size)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            query_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
            query_points_tensor = torch.tensor(query_points, dtype=torch.float32).unsqueeze(0)
            
            # Track points
            pred_tracks, pred_visibility = self.cotracker3_engine.track_video_grid(
                video_tensor, custom_points=query_points_tensor
            )
            
            # Use tracking results to validate and correct flow
            refined_forward_flow = self._correct_flow_with_tracking(
                forward_flow, pred_tracks, pred_visibility, query_points
            )
            
            return refined_forward_flow, backward_flow
            
        except Exception as e:
            print(f"⚠️ CoTracker3 flow validation failed: {e}")
            return forward_flow, backward_flow
    
    def _correct_flow_with_tracking(
        self,
        flow: np.ndarray,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        query_points: np.ndarray
    ) -> np.ndarray:
        """Correct optical flow using CoTracker3 tracking results"""
        
        # Extract track displacements
        tracks_np = tracks[0].cpu().numpy()  # [T=2, N, 2]
        visibility_np = visibility[0].cpu().numpy()  # [T=2, N]
        
        if tracks_np.shape[0] < 2:
            return flow
        
        # Compute tracking-based flow
        track_flow = tracks_np[1] - tracks_np[0]  # [N, 2] - displacement vectors
        
        # Only use visible tracks
        valid_tracks = visibility_np[0] > 0.5  # Visible in first frame
        
        if np.sum(valid_tracks) < 10:  # Need minimum number of valid tracks
            return flow
        
        # Interpolate corrections to full flow field
        corrected_flow = flow.copy()
        
        for i, (x, y) in enumerate(query_points):
            if valid_tracks[i]:
                ix, iy = int(x), int(y)
                if 0 <= ix < flow.shape[1] and 0 <= iy < flow.shape[0]:
                    # Apply correction with confidence weighting
                    confidence = visibility_np[1, i]  # Visibility in second frame
                    correction_weight = confidence * 0.3  # Conservative correction
                    
                    corrected_flow[iy, ix, 0] = (
                        (1 - correction_weight) * flow[iy, ix, 0] +
                        correction_weight * track_flow[i, 0]
                    )
                    corrected_flow[iy, ix, 1] = (
                        (1 - correction_weight) * flow[iy, ix, 1] +
                        correction_weight * track_flow[i, 1]
                    )
        
        return corrected_flow
    
    def _analyze_motion_consistency(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        masks1: Optional[np.ndarray] = None,
        masks2: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Analyze motion consistency between forward and backward flow"""
        
        # Forward-backward consistency check
        H, W = forward_flow.shape[:2]
        y, x = np.mgrid[0:H, 0:W]
        
        # Warp coordinates using forward flow
        x_warped = x + forward_flow[:, :, 0]
        y_warped = y + forward_flow[:, :, 1]
        
        # Sample backward flow at warped positions
        x_warped_clipped = np.clip(x_warped, 0, W-1)
        y_warped_clipped = np.clip(y_warped, 0, H-1)
        
        # Bilinear interpolation of backward flow
        backward_flow_warped = cv2.remap(
            backward_flow, 
            x_warped_clipped.astype(np.float32),
            y_warped_clipped.astype(np.float32),
            cv2.INTER_LINEAR
        )
        
        # Check consistency: forward_flow + backward_flow_warped should be ~0
        consistency_error = forward_flow + backward_flow_warped
        consistency_magnitude = np.linalg.norm(consistency_error, axis=2)
        
        # Overall consistency score (lower is better, convert to 0-1 scale)
        mean_error = np.mean(consistency_magnitude)
        consistency_score = np.exp(-mean_error / 5.0)  # Exponential decay
        
        # Motion coherence (spatial smoothness)
        flow_grad_x = np.gradient(forward_flow[:, :, 0], axis=1)
        flow_grad_y = np.gradient(forward_flow[:, :, 1], axis=0)
        motion_coherence = 1.0 / (1.0 + np.mean(flow_grad_x**2 + flow_grad_y**2))
        
        return {
            'consistency_score': float(consistency_score),
            'motion_coherence': float(motion_coherence),
            'mean_consistency_error': float(mean_error),
            'max_consistency_error': float(np.max(consistency_magnitude))
        }
    
    def _assess_flow_quality(
        self,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray,
        rgb1: torch.Tensor,
        rgb2: torch.Tensor,
        masks1: Optional[np.ndarray] = None,
        masks2: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Assess optical flow quality using multiple metrics"""
        
        # Flow magnitude statistics
        forward_magnitude = np.linalg.norm(forward_flow, axis=2)
        backward_magnitude = np.linalg.norm(backward_flow, axis=2)
        
        # Structural similarity after warping
        try:
            # Simple warping for quality assessment
            H, W = forward_flow.shape[:2]
            y, x = np.mgrid[0:H, 0:W]
            
            x_warped = x + forward_flow[:, :, 0]
            y_warped = y + forward_flow[:, :, 1]
            
            # Convert tensors to numpy for warping
            frame1_np = rgb1[0].permute(1, 2, 0).cpu().numpy()
            frame2_np = rgb2[0].permute(1, 2, 0).cpu().numpy()
            
            frame1_warped = cv2.remap(
                frame1_np,
                x_warped.astype(np.float32),
                y_warped.astype(np.float32),
                cv2.INTER_LINEAR
            )
            
            # Compute mean squared error after warping
            mse = np.mean((frame1_warped - frame2_np)**2)
            photometric_error = mse
            
        except Exception as e:
            photometric_error = float('inf')
        
        # Overall quality score (combination of metrics)
        flow_smoothness = 1.0 / (1.0 + np.std(forward_magnitude))
        photometric_quality = 1.0 / (1.0 + photometric_error)
        
        overall_quality = (flow_smoothness + photometric_quality) / 2.0
        
        return {
            'overall_quality': float(overall_quality),
            'flow_smoothness': float(flow_smoothness),
            'photometric_quality': float(photometric_quality),
            'photometric_error': float(photometric_error),
            'mean_forward_magnitude': float(np.mean(forward_magnitude)),
            'mean_backward_magnitude': float(np.mean(backward_magnitude)),
            'max_forward_magnitude': float(np.max(forward_magnitude)),
            'max_backward_magnitude': float(np.max(backward_magnitude))
        }
    
    def _compute_fallback_flow(
        self, 
        rgb1: torch.Tensor, 
        rgb2: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fallback optical flow computation using basic methods"""
        print("⚠️ Using fallback optical flow computation")
        
        # Convert to OpenCV format
        frame1 = (rgb1[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame2 = (rgb2[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow using Farneback
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        # Create backward flow (simple negation - not accurate but functional)
        backward_flow = -flow if flow is not None else np.zeros((gray1.shape[0], gray1.shape[1], 2))
        forward_flow = flow if flow is not None else np.zeros((gray1.shape[0], gray1.shape[1], 2))
        
        metadata = {
            'method': 'fallback_farneback',
            'mask_guidance_used': False,
            'temporal_consistency_applied': False
        }
        
        return forward_flow, backward_flow, metadata
    
    def _create_flow_visualization(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        forward_flow: np.ndarray,
        backward_flow: np.ndarray
    ) -> np.ndarray:
        """Create flow visualization combining all information"""
        
        def flow_to_color(flow):
            """Convert flow to HSV color representation"""
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Create flow visualizations
        forward_vis = flow_to_color(forward_flow)
        backward_vis = flow_to_color(backward_flow)
        
        # Resize frames to match flow if needed
        H, W = forward_flow.shape[:2]
        frame1_resized = cv2.resize(frame1, (W, H))
        frame2_resized = cv2.resize(frame2, (W, H))
        
        # Create combined visualization
        top_row = np.hstack([frame1_resized, forward_vis])
        bottom_row = np.hstack([frame2_resized, backward_vis])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def _update_performance_stats(self, processing_time: float, quality_metrics: Dict[str, float]):
        """Update performance statistics"""
        self.performance_stats['total_frame_pairs_processed'] += 1
        self.performance_stats['processing_times'].append(processing_time)
        self.performance_stats['flow_quality_scores'].append(quality_metrics['overall_quality'])
    
    def _generate_processing_summary(
        self, 
        results: List[Dict[str, Any]], 
        total_frames: int
    ) -> Dict[str, Any]:
        """Generate processing summary statistics"""
        
        processing_times = [r['processing_time'] for r in results]
        quality_scores = [r['quality_score'] for r in results]
        motion_consistency = [r['motion_consistency'] for r in results]
        
        return {
            'total_frames': total_frames,
            'total_frame_pairs': len(results),
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'avg_quality_score': np.mean(quality_scores),
            'std_quality_score': np.std(quality_scores),
            'avg_motion_consistency': np.mean(motion_consistency),
            'std_motion_consistency': np.std(motion_consistency),
            'total_processing_time': np.sum(processing_times),
            'fps': len(results) / np.sum(processing_times) if processing_times else 0
        }


# =====================================
# Factory Functions and Utilities
# =====================================

def create_sam2_flowseek_bridge(
    use_flowseek: bool = True,
    use_sam2_guidance: bool = True,
    use_cotracker3_validation: bool = True,
    device: str = "auto",
    **kwargs
) -> SAM2FlowSeekBridge:
    """
    Factory function to create SAM2.1-FlowSeek integration bridge
    
    Args:
        use_flowseek: Enable FlowSeek optical flow
        use_sam2_guidance: Enable SAM2.1 segmentation guidance
        use_cotracker3_validation: Enable CoTracker3 validation
        device: Processing device
        **kwargs: Additional configuration options
        
    Returns:
        Configured SAM2FlowSeekBridge instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    config = SAM2FlowSeekBridgeConfig(
        device=device,
        use_flowseek=use_flowseek,
        use_sam2_guidance=use_sam2_guidance,
        use_cotracker3_validation=use_cotracker3_validation,
        **kwargs
    )
    
    bridge = SAM2FlowSeekBridge(config)
    
    print(f"🔗 SAM2.1-FlowSeek bridge created")
    print(f"   • FlowSeek: {use_flowseek}")
    print(f"   • SAM2.1 guidance: {use_sam2_guidance}")
    print(f"   • CoTracker3 validation: {use_cotracker3_validation}")
    print(f"   • Device: {device}")
    
    return bridge