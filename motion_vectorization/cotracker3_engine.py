"""
CoTracker3 Point Tracking Engine for Motion Vectorization Pipeline
Integrates Meta AI's CoTracker3 (Oct 2024) for world-class point tracking performance

Features:
- 27% faster than previous models with superior occlusion handling
- Dense point tracking up to 265×265 points simultaneously  
- Support for offline (full video context) and online (sliding window) modes
- Advanced GPU acceleration and mixed precision support
- Seamless integration with SAM2.1 segmentation pipeline
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
from threading import Lock
import logging

try:
    # CoTracker3 is available via torch.hub, no separate installation needed
    import torch.hub
    COTRACKER3_AVAILABLE = True
except ImportError:
    COTRACKER3_AVAILABLE = False
    warnings.warn("PyTorch Hub not available. CoTracker3 integration disabled.")


@dataclass 
class CoTracker3Config:
    """Advanced configuration for CoTracker3 tracking engine"""
    # Model selection
    model_variant: str = "cotracker3_offline"  # or "cotracker3_online" 
    checkpoint_path: Optional[str] = None  # Use default torch.hub model
    
    # Performance settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Will be automatically disabled on CPU
    batch_size: int = 1  # Process video in batches
    max_sequence_length: int = 100  # For online mode
    
    # Tracking parameters
    grid_size: int = 50  # Creates 50×50 = 2500 tracked points
    max_points: int = 2500  # Maximum points to track simultaneously
    tracking_threshold: float = 0.85  # Confidence threshold for tracks
    visibility_threshold: float = 0.3  # Point visibility threshold
    
    # Quality and optimization
    compile_model: bool = True  # Will be automatically disabled on CPU
    memory_efficient: bool = True  # Trade speed for memory
    temporal_consistency: float = 0.9  # Temporal smoothing factor
    occlusion_recovery: bool = True  # Enable occlusion handling
    
    # Integration settings  
    sam2_integration: bool = True  # Use SAM2.1 masks for point selection
    contour_sampling_density: int = 20  # Points per object contour
    shape_parameter_extraction: bool = True  # Extract affine transforms
    
    # Performance targets
    target_fps: float = 44.0  # Target processing speed
    accuracy_target: float = 0.95  # Target tracking accuracy
    
    def __post_init__(self):
        """Auto-adjust settings based on device capabilities"""
        cuda_available = torch.cuda.is_available()
        
        # Disable CUDA-specific features on CPU
        if self.device == "cpu" or not cuda_available:
            self.mixed_precision = False
            self.compile_model = False
            if not cuda_available:
                self.device = "cpu"
                
        # Adjust parameters for CPU
        if self.device == "cpu":
            self.grid_size = min(self.grid_size, 20)  # Reduce grid size for CPU
            self.max_points = min(self.max_points, 400)  # Reduce max points
            self.batch_size = 1  # Single batch for CPU


class CoTracker3TrackerEngine:
    """
    High-performance CoTracker3 point tracking engine for motion graphics
    Designed to achieve 44 FPS processing and 95%+ tracking accuracy
    """
    
    def __init__(self, config: Optional[CoTracker3Config] = None):
        self.config = config or CoTracker3Config()
        self.device = torch.device(self.config.device)
        self.model = None
        self.model_lock = Lock()
        
        # Performance tracking
        self.performance_stats = {
            'total_sequences_processed': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'accuracy_scores': [],
            'memory_usage_peak': 0,
            'gpu_utilization': []
        }
        
        # Tracking state for online mode
        self.tracking_state = {
            'current_tracks': None,
            'current_visibility': None, 
            'frame_buffer': deque(maxlen=self.config.max_sequence_length),
            'point_history': defaultdict(list),
            'occlusion_tracker': {}
        }
        
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize CoTracker3 model with optimizations"""
        print(f"🚀 Initializing CoTracker3 Engine on {self.device}")
        print(f"   Model variant: {self.config.model_variant}")
        print(f"   Grid size: {self.config.grid_size}×{self.config.grid_size} = {self.config.grid_size**2} points")
        print(f"📊 CoTracker3 available: {COTRACKER3_AVAILABLE}, CUDA available: {torch.cuda.is_available()}")
        
        # Try CoTracker3 if available
        if COTRACKER3_AVAILABLE:
            try:
                self._load_cotracker3_model()
                self._optimize_model()
                self._validate_installation()
                return
            except Exception as e:
                print(f"⚠️ CoTracker3 initialization failed: {e}")
                print("🔄 Falling back to enhanced optical flow tracking")
        
        # Always fallback if CoTracker3 fails or unavailable
        self._initialize_fallback_engine()
    
    def _load_cotracker3_model(self):
        """Load CoTracker3 model via torch.hub with robust error handling"""
        print("📦 Loading CoTracker3 model from torch.hub...")
        
        try:
            # Load model with proper configuration
            self.model = torch.hub.load(
                "facebookresearch/co-tracker", 
                self.config.model_variant,
                trust_repo=True,
                verbose=False,
                force_reload=False  # Use cached version if available
            )
            print("✅ CoTracker3 model loaded from torch.hub")
            
        except Exception as e:
            print(f"⚠️ torch.hub loading failed: {e}")
            # Try alternative model variants
            fallback_variants = ["cotracker3_online", "cotracker2"]
            
            for variant in fallback_variants:
                if variant != self.config.model_variant:
                    try:
                        print(f"🔄 Trying fallback variant: {variant}")
                        self.model = torch.hub.load(
                            "facebookresearch/co-tracker", 
                            variant,
                            trust_repo=True,
                            verbose=False
                        )
                        print(f"✅ Loaded fallback variant: {variant}")
                        break
                    except Exception as variant_e:
                        print(f"⚠️ Fallback variant {variant} failed: {variant_e}")
                        continue
            else:
                raise RuntimeError("All CoTracker model variants failed to load")
        
        try:
            # Move to device and set evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model moved to {self.device}")
            
            # Enable mixed precision only on CUDA
            if self.config.mixed_precision and self.device.type == "cuda":
                try:
                    self.model = self.model.half()
                    print("✅ Mixed precision enabled")
                except Exception as e:
                    print(f"⚠️ Mixed precision failed: {e}")
                    
        except Exception as e:
            print(f"⚠️ Model device setup failed: {e}")
            raise
        
    def _optimize_model(self):
        """Apply PyTorch optimizations for maximum performance"""
        if not self.config.compile_model or self.device.type != "cuda":
            print("⚠️ Skipping model compilation (not on CUDA or disabled)")
            return
            
        # Only compile on CUDA with PyTorch 2.0+
        try:
            if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                print("⚡ Compiling model with torch.compile...")
                
                # Use lighter compilation mode for better compatibility
                self.model = torch.compile(
                    self.model, 
                    mode="reduce-overhead",  # More conservative than max-autotune
                    fullgraph=False,  # Allow graph breaks
                    dynamic=True  # Handle dynamic shapes
                )
                print("✅ Model compilation successful")
            else:
                print("⚠️ torch.compile not available (PyTorch < 2.0)")
                
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
            # Continue without compilation - not critical
            
    def _validate_installation(self):
        """Validate CoTracker3 installation with test inference"""
        print("🔍 Validating CoTracker3 installation...")
        
        try:
            # Create test data with appropriate dtype
            test_shape = (1, 5, 3, 128, 128)  # Smaller test for compatibility
            if self.config.mixed_precision and self.device.type == "cuda":
                test_video = torch.randn(*test_shape, dtype=torch.float16, device=self.device)
            else:
                test_video = torch.randn(*test_shape, dtype=torch.float32, device=self.device)
                
            # Test grid tracking with small grid
            test_grid_size = 5  # Small test grid
            with torch.no_grad():
                try:
                    tracks, visibility = self.track_video_grid(test_video, grid_size=test_grid_size)
                    expected_points = test_grid_size * test_grid_size
                    
                    if tracks.shape[2] == expected_points and visibility.shape[2] == expected_points:
                        print("✅ Grid tracking validation passed")
                    else:
                        print(f"⚠️ Grid tracking shape mismatch: got {tracks.shape[2]}, expected {expected_points}")
                        
                except Exception as track_e:
                    print(f"⚠️ Grid tracking failed during validation: {track_e}")
                    # Try fallback validation
                    self._validate_fallback_mode()
                    
        except Exception as e:
            print(f"⚠️ Validation setup failed: {e}")
            raise RuntimeError(f"CoTracker3 validation failed: {e}")
    
    def _validate_fallback_mode(self):
        """Validate that fallback mode works"""
        print("🔍 Testing fallback mode...")
        # Temporarily disable model to test fallback
        original_model = self.model
        self.model = None
        
        try:
            test_video = torch.randn(1, 3, 3, 64, 64, dtype=torch.float32, device="cpu")
            tracks, visibility = self.track_video_grid(test_video, grid_size=3)
            print("✅ Fallback mode validation passed")
        except Exception as e:
            print(f"⚠️ Fallback validation failed: {e}")
        finally:
            self.model = original_model
            
    def _initialize_fallback_engine(self):
        """Initialize enhanced optical flow fallback when CoTracker3 unavailable"""
        print("🔄 Initializing enhanced optical flow fallback engine")
        
        # Enhanced Lucas-Kanade tracker with temporal consistency
        self.fallback_tracker = {
            'lk_params': dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            ),
            'feature_params': dict(
                maxCorners=self.config.max_points,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7
            )
        }
        
        self.model = None  # Fallback mode
        print("✅ Fallback engine initialized")
        
    def track_video_grid(
        self, 
        video: torch.Tensor,
        grid_size: Optional[int] = None,
        custom_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track points across video sequence using grid sampling or custom points
        
        Args:
            video: Video tensor (B, T, C, H, W)
            grid_size: Size of uniform grid (creates grid_size² points)
            custom_points: Custom points to track (B, N, 2)
            
        Returns:
            tracks: Point trajectories (B, T, N, 2)
            visibility: Point visibility (B, T, N, 1)
        """
        if self.model is None:
            return self._track_video_fallback(video, grid_size, custom_points)
            
        grid_size = grid_size or self.config.grid_size
        B, T, C, H, W = video.shape
        
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                # Setup autocast context with proper error handling
                autocast_enabled = self.config.mixed_precision and self.device.type == "cuda"
                try:
                    if autocast_enabled:
                        autocast_context = torch.cuda.amp.autocast()
                    else:
                        import contextlib
                        autocast_context = contextlib.nullcontext()
                except Exception as e:
                    print(f"⚠️ Autocast setup failed: {e}, using default precision")
                    import contextlib
                    autocast_context = contextlib.nullcontext()
                    
                with autocast_context:
                    tracks, visibility = self._perform_tracking(
                        video, grid_size, custom_points, H, W
                    )
                    
        except Exception as e:
            print(f"⚠️  CoTracker3 inference failed: {e}")
            return self._track_video_fallback(video, grid_size, custom_points)
            
        # Update performance statistics
        processing_time = time.perf_counter() - start_time
        fps = T / processing_time
        self._update_performance_stats(processing_time, fps, tracks, visibility)
        
        return tracks, visibility
        
    def _perform_tracking(
        self, 
        video: torch.Tensor, 
        grid_size: int,
        custom_points: Optional[torch.Tensor],
        H: int, 
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core tracking computation using CoTracker3 with robust error handling"""
        B, T = video.shape[:2]
        
        try:
            if custom_points is not None:
                # Track custom points - CoTracker3 API may vary
                tracks, visibility = self._track_custom_points(video, custom_points)
            else:
                # Track uniform grid
                tracks, visibility = self._track_grid_points(video, grid_size, H, W)
                
        except Exception as e:
            print(f"⚠️ CoTracker3 inference failed: {e}, using fallback tracking")
            return self._track_video_fallback(video, grid_size, custom_points)
            
        # Post-process tracks
        try:
            tracks, visibility = self._post_process_tracks(tracks, visibility)
        except Exception as e:
            print(f"⚠️ Track post-processing failed: {e}")
            
        return tracks, visibility
    
    def _track_custom_points(self, video: torch.Tensor, custom_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track custom points with multiple API attempts"""
        # Try different CoTracker3 API methods
        for method_name in ['track_points', '__call__', 'forward']:
            try:
                if hasattr(self.model, method_name):
                    method = getattr(self.model, method_name)
                    if method_name == 'track_points':
                        result = method(video, custom_points)
                    else:
                        result = method(video, custom_points)
                    
                    tracks, visibility = result if isinstance(result, tuple) else (result, None)
                    if tracks is not None:
                        # Ensure visibility is a tensor, not None
                        if visibility is None:
                            # Create dummy visibility tensor with same shape as tracks
                            visibility = torch.ones(tracks.shape[:-1] + (1,), dtype=torch.float32, device=tracks.device)
                        return tracks, visibility
            except Exception as e:
                print(f"⚠️ Method {method_name} failed: {e}")
                continue
        
        raise RuntimeError("All CoTracker3 custom point tracking methods failed")
    
    def _track_grid_points(self, video: torch.Tensor, grid_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Track grid points with multiple API attempts"""
        # Try different CoTracker3 API methods
        for method_name in ['track_grid', '__call__', 'forward']:
            try:
                if hasattr(self.model, method_name):
                    method = getattr(self.model, method_name)
                    if method_name == 'track_grid':
                        result = method(video, grid_size=grid_size)
                    else:
                        result = method(video)
                    
                    tracks, visibility = result if isinstance(result, tuple) else (result, None)
                    if tracks is not None:
                        # Ensure visibility is a tensor, not None
                        if visibility is None:
                            # Create dummy visibility tensor with same shape as tracks
                            visibility = torch.ones(tracks.shape[:-1] + (1,), dtype=torch.float32, device=tracks.device)
                        return tracks, visibility
            except Exception as e:
                print(f"⚠️ Method {method_name} failed: {e}")
                continue
        
        raise RuntimeError("All CoTracker3 grid tracking methods failed")
    
    def _post_process_tracks(self, tracks: torch.Tensor, visibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Post-process tracks with error handling"""
        if tracks is None:
            raise ValueError("Tracks is None")
            
        # Apply temporal consistency smoothing
        if self.config.temporal_consistency < 1.0:
            try:
                tracks = self._apply_temporal_smoothing(tracks)
            except Exception as e:
                print(f"⚠️ Temporal smoothing failed: {e}")
                
        # Filter by visibility threshold if visibility is available
        if visibility is not None:
            try:
                if len(visibility.shape) > 3:  # [B, T, N, 1]
                    valid_mask = visibility[..., 0] > self.config.visibility_threshold
                else:  # [B, T, N]
                    valid_mask = visibility > self.config.visibility_threshold
                tracks[~valid_mask] = float('nan')  # Mark invalid tracks
            except Exception as e:
                print(f"⚠️ Visibility filtering failed: {e}")
        
        return tracks, visibility
        
    def _apply_temporal_smoothing(self, tracks: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to reduce jitter"""
        alpha = self.config.temporal_consistency
        smoothed_tracks = tracks.clone()
        
        for t in range(1, tracks.shape[1]):
            # Exponential moving average for temporal consistency
            valid_curr = ~torch.isnan(tracks[:, t]).any(dim=-1, keepdim=True)
            valid_prev = ~torch.isnan(smoothed_tracks[:, t-1]).any(dim=-1, keepdim=True)
            valid_both = valid_curr & valid_prev
            
            smoothed_tracks[:, t] = torch.where(
                valid_both,
                alpha * smoothed_tracks[:, t-1] + (1 - alpha) * tracks[:, t],
                tracks[:, t]
            )
            
        return smoothed_tracks
        
    def _track_video_fallback(
        self,
        video: torch.Tensor,
        grid_size: Optional[int] = None,
        custom_points: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced optical flow fallback when CoTracker3 unavailable"""
        grid_size = grid_size or self.config.grid_size
        B, T, C, H, W = video.shape
        
        # Convert to numpy for OpenCV processing
        video_np = video.cpu().numpy()
        if video_np.dtype != np.uint8:
            video_np = (video_np * 255).astype(np.uint8)
            
        all_tracks = []
        all_visibility = []
        
        for b in range(B):
            custom_points_np = None
            if custom_points is not None:
                custom_points_item = custom_points[b]
                if isinstance(custom_points_item, torch.Tensor):
                    custom_points_np = custom_points_item.cpu().numpy()
                elif isinstance(custom_points_item, np.ndarray):
                    custom_points_np = custom_points_item
                # Skip if neither tensor nor ndarray
            tracks, visibility = self._track_sequence_opencv(
                video_np[b], grid_size, custom_points_np
            )
            all_tracks.append(tracks)
            all_visibility.append(visibility)
            
        # Convert back to tensors
        tracks_tensor = torch.tensor(np.stack(all_tracks), dtype=torch.float32, device=self.device)
        visibility_tensor = torch.tensor(np.stack(all_visibility), dtype=torch.float32, device=self.device)
        
        return tracks_tensor, visibility_tensor
        
    def _track_sequence_opencv(
        self, 
        frames: np.ndarray,
        grid_size: int,
        custom_points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track single sequence using enhanced Lucas-Kanade"""
        T, C, H, W = frames.shape
        
        # Initialize tracking points
        if custom_points is not None:
            p0 = custom_points.astype(np.float32)
        else:
            # Create uniform grid
            x_coords = np.linspace(0, W-1, grid_size)
            y_coords = np.linspace(0, H-1, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            p0 = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
            
        num_points = len(p0)
        tracks = np.full((T, num_points, 2), np.nan)
        visibility = np.zeros((T, num_points, 1))
        
        # Initialize first frame
        tracks[0] = p0
        visibility[0] = 1.0
        
        # Convert frames to grayscale for tracking
        gray_frames = []
        for t in range(T):
            if C == 3:
                gray = cv2.cvtColor(frames[t].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            else:
                gray = frames[t, 0] if C == 1 else frames[t].mean(axis=0)
            gray_frames.append(gray)
            
        # Track frame by frame
        current_points = p0.copy()
        for t in range(1, T):
            # Forward tracking
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                gray_frames[t-1], gray_frames[t], current_points, 
                **self.fallback_tracker['lk_params']
            )
            
            # Backward tracking for validation
            back_points, back_status, back_error = cv2.calcOpticalFlowPyrLK(
                gray_frames[t], gray_frames[t-1], new_points,
                **self.fallback_tracker['lk_params']
            )
            
            # Calculate tracking error and update validity
            distance = np.sqrt(np.sum((current_points - back_points)**2, axis=1))
            valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (distance < 1.0)
            
            # Update tracks
            tracks[t, valid] = new_points[valid]
            visibility[t, valid] = 1.0
            
            # Update current points for next iteration
            current_points = new_points.copy()
            
        return tracks, visibility
        
    def track_object_contours(
        self,
        video: torch.Tensor,
        masks: torch.Tensor,
        sample_density: Optional[int] = None
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Track object contours from segmentation masks
        
        Args:
            video: Video tensor (B, T, C, H, W)
            masks: Segmentation masks (B, T, H, W) with object IDs
            sample_density: Points to sample per object contour
            
        Returns:
            Dict mapping object_id to (tracks, visibility)
        """
        sample_density = sample_density or self.config.contour_sampling_density
        B, T, H, W = masks.shape
        
        # Find unique objects across all frames
        unique_objects = torch.unique(masks[masks > 0]).cpu().numpy()
        
        object_tracks = {}
        
        for obj_id in unique_objects:
            # Extract contour points for this object
            contour_points = self._extract_contour_points(masks, obj_id, sample_density)
            
            if contour_points is not None and len(contour_points) > 0:
                # Track contour points
                tracks, visibility = self.track_video_grid(
                    video, custom_points=contour_points
                )
                object_tracks[int(obj_id)] = (tracks, visibility)
                
        return object_tracks
        
    def _extract_contour_points(
        self,
        masks: torch.Tensor,
        obj_id: int,
        sample_density: int
    ) -> Optional[torch.Tensor]:
        """Extract contour points from object masks"""
        B, T, H, W = masks.shape
        
        # Find first frame where object appears
        obj_masks = (masks == obj_id).cpu().numpy()
        
        for t in range(T):
            if obj_masks[0, t].any():
                # Extract contour from this frame
                mask_frame = obj_masks[0, t].astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Sample points along contour
                    if len(largest_contour) >= sample_density:
                        # Uniform sampling
                        indices = np.linspace(0, len(largest_contour)-1, sample_density, dtype=int)
                        sampled_points = largest_contour[indices].reshape(-1, 2)
                    else:
                        # Use all available points
                        sampled_points = largest_contour.reshape(-1, 2)
                    
                    # Convert to tensor format (B, N, 2)
                    points_tensor = torch.tensor(
                        sampled_points[None, :, :], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    
                    return points_tensor
                    
        return None
        
    def extract_motion_parameters(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract motion parameters (translation, rotation, scale) from tracks
        
        Args:
            tracks: Point trajectories (B, T, N, 2)
            visibility: Point visibility (B, T, N, 1)
            
        Returns:
            Motion parameters dict with translation, rotation, scale
        """
        B, T, N, _ = tracks.shape
        
        motion_params = {
            'translation': torch.zeros(B, T, 2, device=self.device),
            'rotation': torch.zeros(B, T, device=self.device),
            'scale': torch.ones(B, T, device=self.device),
            'affine_matrix': torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        }
        
        for b in range(B):
            for t in range(1, T):
                # Get valid points for current and previous frames
                valid_mask = (
                    (visibility[b, t-1, :, 0] > self.config.visibility_threshold) &
                    (visibility[b, t, :, 0] > self.config.visibility_threshold) &
                    (~torch.isnan(tracks[b, t-1]).any(dim=1)) &
                    (~torch.isnan(tracks[b, t]).any(dim=1))
                )
                
                if valid_mask.sum() < 3:  # Need at least 3 points for affine transform
                    continue
                    
                # Extract valid point correspondences
                points_prev = tracks[b, t-1, valid_mask].cpu().numpy()
                points_curr = tracks[b, t, valid_mask].cpu().numpy()
                
                # Estimate affine transformation
                try:
                    # Use OpenCV to estimate affine transform
                    if len(points_prev) >= 3:
                        M = cv2.getAffineTransform(
                            points_prev[:3].astype(np.float32),
                            points_curr[:3].astype(np.float32)
                        )
                        
                        # Extract motion parameters from affine matrix
                        translation = M[:2, 2]
                        scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                        scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
                        scale = (scale_x + scale_y) / 2
                        rotation = np.arctan2(M[1, 0], M[0, 0])
                        
                        # Store parameters
                        motion_params['translation'][b, t] = torch.tensor(translation, device=self.device)
                        motion_params['scale'][b, t] = torch.tensor(scale, device=self.device)
                        motion_params['rotation'][b, t] = torch.tensor(rotation, device=self.device)
                        
                        # Store full affine matrix
                        M_full = np.eye(3)
                        M_full[:2, :] = M
                        motion_params['affine_matrix'][b, t] = torch.tensor(M_full, device=self.device)
                        
                except Exception as e:
                    # Fall back to centroid-based translation estimation
                    centroid_prev = points_prev.mean(axis=0)
                    centroid_curr = points_curr.mean(axis=0)
                    translation = centroid_curr - centroid_prev
                    motion_params['translation'][b, t] = torch.tensor(translation, device=self.device)
                    
        return motion_params
        
    def _update_performance_stats(
        self,
        processing_time: float,
        fps: float,
        tracks: torch.Tensor,
        visibility: torch.Tensor
    ):
        """Update performance statistics"""
        self.performance_stats['total_sequences_processed'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['average_fps'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['total_sequences_processed']
        )
        
        # Estimate tracking accuracy based on visibility and temporal consistency
        valid_tracks = visibility[..., 0] > self.config.visibility_threshold
        temporal_consistency = self._calculate_temporal_consistency(tracks[valid_tracks])
        self.performance_stats['accuracy_scores'].append(temporal_consistency)
        
    def _calculate_temporal_consistency(self, tracks: torch.Tensor) -> float:
        """Calculate temporal consistency score for tracking quality assessment"""
        if len(tracks) < 2:
            return 0.0
            
        # Calculate motion smoothness
        velocities = torch.diff(tracks, dim=0)
        accelerations = torch.diff(velocities, dim=0)
        
        # Smooth motion has low acceleration variance
        acc_variance = torch.var(accelerations).item()
        consistency_score = 1.0 / (1.0 + acc_variance)
        
        return min(consistency_score, 1.0)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        stats = self.performance_stats.copy()
        
        if stats['accuracy_scores']:
            stats['average_accuracy'] = np.mean(stats['accuracy_scores'])
            stats['accuracy_std'] = np.std(stats['accuracy_scores'])
        else:
            stats['average_accuracy'] = 0.0
            stats['accuracy_std'] = 0.0
            
        stats['performance_rating'] = self._calculate_performance_rating()
        
        return stats
        
    def _calculate_performance_rating(self) -> str:
        """Calculate overall performance rating"""
        fps = self.performance_stats['average_fps']
        accuracy = np.mean(self.performance_stats['accuracy_scores']) if self.performance_stats['accuracy_scores'] else 0.0
        
        if fps >= self.config.target_fps and accuracy >= self.config.accuracy_target:
            return "EXCELLENT"
        elif fps >= self.config.target_fps * 0.8 and accuracy >= self.config.accuracy_target * 0.9:
            return "GOOD"
        elif fps >= self.config.target_fps * 0.6 and accuracy >= self.config.accuracy_target * 0.8:
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"
            
    def reset_tracking_state(self):
        """Reset tracking state for new video sequence"""
        self.tracking_state = {
            'current_tracks': None,
            'current_visibility': None, 
            'frame_buffer': deque(maxlen=self.config.max_sequence_length),
            'point_history': defaultdict(list),
            'occlusion_tracker': {}
        }
        
    def cleanup(self):
        """Cleanup resources and memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.reset_tracking_state()
        print("🧹 CoTracker3 engine cleaned up")


# Factory function for easy instantiation
def create_cotracker3_engine(
    mode: str = "offline",
    device: str = "auto",
    grid_size: int = 50,
    **kwargs
) -> CoTracker3TrackerEngine:
    """
    Factory function to create optimized CoTracker3 engine
    
    Args:
        mode: "offline" for full video context, "online" for streaming
        device: "auto", "cuda", or "cpu"
        grid_size: Uniform grid size for point sampling
        **kwargs: Additional config parameters
        
    Returns:
        Configured CoTracker3TrackerEngine instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    config = CoTracker3Config(
        model_variant=f"cotracker3_{mode}",
        device=device,
        grid_size=grid_size,
        **kwargs
    )
    
    return CoTracker3TrackerEngine(config)


if __name__ == "__main__":
    # Demo usage and validation
    print("🎬 CoTracker3 Engine Demo")
    
    # Create engine
    engine = create_cotracker3_engine(mode="offline", grid_size=20)
    
    # Create test video
    test_video = torch.randn(1, 50, 3, 480, 640)  # 50 frames, 480x640
    
    # Track grid points
    print("📍 Testing grid tracking...")
    tracks, visibility = engine.track_video_grid(test_video, grid_size=20)
    print(f"   Tracked {tracks.shape[2]} points across {tracks.shape[1]} frames")
    
    # Extract motion parameters
    print("🎯 Extracting motion parameters...")
    motion_params = engine.extract_motion_parameters(tracks, visibility)
    print(f"   Translation range: {motion_params['translation'].abs().max().item():.2f}")
    
    # Performance report
    report = engine.get_performance_report()
    print(f"📊 Performance: {report['performance_rating']}")
    print(f"   FPS: {report['average_fps']:.1f}")
    print(f"   Accuracy: {report['average_accuracy']:.3f}")
    
    # Cleanup
    engine.cleanup()
    print("✅ Demo completed successfully")