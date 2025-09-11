import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from .utils import warp_flo

# Configure logging for better error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CoTracker3 components with proper type fallbacks
try:
    from .cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config, create_cotracker3_engine
    from .sam2_cotracker_bridge import SAM2CoTrackerBridge, create_sam2_cotracker_bridge
    COTRACKER3_AVAILABLE = True
except ImportError:
    # Define fallback types and functions for when CoTracker3 is not available
    CoTracker3TrackerEngine = None
    CoTracker3Config = None
    SAM2CoTrackerBridge = None
    
    def create_cotracker3_engine(mode="offline", device="auto", grid_size=50, **kwargs):
        """Fallback function that raises ImportError"""
        raise ImportError("CoTracker3 engine not available")
    
    def create_sam2_cotracker_bridge(sam2_accuracy="high", cotracker_mode="offline", contour_density=30, device="auto", **kwargs):
        """Fallback function that raises ImportError"""
        raise ImportError("SAM2-CoTracker3 bridge not available")
    
    COTRACKER3_AVAILABLE = False
    warnings.warn("CoTracker3 integration not available. Using minimal fallback tracking.")


class Processor:
    """
    AI-Focused Motion Graphics Processor with CoTracker3 Integration
    
    This processor is designed around modern AI tracking methods, specifically
    Meta AI's CoTracker3 for superior point tracking and motion analysis.
    """
    
    def __init__(self, use_cotracker3=False, cotracker3_mode="offline", device="auto"):
        # CoTracker3 integration
        self.use_cotracker3 = use_cotracker3 and COTRACKER3_AVAILABLE
        self.cotracker3_engine = None
        self.sam2_cotracker_bridge = None
        
        if self.use_cotracker3:
            self._initialize_cotracker3(cotracker3_mode, device)
        else:
            if use_cotracker3 and not COTRACKER3_AVAILABLE:
                print("⚠️  CoTracker3 requested but not available. Using minimal fallback.")
    
    def _initialize_cotracker3(self, mode="offline", device="auto"):
        """Initialize CoTracker3 engine and bridge"""
        try:
            print(f"🚀 Initializing CoTracker3 integration (mode: {mode})")
            
            # Create CoTracker3 engine
            self.cotracker3_engine = create_cotracker3_engine(
                mode=mode,
                device=device,
                grid_size=40,  # Reasonable grid size for motion graphics
                mixed_precision=True,
                compile_model=True
            )
            
            # Create SAM2-CoTracker3 bridge  
            self.sam2_cotracker_bridge = create_sam2_cotracker_bridge(
                sam2_accuracy="high",
                cotracker_mode=mode,
                contour_density=25,
                device=device
            )
            
            print("✅ CoTracker3 integration initialized successfully")
            
        except Exception as e:
            print(f"⚠️  CoTracker3 initialization failed: {e}")
            self.use_cotracker3 = False
            self.cotracker3_engine = None
            self.sam2_cotracker_bridge = None

    # ================================
    # Core Infrastructure Methods (Keep)
    # ================================

    @staticmethod
    def warp_labels(curr_labels, prev_labels, forw_flo, back_flo):
        """
        Warp labels using optical flow with comprehensive validation.
        
        Args:
          curr_labels: Current frame labels (H, W)
          prev_labels: Previous frame labels (H, W)  
          forw_flo: Forward optical flow (H, W, 2)
          back_flo: Backward optical flow (H, W, 2)
          
        Returns:
          Tuple of (curr_warped_labels, prev_warped_labels)
          
        Raises:
          ValueError: If input arrays have invalid shapes or types
          TypeError: If inputs are not numpy arrays
        """
        # Validate inputs
        inputs = {
            'curr_labels': curr_labels,
            'prev_labels': prev_labels, 
            'forw_flo': forw_flo,
            'back_flo': back_flo
        }
        
        for name, array in inputs.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(array)}")
            if array.size == 0:
                raise ValueError(f"{name} is empty")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        
        # Validate label arrays
        for name, labels in [('curr_labels', curr_labels), ('prev_labels', prev_labels)]:
            if len(labels.shape) != 2:
                raise ValueError(f"{name} must be 2D (H, W), got shape {labels.shape}")
            if not np.issubdtype(labels.dtype, np.integer):
                warnings.warn(f"{name} should be integer type, got {labels.dtype}")
        
        # Validate flow arrays
        for name, flow in [('forw_flo', forw_flo), ('back_flo', back_flo)]:
            if len(flow.shape) != 3 or flow.shape[2] != 2:
                raise ValueError(f"{name} must be 3D (H, W, 2), got shape {flow.shape}")
            if not np.issubdtype(flow.dtype, np.floating):
                warnings.warn(f"{name} should be float type, got {flow.dtype}")
            
            # Check flow magnitude for sanity
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            max_mag = flow_mag.max()
            if max_mag > max(flow.shape[:2]) * 2:
                warnings.warn(f"Very large {name} values detected: max magnitude {max_mag:.1f}")
        
        # Validate spatial compatibility
        if curr_labels.shape != prev_labels.shape:
            raise ValueError(
                f"Label arrays must have same shape: curr {curr_labels.shape} != prev {prev_labels.shape}"
            )
        
        try:
            all_warped_labels = []
            for (labels, fg, flow) in zip([curr_labels, prev_labels], [prev_labels, curr_labels], [forw_flo, back_flo]):
                # Validate individual processing arrays
                if labels.shape[:2] != fg.shape[:2]:
                    raise ValueError(f"Labels and fg shapes incompatible: {labels.shape[:2]} vs {fg.shape[:2]}")
                
                # Create tensors with error handling
                try:
                    labels_tensor = torch.tensor(labels[..., None] + 1).permute(2, 0, 1)[None, ...].float()
                except Exception as e:
                    raise ValueError(f"Failed to create labels tensor: {e}")
                    
                # Handle size differences between labels and flow
                pad_h = labels.shape[0] - flow.shape[0]
                pad_w = labels.shape[1] - flow.shape[1]
                
                # Crop flow if larger than labels
                if pad_h < 0:
                    flow = flow[-pad_h // 2:pad_h + (-pad_h) // 2]
                if pad_w < 0:
                    flow = flow[:, -pad_w // 2:pad_w + (-pad_w) // 2]
                    
                # Pad flow if smaller than labels
                pad_h = max(0, pad_h)
                pad_w = max(0, pad_w)
                pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
                
                try:
                    flow_pad = np.pad(flow, pad)
                    flow_tensor = torch.tensor(flow_pad, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                except Exception as e:
                    raise ValueError(f"Failed to create flow tensor: {e}")
                
                # Perform warping with error handling
                try:
                    # Create mask with proper typing
                    fg_mask = np.asarray(fg >= 0, dtype=np.uint8)
                    fg_mask_expanded = fg_mask[..., None]
                    warped_labels = warp_flo(labels_tensor, flow_tensor) * fg_mask_expanded
                    warped_labels = np.int32(warped_labels[:, :, 0] - 1)
                except Exception as e:
                    raise ValueError(f"Failed during warping operation: {e}")
                    
                all_warped_labels.append(warped_labels)
            
            curr_warped_labels, prev_warped_labels = all_warped_labels
            
            # Apply masks
            curr_warped_labels[prev_labels<0] = -1
            prev_warped_labels[curr_labels<0] = -1
            
            # Final validation
            for name, result in [('curr_warped', curr_warped_labels), ('prev_warped', prev_warped_labels)]:
                if not np.isfinite(result).all():
                    raise ValueError(f"{name} labels contain non-finite values after warping")
            
            logger.debug(f"Successfully warped labels: shapes {curr_warped_labels.shape}, {prev_warped_labels.shape}")
            return curr_warped_labels, prev_warped_labels
            
        except Exception as e:
            logger.error(f"Label warping failed: {e}")
            raise

    @staticmethod
    def compute_match_graphs(curr_labels, prev_labels, curr_fg_labels, prev_fg_labels, curr_warped_labels, prev_warped_labels):
        """
        Compute matching graphs between current and previous labels with validation.
        
        Args:
          curr_labels: Current frame label IDs (list/array)
          prev_labels: Previous frame label IDs (list/array)
          curr_fg_labels: Current foreground labels (H, W)
          prev_fg_labels: Previous foreground labels (H, W)
          curr_warped_labels: Current warped labels (H, W)
          prev_warped_labels: Previous warped labels (H, W)
          
        Returns:
          Tuple of (prev_in_curr, curr_in_prev) matching matrices
          
        Raises:
          ValueError: If input arrays have incompatible shapes or invalid data
          TypeError: If inputs are not proper array types
        """
        # Validate label IDs
        for name, labels in [('curr_labels', curr_labels), ('prev_labels', prev_labels)]:
            if not hasattr(labels, '__len__'):
                raise TypeError(f"{name} must be list or array-like")
            if len(labels) == 0:
                warnings.warn(f"{name} is empty - no labels to process")
        
        # Validate spatial label arrays
        spatial_arrays = {
            'curr_fg_labels': curr_fg_labels,
            'prev_fg_labels': prev_fg_labels,
            'curr_warped_labels': curr_warped_labels,
            'prev_warped_labels': prev_warped_labels
        }
        
        reference_shape = None
        for name, array in spatial_arrays.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be numpy array, got {type(array)}")
            if len(array.shape) != 2:
                raise ValueError(f"{name} must be 2D (H, W), got shape {array.shape}")
            if array.size == 0:
                raise ValueError(f"{name} is empty")
            
            # Check shape consistency
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError(
                    f"{name} shape {array.shape} doesn't match reference shape {reference_shape}"
                )
            
            # Validate data integrity
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        
        try:
            # Convert to arrays for consistent processing
            curr_labels = np.asarray(curr_labels)
            prev_labels = np.asarray(prev_labels)
            
            # Validate label value ranges
            for name, labels, fg_labels in [
                ('curr', curr_labels, curr_fg_labels),
                ('prev', prev_labels, prev_fg_labels)
            ]:
                unique_labels = np.unique(labels)
                unique_fg = np.unique(fg_labels)
                
                # Check if label IDs exist in foreground arrays
                for label_id in unique_labels:
                    if label_id >= 0 and label_id not in unique_fg:
                        warnings.warn(
                            f"{name} label ID {label_id} not found in foreground labels"
                        )
            
            # Memory check before processing
            total_memory_mb = sum(array.nbytes for array in spatial_arrays.values()) / (1024 * 1024)
            if total_memory_mb > 1000:  # > 1GB
                warnings.warn(
                    f"High memory usage for match graph computation: {total_memory_mb:.1f} MB"
                )
            
            # Compute stacks with size validation
            try:
                curr_labels_stack = np.tile(curr_fg_labels, [len(curr_labels), 1, 1])
                curr_labels_stack = np.reshape(curr_labels_stack, (len(curr_labels), -1))
                
                prev_labels_stack = np.tile(prev_fg_labels, [len(prev_labels), 1, 1])
                prev_labels_stack = np.reshape(prev_labels_stack, (len(prev_labels), -1))
            except MemoryError:
                raise ValueError(
                    f"Insufficient memory to create label stacks: "
                    f"{len(curr_labels)} x {len(prev_labels)} x {curr_fg_labels.size} elements"
                )
            
            # Compute masks with error handling
            try:
                # Create binary masks with proper typing
                curr_labels_mask = np.asarray(curr_labels_stack == curr_labels[..., None], dtype=np.uint8).astype(np.float64)
                prev_labels_mask = np.asarray(prev_labels_stack == prev_labels[..., None], dtype=np.uint8).astype(np.float64)
                
                curr_warped_labels_stack = np.tile(curr_warped_labels, [len(curr_labels), 1, 1])
                curr_warped_labels_stack = np.reshape(curr_warped_labels_stack, (len(curr_labels), -1))
                
                prev_warped_labels_stack = np.tile(prev_warped_labels, [len(prev_labels), 1, 1])
                prev_warped_labels_stack = np.reshape(prev_warped_labels_stack, (len(prev_labels), -1))
                
                curr_warped_labels_mask = np.asarray(curr_warped_labels_stack == curr_labels[..., None], dtype=np.uint8).astype(np.float64)
                prev_warped_labels_mask = np.asarray(prev_warped_labels_stack == prev_labels[..., None], dtype=np.uint8).astype(np.float64)
            except Exception as e:
                raise ValueError(f"Failed to compute label masks: {e}")
            
            # Compute totals with division by zero protection
            prev_total = np.sum(prev_warped_labels_mask, axis=1)
            curr_total = np.sum(curr_warped_labels_mask, axis=1)
            
            # Check for zero totals
            zero_prev = (prev_total == 0).sum()
            zero_curr = (curr_total == 0).sum()
            if zero_prev > 0:
                warnings.warn(f"{zero_prev} previous labels have zero total area")
            if zero_curr > 0:
                warnings.warn(f"{zero_curr} current labels have zero total area")
            
            # Compute matching scores with robust division and proper typing
            with np.errstate(divide='ignore', invalid='ignore'):
                # Ensure arrays are float64 for matrix multiplication
                prev_mask_f64 = prev_warped_labels_mask.astype(np.float64)
                curr_mask_f64 = curr_labels_mask.astype(np.float64)
                curr_warped_mask_f64 = curr_warped_labels_mask.astype(np.float64)
                prev_labels_mask_f64 = prev_labels_mask.astype(np.float64)
                
                prev_in_curr = np.matmul(prev_mask_f64, curr_mask_f64.T) / prev_total[..., None]
                curr_in_prev = np.matmul(curr_warped_mask_f64, prev_labels_mask_f64.T) / curr_total[..., None]
            
            # Handle NaN values
            prev_in_curr = np.nan_to_num(prev_in_curr, nan=0.0, posinf=0.0, neginf=0.0)
            curr_in_prev = np.nan_to_num(curr_in_prev, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate results
            if not (0 <= prev_in_curr).all() or not (prev_in_curr <= 1).all():
                warnings.warn("prev_in_curr scores outside [0,1] range")
            if not (0 <= curr_in_prev).all() or not (curr_in_prev <= 1).all():
                warnings.warn("curr_in_prev scores outside [0,1] range")
            
            logger.debug(
                f"Computed match graphs: prev_in_curr {prev_in_curr.shape}, "
                f"curr_in_prev {curr_in_prev.shape}"
            )
            
            return prev_in_curr, curr_in_prev
            
        except Exception as e:
            logger.error(f"Match graph computation failed: {e}")
            raise

    @staticmethod
    def hungarian_matching(scores):
        """Hungarian algorithm for optimal bipartite matching"""
        row_ind, col_ind = linear_sum_assignment(scores)
        unmatched_prev = set([i for i in range(scores.shape[0])])
        unmatched_curr = set([j for j in range(scores.shape[1])])
        matching = []
        for r, c in zip(row_ind, col_ind):
            unmatched_prev.remove(r)
            matching.append([[r], [c]])
        return matching, unmatched_prev, unmatched_curr

    @staticmethod
    def main_matching(scores):
        """Graph-based matching for complex correspondences"""
        unmatched_prev = set([i for i in range(scores.shape[0])])
        unmatched_curr = set([j for j in range(scores.shape[1])])

        # Make matching graph.
        B = nx.Graph()
        B.add_nodes_from([f'p{i}' for i in range(scores.shape[0])], bipartite=0)
        B.add_nodes_from([f'c{j}' for j in range(scores.shape[1])], bipartite=1)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i, j] > 0:
                    B.add_edge(f'p{i}', f'c{j}')
                    if i in unmatched_prev:
                        unmatched_prev.remove(i)
                    if j in unmatched_curr:
                        unmatched_curr.remove(j)

        matching = []
        cc = nx.connected_components(B)
        for C in cc:
            prev, curr = [], []
            for v in C:
                if v[0] == 'p':
                    prev.append(int(v[1:]))
                if v[0] == 'c':
                    curr.append(int(v[1:]))
            if len(prev) > 0 and len(curr) > 0:
                matching.append([prev, curr])
        return matching, unmatched_prev, unmatched_curr

    # ================================
    # Modern CoTracker3 AI Methods (Keep)
    # ================================

    def get_cotracker3_appearance_analysis(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids):
        """
        Modern appearance analysis using CoTracker3 features with comprehensive validation.
        
        Args:
          prev_shapes: List of previous frame shape images (RGBA)
          curr_shapes: List of current frame shape images (RGBA)
          prev_centroids: List of previous frame centroids
          curr_centroids: List of current frame centroids
          
        Returns:
          Tuple of (shape_diffs, rgb_diffs) similarity matrices
          
        Raises:
          ValueError: If input shapes or centroids are invalid
          TypeError: If inputs are not proper types
        """
        # Validate inputs
        for name, shapes in [('prev_shapes', prev_shapes), ('curr_shapes', curr_shapes)]:
            if not isinstance(shapes, (list, tuple)):
                raise TypeError(f"{name} must be list or tuple, got {type(shapes)}")
            if len(shapes) == 0:
                warnings.warn(f"{name} is empty")
                continue
            
            # Validate each shape
            for i, shape in enumerate(shapes):
                if not isinstance(shape, np.ndarray):
                    raise TypeError(f"{name}[{i}] must be numpy array, got {type(shape)}")
                if len(shape.shape) != 3:
                    raise ValueError(f"{name}[{i}] must be 3D (H,W,C), got shape {shape.shape}")
                if shape.shape[2] not in [3, 4]:  # BGR or BGRA
                    raise ValueError(f"{name}[{i}] must have 3 or 4 channels, got {shape.shape[2]}")
                if shape.size == 0:
                    raise ValueError(f"{name}[{i}] is empty")
                if not np.isfinite(shape).all():
                    raise ValueError(f"{name}[{i}] contains non-finite values")
        
        # Validate centroids
        for name, centroids in [('prev_centroids', prev_centroids), ('curr_centroids', curr_centroids)]:
            if not isinstance(centroids, (list, tuple)):
                raise TypeError(f"{name} must be list or tuple, got {type(centroids)}")
            
            # Check centroid format
            for i, centroid in enumerate(centroids):
                if not isinstance(centroid, (tuple, list, np.ndarray)):
                    raise TypeError(f"{name}[{i}] must be tuple/list/array, got {type(centroid)}")
                if len(centroid) != 2:
                    raise ValueError(f"{name}[{i}] must have 2 coordinates, got {len(centroid)}")
                if not all(np.isfinite(c) for c in centroid):
                    raise ValueError(f"{name}[{i}] contains non-finite coordinates")
        
        # Check shape-centroid correspondence
        if len(prev_shapes) != len(prev_centroids):
            raise ValueError(
                f"Shape/centroid count mismatch: {len(prev_shapes)} prev shapes vs {len(prev_centroids)} centroids"
            )
        if len(curr_shapes) != len(curr_centroids):
            raise ValueError(
                f"Shape/centroid count mismatch: {len(curr_shapes)} curr shapes vs {len(curr_centroids)} centroids"
            )
        
        if not self.use_cotracker3 or not self.cotracker3_engine:
            # Minimal fallback when CoTracker3 not available
            logger.debug("Using minimal fallback similarity (CoTracker3 not available)")
            return self._get_minimal_fallback_similarity(prev_shapes, curr_shapes)
        
        # Use CoTracker3 for advanced feature extraction
        try:
            logger.debug(f"Computing CoTracker3 features for {len(prev_shapes)} -> {len(curr_shapes)} shapes")
            
            # Use CoTracker3 features if available, otherwise minimal fallback
            if hasattr(self.cotracker3_engine, 'extract_shape_features'):
                # CoTracker3 provides superior shape understanding
                shape_features_prev = self.cotracker3_engine.extract_shape_features(prev_shapes)
                shape_features_curr = self.cotracker3_engine.extract_shape_features(curr_shapes)
            else:
                # Fallback to basic shape analysis
                logger.warning("CoTracker3 engine missing extract_shape_features method, using fallback")
                shape_features_prev = [self._extract_minimal_shape_features(shape) for shape in prev_shapes]
                shape_features_curr = [self._extract_minimal_shape_features(shape) for shape in curr_shapes]
            
            # Validate extracted features
            if shape_features_prev is None or shape_features_curr is None:
                raise ValueError("CoTracker3 feature extraction returned None")
            
            if len(shape_features_prev) != len(prev_shapes):
                raise ValueError(
                    f"Feature count mismatch: {len(shape_features_prev)} features vs {len(prev_shapes)} shapes"
                )
            
            # Compute similarity matrix using CoTracker3 features
            shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
            
            for i in range(len(prev_shapes)):
                for j in range(len(curr_shapes)):
                    try:
                        if hasattr(self.cotracker3_engine, 'compute_feature_similarity'):
                            similarity = self.cotracker3_engine.compute_feature_similarity(
                                shape_features_prev[i], shape_features_curr[j]
                            )
                        else:
                            # Fallback to basic similarity computation
                            similarity = self._compute_minimal_feature_similarity(
                                shape_features_prev[i], shape_features_curr[j]
                            )
                        
                        # Validate similarity score
                        if not np.isfinite(similarity):
                            similarity = 0.0
                        elif similarity < 0 or similarity > 1:
                            warnings.warn(f"Similarity score {similarity} outside [0,1] range")
                            similarity = np.clip(similarity, 0.0, 1.0)
                        
                        shape_diffs[i, j] = similarity
                        
                    except Exception as e:
                        logger.warning(f"Feature similarity computation failed for shapes {i},{j}: {e}")
                        shape_diffs[i, j] = 0.0
            
            # Enhanced RGB analysis with CoTracker3 color understanding
            rgb_diffs = self._get_minimal_rgb_similarity(prev_shapes, curr_shapes)
            
            # Validate output matrices
            if shape_diffs.shape != (len(prev_shapes), len(curr_shapes)):
                raise ValueError(f"Invalid shape_diffs shape: {shape_diffs.shape}")
            if rgb_diffs.shape != (len(prev_shapes), len(curr_shapes)):
                raise ValueError(f"Invalid rgb_diffs shape: {rgb_diffs.shape}")
            
            logger.debug(f"✅ CoTracker3 appearance analysis completed successfully")
            return shape_diffs, rgb_diffs
            
        except Exception as e:
            logger.error(f"⚠️ CoTracker3 feature extraction failed: {e}")
            logger.debug("Falling back to minimal similarity")
            return self._get_minimal_fallback_similarity(prev_shapes, curr_shapes)

    def _get_cotracker3_correspondences(
        self,
        prev_shapes: List[np.ndarray],
        curr_shapes: List[np.ndarray], 
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_centroids: List[Tuple[float, float]],
        curr_centroids: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get shape correspondences using CoTracker3 point tracking
        
        This method replaces get_appearance_graphs and provides superior tracking
        using Meta AI's CoTracker3 for dense point correspondence.
        
        Args:
          prev_shapes: List of previous frame shape images (RGBA)
          curr_shapes: List of current frame shape images (RGBA) 
          prev_frame: Previous frame (BGR)
          curr_frame: Current frame (BGR)
          prev_centroids: Previous frame shape centroids
          curr_centroids: Current frame shape centroids
          
        Returns:
          shape_similarities: Shape similarity matrix (N_prev, N_curr)
          motion_scores: Motion consistency scores (N_prev, N_curr)
          tracking_metadata: Additional tracking information
        """
        if not self.use_cotracker3 or self.cotracker3_engine is None:
            # Fallback to minimal method
            shape_diffs, rgb_diffs = self._get_minimal_fallback_similarity(prev_shapes, curr_shapes)
            return shape_diffs, rgb_diffs, {}
            
        print("🎯 Using CoTracker3 for shape correspondence")
        
        # Prepare video tensor (2 frames for pairwise tracking)
        video_tensor = self._prepare_video_tensor(prev_frame, curr_frame)
        
        # Extract contour points for each shape
        prev_contour_points = self._extract_shape_contours(prev_shapes)
        curr_contour_points = self._extract_shape_contours(curr_shapes)
        
        # Perform CoTracker3 tracking
        shape_similarities = np.zeros((len(prev_shapes), len(curr_shapes)))
        motion_scores = np.zeros((len(prev_shapes), len(curr_shapes)))
        tracking_metadata = {
            'tracks': {},
            'visibility': {},
            'motion_params': {},
            'quality_scores': {}
        }
        
        # Track each previous shape against all current shapes
        for i, prev_points in enumerate(prev_contour_points):
            if prev_points is None or len(prev_points[0]) < 3:
                continue
                
            try:
                # Track points across the two frames (with fallback handling)
                try:
                    tracks, visibility = self.cotracker3_engine.track_video_grid(
                        video_tensor, custom_points=prev_points
                    )
                except AttributeError:
                    # Fallback if method doesn't exist
                    tracks, visibility = [], []
                
                # Extract motion parameters
                motion_params = self.cotracker3_engine.extract_motion_parameters(
                    tracks, visibility
                )
                
                tracking_metadata['tracks'][i] = tracks
                tracking_metadata['visibility'][i] = visibility
                tracking_metadata['motion_params'][i] = motion_params
                
                # Calculate similarity with each current shape
                for j, curr_points in enumerate(curr_contour_points):
                    if curr_points is None or len(curr_points[0]) < 3:
                        continue
                        
                    # Calculate shape similarity based on tracking quality
                    shape_sim = self._calculate_tracking_similarity(
                        tracks, visibility, prev_points, curr_points
                    )
                    
                    # Calculate motion consistency score
                    motion_score = self._calculate_motion_consistency(
                        motion_params, prev_centroids[i], curr_centroids[j]
                    )
                    
                    shape_similarities[i, j] = shape_sim
                    motion_scores[i, j] = motion_score
                    
                # Calculate overall tracking quality
                quality_score = self._calculate_tracking_quality_score(tracks, visibility)
                tracking_metadata['quality_scores'][i] = quality_score
                
            except Exception as e:
                print(f"⚠️  CoTracker3 tracking failed for shape {i}: {e}")
                # Use centroid distance as fallback
                for j in range(len(curr_shapes)):
                    dist = np.linalg.norm(
                        np.array(prev_centroids[i]) - np.array(curr_centroids[j])
                    )
                    shape_similarities[i, j] = max(0.0, 1.0 - dist / 100.0)  # Normalize distance
                    motion_scores[i, j] = shape_similarities[i, j]
        
        print(f"✅ CoTracker3 correspondence complete: {len(prev_shapes)}→{len(curr_shapes)} shapes")
        return shape_similarities, motion_scores, tracking_metadata

    def cotracker3_joint_tracking(
        self,
        video_frames: List[np.ndarray],
        shape_masks: List[np.ndarray],
        object_ids: List[int],
        start_frame: int = 0
    ) -> Dict[int, Dict[str, Any]]:
        """
        Joint tracking of multiple objects across video sequence using CoTracker3
        
        This method provides temporal consistency across multiple frames,
        superior to frame-by-frame matching.
        
        Args:
          video_frames: List of video frames (BGR format)
          shape_masks: List of segmentation masks with object IDs
          object_ids: List of unique object IDs to track
          start_frame: Frame index to start tracking from
          
        Returns:
          Dictionary mapping object_id to tracking results
        """
        if not self.use_cotracker3 or self.sam2_cotracker_bridge is None:
            print("⚠️  CoTracker3 joint tracking not available, using fallback")
            return {}
            
        print(f"🎬 CoTracker3 joint tracking: {len(video_frames)} frames, {len(object_ids)} objects")
        
        # Prepare video tensor
        video_tensor = self._prepare_video_sequence(video_frames)
        
        # Prepare mask tensor
        mask_tensor = self._prepare_mask_sequence(shape_masks)
        
        # Use SAM2-CoTracker3 bridge for complete processing
        try:
            results = self.sam2_cotracker_bridge.process_video(
                video_tensor,
                prompts=None,  # Masks already provided
                return_intermediate=True
            )
            
            # Extract relevant tracking data for each object
            object_tracking = {}
            for obj_id in object_ids:
                if obj_id in results['motion_parameters']:
                    object_tracking[obj_id] = {
                        'tracks': results['object_tracks'].get(obj_id, {}).get('tracks'),
                        'visibility': results['object_tracks'].get(obj_id, {}).get('visibility'),
                        'motion_params': results['motion_parameters'][obj_id],
                        'quality_score': results['motion_parameters'][obj_id].get('quality_score', 0.0)
                    }
                    
            print(f"✅ Joint tracking complete: {len(object_tracking)} objects tracked")
            return object_tracking
            
        except Exception as e:
            print(f"⚠️  CoTracker3 joint tracking failed: {e}")
            return {}

    # ================================
    # CoTracker3 Helper Methods
    # ================================

    def _prepare_video_tensor(self, frame1: np.ndarray, frame2: np.ndarray) -> torch.Tensor:
        """Prepare 2-frame video tensor for CoTracker3"""
        # Convert BGR to RGB and normalize
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Create tensor (B=1, T=2, C=3, H, W)
        video = np.stack([frame1_rgb, frame2_rgb], axis=0)
        video_tensor = torch.tensor(video, dtype=torch.float32)
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 2, 3, H, W)
        
        return video_tensor.to(self.cotracker3_engine.device if self.cotracker3_engine else 'cpu')
        
    def _prepare_video_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Prepare multi-frame video tensor for joint tracking"""
        rgb_frames = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb_frames.append(frame_rgb)
            
        video = np.stack(rgb_frames, axis=0)
        video_tensor = torch.tensor(video, dtype=torch.float32)
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
        
        return video_tensor.to(self.cotracker3_engine.device if self.cotracker3_engine else 'cpu')
        
    def _prepare_mask_sequence(self, masks: List[np.ndarray]) -> torch.Tensor:
        """Prepare mask sequence tensor"""
        mask_tensor = torch.tensor(np.stack(masks, axis=0), dtype=torch.long)
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, T, H, W)
        
        return mask_tensor.to(self.cotracker3_engine.device if self.cotracker3_engine else 'cpu')
        
    def _extract_shape_contours(self, shapes: List[np.ndarray]) -> List[Optional[torch.Tensor]]:
        """Extract contour points from shape images for CoTracker3"""
        contour_points = []
        
        for shape in shapes:
            if shape.shape[-1] != 4:  # Not RGBA
                contour_points.append(None)
                continue
                
            # Extract alpha channel
            alpha = shape[:, :, 3]
            
            if alpha.max() == 0:  # Empty shape
                contour_points.append(None)
                continue
                
            # Find contours
            contours, _ = cv2.findContours(
                (alpha > 0).astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                contour_points.append(None)
                continue
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Sample points along contour
            num_points = min(25, len(largest_contour))  # Reasonable number of points
            if len(largest_contour) >= num_points:
                indices = np.linspace(0, len(largest_contour) - 1, num_points, dtype=int)
                sampled_points = largest_contour[indices].reshape(-1, 2)
            else:
                sampled_points = largest_contour.reshape(-1, 2)
                
            # Convert to tensor format (1, N, 2)
            points_tensor = torch.tensor(
                sampled_points[None, :, :], 
                dtype=torch.float32,
                device=self.cotracker3_engine.device if self.cotracker3_engine else 'cpu'
            )
            
            contour_points.append(points_tensor)
            
        return contour_points

    def _calculate_tracking_similarity(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        prev_points: torch.Tensor,
        curr_points: torch.Tensor
    ) -> float:
        """Calculate shape similarity based on tracking quality"""
        if tracks is None or len(tracks.shape) != 4:
            return 0.0
            
        # Extract final tracked positions
        final_tracks = tracks[0, -1]  # (N, 2)
        final_visibility = visibility[0, -1, :, 0]  # (N,)
        
        # Calculate how well tracking matches expected positions
        if len(curr_points[0]) == 0:
            return 0.0
            
        # Visible tracks only
        valid_mask = final_visibility > 0.5
        if not valid_mask.any():
            return 0.0
            
        valid_tracks = final_tracks[valid_mask]
        
        # Find closest matches between tracked and expected positions
        curr_points_np = curr_points[0].cpu().numpy()
        valid_tracks_np = valid_tracks.cpu().numpy()
        
        if len(valid_tracks_np) == 0 or len(curr_points_np) == 0:
            return 0.0
            
        # Calculate minimum distances
        distances = cdist(valid_tracks_np, curr_points_np)
        min_distances = distances.min(axis=1)
        
        # Convert to similarity (closer = higher similarity)
        avg_distance = min_distances.mean()
        similarity = np.exp(-avg_distance / 20.0)  # Exponential decay
        
        return float(similarity)

    def _calculate_motion_consistency(
        self,
        motion_params: Dict[str, torch.Tensor],
        prev_centroid: Tuple[float, float],
        curr_centroid: Tuple[float, float]
    ) -> float:
        """Calculate motion consistency score"""
        if 'translation' not in motion_params:
            return 0.0
            
        # Extract translation from motion parameters
        translation = motion_params['translation'][0, -1].cpu().numpy()  # (2,)
        
        # Expected translation from centroids
        expected_translation = np.array(curr_centroid) - np.array(prev_centroid)
        
        # Calculate consistency
        translation_error = np.linalg.norm(translation - expected_translation)
        consistency = np.exp(-translation_error / 50.0)  # Exponential decay
        
        return float(consistency)

    def _calculate_tracking_quality_score(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor
    ) -> float:
        """Calculate overall tracking quality score"""
        if tracks is None or visibility is None:
            return 0.0
            
        # Visibility-based quality
        visibility_score = visibility.mean().item()
        
        # Motion smoothness quality
        if tracks.shape[1] > 1:
            velocities = torch.diff(tracks, dim=1)
            if velocities.numel() > 0:
                velocity_variance = torch.var(velocities).item()
                smoothness_score = 1.0 / (1.0 + velocity_variance)
            else:
                smoothness_score = 0.0
        else:
            smoothness_score = 1.0
            
        # Combined score
        quality_score = 0.6 * visibility_score + 0.4 * smoothness_score
        
        return float(quality_score)

    # ================================
    # Minimal Fallback Methods (Keep Minimal)
    # ================================

    def _get_minimal_fallback_similarity(self, prev_shapes, curr_shapes):
        """
        Minimal fallback similarity when CoTracker3 is not available
        """
        # Initialize similarity matrices
        shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
        rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
        
        if len(prev_shapes) == 0 or len(curr_shapes) == 0:
            return shape_diffs, rgb_diffs
        
        try:
            # Very basic size and mean color comparison
            for i, prev_shape in enumerate(prev_shapes):
                if prev_shape is None or prev_shape.size == 0:
                    continue
                    
                prev_size = prev_shape.shape[0] * prev_shape.shape[1]
                prev_mean = np.mean(prev_shape[:, :, :3]) if len(prev_shape.shape) >= 3 else np.mean(prev_shape)
                
                for j, curr_shape in enumerate(curr_shapes):
                    if curr_shape is None or curr_shape.size == 0:
                        continue
                        
                    curr_size = curr_shape.shape[0] * curr_shape.shape[1]
                    curr_mean = np.mean(curr_shape[:, :, :3]) if len(curr_shape.shape) >= 3 else np.mean(curr_shape)
                    
                    # Size similarity
                    size_ratio = min(prev_size, curr_size) / max(prev_size, curr_size) if max(prev_size, curr_size) > 0 else 0
                    
                    # Color similarity
                    color_diff = abs(prev_mean - curr_mean)
                    color_similarity = max(0.0, 1.0 - color_diff / 255.0)
                    
                    # Combined similarity
                    shape_diffs[i, j] = (size_ratio + color_similarity) / 2.0
                    rgb_diffs[i, j] = color_similarity
            
            return shape_diffs, rgb_diffs
            
        except Exception as e:
            logger.warning(f"Minimal fallback similarity failed: {e}")
            return shape_diffs, rgb_diffs

    def _get_minimal_rgb_similarity(self, prev_shapes, curr_shapes):
        """
        Minimal RGB similarity for CoTracker3 fallback scenarios only.
        """
        # Initialize similarity matrix
        rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
        
        if len(prev_shapes) == 0 or len(curr_shapes) == 0:
            return rgb_diffs
        
        try:
            # Simple mean RGB comparison for minimal fallback
            for i, prev_shape in enumerate(prev_shapes):
                if prev_shape is None or prev_shape.size == 0:
                    continue
                    
                prev_mean = np.mean(prev_shape[:, :, :3]) if len(prev_shape.shape) >= 3 else np.mean(prev_shape)
                
                for j, curr_shape in enumerate(curr_shapes):
                    if curr_shape is None or curr_shape.size == 0:
                        continue
                        
                    curr_mean = np.mean(curr_shape[:, :, :3]) if len(curr_shape.shape) >= 3 else np.mean(curr_shape)
                    
                    # Simple similarity based on mean values
                    diff = abs(prev_mean - curr_mean)
                    similarity = max(0.0, 1.0 - diff / 255.0)
                    rgb_diffs[i, j] = similarity
            
            return rgb_diffs
            
        except Exception as e:
            logger.warning(f"Minimal RGB similarity failed: {e}")
            return rgb_diffs

    def _extract_minimal_shape_features(self, shape):
        """Extract minimal shape features as fallback for CoTracker3"""
        if shape.size == 0:
            return np.zeros(4)  # Return minimal features
        
        try:
            # Basic geometric features only
            height, width = shape.shape[:2]
            area = height * width
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Basic intensity
            mean_intensity = np.mean(shape)
            
            return np.array([area, aspect_ratio, mean_intensity, height])
        except Exception as e:
            logger.warning(f"Failed to extract minimal shape features: {e}")
            return np.zeros(4)

    def _compute_minimal_feature_similarity(self, features_a, features_b):
        """Compute minimal similarity between feature vectors"""
        try:
            features_a = np.asarray(features_a, dtype=np.float64)
            features_b = np.asarray(features_b, dtype=np.float64)
            
            if features_a.size == 0 or features_b.size == 0:
                return 0.0
            
            # Simple normalized correlation
            norm_a = np.linalg.norm(features_a)
            norm_b = np.linalg.norm(features_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = np.dot(features_a, features_b) / (norm_a * norm_b)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Failed to compute minimal feature similarity: {e}")
            return 0.0

    # ================================
    # Cleanup and Legacy Compatibility 
    # ================================

    def cleanup_cotracker3(self):
        """Cleanup CoTracker3 resources"""
        if self.cotracker3_engine:
            self.cotracker3_engine.cleanup()
        if self.sam2_cotracker_bridge:
            self.sam2_cotracker_bridge.cleanup()
            
        self.cotracker3_engine = None
        self.sam2_cotracker_bridge = None
        print("🧹 CoTracker3 processor integration cleaned up")

    def get_cotracker3_correspondences(self, prev_shapes, curr_shapes, prev_frame, curr_frame, prev_centroids, curr_centroids):
        """Get shape correspondences using CoTracker3 or fallback methods"""
        try:
            if self.use_cotracker3 and self.cotracker3_engine is not None:
                # Use CoTracker3 for advanced tracking
                return self._get_cotracker3_correspondences(
                    prev_shapes, curr_shapes, prev_frame, curr_frame, prev_centroids, curr_centroids
                )
        except Exception as e:
            logger.warning(f"CoTracker3 correspondence failed: {e}")
        
        # Fallback to appearance analysis
        return self.get_cotracker3_appearance_analysis(prev_shapes, curr_shapes, prev_centroids, curr_centroids)

    # Legacy compatibility methods (minimal implementations)
    def get_appearance_graphs(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids):
        """Legacy compatibility method - delegates to CoTracker3 analysis"""
        return self.get_cotracker3_appearance_analysis(prev_shapes, curr_shapes, prev_centroids, curr_centroids)

    @staticmethod 
    def compute_matching_comp_groups(matching, prev_labels, curr_labels, prev_fg_comp_to_label, curr_fg_comp_to_label):
        """Compute matching component groups using graph connectivity"""
        ceg = nx.Graph()
        ceg.add_nodes_from([f'p{i}' for i in prev_labels])
        ceg.add_nodes_from([f'c{j}' for j in curr_labels])
        ceg.add_nodes_from([f'M{c}' for c in prev_fg_comp_to_label])
        ceg.add_nodes_from([f'N{c}' for c in curr_fg_comp_to_label])
        for prev, curr in matching:
            for i in prev:
                for j in curr:
                    ceg.add_edges_from([(f'p{prev_labels[i]}', f'c{curr_labels[j]}')])
        for c in prev_fg_comp_to_label:
            ceg.add_edges_from([(f'M{c}', f'p{i}') for i in prev_fg_comp_to_label[c]])
        for c in curr_fg_comp_to_label:
            ceg.add_edges_from([(f'N{c}', f'c{j}') for j in curr_fg_comp_to_label[c]])

        matching_comp_groups = []
        for C in nx.connected_components(ceg):
            prev_comps = []
            curr_comps = []
            for v in C:
                if v[0] == 'M':
                    prev_comps.append(int(v[1:]))
                if v[0] == 'N':
                    curr_comps.append(int(v[1:]))
            matching_comp_groups.append([prev_comps, curr_comps])
        return matching_comp_groups, ceg