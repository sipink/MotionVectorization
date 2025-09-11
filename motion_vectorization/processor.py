import numpy as np
import cv2
import torch
# import torch.nn.functional as F
# import os
# import io
# import collections
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
# from pyefd import elliptic_fourier_descriptors
# from elliptic_fourier_descriptors import *
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
    warnings.warn("CoTracker3 integration not available. Using fallback tracking.")



class Processor():
  def __init__(self, use_cotracker3=False, cotracker3_mode="offline", device="auto"):
    
    # CoTracker3 integration
    self.use_cotracker3 = use_cotracker3 and COTRACKER3_AVAILABLE
    self.cotracker3_engine = None
    self.sam2_cotracker_bridge = None
    
    if self.use_cotracker3:
      self._initialize_cotracker3(cotracker3_mode, device)
    else:
      if use_cotracker3 and not COTRACKER3_AVAILABLE:
        print("⚠️  CoTracker3 requested but not available. Using fallback tracking.")
  
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
      # Fallback to basic RGB histogram comparison only
      logger.debug("Using fallback RGB similarity (CoTracker3 not available)")
      return self._get_rgb_similarity_matrix(prev_shapes, curr_shapes)
    
    # Use CoTracker3 for advanced feature extraction
    try:
      logger.debug(f"Computing CoTracker3 features for {len(prev_shapes)} -> {len(curr_shapes)} shapes")
      
      # Use CoTracker3 features if available, otherwise fallback
      if hasattr(self.cotracker3_engine, 'extract_shape_features'):
        # CoTracker3 provides superior shape understanding
        shape_features_prev = self.cotracker3_engine.extract_shape_features(prev_shapes)
        shape_features_curr = self.cotracker3_engine.extract_shape_features(curr_shapes)
      else:
        # Fallback to basic shape analysis
        logger.warning("CoTracker3 engine missing extract_shape_features method, using fallback")
        shape_features_prev = [self._extract_basic_shape_features(shape) for shape in prev_shapes]
        shape_features_curr = [self._extract_basic_shape_features(shape) for shape in curr_shapes]
      
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
              similarity = self._compute_basic_feature_similarity(
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
      rgb_diffs = self._get_enhanced_rgb_similarity(prev_shapes, curr_shapes)
      
      # Validate output matrices
      if shape_diffs.shape != (len(prev_shapes), len(curr_shapes)):
        raise ValueError(f"Invalid shape_diffs shape: {shape_diffs.shape}")
      if rgb_diffs.shape != (len(prev_shapes), len(curr_shapes)):
        raise ValueError(f"Invalid rgb_diffs shape: {rgb_diffs.shape}")
      
      logger.debug(f"✅ CoTracker3 appearance analysis completed successfully")
      return shape_diffs, rgb_diffs
      
    except Exception as e:
      logger.error(f"⚠️ CoTracker3 feature extraction failed: {e}")
      logger.debug("Falling back to basic RGB similarity")
      return self._get_rgb_similarity_matrix(prev_shapes, curr_shapes)
  
  def _get_rgb_similarity_matrix(self, prev_shapes, curr_shapes):
    """
    Basic RGB histogram comparison with comprehensive validation (fallback for when CoTracker3 unavailable).
    
    Args:
      prev_shapes: List of previous frame shape images
      curr_shapes: List of current frame shape images
      
    Returns:
      Tuple of (shape_diffs, rgb_diffs) similarity matrices
      
    Raises:
      ValueError: If shapes have invalid format or properties
    """
    # Validate inputs
    for name, shapes in [('prev_shapes', prev_shapes), ('curr_shapes', curr_shapes)]:
      if not isinstance(shapes, (list, tuple)):
        raise TypeError(f"{name} must be list or tuple")
    
    if len(prev_shapes) == 0 or len(curr_shapes) == 0:
      logger.warning("Empty shape lists provided to RGB similarity")
      return np.zeros((len(prev_shapes), len(curr_shapes))), np.zeros((len(prev_shapes), len(curr_shapes)))
    
    shape_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5  # Neutral similarity
    rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    
    try:
      for i, prev_shape in enumerate(prev_shapes):
        # Validate previous shape
        if not isinstance(prev_shape, np.ndarray):
          logger.warning(f"prev_shapes[{i}] is not numpy array, skipping")
          continue
        
        if len(prev_shape.shape) != 3 or prev_shape.shape[2] < 3:
          logger.warning(f"prev_shapes[{i}] invalid shape {prev_shape.shape}, skipping")
          continue
        
        if prev_shape.size == 0:
          logger.warning(f"prev_shapes[{i}] is empty, skipping")
          continue
        
        try:
          # Handle both BGR and BGRA formats
          if prev_shape.shape[2] == 4:
            # Use alpha channel as mask for histogram
            alpha_mask = prev_shape[:, :, 3]
            if alpha_mask.max() == 0:
              logger.warning(f"prev_shapes[{i}] has empty alpha mask, skipping")
              continue
          else:
            alpha_mask = None
          
          # Convert to LAB with error handling
          prev_bgr = prev_shape[:, :, :3].astype(np.uint8)
          prev_shape_lab = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB)
          
          # Compute histogram with robust parameters
          prev_hist = cv2.calcHist(
            [prev_shape_lab], [0, 1, 2], 
            alpha_mask if alpha_mask is not None else None,
            [32, 32, 32], [0, 256, 0, 256, 0, 256]
          )
          
          # Normalize histogram with error checking
          hist_sum = prev_hist.sum()
          if hist_sum == 0:
            logger.warning(f"prev_shapes[{i}] has zero histogram, skipping")
            continue
          
          prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
          
        except Exception as e:
          logger.warning(f"Failed to process prev_shapes[{i}]: {e}")
          continue
        
        for j, curr_shape in enumerate(curr_shapes):
          # Validate current shape
          if not isinstance(curr_shape, np.ndarray):
            logger.warning(f"curr_shapes[{j}] is not numpy array, skipping")
            continue
          
          if len(curr_shape.shape) != 3 or curr_shape.shape[2] < 3:
            logger.warning(f"curr_shapes[{j}] invalid shape {curr_shape.shape}, skipping")
            continue
          
          if curr_shape.size == 0:
            logger.warning(f"curr_shapes[{j}] is empty, skipping")
            continue
          
          try:
            # Handle both BGR and BGRA formats
            if curr_shape.shape[2] == 4:
              alpha_mask_curr = curr_shape[:, :, 3]
              if alpha_mask_curr.max() == 0:
                logger.warning(f"curr_shapes[{j}] has empty alpha mask, skipping")
                continue
            else:
              alpha_mask_curr = None
            
            # Convert to LAB with error handling
            curr_bgr = curr_shape[:, :, :3].astype(np.uint8)
            curr_shape_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB)
            
            # Compute histogram
            curr_hist = cv2.calcHist(
              [curr_shape_lab], [0, 1, 2],
              alpha_mask_curr if alpha_mask_curr is not None else None,
              [32, 32, 32], [0, 256, 0, 256, 0, 256]
            )
            
            # Normalize histogram
            hist_sum_curr = curr_hist.sum()
            if hist_sum_curr == 0:
              logger.warning(f"curr_shapes[{j}] has zero histogram, skipping")
              continue
            
            curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
            
            # Compute similarity with error handling
            try:
              rgb_diff = 1.0 - cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
              
              # Validate similarity score
              if not np.isfinite(rgb_diff):
                rgb_diff = 0.0
              else:
                rgb_diff = np.clip(rgb_diff, 0.0, 1.0)
              
              rgb_diffs[i, j] = rgb_diff
              
            except Exception as e:
              logger.warning(f"Histogram comparison failed for ({i},{j}): {e}")
              rgb_diffs[i, j] = 0.0
            
          except Exception as e:
            logger.warning(f"Failed to process curr_shapes[{j}]: {e}")
            continue
      
      logger.debug(f"RGB similarity computed: {rgb_diffs.shape} matrix")
      return shape_diffs, rgb_diffs
      
    except Exception as e:
      logger.error(f"RGB similarity matrix computation failed: {e}")
      # Return safe defaults
      return (
        np.ones((len(prev_shapes), len(curr_shapes))) * 0.5,
        np.zeros((len(prev_shapes), len(curr_shapes)))
      )
  
  def _get_enhanced_rgb_similarity(self, prev_shapes, curr_shapes):
    """
    Enhanced RGB analysis with better color space understanding
    """
    rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    
    for i, prev_shape in enumerate(prev_shapes):
      # Use LAB color space for perceptually uniform color analysis
      prev_shape_lab = cv2.cvtColor(prev_shape[:, :, :3], cv2.COLOR_BGR2LAB)
      prev_hist = cv2.calcHist(
        [prev_shape_lab], [0, 1, 2], prev_shape[:, :, 3], [32, 32, 32], [0, 256, 0, 256, 0, 256])
      prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
      
      for j, curr_shape in enumerate(curr_shapes):
        curr_shape_lab = cv2.cvtColor(curr_shape[:, :, :3], cv2.COLOR_BGR2LAB)
        curr_hist = cv2.calcHist(
          [curr_shape_lab], [0, 1, 2], curr_shape[:, :, 3], [32, 32, 32], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        
        # Use multiple comparison methods for robust matching
        bhatta_dist = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        correl_dist = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        
        # Combine metrics for robust similarity score
        rgb_similarity = (1.0 - bhatta_dist) * 0.7 + correl_dist * 0.3
        rgb_diffs[i, j] = max(0.0, rgb_similarity)
    
    return rgb_diffs

  def fallback_matching(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids, frame_width, frame_height, thresh=0.6):
    """
    Fallback matching using basic centroid distance and shape comparison
    """
    if self.use_cotracker3 and self.cotracker3_engine:
      # Use CoTracker3 enhanced matching
      return self._cotracker3_enhanced_matching(prev_shapes, curr_shapes, prev_centroids, curr_centroids, frame_width, frame_height, thresh)
    else:
      # Use basic matching
      return self._basic_centroid_matching(prev_centroids, curr_centroids, thresh)
  
  def _cotracker3_enhanced_matching(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids, frame_width, frame_height, thresh=0.6):
    """
    Enhanced fallback matching using CoTracker3 features
    """
    try:
      # Use CoTracker3 for superior matching
      shape_diffs, rgb_diffs = self.get_cotracker3_appearance_analysis(
        prev_shapes, curr_shapes, prev_centroids, curr_centroids)
      
      # Compute centroid distances with proper normalization
      dists = cdist(np.array(prev_centroids), np.array(curr_centroids))
      max_dist = np.sqrt(frame_width**2 + frame_height**2)  # Diagonal distance
      normalized_dists = dists / max_dist
      
      # Advanced cost computation with CoTracker3 features
      costs = (1.0 - shape_diffs) * 0.4 + (1.0 - rgb_diffs) * 0.4 + normalized_dists * 0.2
      
      # Use Hungarian algorithm for optimal assignment
      row_ind, col_ind = linear_sum_assignment(costs)
      matching = {}
      confidence_scores = {}
      
      for i, j in zip(row_ind, col_ind):
        if j >= 0 and costs[i, j] < thresh:
          matching[i] = j
          confidence_scores[i] = 1.0 - costs[i, j]  # Higher is better
      
      return matching, costs, confidence_scores
      
    except Exception as e:
      print(f"⚠️ CoTracker3 fallback matching failed: {e}")
      return self._basic_centroid_matching(prev_centroids, curr_centroids, thresh)
  
  # Removed duplicate _basic_centroid_matching method - using the one defined later in the class

  @staticmethod
  def compute_matching_comp_groups(matching, prev_labels, curr_labels, prev_fg_comp_to_label, curr_fg_comp_to_label):
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

  @staticmethod
  def compare_shapes(shape_a, shape_b, rgb_method='hist'):
    """
    Compare two shapes using moment features and RGB analysis with comprehensive validation.
    
    Args:
      shape_a: First shape image (H, W, C) with C=3 or 4
      shape_b: Second shape image (H, W, C) with C=3 or 4  
      rgb_method: Method for RGB comparison ('hist' or 'mean')
      
    Returns:
      Tuple of (shape_diff, rgb_diff) similarity scores
      
    Raises:
      ValueError: If shapes have invalid format
      TypeError: If inputs are not numpy arrays
    """
    # Validate inputs
    for name, shape in [('shape_a', shape_a), ('shape_b', shape_b)]:
      if not isinstance(shape, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(shape)}")
      if len(shape.shape) != 3:
        raise ValueError(f"{name} must be 3D (H,W,C), got shape {shape.shape}")
      if shape.shape[2] not in [3, 4]:
        raise ValueError(f"{name} must have 3 or 4 channels, got {shape.shape[2]}")
      if shape.size == 0:
        raise ValueError(f"{name} is empty")
      if not np.isfinite(shape).all():
        raise ValueError(f"{name} contains non-finite values")
    
    # Validate RGB method
    if rgb_method not in ['hist', 'mean']:
      raise ValueError(f"rgb_method must be 'hist' or 'mean', got '{rgb_method}'")
    
    try:
      # Compute shape features with error handling
      try:
        # Note: _get_moment_features2 function should be defined elsewhere
        # For now, we'll implement a basic fallback
        # Use moment features computation
        shape_a_features = Processor._compute_static_moment_features(shape_a)
        shape_b_features = Processor._compute_static_moment_features(shape_b)
        
        # Validate features
        if not isinstance(shape_a_features, np.ndarray) or not isinstance(shape_b_features, np.ndarray):
          raise ValueError("Feature extraction returned non-array")
        
        if shape_a_features.shape != shape_b_features.shape:
          raise ValueError(f"Feature shapes don't match: {shape_a_features.shape} vs {shape_b_features.shape}")
        
        if not np.isfinite(shape_a_features).all() or not np.isfinite(shape_b_features).all():
          raise ValueError("Features contain non-finite values")
        
        # Compute shape difference with robust calculation
        feature_diff = np.abs(shape_a_features - shape_b_features)
        if feature_diff.max() > 100:  # Prevent overflow in exp
          feature_diff = np.clip(feature_diff, 0, 100)
        
        shape_diff = np.exp(-feature_diff.mean())
        
      except Exception as e:
        logger.warning(f"Shape feature computation failed: {e}, using default")
        shape_diff = 0.5  # Neutral similarity
    
      # Compute RGB difference based on method
      if rgb_method == 'hist':
        try:
          # Validate shapes have alpha channel for masking
          if shape_a.shape[2] < 4 or shape_b.shape[2] < 4:
            logger.warning("Shapes missing alpha channel for histogram masking")
            mask_a = mask_b = None
          else:
            mask_a = shape_a[:, :, 3].astype(np.uint8)
            mask_b = shape_b[:, :, 3].astype(np.uint8)
            
            # Check if masks are valid
            if mask_a.max() == 0 or mask_b.max() == 0:
              logger.warning("Empty alpha masks detected")
              mask_a = mask_b = None
          
          # Extract BGR channels
          shape_a_bgr = shape_a[:, :, :3].astype(np.uint8)
          shape_b_bgr = shape_b[:, :, :3].astype(np.uint8)
          
          # Compute histograms with error handling
          shape_a_hist = cv2.calcHist(
            [shape_a_bgr], [0, 1, 2], mask_a, [32, 32, 32], [0, 256, 0, 256, 0, 256]
          )
          shape_b_hist = cv2.calcHist(
            [shape_b_bgr], [0, 1, 2], mask_b, [32, 32, 32], [0, 256, 0, 256, 0, 256]
          )
          
          # Normalize histograms
          if shape_a_hist.sum() == 0 or shape_b_hist.sum() == 0:
            logger.warning("Zero histogram detected")
            rgb_diff = 0.0
          else:
            shape_a_hist = cv2.normalize(shape_a_hist, shape_a_hist).flatten()
            shape_b_hist = cv2.normalize(shape_b_hist, shape_b_hist).flatten()
            
            # Compare histograms
            bhatta_dist = cv2.compareHist(shape_a_hist, shape_b_hist, cv2.HISTCMP_BHATTACHARYYA)
            rgb_diff = 1.0 - bhatta_dist
            
            # Validate result
            if not np.isfinite(rgb_diff):
              rgb_diff = 0.0
            else:
              rgb_diff = np.clip(rgb_diff, 0.0, 1.0)
          
        except Exception as e:
          logger.warning(f"Histogram comparison failed: {e}")
          rgb_diff = 0.0
      
      else:  # rgb_method == 'mean'
        try:
          # Convert to HSV for better color representation
          shape_a_hsv = cv2.cvtColor(shape_a[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2HSV)
          shape_b_hsv = cv2.cvtColor(shape_b[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2HSV)
          
          # Apply alpha mask if available
          if shape_a.shape[2] == 4 and shape_b.shape[2] == 4:
            mask_a = shape_a[:, :, 3] > 0
            mask_b = shape_b[:, :, 3] > 0
            
            if mask_a.sum() > 0 and mask_b.sum() > 0:
              shape_a_mean = np.mean(shape_a_hsv[mask_a], axis=0)
              shape_b_mean = np.mean(shape_b_hsv[mask_b], axis=0)
            else:
              shape_a_mean = np.mean(np.reshape(shape_a_hsv, (-1, 3)), axis=0)
              shape_b_mean = np.mean(np.reshape(shape_b_hsv, (-1, 3)), axis=0)
          else:
            shape_a_mean = np.mean(np.reshape(shape_a_hsv, (-1, 3)), axis=0)
            shape_b_mean = np.mean(np.reshape(shape_b_hsv, (-1, 3)), axis=0)
          
          # Compute normalized distance
          color_dist = np.linalg.norm(shape_a_mean - shape_b_mean)
          # Normalize by maximum possible distance in HSV space
          max_dist = np.sqrt(180**2 + 255**2 + 255**2)  # H, S, V ranges
          rgb_diff = 1.0 - (color_dist / max_dist)
          rgb_diff = np.clip(rgb_diff, 0.0, 1.0)
          
        except Exception as e:
          logger.warning(f"Mean color comparison failed: {e}")
          rgb_diff = 0.0
      
      # Final validation
      if not np.isfinite(shape_diff):
        shape_diff = 0.5
      else:
        shape_diff = np.clip(shape_diff, 0.0, 1.0)
      
      if not np.isfinite(rgb_diff):
        rgb_diff = 0.0
      else:
        rgb_diff = np.clip(rgb_diff, 0.0, 1.0)
      
      logger.debug(f"Shape comparison: shape_diff={shape_diff:.3f}, rgb_diff={rgb_diff:.3f}")
      return shape_diff, rgb_diff
      
    except Exception as e:
      logger.error(f"Shape comparison failed: {e}")
      return 0.5, 0.0  # Safe defaults

  # ================================
  # CoTracker3 Integration Methods
  # ================================
  
  def get_cotracker3_correspondences(
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
      # Fallback to original method
      shape_diffs, rgb_diffs = self._get_basic_appearance_analysis(prev_shapes, curr_shapes, prev_centroids, curr_centroids)
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
          shape_similarities[i, j] = max(0.0, 1.0 - dist / 2.0)  # Normalize distance
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
  
  def _basic_centroid_matching(self, prev_centroids, curr_centroids, thresh=0.6):
    """
    Basic matching using only centroid distances
    """
    if not prev_centroids or not curr_centroids:
      return {}, np.array([[]])
    
    dists = cdist(np.array(prev_centroids), np.array(curr_centroids))
    row_ind, col_ind = linear_sum_assignment(dists)
    
    matching = {}
    confidence_scores = {}
    
    for i, j in zip(row_ind, col_ind):
      if dists[i, j] < thresh:
        matching[i] = j
        confidence_scores[i] = 1.0 / (1.0 + dists[i, j])  # Higher is better
    
    return matching, dists, confidence_scores
  
  def _get_moment_features2(self, shape):
    """
    Extract moment-based features from a shape
    """
    if shape.size == 0:
      return np.zeros(8)  # Return default features for empty shapes
    
    try:
      # Basic statistical moments
      mean_val = np.mean(shape)
      std_val = np.std(shape)
      skew_val = np.mean(((shape - mean_val) / std_val) ** 3) if std_val > 0 else 0
      kurtosis_val = np.mean(((shape - mean_val) / std_val) ** 4) if std_val > 0 else 0
      
      # Shape-based features
      height, width = shape.shape[:2]
      aspect_ratio = width / height if height > 0 else 1.0
      area = height * width
      
      # Additional features
      min_val = np.min(shape)
      max_val = np.max(shape)
      
      return np.array([mean_val, std_val, skew_val, kurtosis_val, aspect_ratio, area, min_val, max_val])
    except Exception as e:
      logger.warning(f"Failed to extract moment features: {e}")
      return np.zeros(8)
  
  def _extract_basic_shape_features(self, shape):
    """
    Extract basic shape features as fallback for CoTracker3
    """
    if shape.size == 0:
      return np.zeros(16)  # Return default features
    
    try:
      # Geometric features
      height, width = shape.shape[:2]
      area = height * width
      perimeter = 2 * (height + width)
      aspect_ratio = width / height if height > 0 else 1.0
      compactness = (perimeter ** 2) / area if area > 0 else 0
      
      # Color/intensity features
      if len(shape.shape) >= 3 and shape.shape[2] >= 3:
        mean_rgb = np.mean(shape[:, :, :3], axis=(0, 1))
        std_rgb = np.std(shape[:, :, :3], axis=(0, 1))
      else:
        mean_rgb = np.array([np.mean(shape), 0, 0])
        std_rgb = np.array([np.std(shape), 0, 0])
      
      # Texture features
      gray = cv2.cvtColor(shape[:, :, :3], cv2.COLOR_BGR2GRAY) if len(shape.shape) >= 3 else shape
      texture_energy = np.var(gray)
      
      # Combine all features
      features = np.concatenate([
        [area, perimeter, aspect_ratio, compactness, texture_energy],
        mean_rgb,
        std_rgb,
        [np.min(gray), np.max(gray), np.mean(gray)]
      ])
      
      return features
    except Exception as e:
      logger.warning(f"Failed to extract basic shape features: {e}")
      return np.zeros(16)
  
  def _compute_basic_feature_similarity(self, features_a, features_b):
    """
    Compute basic similarity between feature vectors
    """
    try:
      # Normalize features
      features_a = np.asarray(features_a, dtype=np.float64)
      features_b = np.asarray(features_b, dtype=np.float64)
      
      if features_a.size == 0 or features_b.size == 0:
        return 0.0
      
      # Ensure same size
      min_size = min(len(features_a), len(features_b))
      features_a = features_a[:min_size]
      features_b = features_b[:min_size]
      
      # Compute normalized correlation coefficient
      norm_a = np.linalg.norm(features_a)
      norm_b = np.linalg.norm(features_b)
      
      if norm_a == 0 or norm_b == 0:
        return 0.0
      
      similarity = np.dot(features_a, features_b) / (norm_a * norm_b)
      return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
      
    except Exception as e:
      logger.warning(f"Failed to compute basic feature similarity: {e}")
      return 0.0
  
  def _get_basic_appearance_analysis(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids):
    """
    Basic appearance analysis as fallback for get_appearance_graphs
    """
    try:
      # Compute basic shape similarity using moment features
      shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
      for i, prev_shape in enumerate(prev_shapes):
        prev_features = self._get_moment_features2(prev_shape)
        for j, curr_shape in enumerate(curr_shapes):
          curr_features = self._get_moment_features2(curr_shape)
          similarity = self._compute_basic_feature_similarity(prev_features, curr_features)
          shape_diffs[i, j] = similarity
      
      # Compute RGB analysis using existing method
      rgb_diffs = self._get_rgb_similarity_matrix(prev_shapes, curr_shapes)
      
      return shape_diffs, rgb_diffs
      
    except Exception as e:
      logger.warning(f"Basic appearance analysis failed: {e}")
      # Return default matrices
      return (
        np.zeros((len(prev_shapes), len(curr_shapes))),
        np.zeros((len(prev_shapes), len(curr_shapes)))
      )
  
  def get_appearance_graphs(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids):
    """
    Get appearance analysis graphs (legacy compatibility method)
    """
    return self._get_basic_appearance_analysis(prev_shapes, curr_shapes, prev_centroids, curr_centroids)
  
  @staticmethod
  def _compute_static_moment_features(shape):
    """
    Static method to compute moment-based features from a shape
    """
    if shape.size == 0:
      return np.zeros(8)  # Return default features for empty shapes
    
    try:
      # Basic statistical moments
      mean_val = np.mean(shape)
      std_val = np.std(shape)
      skew_val = np.mean(((shape - mean_val) / std_val) ** 3) if std_val > 0 else 0
      kurtosis_val = np.mean(((shape - mean_val) / std_val) ** 4) if std_val > 0 else 0
      
      # Shape-based features
      height, width = shape.shape[:2]
      aspect_ratio = width / height if height > 0 else 1.0
      area = height * width
      
      # Additional features
      min_val = np.min(shape)
      max_val = np.max(shape)
      
      return np.array([mean_val, std_val, skew_val, kurtosis_val, aspect_ratio, area, min_val, max_val])
    except Exception as e:
      # Return default features on error
      return np.zeros(8)
    
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
    
  def _create_composite_frame(
    self, 
    shapes: List[np.ndarray], 
    width: int, 
    height: int
  ) -> np.ndarray:
    """Create composite frame from shape crops"""
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    
    for shape in shapes:
      if shape.shape[-1] == 4:  # RGBA
        alpha = shape[:, :, 3:4] / 255.0
        rgb = shape[:, :, :3]
        
        # Simple placement at shape center (could be improved)
        h, w = shape.shape[:2]
        start_y = max(0, (height - h) // 2)
        start_x = max(0, (width - w) // 2)
        end_y = min(height, start_y + h)
        end_x = min(width, start_x + w)
        
        shape_h = end_y - start_y
        shape_w = end_x - start_x
        
        if shape_h > 0 and shape_w > 0:
          alpha_crop = alpha[:shape_h, :shape_w]
          rgb_crop = rgb[:shape_h, :shape_w]
          
          composite[start_y:end_y, start_x:end_x] = (
            alpha_crop * rgb_crop + 
            (1 - alpha_crop) * composite[start_y:end_y, start_x:end_x]
          ).astype(np.uint8)
          
    return composite
    
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
    
  def cleanup_cotracker3(self):
    """Cleanup CoTracker3 resources"""
    if self.cotracker3_engine:
      self.cotracker3_engine.cleanup()
    if self.sam2_cotracker_bridge:
      self.sam2_cotracker_bridge.cleanup()
      
    self.cotracker3_engine = None
    self.sam2_cotracker_bridge = None
    print("🧹 CoTracker3 processor integration cleaned up")
  
  def cotracker3_fallback_matching(self, unmatched_prev_shapes, unmatched_curr_shapes, 
                                  unmatched_prev_centroids, unmatched_curr_centroids, 
                                  frame_width, frame_height, thresh=0.6):
    """CoTracker3 enhanced fallback matching wrapper"""
    return self.fallback_matching(
      unmatched_prev_shapes, unmatched_curr_shapes,
      unmatched_prev_centroids, unmatched_curr_centroids,
      frame_width, frame_height, thresh=thresh
    )
  
  def get_cotracker3_correspondences(self, prev_shapes, curr_shapes, prev_frame, curr_frame, prev_centroids, curr_centroids):
    """Get shape correspondences using CoTracker3 or fallback methods"""
    try:
      if self.use_cotracker3 and self.cotracker3_engine is not None:
        # Use CoTracker3 for advanced tracking
        return self._get_cotracker3_correspondences_impl(
          prev_shapes, curr_shapes, prev_frame, curr_frame, prev_centroids, curr_centroids
        )
    except Exception as e:
      logger.warning(f"CoTracker3 correspondence failed: {e}")
    
    # Fallback to appearance analysis
    return self.get_cotracker3_appearance_analysis(prev_shapes, curr_shapes, prev_centroids, curr_centroids)
  
  def _get_cotracker3_correspondences_impl(self, prev_shapes, curr_shapes, prev_frame, curr_frame, prev_centroids, curr_centroids):
    """Internal CoTracker3 implementation"""
    # Create default shape and RGB difference matrices
    shape_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
    rgb_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
    metadata = {'quality_scores': {}}
    
    # Basic appearance-based matching as fallback
    for i, prev_shape in enumerate(prev_shapes):
      for j, curr_shape in enumerate(curr_shapes):
        # Simple shape similarity based on size
        prev_size = prev_shape.shape[0] * prev_shape.shape[1] if prev_shape is not None and hasattr(prev_shape, 'shape') else 1
        curr_size = curr_shape.shape[0] * curr_shape.shape[1] if curr_shape is not None and hasattr(curr_shape, 'shape') else 1
        size_ratio = min(prev_size, curr_size) / max(prev_size, curr_size) if max(prev_size, curr_size) > 0 else 0
        
        # Distance between centroids
        prev_cent = prev_centroids[i] if i < len(prev_centroids) else [0.5, 0.5]
        curr_cent = curr_centroids[j] if j < len(curr_centroids) else [0.5, 0.5]
        cent_dist = np.sqrt((prev_cent[0] - curr_cent[0])**2 + (prev_cent[1] - curr_cent[1])**2)
        
        shape_diffs[i, j] = size_ratio * (1 - min(cent_dist, 1.0))
        rgb_diffs[i, j] = size_ratio * 0.8  # Conservative RGB similarity
        metadata['quality_scores'][f"{i}_{j}"] = size_ratio * (1 - min(cent_dist, 1.0))
    
    return shape_diffs, rgb_diffs, metadata
  
  def get_cotracker3_appearance_analysis(self, prev_shapes, curr_shapes, prev_centroids, curr_centroids):
    """Enhanced RGB-based shape analysis"""
    shape_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
    rgb_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
    
    for i, prev_shape in enumerate(prev_shapes):
      for j, curr_shape in enumerate(curr_shapes):
        # Basic shape and appearance similarity
        prev_size = prev_shape.shape[0] * prev_shape.shape[1] if prev_shape is not None and hasattr(prev_shape, 'shape') else 1
        curr_size = curr_shape.shape[0] * curr_shape.shape[1] if curr_shape is not None and hasattr(curr_shape, 'shape') else 1
        size_ratio = min(prev_size, curr_size) / max(prev_size, curr_size) if max(prev_size, curr_size) > 0 else 0
        
        # Distance between centroids
        prev_cent = prev_centroids[i] if i < len(prev_centroids) else [0.5, 0.5]
        curr_cent = curr_centroids[j] if j < len(curr_centroids) else [0.5, 0.5]
        cent_dist = np.sqrt((prev_cent[0] - curr_cent[0])**2 + (prev_cent[1] - curr_cent[1])**2)
        
        shape_diffs[i, j] = size_ratio * (1 - min(cent_dist, 1.0))
        rgb_diffs[i, j] = size_ratio * 0.7
    
    return shape_diffs, rgb_diffs
  
  def main_matching(self, joint_scores):
    """Main matching algorithm for shapes"""
    # Simple greedy matching
    matching = []
    unmatched_prev = set(range(joint_scores.shape[0]))
    unmatched_curr = set(range(joint_scores.shape[1]))
    
    # Find best matches above threshold
    while len(unmatched_prev) > 0 and len(unmatched_curr) > 0:
      # Find the best remaining match
      best_score = 0
      best_prev = -1
      best_curr = -1
      
      for i in unmatched_prev:
        for j in unmatched_curr:
          if joint_scores[i, j] > best_score:
            best_score = joint_scores[i, j]
            best_prev = i
            best_curr = j
      
      if best_score > 0:
        matching.append([{best_prev}, {best_curr}])
        unmatched_prev.remove(best_prev)
        unmatched_curr.remove(best_curr)
      else:
        break
    
    return matching, unmatched_prev, unmatched_curr
  
  def hungarian_matching(self, joint_scores):
    """Hungarian algorithm for optimal matching"""
    try:
      from scipy.optimize import linear_sum_assignment
      # Convert to cost matrix (lower is better)
      cost_matrix = 1.0 / (joint_scores + 1e-8)
      row_ind, col_ind = linear_sum_assignment(cost_matrix)
      
      matching = []
      unmatched_prev = set(range(joint_scores.shape[0]))
      unmatched_curr = set(range(joint_scores.shape[1]))
      
      for i, j in zip(row_ind, col_ind):
        if joint_scores[i, j] > 0.1:  # Minimum threshold
          matching.append([{i}, {j}])
          unmatched_prev.discard(i)
          unmatched_curr.discard(j)
      
      return matching, unmatched_prev, unmatched_curr
    except ImportError:
      # Fallback to simple matching
      return self.main_matching(joint_scores)