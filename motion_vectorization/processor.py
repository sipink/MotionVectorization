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
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from .utils import warp_flo
from .shape_context import ShapeContext

# Import CoTracker3 components
try:
    from .cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config, create_cotracker3_engine
    from .sam2_cotracker_bridge import SAM2CoTrackerBridge, create_sam2_cotracker_bridge
    COTRACKER3_AVAILABLE = True
except ImportError:
    COTRACKER3_AVAILABLE = False
    warnings.warn("CoTracker3 integration not available. Using fallback tracking.")


def _get_moment_features3(sc, img, color=None):
  points, _ = sc.get_points_from_img(img[:, :, 3])
  desc = sc.compute(np.array(points))
  return desc


class Processor():
  def __init__(self, use_cotracker3=False, cotracker3_mode="offline", device="auto"): 
    self.sc = ShapeContext()
    
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
    '''
    labels: NxHxW
    flow: Nx2xHxW
    '''
    all_warped_labels = []
    for (labels, fg, flow) in zip([curr_labels, prev_labels], [prev_labels, curr_labels], [forw_flo, back_flo]):
      labels_tensor = torch.tensor(labels[..., None] + 1).permute(2, 0, 1)[None, ...].float()
      pad_h = labels.shape[0] - flow.shape[0]
      pad_w = labels.shape[1] - flow.shape[1]
      if pad_h < 0:
        flow = flow[-pad_h // 2:pad_h + (-pad_h) // 2]
      if pad_w < 0:
        flow = flow[:, -pad_w // 2:pad_w + (-pad_w) // 2]
      pad_h = max(0, pad_h)
      pad_w = max(0, pad_w)
      pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
      flow_pad = np.pad(flow, pad)
      flow_tensor = torch.tensor(flow_pad, dtype=torch.float32).permute(2, 0, 1)[None, ...]
      warped_labels = warp_flo(labels_tensor, flow_tensor) * np.uint8(fg>=0)[..., None]
      warped_labels = np.int32(warped_labels[:, :, 0] - 1)
      all_warped_labels.append(warped_labels)
    curr_warped_labels, prev_warped_labels = all_warped_labels
    curr_warped_labels[prev_labels<0] = -1
    prev_warped_labels[curr_labels<0] = -1
    return curr_warped_labels, prev_warped_labels

  @staticmethod
  def compute_match_graphs(curr_labels, prev_labels, curr_fg_labels, prev_fg_labels, curr_warped_labels, prev_warped_labels):
    curr_labels_stack = np.tile(curr_fg_labels, [len(curr_labels), 1, 1])
    curr_labels_stack = np.reshape(curr_labels_stack, (len(curr_labels), -1))
    prev_labels_stack = np.tile(prev_fg_labels, [len(prev_labels), 1, 1])
    prev_labels_stack = np.reshape(prev_labels_stack, (len(prev_labels), -1))
    curr_labels_mask = np.uint8(curr_labels_stack==curr_labels[..., None]) / 1.0
    prev_labels_mask = np.uint8(prev_labels_stack==prev_labels[..., None]) / 1.0
    curr_warped_labels_stack = np.tile(curr_warped_labels, [len(curr_labels), 1, 1])
    curr_warped_labels_stack = np.reshape(curr_warped_labels_stack, (len(curr_labels), -1))
    prev_warped_labels_stack = np.tile(prev_warped_labels, [len(prev_labels), 1, 1])
    prev_warped_labels_stack = np.reshape(prev_warped_labels_stack, (len(prev_labels), -1))
    curr_warped_labels_mask = np.uint8(curr_warped_labels_stack==curr_labels[..., None]) / 1.0
    prev_warped_labels_mask = np.uint8(prev_warped_labels_stack==prev_labels[..., None]) / 1.0
    prev_total = np.sum(prev_warped_labels_mask, axis=1)
    curr_total = np.sum(curr_warped_labels_mask, axis=1)
    prev_in_curr = prev_warped_labels_mask @ curr_labels_mask.T / prev_total[..., None]
    prev_in_curr[np.isnan(prev_in_curr)] = 0.0
    curr_in_prev = curr_warped_labels_mask @ prev_labels_mask.T / curr_total[..., None]
    curr_in_prev[np.isnan(curr_in_prev)] = 0.0
    return prev_in_curr, curr_in_prev

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

  @staticmethod
  def get_appearance_graphs(prev_shapes, curr_shapes, prev_centroids, curr_centroids):
    # Compute shape features.
    sc = ShapeContext()
    prev_shape_features = []
    for prev_shape in prev_shapes:
      feature = _get_moment_features3(sc, prev_shape)
      prev_shape_features.append(feature)
    curr_shape_features = []
    for curr_shape in curr_shapes:
      feature = _get_moment_features3(sc, curr_shape)
      curr_shape_features.append(feature)
    shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    t0 = time.perf_counter()
    for i in range(len(prev_shapes)):
      for j in range(len(curr_shapes)):
        shape_diffs[i, j] = np.exp(-sc.diff(prev_shape_features[i], curr_shape_features[j], idxs=False))
    t1 = time.perf_counter()

    # Compute shape RGB histograms.
    rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    for i, prev_shape in enumerate(prev_shapes):
      prev_shape_lab = cv2.cvtColor(prev_shape[:, :, :3], cv2.COLOR_BGR2LAB)
      prev_hist = cv2.calcHist(
        [prev_shape_lab[:, :, :3]], [0, 1, 2], prev_shape[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
      for j, curr_shape in enumerate(curr_shapes):
        curr_shape_lab = cv2.cvtColor(curr_shape[:, :, :3], cv2.COLOR_BGR2LAB)
        curr_hist = cv2.calcHist(
          [curr_shape_lab[:, :, :3]], [0, 1, 2], curr_shape[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        rgb_diff = 1.0 - cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        rgb_diffs[i, j] = rgb_diff

    return shape_diffs, rgb_diffs

  @staticmethod
  def fallback_matching(prev_shapes, curr_shapes, prev_centroids, curr_centroids, frame_width, frame_height, thresh=0.6):
    """Original fallback matching method (now legacy - use cotracker3_fallback_matching)"""
    shape_diffs, rgb_diffs = Processor().get_appearance_graphs(
      prev_shapes, curr_shapes, prev_centroids, curr_centroids)

    # Compute centroid distances.
    dists = cdist(np.array(prev_centroids), np.array(curr_centroids))
    costs = (-np.log(shape_diffs) + (1.0 - rgb_diffs) + dists) / 3
    row_ind, col_ind = linear_sum_assignment(costs)
    matching = {}
    for i, j in zip(row_ind, col_ind):
      if j >= 0 and costs[i, j] < thresh:
        matching[i] = j
    return matching, costs

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
    shape_a_features = _get_moment_features2(shape_a)
    shape_b_features = _get_moment_features2(shape_b)
    # shape_diff = 1.0 - shape_a_features @ shape_b_features.T
    shape_diff = np.exp(-np.abs(shape_a_features - shape_b_features))

    if rgb_method == 'hist':
      shape_a_hist = cv2.calcHist(
        [shape_a[:, :, :3]], [0, 1, 2], shape_a[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      shape_a_hist = cv2.normalize(shape_a_hist, shape_a_hist).flatten()
      shape_b_hist = cv2.calcHist(
        [shape_b[:, :, :3]], [0, 1, 2], shape_b[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      shape_b_hist = cv2.normalize(shape_b_hist, shape_b_hist).flatten()
      rgb_diff = 1.0 - cv2.compareHist(shape_a_hist, shape_b_hist, cv2.HISTCMP_BHATTACHARYYA)
    else:
      shape_a_lab = cv2.cvtColor(shape_a[:, :, :3], cv2.COLOR_BGR2HSV)
      shape_b_lab = cv2.cvtColor(shape_b[:, :, :3], cv2.COLOR_BGR2HSV)
      shape_a_lab = np.mean(np.reshape(shape_a_lab, (-1, 3)), axis=0)
      shape_b_lab = np.mean(np.reshape(shape_b_lab, (-1, 3)), axis=0)
      rgb_diff = np.linalg.norm(shape_a_lab - shape_b_lab)

    return shape_diff, rgb_diff

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
      return self.get_appearance_graphs(prev_shapes, curr_shapes, prev_centroids, curr_centroids)
      
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
        # Track points across the two frames
        tracks, visibility = self.cotracker3_engine.track_video_grid(
          video_tensor, custom_points=prev_points
        )
        
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
          shape_similarities[i, j] = max(0, 1.0 - dist / 2.0)  # Normalize distance
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
  
  def cotracker3_fallback_matching(
    self,
    prev_shapes: List[np.ndarray],
    curr_shapes: List[np.ndarray],
    prev_centroids: List[Tuple[float, float]],
    curr_centroids: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int,
    thresh: float = 0.6
  ) -> Tuple[Dict[int, int], np.ndarray]:
    """
    CoTracker3-enhanced fallback matching for unmatched shapes
    
    This method replaces the original fallback_matching with CoTracker3
    capabilities while maintaining the same API.
    
    Args:
      prev_shapes: Previous frame shapes
      curr_shapes: Current frame shapes  
      prev_centroids: Previous frame centroids
      curr_centroids: Current frame centroids
      frame_width: Frame width for normalization
      frame_height: Frame height for normalization
      thresh: Matching threshold
      
    Returns:
      matching: Dictionary mapping prev_idx -> curr_idx
      costs: Cost matrix for all pairs
    """
    if not self.use_cotracker3:
      # Use original fallback matching
      return self.fallback_matching(
        prev_shapes, curr_shapes, prev_centroids, curr_centroids,
        frame_width, frame_height, thresh
      )
      
    print("🔄 CoTracker3 enhanced fallback matching")
    
    # Create mini-video from shape crops for tracking
    prev_frame = self._create_composite_frame(prev_shapes, frame_width, frame_height)
    curr_frame = self._create_composite_frame(curr_shapes, frame_width, frame_height)
    
    # Get CoTracker3 correspondences
    shape_similarities, motion_scores, metadata = self.get_cotracker3_correspondences(
      prev_shapes, curr_shapes, prev_frame, curr_frame,
      prev_centroids, curr_centroids
    )
    
    # Compute centroid distances (normalized)
    dists = cdist(np.array(prev_centroids), np.array(curr_centroids))
    dists_norm = dists / np.sqrt(frame_width**2 + frame_height**2)
    
    # Enhanced cost function combining CoTracker3 features
    costs = (
      (1.0 - shape_similarities) * 0.4 +  # Shape tracking similarity
      (1.0 - motion_scores) * 0.4 +       # Motion consistency 
      dists_norm * 0.2                    # Spatial distance
    )
    
    # Apply Hungarian matching with CoTracker3 quality weighting
    row_ind, col_ind = linear_sum_assignment(costs)
    matching = {}
    
    for i, j in zip(row_ind, col_ind):
      if j >= 0 and costs[i, j] < thresh:
        # Additional quality check using CoTracker3 metrics
        quality_score = metadata['quality_scores'].get(i, 0.0)
        if quality_score > 0.3:  # Minimum quality threshold
          matching[i] = j
          
    print(f"✅ CoTracker3 fallback matching: {len(matching)} correspondences found")
    return matching, costs
    
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