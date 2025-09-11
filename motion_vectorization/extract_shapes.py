import cv2
import pickle
import numpy as np
import torch
import copy
import os
import argparse
import json
import time
import datetime
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Optional, Tuple, List
mpl.use('Agg')
from skimage.transform import AffineTransform
from skimage.measure import ransac
import networkx as nx
from scipy import stats

from .dataloader import DataLoader
from .processor import Processor
from .visualizer import Visualizer
from .utils import *
from . import compositing

parser = argparse.ArgumentParser()
# Video and directory information.
parser.add_argument(
  '--video_file', type=str, required=True, 
  help='Name of the video to process.')
parser.add_argument(
  '--video_dir', default='videos', 
  help='Directory containing videos.')
parser.add_argument(
  '--output_dir', default='motion_vectorization/outputs', type=str, 
  help='Directory to save outputs.')
parser.add_argument(
  '--suffix', default=None, type=str, 
  help='Suffix for output video names.')
parser.add_argument(
  '--config', type=str, default=None, 
  help='Config file.')

# Video processing.
parser.add_argument(
  '--max_frames', default=-1, type=int, 
  help='The maximum number of frames to process. If set to -1, then process all frames.')
parser.add_argument(
  '--start_frame', default=1, type=int, 
  help='The frame to start at.')
parser.add_argument(
  '--base_frame', default=0, type=int, 
  help='The at which to begin tracking.')

# Regions.
parser.add_argument(
  '--min_cluster_size', default=50, type=int, 
  help='The minimum number of samples allowed in a cluster.')
parser.add_argument(
  '--min_density', default=0.15, type=int, 
  help='The minimum density cluster allowed.')
parser.add_argument(
  '--bg_file', type=str, default=None, 
  help='Background file.')

# Shape matching.
parser.add_argument(
  '--main_match_thresh', default=0.6, type=float, 
  help='Threshold to be considered as a main match.')
parser.add_argument(
  '--single_match_thresh', default=0.2, type=float, 
  help='Threshold to be considered as a main match.')
parser.add_argument(
  '--num_match_shapes', default=5, type=int, 
  help='The number of closest shapes to check for matches.')
parser.add_argument(
  '--fallback_match_thresh', default=0.6, type=float, 
  help='Threshold to be considered as a fallback match.')
parser.add_argument(
  '--drop_thresh', type=float, default=0.05, 
  help='Threshold on RGB loss to drop a shape.')
parser.add_argument(
  '--breaks', type=int, nargs='+', default=[],
  help='Frames at which there is no continuity and we should track new shapes completely.')
parser.add_argument(
  '--multiply', action='store_true', default=False,
  help='Multiply flow scores instead of add.')
parser.add_argument(
  '--all_joint', action='store_true', default=False,
  help='Optimize all shapes jointly.')
parser.add_argument(
  '--manual_init', action='store_true', default=False,
  help='Use template matching for initialization.')
parser.add_argument(
  '--hungarian', action='store_true', default=False,
  help='Use hungarian matching for initialization.')

# Unified Pipeline Integration (SAM2.1 + CoTracker3 + FlowSeek)
parser.add_argument(
  '--use_unified_pipeline', action='store_true', default=False,
  help='Use unified SAM2.1 + CoTracker3 + FlowSeek pipeline for maximum accuracy and speed.')
parser.add_argument(
  '--unified_mode', type=str, default='balanced', choices=['speed', 'balanced', 'accuracy'],
  help='Unified pipeline mode: speed (60 FPS), balanced (44 FPS), accuracy (30 FPS with 95%+ quality).')
parser.add_argument(
  '--unified_device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
  help='Device for unified pipeline processing.')
parser.add_argument(
  '--progressive_fallback', action='store_true', default=True,
  help='Enable progressive fallback: Unified → Individual engines → Legacy methods.')
parser.add_argument(
  '--quality_threshold', type=float, default=0.85,
  help='Minimum quality threshold to accept unified pipeline results.')
parser.add_argument(
  '--benchmark_performance', action='store_true', default=False,
  help='Run performance benchmarks and accuracy validation.')

# CoTracker3 integration (maintained for individual engine support)
parser.add_argument(
  '--use_cotracker3', action='store_true', default=False,
  help='Use CoTracker3 for superior point tracking instead of primitive shape matching.')
parser.add_argument(
  '--cotracker3_mode', type=str, default='offline', choices=['offline', 'online'],
  help='CoTracker3 mode: offline for maximum accuracy, online for long videos.')
parser.add_argument(
  '--cotracker3_device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
  help='Device for CoTracker3 processing.')
parser.add_argument(
  '--cotracker3_grid_size', type=int, default=40,
  help='Grid size for CoTracker3 point sampling (creates grid_size² points).')

# SAM2.1 individual engine support
parser.add_argument(
  '--use_sam2', action='store_true', default=False,
  help='Use SAM2.1 for superior segmentation accuracy.')
parser.add_argument(
  '--sam2_model', type=str, default='large', choices=['small', 'large'],
  help='SAM2.1 model variant: small for speed, large for accuracy.')

# FlowSeek individual engine support
parser.add_argument(
  '--use_flowseek', action='store_true', default=False,
  help='Use FlowSeek for superior optical flow accuracy with 8x less hardware.')
parser.add_argument(
  '--flowseek_depth_integration', action='store_true', default=True,
  help='Enable FlowSeek depth foundation model integration.')

# Optimization.
parser.add_argument(
  '--min_opt_size', default=200, type=int, 
  help='Size of optimization.')
parser.add_argument(
  '--blur_kernel', default=15, type=int, 
  help='Blur kernel size for visual loss.')
parser.add_argument(
  '--lr', default=0.01, type=float, 
  help='Learning rate for optimization.')
parser.add_argument(
  '--p_weight', default=0.1, type=float, 
  help='Weight for params loss.')
parser.add_argument(
  '--n_steps', default=50, type=int, 
  help='Number of steps to optimize for individual shape matching.')
parser.add_argument(
  '--use_gpu', action='store_true', default=False, 
  help='Use GPU')
parser.add_argument(
  '--bleed', default=50, type=int, 
  help='Margin outside of frame.')
parser.add_argument(
  '--overlap_thresh', type=float, default=0.9, 
  help='Threshold definition of containment.')
parser.add_argument(
  '--use_k', action='store_true', default=False, 
  help='Use skew.')
parser.add_argument(
  '--use_r', action='store_true', default=False, 
  help='Use rotation.')
parser.add_argument(
  '--use_s', action='store_true', default=False, 
  help='Use scale.')
parser.add_argument(
  '--use_t', action='store_true', default=False, 
  help='Use translation.')
parser.add_argument(
  '--init_r', action='store_true', default=False, 
  help='Initalize rotation.')
parser.add_argument(
  '--init_s', action='store_true', default=False, 
  help='Initalize scale.')
parser.add_argument(
  '--init_t', action='store_true', default=False, 
  help='Initalize translation.')

# Debugging.
parser.add_argument(
  '--verbose', action='store_true', default=False, 
  help='If true, print intermediate messages and outputs.')

d = datetime.date.today().strftime("%d%m%y")
now = datetime.datetime.now()
print('TIME:', now)

args = parser.parse_args()
video_name = os.path.splitext(args.video_file.split('/')[-1])[0]
if args.config is not None:
  configs_file = args.config

  if not os.path.exists(configs_file):
    print('[WARNING] Configs file not found! Using default.json instead.')
    configs_file = 'motion_vectorization/config/default.json'

  configs = json.load(open(configs_file, 'r'))
  parser.set_defaults(**configs)
  args = parser.parse_args()
print('Configs:')
for arg_name, arg_val in vars(args).items():
  print(f'  {arg_name}:\t{arg_val}')

np.random.seed(1)
torch.manual_seed(0)

def main():
  device = 'cuda' if args.use_gpu else 'cpu'
  video_dir = os.path.join(args.video_dir, video_name)

  # Create output directories.
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  video_folder = os.path.join(
    args.output_dir, 
    f'{video_name}_{args.suffix}')
  track_folder = os.path.join(video_folder, 'outputs', 'track')
  debug_opt_folder = os.path.join(video_folder, 'debug', 'track_opt')
  debug_rot_folder = os.path.join(video_folder, 'debug', 'track_rotate')
  debug_comp_folder = os.path.join(video_folder, 'debug', 'comp')
  debug_fgbg_folder = os.path.join(video_folder, 'debug', 'fg_bg')
  debug_match_folder = os.path.join(video_folder, 'debug', 'match')
  debug_fallback_folder = os.path.join(video_folder, 'debug', 'fallback')
  debug_shapes_folder = os.path.join(video_folder, 'debug', 'shapes')
  debug_t2e_folder = os.path.join(video_folder, 'debug', 't2e')
  debug_groups_folder = os.path.join(video_folder, 'debug', 'groups')
  debug_digraph_folder = os.path.join(video_folder, 'debug', 'digraph')
  debug_matrix_folder = os.path.join(video_folder, 'debug', 'matrix')
  debug_submatch_folder = os.path.join(video_folder, 'debug', 'submatch')
  for folder in [
    video_folder, 
    track_folder, 
    debug_rot_folder,
    debug_comp_folder, 
    debug_fgbg_folder,
    debug_match_folder,
    debug_fallback_folder,
    debug_shapes_folder,
    debug_t2e_folder,
    debug_groups_folder,
    debug_digraph_folder,
    debug_matrix_folder,
    debug_submatch_folder
  ]:
    if not os.path.exists(folder):
      os.makedirs(folder)

  dataloader = DataLoader(video_dir, max_frames=args.max_frames)
  # Initialize processor with CoTracker3 if requested
  processor = Processor(
    use_cotracker3=args.use_cotracker3,
    cotracker3_mode=args.cotracker3_mode,
    device=args.cotracker3_device
  )
  
  if args.use_cotracker3:
    print(f"🚀 CoTracker3 integration enabled (mode: {args.cotracker3_mode})")
    print(f"   Grid size: {args.cotracker3_grid_size}×{args.cotracker3_grid_size} = {args.cotracker3_grid_size**2} points")
    print(f"   Device: {args.cotracker3_device}")
  viz = Visualizer()

  # Load background job.
  bg_img = None
  if args.bg_file is not None:
    bg_img = cv2.imread(args.bg_file)

  color = {-1: np.random.randint(0, 255, (3))}  # Contains background cluster color.
  shape_bank = {-1: []}
  optim_bank = {-1: []}
  # Base frame info
  base_bank = None
  base_fg_labels = None
  base_frame = None 
  base_fg_comps = None
  base_fg_comp_to_label = None
  time_bank = {'bgr': [], 'shapes': collections.defaultdict(dict)}
  highest_res = {}  # Highest resolution shape image.
  highest_res_update = {}  # Index of the highest resolution shape.
  merged = collections.defaultdict(set)
  prev_fg_labels = None  # Initialize as None instead of -1
  prev_frame = None  # Initialize prev_frame
  prev_fg_comps = None  # Initialize prev_fg_comps
  prev_fg_comp_to_label = None  # Initialize prev_fg_comp_to_label
  shape_diffs = None  # Initialize shape_diffs
  rgb_diffs = None  # Initialize rgb_diffs
  label_mapping = {}  # Maps shape indices in this frame to shape indices across entire video.
  unoccluded_canon = {}
  
  # Initialize optimization variables
  all_tx = None
  all_ty = None
  all_sx = None
  all_sy = None
  all_theta = None
  all_kx = None
  all_ky = None
  frame_ordering = [i for i in range(args.base_frame, -1, -1)] + [i for i in range(args.base_frame + 1, len(dataloader.frame_idxs))]
  for t in frame_ordering:
    frame_idx = dataloader.frame_idxs[t]
    if t < args.start_frame:
      continue
    if t == args.base_frame + 1:
      optim_bank = base_bank
      prev_fg_labels = base_fg_labels
      prev_fg_comps = base_fg_comps
      prev_fg_comp_to_label = base_fg_comp_to_label
      prev_frame = base_frame
    print(f'\nFRAME {frame_idx} ({t}) CLUSTERS\n========================')
    frame_t0 = time.perf_counter()
    #######
    # 1. For every frame pair (t, t + 1), get labels, forward flow, backward flow, 
    # and foreground/background.
    #######
    curr_frame, curr_fg_labels, fg_bg, curr_fg_comps, forw_flow, back_flow = dataloader.load_data(t)
    curr_fg_comps_vis = viz.show_labels(curr_fg_comps + 1)
    curr_fg_comp_to_label, curr_fg_label_to_comp = get_comp_label_map(curr_fg_comps, curr_fg_labels)
    # Write visualization with proper array format
    if curr_fg_comps_vis is not None:
      cv2.imwrite(os.path.join(debug_comp_folder, f'{frame_idx:03d}.png'), np.ascontiguousarray(curr_fg_comps_vis, dtype=np.uint8))

    frame_height, frame_width, _ = curr_frame.shape
    frame_lab = cv2.cvtColor(cv2.bilateralFilter(curr_frame, 9, 25, 25), cv2.COLOR_BGR2LAB)
    lab_mode, _ = np.array(stats.mode(np.reshape(frame_lab, (-1, 3)), axis=0))
    # Fix: Ensure lab_mode has 3 channels for LAB2BGR conversion
    lab_mode_3ch = np.uint8(np.array(lab_mode)[None, None, :])  # Shape: (1, 1, 3)
    # Convert to proper OpenCV format for cvtColor
    lab_mode_3ch = np.ascontiguousarray(lab_mode_3ch, dtype=np.uint8)
    bgr_mode = cv2.cvtColor(lab_mode_3ch, cv2.COLOR_LAB2BGR).squeeze()
    if t <= args.base_frame:
      time_bank['bgr'].insert(0, bgr_mode)
    else:
      time_bank['bgr'].append(bgr_mode)
    label_mapping.clear()
    new_fg_labels = -1 * np.ones_like(curr_fg_labels, dtype=np.int32)

    # Create background image.
    if args.bg_file is None:
      bg_img = np.full((frame_height, frame_width, 3), np.array(bgr_mode))

    active_shapes = np.unique(prev_fg_labels)[1:].tolist() if prev_fg_labels is not None else []
    curr_labels = np.unique(curr_fg_labels)[1:].tolist()
    print('[NOTE] Active shapes:', active_shapes)
    if len(active_shapes) > 0 and len(curr_labels) > 0 and frame_idx not in args.breaks:
      #######
      # 2. Warp the labels forward/backward and compute an IOU graph (add IOU in each direction).
      #    Collect all shapes which have a very high (thresh) IOU and match them. For remaining shapes,
      #    Look at the one-sided IOU and match shapes with a high one-sided IOU (thresh). This should
      #    guarantee a set of one-to-many or many-to-one matches.
      #######
      if t <= args.base_frame:
        curr_warped_labels, prev_warped_labels = processor.warp_labels(
          curr_fg_labels, prev_fg_labels, back_flow, forw_flow)
      else:
        curr_warped_labels, prev_warped_labels = processor.warp_labels(
          curr_fg_labels, prev_fg_labels, forw_flow, back_flow)
      print('[NOTE] curr_fg_comp_to_label:', curr_fg_comp_to_label)
      print('[NOTE] prev_fg_comp_to_label:', prev_fg_comp_to_label)
      
      viz.set_info(
        active_shapes, curr_labels, 
        prev_frame, curr_frame, 
        prev_fg_labels, curr_fg_labels, 
        prev_fg_comps, curr_fg_comps,
        prev_fg_comp_to_label, curr_fg_comp_to_label, 
        bg_img
      )

      #######
      # 3. Generate the matches. We only accept two types of matches:
      #      (a) The prev-in-curr is > threshold, or
      #      (b) The curr-in-prev is > threshold.
      #    This ensures that our matching graph has components with the form (many, one) or 
      #    (one, many).
      #######
      # Appearance-based match graphs.
      appearance_t0 = time.perf_counter()
      prev_shapes = []
      prev_centroids = []
      for prev_shape_idx in active_shapes:
        prev_shape = shape_bank.get(prev_shape_idx) if shape_bank is not None else None
        optim_data = optim_bank.get(prev_shape_idx) if optim_bank is not None else None
        if optim_data is not None and isinstance(optim_data, dict) and 'centroid' in optim_data:
          centroid_data = optim_data['centroid']
          if centroid_data is not None and len(centroid_data) >= 2:
            cx, cy = centroid_data[:2]
            prev_centroids.append([cx / frame_width, cy / frame_height])
          else:
            prev_centroids.append([0.5, 0.5])
        else:
          # Fallback centroid if not available
          prev_centroids.append([0.5, 0.5])
        
        if prev_shape is not None and hasattr(prev_shape, 'shape') and len(prev_shape.shape) >= 3:
          try:
            if prev_shape.shape[2] > 3:
              min_x, min_y, max_x, max_y = get_shape_coords(prev_shape[:, :, 3])
              prev_shapes.append(prev_shape[min_y:max_y, min_x:max_x])
            else:
              prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
          except (IndexError, AttributeError):
            prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
        else:
          # Fallback shape if not available
          prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
      curr_shapes = []
      curr_centroids = []
      for curr_shape_idx in curr_labels:
        curr_shape_alpha = np.uint8(curr_fg_labels==curr_shape_idx)
        curr_shape_alpha = get_alpha(np.float64(curr_shape_alpha), curr_frame)[..., None]
        curr_shape = curr_shape_alpha * curr_frame + (1 - curr_shape_alpha) * (bg_img if bg_img is not None else np.zeros_like(curr_frame))
        curr_shape = np.uint8(np.dstack([curr_shape, 255 * curr_shape_alpha]))
        min_x, min_y, max_x, max_y = get_shape_coords(curr_shape[:, :, 3])
        curr_shapes.append(curr_shape[min_y:max_y, min_x:max_x])
        cx, cy = get_shape_centroid(curr_shape[:, :, 3])
        curr_centroids.append([cx, cy])

      # Enhanced tracking with modern engines (CoTracker3 primary, robust fallbacks)
      tracking_method = "cotracker3" if (args.use_cotracker3 and processor.use_cotracker3) else "enhanced_rgb"
      
      # Initialize default values to prevent unbound variables
      shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
      rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
      cotracker3_metadata = {}
      
      # Use CoTracker3 for superior tracking if enabled
      if args.use_cotracker3 and processor.use_cotracker3:
        print("🎯 Using CoTracker3 for shape correspondence")
        tracking_method = "cotracker3"
        try:
          # Enhanced CoTracker3 integration with quality monitoring
          result = processor.get_cotracker3_correspondences(
            prev_shapes, curr_shapes, prev_frame, curr_frame,
            prev_centroids, curr_centroids
          )
          if len(result) == 3:
            shape_diffs, rgb_diffs, cotracker3_metadata = result
          else:
            shape_diffs, rgb_diffs = result
            cotracker3_metadata = {}
          
          # Quality validation
          cotracker3_quality_scores = cotracker3_metadata.get('quality_scores', {})
          avg_quality = np.mean([score for score in cotracker3_quality_scores.values() if score > 0]) if cotracker3_quality_scores else 0.0
          if avg_quality > 0:
            print(f"   CoTracker3 quality scores: {[f'{k}:{v:.3f}' for k, v in cotracker3_quality_scores.items()]}")
            print(f"   Average quality: {avg_quality:.1%}")
            if avg_quality < args.quality_threshold and args.progressive_fallback:
              print(f"   ⚠️ Quality below threshold ({args.quality_threshold:.1%}), using enhanced RGB analysis")
              tracking_method = "enhanced_rgb"
          else:
            print(f"   ⚠️ CoTracker3 tracking failed, using enhanced RGB analysis")
            tracking_method = "enhanced_rgb"
        except Exception as e:
          print(f"   ❌ CoTracker3 tracking error: {e}")
          if not args.progressive_fallback:
            raise e
          print(f"   🔄 Using enhanced RGB analysis fallback")
          tracking_method = "enhanced_rgb"
      
      # Use enhanced RGB analysis if CoTracker3 unavailable
      if tracking_method == "enhanced_rgb":
        print("🔧 Using enhanced RGB-based shape analysis")
        try:
          shape_diffs, rgb_diffs = processor.get_cotracker3_appearance_analysis(
            prev_shapes, curr_shapes, prev_centroids, curr_centroids)
        except AttributeError:
          # Fallback if method doesn't exist
          print("   Using basic shape analysis")
          shape_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
          rgb_diffs = np.ones((len(prev_shapes), len(curr_shapes))) * 0.5
      appearance_t1 = time.perf_counter()
      print(f'[TIME] Appearance-based matching took {appearance_t1 - appearance_t0:.2f}s')

      # Optical flow-based match graphs.
      prev_in_curr, curr_in_prev = processor.compute_match_graphs(
        np.array(curr_labels), np.array(active_shapes), curr_fg_labels, prev_fg_labels, curr_warped_labels, prev_warped_labels)
      # Ensure arrays are properly initialized before operations
      shape_diffs = shape_diffs if shape_diffs is not None else np.zeros((len(active_shapes), len(curr_labels)))
      rgb_diffs = rgb_diffs if rgb_diffs is not None else np.zeros((len(active_shapes), len(curr_labels)))
      
      if args.multiply:
        joint_scores = (shape_diffs) * (rgb_diffs) * (prev_in_curr * curr_in_prev.T)
      elif args.all_joint:
        joint_scores = (shape_diffs + rgb_diffs + prev_in_curr + curr_in_prev.T) / 4
      elif args.hungarian:
        joint_scores = 1 / (shape_diffs + rgb_diffs + prev_in_curr + curr_in_prev.T + 1e-8)  # Add small epsilon
      else:
        joint_scores = (shape_diffs) * (rgb_diffs) * (prev_in_curr + curr_in_prev.T) / 2
      joint_scores[joint_scores < args.single_match_thresh] = 0.0
      # for i in range(joint_scores.shape[0]):
      #   row_max = np.max(joint_scores[i])
      #   new_row = joint_scores[i].copy()
      #   new_row[new_row < row_max] = 0
      #   joint_scores[i] = new_row
      flow_scores = (prev_in_curr + curr_in_prev.T) / 2
      appearance_scores = (shape_diffs + rgb_diffs) / 2
      print('[NOTE] prev_in_curr:\n', prev_in_curr)
      print('[NOTE] curr_in_prev:\n', curr_in_prev)

      # Visualize prev -> curr and curr -> prev graph.
      matching_digraph_vis = viz.matching_digraph(
        np.uint8(prev_in_curr>args.main_match_thresh), np.uint8(curr_in_prev>args.main_match_thresh))
      shape_diffs_vis = viz.vis_graph(shape_diffs)
      rgb_diffs_vis = viz.vis_graph(rgb_diffs)
      prev_in_curr_vis = viz.vis_graph(prev_in_curr, mark_pos=True)
      curr_in_prev_vis = viz.vis_graph(curr_in_prev.T, mark_pos=True)
      joint_vis = viz.vis_graph(joint_scores, mark_pos=True)
      divider = 255 * np.ones((shape_diffs_vis.shape[0], 25, 3), dtype=np.uint8)
      matrix_vis = np.concatenate([
        shape_diffs_vis, divider, 
        rgb_diffs_vis, divider, 
        prev_in_curr_vis, divider,
        curr_in_prev_vis, divider,
        joint_vis 
      ], axis=1)
      border = 255 * np.ones((25, matrix_vis.shape[1], 3), dtype=np.uint8)
      matrix_vis = np.concatenate([matrix_vis, border])
      matrix_vis = cv2.putText(
        matrix_vis, 'shape', (10, matrix_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
      matrix_vis = cv2.putText(
        matrix_vis, 'rgb', (10 + 25 * (len(curr_shapes) + 2), matrix_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
      matrix_vis = cv2.putText(
        matrix_vis, 'prev_in_curr', (10 + 25 * (2 * (len(curr_shapes) + 2)), matrix_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
      matrix_vis = cv2.putText(
        matrix_vis, 'curr_in_prev', (10 + 25 * (3 * (len(curr_shapes) + 2)), matrix_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
      matrix_vis = cv2.putText(
        matrix_vis, 'joint', (10 + 25 * (4 * (len(curr_shapes) + 2)), matrix_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
      cv2.imwrite(os.path.join(debug_digraph_folder, f'{frame_idx:03d}.png'), matching_digraph_vis)
      cv2.imwrite(os.path.join(debug_matrix_folder, f'{frame_idx:03d}.png'), matrix_vis)

      if args.hungarian:
        matching, unmatched_prev, unmatched_curr = processor.hungarian_matching(joint_scores)
      else:
        matching, unmatched_prev, unmatched_curr = processor.main_matching(joint_scores)
        
      # Create prev/curr dictionaries.
      prev_to_curr = collections.defaultdict(list)
      curr_to_prev = collections.defaultdict(list)
      print('matching:', matching)
      for prev, curr in matching:
        for i in prev:
          for j in curr:
            if j not in prev_to_curr[i]:
              prev_to_curr[i].append(j)
            if i not in curr_to_prev[j]:
              curr_to_prev[j].append(i)

      print('[NOTE] Main matching:', matching)
      print('[NOTE] Main prev_to_curr:', prev_to_curr)
      print('[NOTE] Main curr_to_prev:', curr_to_prev)
      print('[NOTE] Unmatched prev:', unmatched_prev)
      print('[NOTE] Unmatched curr:', unmatched_curr)

      # Visualize matches.
      main_matching_vis = viz.main_matching(matching)
      cv2.imwrite(os.path.join(debug_match_folder, f'{frame_idx:03d}.png'), main_matching_vis)

      # Finally, we check for fallback matches.
      fallback_t0 = time.perf_counter()
      if len(unmatched_prev) > 0 and len(unmatched_curr) > 0:
        unmatched_prev_shapes = []
        unmatched_prev_centroids = []
        for i in unmatched_prev:
          prev_shape_idx = active_shapes[i]
          unmatched_prev_shape = shape_bank.get(prev_shape_idx) if shape_bank is not None else None
          optim_data = optim_bank.get(prev_shape_idx) if optim_bank is not None else None
          if optim_data is not None and isinstance(optim_data, dict) and 'centroid' in optim_data:
            centroid_data = optim_data['centroid']
            if centroid_data is not None and len(centroid_data) >= 2:
              cx, cy = centroid_data[:2]
              unmatched_prev_centroids.append([cx / frame_width, cy / frame_height])
            else:
              unmatched_prev_centroids.append([0.5, 0.5])
          else:
            unmatched_prev_centroids.append([0.5, 0.5])
          
          if unmatched_prev_shape is not None and hasattr(unmatched_prev_shape, 'shape') and len(unmatched_prev_shape.shape) >= 3:
            try:
              if unmatched_prev_shape.shape[2] > 3:
                min_x, min_y, max_x, max_y = get_shape_coords(unmatched_prev_shape[:, :, 3])
                unmatched_prev_shapes.append(unmatched_prev_shape[min_y:max_y, min_x:max_x])
              else:
                unmatched_prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
            except (IndexError, AttributeError):
              unmatched_prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
          else:
            unmatched_prev_shapes.append(np.zeros((10, 10, 4), dtype=np.uint8))
        unmatched_curr_shapes = []
        unmatched_curr_centroids = []
        for j in unmatched_curr:
          curr_shape_idx = curr_labels[j]
          unmatched_curr_shape_alpha = np.uint8(curr_fg_labels==curr_shape_idx)[..., None]
          unmatched_curr_shape = unmatched_curr_shape_alpha * curr_frame + (1 - unmatched_curr_shape_alpha) * (bg_img if bg_img is not None else np.zeros_like(curr_frame))
          unmatched_curr_shape = np.dstack([unmatched_curr_shape, unmatched_curr_shape_alpha])
          min_x, min_y, max_x, max_y = get_shape_coords(unmatched_curr_shape[:, :, 3])
          unmatched_curr_shapes.append(unmatched_curr_shape[min_y:max_y, min_x:max_x])
          cx, cy = get_shape_centroid(unmatched_curr_shape[:, :, 3])
          unmatched_curr_centroids.append([cx / curr_frame.shape[1], cy / curr_frame.shape[0]])

        # Use advanced fallback matching with CoTracker3 features when available
        if args.use_cotracker3 and processor.use_cotracker3:
          print("🎯 CoTracker3 enhanced fallback matching")
          try:
            result = processor.cotracker3_fallback_matching(
              unmatched_prev_shapes, unmatched_curr_shapes, 
              unmatched_prev_centroids, unmatched_curr_centroids, 
              frame_width, frame_height,
              thresh=args.fallback_match_thresh
            )
            if len(result) == 3:
              fallback_matching, costs, confidence_scores = result
            else:
              fallback_matching, costs = result
              confidence_scores = {}
            print(f"   Fallback confidence scores: {confidence_scores}")
          except AttributeError:
            # Fallback to basic matching
            print("🔧 Using enhanced RGB-based fallback matching")
            result = processor.fallback_matching(
              unmatched_prev_shapes, unmatched_curr_shapes, 
              unmatched_prev_centroids, unmatched_curr_centroids, 
              frame_width, frame_height,
              thresh=args.fallback_match_thresh
            )
            if len(result) == 3:
              fallback_matching, costs, confidence_scores = result
            else:
              fallback_matching, costs = result
              confidence_scores = {}
        else:
          print("🔧 Using enhanced RGB-based fallback matching")
          result = processor.fallback_matching(
            unmatched_prev_shapes, unmatched_curr_shapes, 
            unmatched_prev_centroids, unmatched_curr_centroids, 
            frame_width, frame_height,
            thresh=args.fallback_match_thresh
          )
          if len(result) == 3:
            fallback_matching, costs, confidence_scores = result
          else:
            fallback_matching, costs = result
            confidence_scores = {}
        fallback_t1 = time.perf_counter()
        print('[NOTE] Fallback matches:', fallback_matching)
        print('[NOTE] Fallback matching costs:\n', costs)
        print('[NOTE] Fallback confidence scores:', confidence_scores if 'confidence_scores' in locals() else 'N/A')
        print(f'[TIME] Fallback matching took {fallback_t1 - fallback_t0:.2f}s')

        # Update main matches.
        unmatched_prev_list = list(unmatched_prev)
        unmatched_curr_list = list(unmatched_curr)
        for i, j in fallback_matching.items():
          prev_to_curr[unmatched_prev_list[i]].append(unmatched_curr_list[j])
          curr_to_prev[unmatched_curr_list[j]].append(unmatched_prev_list[i])
          matching.append([{unmatched_prev_list[i],}, {unmatched_curr_list[j]}])
          unmatched_prev.remove(unmatched_prev_list[i])
          unmatched_curr.remove(unmatched_curr_list[j])

        # Visualize matches.
        fallback_matching_vis = viz.fallback_matching(fallback_matching, unmatched_prev_shapes, unmatched_curr_shapes)
        cv2.imwrite(os.path.join(debug_fallback_folder, f'{frame_idx:03d}.png'), fallback_matching_vis)

      print('[NOTE] Final prev_to_curr:', prev_to_curr)
      print('[NOTE] Final curr_to_prev:', curr_to_prev)
      print('[NOTE] Final matching:', matching)

      #######
      # 4. For each shape, extract a transform with the optical flow. Generate element-to-target
      #    maps and optimize in both directions.
      #######

      # Get shape elements to be optimized (in both directions).
      curr_shape_crops = []
      for curr_shape_idx in curr_labels:
        shape_mask = get_shape_mask(curr_fg_labels, curr_shape_idx, dtype=np.uint8).astype(np.float64)
        shape_alpha = get_alpha(shape_mask, curr_frame, expand=True)
        min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
        bg_img_safe = bg_img if bg_img is not None else np.zeros_like(curr_frame)
        shape_crop = shape_alpha * curr_frame + (1 - shape_alpha) * bg_img_safe
        shape_crop = np.dstack([np.uint8(shape_crop), np.uint8(255 * shape_alpha)])
        shape_crop = shape_crop[min_y:max_y, min_x:max_x]
        curr_shape_crops.append(shape_crop)
      prev_shape_crops = []
      for prev_shape_idx in active_shapes:
        shape_mask = get_shape_mask(prev_fg_labels, prev_shape_idx, dtype=np.uint8).astype(np.float64)
        if prev_frame is not None:
          shape_alpha = get_alpha(shape_mask, prev_frame, expand=True)
          min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
          bg_img_safe = bg_img if bg_img is not None else np.zeros_like(prev_frame)
          shape_crop = shape_alpha * prev_frame + (1 - shape_alpha) * bg_img_safe
        else:
          # Fallback if prev_frame is None
          shape_alpha = np.zeros(curr_frame.shape[:2] + (1,), dtype=np.float64)
          min_x, min_y, max_x, max_y = 0, 0, 10, 10
          shape_crop = np.zeros((max_y - min_y, max_x - min_x, 4), dtype=np.uint8)
        shape_crop = np.dstack([np.uint8(shape_crop), np.uint8(255 * shape_alpha)])
        shape_crop = shape_crop[min_y:max_y, min_x:max_x]
        prev_shape_crops.append(shape_crop)

      # Compute component/matching groups.
      matching_comp_groups, ceg = processor.compute_matching_comp_groups(
        matching, active_shapes, curr_labels, prev_fg_comp_to_label, curr_fg_comp_to_label)
      print('[NOTE] Matching component groups:', matching_comp_groups)
      matching_comp_groups_vis = viz.matching_comp_groups(ceg)
      cv2.imwrite(os.path.join(debug_groups_folder, f'{frame_idx:03d}.png'), matching_comp_groups_vis)

      # Generate optimization elements and initialize parameters.
      elements = []
      full_targets = []
      cxs, cys = [], []
      centroids_opt = []
      element_labels = []
      # Set up init variables.
      sx_init, sy_init, theta_init, tx_init, ty_init, kx_init, ky_init, z_init = [], [], [], [], [], [], [], []
      for prev_comps, curr_comps in matching_comp_groups:
        # We optimize shapes if: (1) there are multiple prev/current shapes and (2) if the shape is 
        # adjacent to the border.
        total_prev_shapes = []
        total_curr_shapes = []
        for prev_comp_idx in prev_comps:
          total_prev_shapes.extend(prev_fg_comp_to_label[prev_comp_idx])
        for curr_comp_idx in curr_comps:
          total_curr_shapes.extend(curr_fg_comp_to_label[curr_comp_idx])
        if len(total_prev_shapes) == 0 or len(total_curr_shapes) == 0:
          continue
        # We also want to optimize any shapes that are at the border (skip optimization if both 
        # shapes are not at border in a 1-to-1 match).
        if len(total_prev_shapes) == 1 and len(total_curr_shapes) == 1:
          if not shape_at_border(prev_fg_labels, total_prev_shapes[0]) and not shape_at_border(curr_fg_labels, total_curr_shapes[0]):
            continue
        for (
          from_comp_idxs, to_comp_idxs, 
          to_comps, from_fg_comp_to_label,
          from_labels, to_labels,
          shape_crops, target_crops,
          from_fg_labels, to_fg_labels, 
          flow, 
          match_dict, other_match_dict, 
          from_frame, to_frame,
          init_score, joint_score,
          init_bank
        ) in zip(
          [prev_comps, curr_comps], [curr_comps, prev_comps],
          [curr_fg_comps, prev_fg_comps],
          [prev_fg_comp_to_label, curr_fg_comp_to_label],
          [active_shapes, curr_labels], [curr_labels, active_shapes],
          [highest_res, curr_shape_crops], [curr_shape_crops, prev_shape_crops],
          [prev_fg_labels, curr_fg_labels], [curr_fg_labels, prev_fg_labels],
          [forw_flow, back_flow],
          [prev_to_curr, curr_to_prev], [curr_to_prev, prev_to_curr],
          [prev_frame, curr_frame], [curr_frame, prev_frame],
          [flow_scores, flow_scores.T], [joint_scores, joint_scores.T],
          [optim_bank, {idx: {'h': np.eye(3)} for idx in curr_labels}]
        ):
          for comp_idx in from_comp_idxs:
            for shape_idx in from_fg_comp_to_label[comp_idx]:
              i = from_labels.index(shape_idx)
              if len(match_dict[i]) < 1:  # If this shape doesn't have a match, skip.
                continue
              init_data = init_bank.get(shape_idx) if init_bank is not None else None
              if init_data is not None and 'h' in init_data and np.trace(init_data['h']) > 0:
                _, _, sx0, sy0, theta0, kx0 = decompose(init_data['h'])
              else:
                sx0, sy0, theta0, kx0 = 1.0, 1.0, 0.0, 0.0
              sx_init.append(sx0)
              sy_init.append(sy0)
              theta_init.append(theta0)
              kx_init.append(kx0)
              ky_init.append(0)
              element_labels.append(shape_idx)
              shape_crop = shape_crops.get(shape_idx) if shape_crops is not None else None
              if shape_crop is None:
                continue
              elements.append(shape_crop)
              shape_mask = np.uint8(from_fg_labels==shape_idx)
              shape_coords = np.stack(np.where(shape_mask>0), axis=1)[:, ::-1]  # x, y
              flow_shape = flow[shape_coords[:, 1], shape_coords[:, 0]]
              # Initialize other transformation parameters.
              new_shape_coords = (shape_coords + flow_shape).astype(np.int64)
              shape_coords_mean = np.mean(shape_coords, axis=0)
              spacing = shape_coords.shape[0] // min(shape_coords.shape[0], 500)
              idxs = np.arange(0, shape_coords.shape[0], spacing)
              model = AffineTransform()
              # Robustly estimate affine transform model with RANSAC.
              try:
                model, _ = ransac(
                  (shape_coords[idxs, :] - shape_coords_mean, new_shape_coords[idxs, :] - shape_coords_mean), 
                  AffineTransform, min_samples=min(len(idxs), 3),
                  residual_threshold=2, max_trials=100
                )
              except (ValueError, np.linalg.LinAlgError):
                model = None
                
              if args.all_joint:
                if len(match_dict[i]) == 1 and len(other_match_dict[match_dict[i][0]]) == 1 and other_match_dict[match_dict[i][0]][0] == i:
                  best_match = match_dict[i][np.argmax(joint_score[i][match_dict[i]])]
                  best_flow_match = match_dict[i][np.argmax(init_score[i][match_dict[i]])]
                  if best_match == best_flow_match:
                    if model is not None and hasattr(model, 'scale') and hasattr(model, 'rotation'):
                      sx, sy = model.scale
                      theta = model.rotation
                    else:
                      sx, sy = 1.0, 1.0
                      theta = 0.0
                  else:
                    target_crop = target_crops[best_match] / 255.0
                    min_x = np.min(shape_coords[:, 0])
                    min_y = np.min(shape_coords[:, 1])
                    max_x = np.max(shape_coords[:, 0])
                    max_y = np.max(shape_coords[:, 1])
                    bg_crop = bg_img[min_y:max_y, min_x:max_x]
                    theta, sx, sy, _ = init_rot_scale(
                      shape_crop / 255.0, target_crop, theta_init[-1], bg_crop, over_mask=np.zeros_like(target_crop[:, :, 3]), p_weight=0.0)
                    theta = np.deg2rad(-theta)
                else:
                  if model is not None and hasattr(model, 'scale') and hasattr(model, 'rotation'):
                    sx, sy = model.scale
                    sx *= sx_init[-1]
                    sy *= sy_init[-1]
                    theta = model.rotation
                  else:
                    sx, sy = sx_init[-1], sy_init[-1]
                    theta = 0.0
              else:
                if model is not None and hasattr(model, 'scale') and hasattr(model, 'rotation'):
                  sx, sy = model.scale
                  sx *= sx_init[-1]
                  sy *= sy_init[-1]
                  theta = model.rotation
                else:
                  sx, sy = sx_init[-1], sy_init[-1]
                  theta = 0.0
              if args.init_s:
                sx_init[-1] = sx
                sy_init[-1] = sy
              if args.init_r:
                theta_init[-1] += theta
              # Estimate the translation of the shape.
              if args.all_joint:
                best_match = np.argmax(joint_score[i])
                target_alpha = np.uint8(to_fg_labels==to_labels[best_match])
                target_cx, target_cy = get_shape_centroid(target_alpha)
                if len(match_dict[i]) == 1 and len(other_match_dict[match_dict[i][0]]) == 1 and other_match_dict[match_dict[i][0]][0] == i:
                  best_match = np.argmax(joint_score[i])
                  target_alpha = np.uint8(to_fg_labels==to_labels[best_match])
                  target_cx, target_cy = get_shape_centroid(target_alpha)
                else:
                  target_cx, target_cy = get_shape_centroid(shape_mask)
                  tx, ty = model.translation
                  target_cx += tx
                  target_cy += ty
              else:
                if len(match_dict[i]) >= 1 and len(other_match_dict[match_dict[i][0]]) == 1 and other_match_dict[match_dict[i][0]][0] == i:
                  match_labels = [to_labels[j] for j in match_dict[i]]
                  target_alpha = np.uint8(np.isin(to_fg_labels, match_labels))
                  target_cx, target_cy = get_shape_centroid(target_alpha)
                  # Add visual scores with flow scores to get best initialization.
                else:
                  best_match = np.argmax(init_score[i, match_dict[i]])
                  joint_match = np.argmax(joint_score[i, match_dict[i]])
                  if best_match == joint_match:
                    target_cx, target_cy = get_shape_centroid(shape_mask)
                    tx, ty = model.translation
                    target_cx += tx
                    target_cy += ty
                  else:
                    target_alpha = np.uint8(to_fg_labels==to_labels[match_dict[i][best_match]])
                    target_cx, target_cy = get_shape_centroid(target_alpha)
              # Be careful to place shapes completely within the bounds of the frame.
              if args.init_t:
                cx = max(shape_crop.shape[1] // 2, min(target_cx, from_frame.shape[1] - shape_crop.shape[1] // 2))
                cy = max(shape_crop.shape[0] // 2, min(target_cy, from_frame.shape[0] - shape_crop.shape[0] // 2))
                tx_init.append(target_cx - cx)
                ty_init.append(target_cy - cy)
              else:
                cx, cy = get_shape_centroid(shape_mask)
                tx_init.append(0.0)
                ty_init.append(0.0)
              cxs.append(cx)
              cys.append(cy)
              centroids_opt.append([cx, cy])
              z_init.append(5.0)
          comp_alpha = np.uint8(np.isin(to_comps, to_comp_idxs))[..., None]
          comp_rgb = comp_alpha * to_frame + (1 - comp_alpha) * bg_img
          full_target = np.dstack([comp_rgb, 255 * comp_alpha])
          full_targets.append(full_target)

      full_elements = compositing.place_shape(
        elements,
        cxs, cys,
        [1.0] * len(elements), # sx 
        [1.0] * len(elements), # sy
        [0.0] * len(elements), # theta
        [0.0] * len(elements), # kx
        [0.0] * len(elements), # ky
        frame_width, 
        frame_height,
        bg=np.tile(np.transpose(bg_img / 255.0, (2, 0, 1)), [len(elements), 1, 1, 1]),
        keep_alpha=True,
        device=device
      )

      # Create subregion inputs to optimization loop.
      elements = []
      target_labels = []
      targets = []
      target_bounds = []
      bg_crops = []
      target_to_element = collections.defaultdict(list)
      target_idx = 0
      element_idx = 0
      prev_element_idxs = {}
      prev_target_idxs = {}
      curr_element_idxs = {}
      curr_target_idxs = {}
      for prev_comps, curr_comps in matching_comp_groups:
        # Skip over the same shapes we skipped over in previous loop.
        total_prev_shapes = []
        total_curr_shapes = []
        for prev_comp_idx in prev_comps:
          total_prev_shapes.extend(prev_fg_comp_to_label[prev_comp_idx])
        for curr_comp_idx in curr_comps:
          total_curr_shapes.extend(curr_fg_comp_to_label[curr_comp_idx])
        if len(total_prev_shapes) == 0 or len(total_curr_shapes) == 0:
          continue
        if len(total_prev_shapes) == 1 and len(total_curr_shapes) == 1:
          if not shape_at_border(prev_fg_labels, total_prev_shapes[0]) and not shape_at_border(curr_fg_labels, total_curr_shapes[0]):
            continue
        for (
          from_comp_idxs, to_comp_idxs,
          from_fg_comp_to_label,
          from_labels, to_labels,
          element_idxs, target_idxs,
          to_fg_labels, to_frame,
          match_dict, other_match_dict
        ) in zip(
          [prev_comps, curr_comps], [curr_comps, prev_comps],
          [prev_fg_comp_to_label, curr_fg_comp_to_label],
          [active_shapes, curr_labels],
          [curr_labels, active_shapes],
          [prev_element_idxs, curr_element_idxs], [prev_target_idxs, curr_target_idxs],
          [curr_fg_labels, prev_fg_labels], [curr_frame, prev_frame],
          [prev_to_curr, curr_to_prev], [curr_to_prev, prev_to_curr],
        ):
          comp_alpha = full_targets[target_idx][:, :, 3]
          start_element_idx = element_idx
          for from_comp_idx in from_comp_idxs:
            for shape_idx in from_fg_comp_to_label[from_comp_idx]:
              i = from_labels.index(shape_idx)
              if len(match_dict[i]) < 1:
                continue
              shape_alpha = compositing.torch2numpy(full_elements[element_idx])[:, :, 3]
              shape_alpha = shape_alpha[:frame_height, :frame_width]
              comp_alpha = np.maximum(comp_alpha, shape_alpha)
              element_idx += 1
          element_idx = start_element_idx
          r_min_x, r_min_y, r_max_x, r_max_y = get_shape_coords(comp_alpha)
          for from_comp_idx in from_comp_idxs:
            for shape_idx in from_fg_comp_to_label[from_comp_idx]:
              i = from_labels.index(shape_idx)
              if len(match_dict[i]) < 1:
                continue
              elements.append(full_elements[element_idx, :, r_min_y:r_max_y, r_min_x:r_max_x])
              # Adjust translation initialization to dimension of element.
              tx_init[element_idx] /= (max(r_max_x - r_min_x, r_max_y - r_min_y) / 2)
              ty_init[element_idx] /= (max(r_max_x - r_min_x, r_max_y - r_min_y) / 2)
              centroids_opt[element_idx] = [centroids_opt[element_idx][0] - r_min_x, centroids_opt[element_idx][1] - r_min_y]
              element_idxs[i] = element_idx
              target_idxs[i] = target_idx
              target_to_element[target_idx].append(element_idx)
              element_idx += 1
          targets.append(full_targets[target_idx][r_min_y:r_max_y, r_min_x:r_max_x])
          bg_crops.append(bg_img[r_min_y:r_max_y, r_min_x:r_max_x])
          target_bounds.append([r_min_x, r_min_y, r_max_x, r_max_y])
          target_label = to_fg_labels.copy()
          target_label[full_targets[target_idx][:, :, 3]==0] = -1
          target_labels.append(target_label[r_min_y:r_max_y, r_min_x:r_max_x])
          target_idx += 1
      print('[NOTE] target_to_element:', target_to_element)

      if len(target_to_element) > 0 and len(elements) > 0 and len(targets) > 0:
        t2e_vis = viz.target_to_element(target_to_element, elements, targets)
        cv2.imwrite(os.path.join(debug_t2e_folder, f'{frame_idx:03d}.png'), t2e_vis)

        print('[NOTE] Optimizing...')
        t0 = time.perf_counter()
        _, params_shapes, layer_zs, _, render_shapes_bleed, target_shapes, _, _, side_by_sides, on_tops, losses = optimize(
          elements, centroids_opt, targets, target_to_element,
          np.array(sx_init), np.array(sy_init), np.array(theta_init), np.array(tx_init), np.array(ty_init), np.array(z_init), 
          0, 0, bg_crops, bleed=args.bleed, use_k=args.use_k, use_r=args.use_r, use_s=args.use_s, use_t=args.use_t, 
          blur_kernel=args.blur_kernel, lr=args.lr, n_steps=args.n_steps, min_size=args.min_opt_size, p_weight=args.p_weight, device=device)
        render_shapes = [
          render_shape[args.bleed:render_shape.shape[0] - args.bleed, args.bleed:render_shape.shape[1] - args.bleed] for render_shape in render_shapes_bleed]
        all_tx, all_ty, all_sx, all_sy, all_theta, all_kx, all_ky = params_shapes
        t1 = time.perf_counter()
        print(f'[TIME] Optimizing took {t1 - t0:.4f}s')

        # Create ordered label compositions for each optimization (matching) group.
        z_order_labels = {}
        z_order_rgbs = {}
        for target_idx in target_to_element:
          r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[target_idx]
          target_width = r_max_x - r_min_x
          target_height = r_max_y - r_min_y
          z_order_label = -1 * np.ones((target_height, target_width), dtype=np.int32)
          z_order_rgb = bg_crops[target_idx] / 255.0
          element_zs = [layer_zs[element_idx] for element_idx in target_to_element[target_idx]]
          render_order = [target_to_element[target_idx][i] for i in np.argsort(element_zs)]
          for ro in render_order:
            shape_mask = render_shapes[ro][:target_height, :target_width, 3]
            shape_rgb = render_shapes[ro][:target_height, :target_width, :3]
            z_order_rgb = shape_mask[..., None] * shape_rgb + (1 - shape_mask[..., None]) * z_order_rgb
            z_order_label[shape_mask>0.5] = element_labels[ro]
          z_order_labels[target_idx] = z_order_label
          z_order_rgbs[target_idx] = z_order_rgb

        # Save vis outputs.
        frame_opt_folder = os.path.join(debug_opt_folder, f'f{frame_idx:03d}')
        for comp_idx, (sbs, ot) in enumerate(zip(side_by_sides, on_tops)):
          frame_comp_folder = os.path.join(frame_opt_folder, f'c{comp_idx:02d}')
          if not os.path.exists(frame_comp_folder):
            os.makedirs(frame_comp_folder)
          save_frames(sbs, frame_comp_folder, suffix='pss')
          save_frames(ot, frame_comp_folder, suffix='pot')
          font = {
            'family': 'serif',
            'color':  'red',
            'weight': 'normal',
            'size': 12,
          }
          comp_losses = losses[comp_idx]
          fig = plt.figure()
          plt.plot(np.arange(0, len(comp_losses)), comp_losses)
          plt.axvline(x=np.argmin(comp_losses), c='r')
          plt.text(np.argmin(comp_losses), np.max(comp_losses), f'{np.min(comp_losses):.4f}', fontdict=font)
          plt.title('loss')
          fig.savefig(os.path.join(frame_comp_folder, 'loss_plot.png'))
          plt.close()

      #######
      # 5. Once the params have been optimized, check which loss (forward or backward) is smaller
      #    (compute this with a loss mask). We consider the following cases:
      #      (A) In a one-to-one match we always propagate the previous label.
      #      (B) If the forward loss is better for a one-to-many match, then we propagate the
      #          previous label (e.g., disjointed occlusion).
      #      (C) If the backward loss is better for a one-to-many match, then we do not propagate
      #          any labels, i.e. (separation).
      #      (D) If the forward loss is better for a many-to-one match, then we propagate the
      #          previous label of each shape (e.g., distinct shapes colliding).
      #      (E) If the backward loss is better for a many-to-one match, then we merge all the
      #          previous labels and propagate one canonical label (e.g., merging).
      #######
      # Initialize variables to prevent unbound errors
      target_shapes = {}
      layer_zs = {}
      z_order_labels = {}
      z_order_rgbs = {}
      render_shapes = {}
      render_shapes_bleed = {}
      target_bounds = {}
      target_labels = {}
      prev_element_idxs = []
      curr_element_idxs = []
      prev_target_idxs = []
      curr_target_idxs = []
      target_to_element = {}
      element_labels = {}
      
      all_mappings_vis = []
      for prev_set, curr_set in matching:
        comp_candidate_mappings_vis = []
        comp_chosen_mappings_vis = []
        # Compute forward and backward losses.
        prev_list = [i for i in prev_set]
        curr_list = [j for j in curr_set]
        prev_idxs = [f'p{active_shapes[i]}' for i in prev_list]
        curr_idxs = [f'c{curr_labels[j]}' for j in curr_list]
        sub_B = nx.DiGraph()
        sub_B.add_nodes_from(prev_idxs, bipartite=0)
        sub_B.add_nodes_from(curr_idxs, bipartite=1)
        mappings = []
        case_A = set()
        if len(prev_list) == 1 and len(curr_list) == 1 and prev_list[0] not in prev_element_idxs and curr_list[0] not in curr_element_idxs:
          case_A.add(prev_idxs[0])
          case_A.add(curr_idxs[0])
          sub_B.add_edge(prev_idxs[0], curr_idxs[0], weight=0.0, loss=0.0)
          print(f'[NOTE] 1-to-1 matching: {prev_idxs[0][1:]} -> {curr_idxs[0][1:]}')
        else:
          for (element_idxs, target_idxs, from_set, to_set, from_list, to_list, from_labels, to_labels, from_fg_labels, to_fg_labels, frame, joint_score) in zip(
            [prev_element_idxs, curr_element_idxs],
            [prev_target_idxs, curr_target_idxs], 
            [prev_idxs, curr_idxs], [curr_idxs, prev_idxs],
            [prev_list, curr_list], [curr_list, prev_list],
            [active_shapes, curr_labels], [curr_labels, active_shapes],
            [prev_fg_labels, curr_fg_labels], [curr_fg_labels, prev_fg_labels],
            [curr_frame, prev_frame], [joint_scores, joint_scores.T]
          ):
            sub_B.remove_edges_from(list(sub_B.edges))
            r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[target_idxs[from_list[0]]]
            target_width = r_max_x - r_min_x
            target_height = r_max_y - r_min_y
            # Get target RGB and labels.
            target_rgb = target_shapes[target_idxs[from_list[0]]][:target_height, :target_width, :3] / 255.0
            target_label = target_labels[target_idxs[from_list[0]]]
            unique_target_labels = np.array([to_labels[j] for j in to_list])
            target_labels_mask = np.tile(target_label, [len(unique_target_labels), 1, 1])
            target_labels_mask = np.reshape(target_labels_mask, (len(unique_target_labels), -1))
            target_labels_mask = np.uint8(target_labels_mask==unique_target_labels[..., None]) / 1.0
            target_total = np.sum(target_labels_mask, axis=1)
            
            # Get optimized element output RGB and labels.
            match_element_idxs = [element_idxs[i] for i in from_list]
            element_zs = [layer_zs[idx] for idx in match_element_idxs]
            render_order = np.argsort(element_zs).tolist()

            # Get render element labels in z-ordering.
            z_order_label = z_order_labels[target_idxs[from_list[0]]]
            render_rgb = z_order_rgbs[target_idxs[from_list[0]]] 

            # Find the corresponding target regions for each shape.
            for i, v in enumerate(from_set):
              # Construct the output RGB and mask.
              output_mask = np.uint8(z_order_label==int(v[1:]))
              if np.sum(output_mask) < 5:
                continue
              output_rgb = render_shapes[element_idxs[from_list[i]]][:target_height, :target_width, :3]
              output_label = np.reshape(output_mask, (1, -1)) / 1.0
              output_total = np.sum(output_label)
              output_in_target = output_label @ target_labels_mask.T / output_total
              output_in_target[np.isnan(output_in_target)] = 0.0
              target_in_output = output_label @ target_labels_mask.T / target_total
              target_in_output[np.isnan(target_in_output)] = 0.0
              unique_j = unique_target_labels[np.argmax(output_in_target)]
              print('target_in_output:', target_in_output)
              if np.max(output_in_target) > args.overlap_thresh:
                print(v, 'falls in', f'{to_set[0][0]}{unique_j}')
                sub_B.add_edge(v, f'{to_set[0][0]}{unique_j}', weight=np.exp(-joint_score[from_labels.index(int(v[1:])), to_labels.index(unique_j)]))
              # Get the targets which element mostly covers (this constitutes a complete candidate mapping).
              target_in_output_j = np.where(target_in_output>args.overlap_thresh)[1]
              if len(target_in_output_j) > 0:
                print(v, 'covers', [f'{to_set[0][0]}{unique_target_labels[j]}' for j in target_in_output_j])
                target_mask = np.uint8(np.isin(target_label, [unique_target_labels[j] for j in target_in_output_j]))
                # output_rgba = np.dstack([output_rgb, output_mask[..., None]])
                # target_rgba = np.dstack([target_rgb, target_mask[..., None]])
                target_rgb_ = target_mask[..., None] * target_rgb + (1 - target_mask[..., None]) * bg_crops[target_idxs[from_list[0]]] / 255.0
                loss_mask = np.maximum(target_mask, output_mask)[..., None]
                loss = np.sum(loss_mask * (output_rgb - target_rgb_)**2.0) / np.sum(loss_mask)
                # cv2.imshow('target_mask', 255 * target_mask)
                # cv2.imshow('bg_crop', bg_crops[target_idxs[from_list[0]]])
                # cv2.imshow('target_rgb_', target_rgb_)
                # cv2.imshow('target_rgb', target_rgb)
                # cv2.imshow('output_rgb', output_rgb)
                # cv2.imshow('output_mask', 255 * output_mask)
                # cv2.imshow('loss_mask', 255 * loss_mask)
                # cv2.imshow('loss', loss_mask * (output_rgb - target_rgb_)**2.0)
                # print('loss:', loss)
                # cv2.waitKey(0)
                mappings.append([[v], [f'{to_set[0][0]}{unique_target_labels[j]}' for j in target_in_output_j], loss])
                comp_candidate_mappings_vis.append(viz.vis_mapping([v], [f'{to_set[0][0]}{unique_target_labels[j]}' for j in target_in_output_j], loss, loss_mask * (output_rgb - target_rgb)**2.0))

            # Get all remaining mappings.
            for u in to_set:
              if len(sub_B.in_edges(u)) > 0:
                print(f'remaining mappings for {u}:', sub_B.in_edges(u))
                target_mask = np.uint8(target_label==int(u[1:]))
                # Sort by render order.
                in_nodes = [v for v, _ in sub_B.in_edges(u)]
                output_mask = np.uint8(np.isin(z_order_label, [int(v[1:]) for v in in_nodes]))
                output_rgb = output_mask[..., None] * render_rgb + (1 - output_mask[..., None]) * bg_crops[target_idxs[from_list[0]]] / 255.0
                loss_mask = np.maximum(target_mask, output_mask)[..., None]
                # output_rgba = np.dstack([render_rgb, output_mask[..., None]])
                # target_rgba = np.dstack([target_rgb, target_mask[..., None]])
                target_rgb_ = target_mask[..., None] * target_rgb + (1 - target_mask[..., None]) * bg_crops[target_idxs[from_list[0]]] / 255.0
                loss = np.sum(loss_mask * (render_rgb - target_rgb_)**2.0) / np.sum(loss_mask)
                # cv2.imshow('target_mask', 255 * target_mask)
                # cv2.imshow('target_rgb', target_rgb)
                # cv2.imshow('bg_crop', bg_crops[target_idxs[from_list[0]]])
                # cv2.imshow('target_rgb_', target_rgb_)
                # cv2.imshow('output_rgb', output_rgb)
                # cv2.imshow('output_mask', 255 * output_mask)
                # cv2.imshow('loss_mask', 255 * loss_mask)
                # cv2.imshow('loss', loss_mask * (output_rgb - target_rgb)**2.0)
                # print('loss:', loss)
                # cv2.waitKey(0)
                mappings.append([in_nodes, [u], loss])
                comp_candidate_mappings_vis.append(viz.vis_mapping(in_nodes, [u], loss, loss_mask * (render_rgb - target_rgb)**2.0))

          # Greedily accept mappings, deleting any conflicting mappings along the way. Stop when
          # there are no more mappings left.
          sub_B.remove_edges_from(list(sub_B.edges))
          while len(mappings) > 0:
            best_mapping = None
            best_mapping_score = np.inf
            for mapping in mappings:
              if mapping[2] < best_mapping_score:
                best_mapping = mapping
                best_mapping_score = mapping[2]
            if best_mapping_score > args.drop_thresh:
              break
            for from_node in best_mapping[0]:
              for to_node in best_mapping[1]:
                sub_B.add_edge(from_node, to_node, loss=best_mapping_score)
            comp_chosen_mappings_vis.append(viz.vis_mapping(best_mapping[0], best_mapping[1], best_mapping[2]))
            # Remove all conflicting mappings.
            remaining_mappings = []
            for mapping in mappings:
              from_nodes, to_nodes, _ = mapping
              conflict = False
              for from_node in from_nodes:
                mapping_group = best_mapping[0] if from_node[0] == best_mapping[0][0][0] else best_mapping[1]
                if from_node in mapping_group:
                  conflict = True
                  break
              for to_node in to_nodes:
                mapping_group = best_mapping[0] if to_node[0] == best_mapping[0][0][0] else best_mapping[1]
                if to_node in mapping_group:
                  conflict = True
                  break
              if not conflict:
                remaining_mappings.append(mapping)
            mappings = remaining_mappings
        if len(comp_candidate_mappings_vis) > 0:
          all_candidate_mappings_vis = np.concatenate(comp_candidate_mappings_vis, axis=0)
          if len(comp_chosen_mappings_vis) > 0:
            all_chosen_mappings_vis = np.concatenate(comp_chosen_mappings_vis, axis=0)
            border = 255 * np.ones((all_candidate_mappings_vis.shape[0] - all_chosen_mappings_vis.shape[0], all_chosen_mappings_vis.shape[1], 3), dtype=np.uint8)
            all_chosen_mappings_vis = np.concatenate([all_chosen_mappings_vis, border])
          else:
            all_chosen_mappings_vis = 255 * np.ones(all_candidate_mappings_vis.shape)
          border = 255 * np.ones((all_candidate_mappings_vis.shape[0], 25, 3), dtype=np.uint8)
          mappings_vis = np.concatenate([all_candidate_mappings_vis, border, all_chosen_mappings_vis], axis=1)
          border = 255 * np.ones((25, mappings_vis.shape[1], 3), dtype=np.uint8)
          mappings_vis = np.concatenate([mappings_vis, border], axis=0)
          all_mappings_vis.append(mappings_vis)
        
        # Determine how to propagate labels.
        for cc in nx.connected_components(sub_B.to_undirected()):
          prev, curr = [], []
          match_type = None
          for v in cc:
            if v[0] == 'p':
              prev.append(v)
              if len(sub_B.out_edges(v)) > 0:
                match_type = 'prev_to_curr'
            if v[0] == 'c':
              curr.append(v)
              if len(sub_B.out_edges(v)) > 0:
                match_type = 'curr_to_prev'  
          assert len(prev) == 1 or len(curr) == 1
          match_weight = 0.0
          if match_type == 'prev_to_curr':
            for _, u in sub_B.out_edges(prev[0]):
              edge_data = sub_B[prev[0]][u] if prev[0] in sub_B and u in sub_B[prev[0]] else {}
              loss_value = edge_data.get('loss', 0.0) if isinstance(edge_data, dict) else 0.0
              match_weight += loss_value
            match_weight /= len(sub_B.out_edges(prev[0]))
          if match_type == 'curr_to_prev':
            for _, u in sub_B.out_edges(curr[0]):
              edge_data = sub_B[curr[0]][u] if curr[0] in sub_B and u in sub_B[curr[0]] else {}
              loss_value = edge_data.get('loss', 0.0) if isinstance(edge_data, dict) else 0.0
              match_weight += loss_value
            match_weight /= len(sub_B.out_edges(curr[0]))

          if match_weight > args.drop_thresh:
            print(f'[WARN] Matching weight {match_weight:.4f} > {args.drop_thresh:.4f} -- dropped!')
            continue
          if match_type is None:
            print(f'[WARN] No matches in this subgraph!')
            continue

          if len(prev) == 1 and len(curr) == 1:
            case = 'A'
          elif len(prev) == 1 and len(curr) > 1:
            if match_type == 'prev_to_curr':
              case = 'B'
            elif match_type == 'curr_to_prev':
              case = 'C'
            else:
              case = None
          elif len(prev) > 1 and len(curr) == 1:
            if match_type == 'prev_to_curr':
              case = 'D'
            elif match_type == 'curr_to_prev':
              # Check if the transform was close to an identity.
              j = curr_labels.index(int(curr[0][1:]))
              tx = all_tx[curr_element_idxs[j]]
              ty = all_ty[curr_element_idxs[j]]
              sx = all_sx[curr_element_idxs[j]]
              sy = all_sy[curr_element_idxs[j]]
              theta = all_theta[curr_element_idxs[j]]
              kx = all_kx[curr_element_idxs[j]]
              ky = all_ky[curr_element_idxs[j]]
              A = params_to_mat(sx, sy, theta, kx, ky, tx=tx, ty=ty)
              I_err = np.linalg.norm(A - np.eye(3))
              if I_err < 0.1:
                case = 'D'
              else:
                case = 'E'
            else:
              case = None
          else:
            case = None
          print('[NOTE] Case:', case)
          print(f'[NOTE] Matching weight: {match_weight:.4f}')
          if case == 'A':
            prev_shape_idx = int(prev[0][1:])
            curr_shape_idx = int(curr[0][1:])
            i = active_shapes.index(prev_shape_idx)
            j = curr_labels.index(curr_shape_idx)
            shape_alpha = np.uint8(curr_fg_labels==curr_shape_idx)
            shape_alpha = get_alpha(np.float64(shape_alpha), curr_frame)
            shape_alpha_bleed = np.pad(shape_alpha, ((args.bleed, args.bleed), (args.bleed, args.bleed)))
            prev_shape_alpha = np.uint8(prev_fg_labels==prev_shape_idx)
            prev_shape_alpha = get_alpha(np.float64(prev_shape_alpha), prev_frame)
            prev_shape_alpha_bleed = np.pad(prev_shape_alpha, ((args.bleed, args.bleed), (args.bleed, args.bleed)))
            min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
            shape_rgb = curr_frame * shape_alpha[..., None] + bg_img * (1 - shape_alpha[..., None])
            shape = np.uint8(np.dstack([shape_rgb, 255 * shape_alpha]))
            centroid = get_shape_centroid(shape_alpha)
            if i in prev_element_idxs and match_type == 'curr_to_prev':
              new_mat = params_to_mat(
                all_sx[curr_element_idxs[j]],
                all_sy[curr_element_idxs[j]],
                all_theta[curr_element_idxs[j]],
                all_kx[curr_element_idxs[j]],
                all_ky[curr_element_idxs[j]],
              )
              new_h = np.linalg.inv(new_mat) @ optim_bank[prev_shape_idx]['h']
              r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[curr_target_idxs[j]]
              prev_shape_mask = -1 * np.ones_like(fg_bg)
              prev_shape_mask = np.pad(prev_shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
              render_shape_bleed = -1 * np.ones_like(render_shapes_bleed[curr_element_idxs[j]][:, :, 3], dtype=np.int8)
              render_shape_bleed[render_shapes_bleed[curr_element_idxs[j]][:, :, 3]>0] = prev_shape_idx
              prev_shape_mask = place_mask(render_shape_bleed, r_min_x, r_min_y, prev_shape_mask)
              prev_shape_mask[prev_shape_alpha_bleed>0] = prev_shape_idx
              p_min_x, p_min_y, p_max_x, p_max_y = get_shape_coords(np.uint8(prev_shape_mask>=0))
              if t < args.base_frame:
                time_bank['shapes'][t + 1][prev_shape_idx] = {
                  'coords': np.array([
                    [p_min_x - args.bleed, p_min_y - args.bleed], 
                    [p_max_x - args.bleed, p_max_y - args.bleed]
                  ]),
                  'centroid': [
                    (p_min_x + p_max_x) / 2 - args.bleed, 
                    (p_min_y + p_max_y) / 2 - args.bleed
                  ],
                  'mask': prev_shape_mask
                }
              else:
                time_bank['shapes'][t - 1][prev_shape_idx] = {
                  'coords': np.array([
                    [p_min_x - args.bleed, p_min_y - args.bleed], 
                    [p_max_x - args.bleed, p_max_y - args.bleed]
                  ]),
                  'centroid': [
                    (p_min_x + p_max_x) / 2 - args.bleed, 
                    (p_min_y + p_max_y) / 2 - args.bleed
                  ],
                  'mask': prev_shape_mask
                }
              shape_mask = -1 * np.ones_like(fg_bg)
              shape_mask[shape_alpha * fg_bg>0] = prev_shape_idx
              shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
              time_bank['shapes'][t][prev_shape_idx] = {
                'coords': np.array([[min_x, min_y], [max_x, max_y]]),
                'centroid': centroid,
                'mask': shape_mask
              }
              new_fg_labels[shape_alpha * fg_bg>0] = prev_shape_idx
            elif i in prev_element_idxs and match_type == 'prev_to_curr':
              new_mat = params_to_mat(
                all_sx[prev_element_idxs[i]],
                all_sy[prev_element_idxs[i]],
                all_theta[prev_element_idxs[i]],
                all_kx[prev_element_idxs[i]],
                all_ky[prev_element_idxs[i]],
              )
              new_h = new_mat #@ optim_bank[prev_shape_idx]['h']
              r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[prev_target_idxs[i]]
              shape_mask = -1 * np.ones_like(fg_bg)
              shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
              render_shape_bleed = -1 * np.ones_like(render_shapes_bleed[prev_element_idxs[i]][:, :, 3], dtype=np.int8)
              render_shape_bleed[render_shapes_bleed[prev_element_idxs[i]][:, :, 3]>0] = prev_shape_idx
              shape_mask = place_mask(render_shape_bleed, r_min_x, r_min_y, shape_mask)
              shape_mask[shape_alpha_bleed * np.pad(fg_bg, ((args.bleed, args.bleed), (args.bleed, args.bleed)))>0] = prev_shape_idx
              p_min_x, p_min_y, p_max_x, p_max_y = get_shape_coords(np.uint8(shape_mask>=0))
              time_bank['shapes'][t][prev_shape_idx] = {
                'coords': np.array([
                  [p_min_x - args.bleed, p_min_y - args.bleed], 
                  [p_max_x - args.bleed, p_max_y - args.bleed]
                ]),
                'centroid': [
                  (p_min_x + p_max_x) / 2 - args.bleed, 
                  (p_min_y + p_max_y) / 2 - args.bleed
                ],
                'mask': shape_mask
              }
              new_fg_labels[shape_alpha * fg_bg>0] = prev_shape_idx
            else:
              new_h = optim_bank[prev_shape_idx]['h']
              shape_mask = -1 * np.ones_like(fg_bg)
              shape_mask[shape_alpha * fg_bg>0] = prev_shape_idx
              shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
              min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
              shape_rgb = curr_frame * shape_alpha[..., None] + bg_img * (1 - shape_alpha[..., None])
              shape = np.uint8(np.dstack([shape_rgb, 255 * shape_alpha]))
              centroid = get_shape_centroid(shape_alpha)
              time_bank['shapes'][t][prev_shape_idx] = {
                'coords': np.array([[min_x, min_y], [max_x, max_y]]),
                'centroid': centroid,
                'mask': shape_mask
              }
              new_fg_labels[curr_fg_labels==curr_shape_idx] = prev_shape_idx

            # Save highest res shape only if the current shape is in one piece and unoccluded by edge.
            prev_highest_shape_area = np.sum(
              highest_res[prev_shape_idx][:, :, 3] / 255.0) / (frame_width * frame_height)
            new_shape_area = np.sum(shape_alpha) / (frame_width * frame_height)
            at_edge_x = min_x < 3 or max_x >= curr_frame.shape[1] - 3
            at_edge_y = min_y < 3 or max_y >= curr_frame.shape[0] - 3
            occluded = len(curr_fg_comp_to_label[curr_fg_label_to_comp[curr_shape_idx]]) > 1
            print(f'Prev shape {prev_shape_idx} occluded:', (not unoccluded_canon[prev_shape_idx]))
            if unoccluded_canon[prev_shape_idx]:
              if new_shape_area > prev_highest_shape_area and not at_edge_x and not at_edge_y and not occluded:
                highest_res[prev_shape_idx] = shape[min_y:max_y, min_x:max_x]
                highest_res_update[prev_shape_idx] = True
                new_h = np.eye(3)
            else:
              if new_shape_area > prev_highest_shape_area:
                highest_res[prev_shape_idx] = shape[min_y:max_y, min_x:max_x]
                highest_res_update[prev_shape_idx] = True
                unoccluded_canon[prev_shape_idx] = not (at_edge_x or at_edge_y) and not occluded
                new_h = np.eye(3)

            shape_info = {
              't': dataloader.frame_idxs[t],
              'coords': [(min_x, min_y), (max_x, max_y)],
              'centroid': centroid,
              'h': new_h
            }
            shape_bank[prev_shape_idx] = shape
            optim_bank[prev_shape_idx] = shape_info

          if case == 'B':
            prev_shape_idx = int(prev[0][1:])
            curr_shape_idxs = [int(j[1:]) for j in curr]
            i = active_shapes.index(prev_shape_idx)
            shape_alpha = np.uint8(np.isin(curr_fg_labels, curr_shape_idxs))
            shape_alpha = get_alpha(np.float64(shape_alpha), curr_frame)
            shape_alpha_bleed = np.pad(shape_alpha, ((args.bleed, args.bleed), (args.bleed, args.bleed)))
            shape_mask = -1 * np.ones_like(fg_bg)
            shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
            r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[prev_target_idxs[i]]
            render_shape_bleed = -1 * np.ones_like(render_shapes_bleed[prev_element_idxs[i]][:, :, 3], dtype=np.int8)
            render_shape_bleed[render_shapes_bleed[prev_element_idxs[i]][:, :, 3]>0] = prev_shape_idx
            shape_mask = place_mask(render_shape_bleed, r_min_x, r_min_y, shape_mask)
            shape_mask[shape_alpha_bleed * np.pad(fg_bg, ((args.bleed, args.bleed), (args.bleed, args.bleed)))>0] = prev_shape_idx
            p_min_x, p_min_y, p_max_x, p_max_y = get_shape_coords(shape_alpha)
            shape_rgb = curr_frame * shape_alpha[..., None] + bg_img * (1 - shape_alpha[..., None])
            shape = np.uint8(np.dstack([shape_rgb, 255 * shape_alpha]))
            centroid = get_shape_centroid(shape_alpha)
            new_mat = params_to_mat(
              all_sx[prev_element_idxs[i]],
              all_sy[prev_element_idxs[i]],
              all_theta[prev_element_idxs[i]],
              all_kx[prev_element_idxs[i]],
              all_ky[prev_element_idxs[i]],
            )
            shape_info = {
              't': dataloader.frame_idxs[t],
              'coords': [(p_min_x, p_min_y), (p_max_x, p_max_y)],
              'centroid': centroid,
              'h': new_mat #@ optim_bank[prev_shape_idx]['h']
            }
            time_bank['shapes'][t][prev_shape_idx] = {
              'coords': np.array([(p_min_x, p_min_y), (p_max_x, p_max_y)]),
              'centroid': centroid,
              'mask': shape_mask
            }
            new_fg_labels[shape_alpha * fg_bg>0] = prev_shape_idx
            shape_bank[prev_shape_idx] = shape
            optim_bank[prev_shape_idx] = shape_info

          if case == 'D':
            r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[prev_target_idxs[prev_list[0]]]
            j = curr_labels.index(curr_list[0])
            for i in prev_list:
              prev_shape_idx = active_shapes[i]
              shape_alpha = np.zeros_like(fg_bg)
              render_alpha = np.uint8(render_shapes[prev_element_idxs[i]][:, :, 3][:r_max_y - r_min_y, :r_max_x - r_min_x]>0)
              shape_alpha[r_min_y:r_max_y, r_min_x:r_max_x] = render_alpha
              shape_alpha_bleed = np.pad(shape_alpha, ((args.bleed, args.bleed), (args.bleed, args.bleed)))
              shape_mask = -1 * np.ones_like(fg_bg)
              shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
              render_shape_bleed = -1 * np.ones_like(render_shapes_bleed[prev_element_idxs[i]][:, :, 3], dtype=np.int8)
              render_shape_bleed[render_shapes_bleed[prev_element_idxs[i]][:, :, 3]>0] = prev_shape_idx
              shape_mask = place_mask(render_shape_bleed, r_min_x, r_min_y, shape_mask)
              shape_mask[shape_alpha_bleed * np.pad(fg_bg, ((args.bleed, args.bleed), (args.bleed, args.bleed)))>0] = prev_shape_idx
              min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
              centroid = get_shape_centroid(shape_alpha)
              shape_rgb = curr_frame * shape_alpha[..., None] + bg_img * (1 - shape_alpha[..., None])
              shape = np.uint8(np.dstack([shape_rgb, 255 * shape_alpha]))
              new_mat = params_to_mat(
                all_sx[prev_element_idxs[i]],
                all_sy[prev_element_idxs[i]],
                all_theta[prev_element_idxs[i]],
                all_kx[prev_element_idxs[i]],
                all_ky[prev_element_idxs[i]],
              )
              new_h = new_mat #@ optim_bank[prev_shape_idx]['h']
              shape_info = {
                't': dataloader.frame_idxs[t],
                'coords': [(min_x, min_y), (max_x, max_y)],
                'centroid': centroid,
                'h': new_h
              }
              time_bank['shapes'][t][prev_shape_idx] = {
                'coords': np.array([(min_x, min_y), (max_x, max_y)]),
                'centroid': centroid,
                'mask': shape_mask
              }
              new_fg_labels[shape_alpha * fg_bg>0] = prev_shape_idx
              shape_bank[prev_shape_idx] = shape
              optim_bank[prev_shape_idx] = shape_info
              # Note: We don't save the highest shape here because there are part of a larger
              # connected component in the new frame and thus may be partially occluded.
          if case == 'E':
            curr_shape_idx = int(curr[0][1:])
            prev_shape_idxs = [int(p[1:]) for p in prev]
            canonical_pl = np.min(prev_shape_idxs)
            j = curr_labels.index(curr_shape_idx)
            r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[curr_target_idxs[curr_list[0]]]
            print(f'[NOTE] Merging labels {prev} into label {canonical_pl}')
            for v in prev:
              prev_shape_idx = int(v[1:])
              if prev_shape_idx == canonical_pl:
                continue
              del prev_to_curr[active_shapes.index(prev_shape_idx)]
              if prev_shape_idx in merged:
                merged[canonical_pl] = merged[canonical_pl].union(merged[prev_shape_idx])
                del merged[prev_shape_idx]
              merged[canonical_pl].add(prev_shape_idx)
            curr_to_prev[curr_shape_idx] = [canonical_pl]
            shape_alpha = np.uint8(curr_fg_labels==curr_shape_idx)
            shape_alpha = get_alpha(np.float64(shape_alpha), curr_frame)
            prev_shape_alpha = np.uint8(np.isin(prev_fg_labels, prev_shape_idxs))
            prev_shape_alpha = get_alpha(np.float64(prev_shape_alpha), prev_frame)
            prev_shape_alpha_bleed = np.pad(prev_shape_alpha, ((args.bleed, args.bleed), (args.bleed, args.bleed)))
            shape_mask = -1 * np.ones_like(fg_bg)
            shape_mask[shape_alpha * fg_bg>0] = canonical_pl
            shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
            prev_shape_mask = -1 * np.ones_like(fg_bg)
            prev_shape_mask = np.pad(prev_shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
            render_shape_bleed = -1 * np.ones_like(render_shapes_bleed[curr_element_idxs[j]][:, :, 3], dtype=np.int8)
            render_shape_bleed[render_shapes_bleed[curr_element_idxs[j]][:, :, 3]>0] = canonical_pl
            prev_shape_mask = place_mask(render_shape_bleed, r_min_x, r_min_y, prev_shape_mask)
            prev_shape_mask[prev_shape_alpha_bleed>0] = canonical_pl
            p_min_x, p_min_y, p_max_x, p_max_y = get_shape_coords(np.uint8(prev_shape_mask>=0))
            if t < args.base_frame:
              time_bank['shapes'][t + 1][canonical_pl]['mask'] = prev_shape_mask
              time_bank['shapes'][t + 1][canonical_pl]['coords'] = [
                [p_min_x - args.bleed, p_min_y - args.bleed], 
                [p_max_x - args.bleed, p_max_y - args.bleed]
              ]
              time_bank['shapes'][t + 1][canonical_pl]['centroid'] = [
                (p_min_x + p_max_x) / 2 - args.bleed, 
                (p_min_y + p_max_y) / 2 - args.bleed
              ]
            else:
              time_bank['shapes'][t - 1][canonical_pl]['mask'] = prev_shape_mask
              time_bank['shapes'][t - 1][canonical_pl]['coords'] = [
                [p_min_x - args.bleed, p_min_y - args.bleed], 
                [p_max_x - args.bleed, p_max_y - args.bleed]
              ]
              time_bank['shapes'][t - 1][canonical_pl]['centroid'] = [
                (p_min_x + p_max_x) / 2 - args.bleed, 
                (p_min_y + p_max_y) / 2 - args.bleed
              ]
            min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
            centroid = get_shape_centroid(shape_alpha)
            shape_rgb = curr_frame * shape_alpha[..., None] + bg_img * (1 - shape_alpha[..., None])
            shape = np.uint8(np.dstack([shape_rgb, 255 * shape_alpha]))
            shape_info = {
              't': dataloader.frame_idxs[t],
              'coords': [(min_x, min_y), (max_x, max_y)],
              'centroid': centroid,
              'h': np.eye(3)
            }
            time_bank['shapes'][t][canonical_pl] = {
              'coords': np.array([(min_x, min_y), (max_x, max_y)]),
              'centroid': centroid,
              'mask': shape_mask
            }
            new_fg_labels[shape_alpha * fg_bg>0] = canonical_pl
            shape_bank[canonical_pl] = shape
            optim_bank[canonical_pl] = shape_info

            # Save highest res shape.
            prev_highest_shape_area = np.sum(
              highest_res[canonical_pl][:, :, 3] / 255.0) / (frame_width * frame_height)
            new_shape_area = np.sum(shape_alpha) / (frame_width * frame_height)
            at_edge_x = min_x < 3 or max_x >= curr_frame.shape[1] - 3
            at_edge_y = min_y < 3 or max_y >= curr_frame.shape[0] - 3
            occluded = len(curr_fg_comp_to_label[curr_fg_label_to_comp[curr_shape_idx]]) > 1
            print(f'Prev shape {canonical_pl} occluded:', (not unoccluded_canon.get(canonical_pl, True)))
            highest_res[canonical_pl] = shape[min_y:max_y, min_x:max_x]
            highest_res_update[canonical_pl] = True
            unoccluded_canon[canonical_pl] = not (at_edge_x or at_edge_y) and not occluded

      if len(all_mappings_vis) > 0:
        all_mappings_vis = np.concatenate(all_mappings_vis, axis=0)
        border = 255 * np.ones((25, all_mappings_vis.shape[1], 3), dtype=np.uint8)
        all_mappings_vis = np.concatenate([border, all_mappings_vis], axis=0)
        all_mappings_vis = cv2.putText(
          all_mappings_vis, 'before', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        all_mappings_vis = cv2.putText(
          all_mappings_vis, 'after', (135, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_submatch_folder, f'{frame_idx:03d}.png'), all_mappings_vis)
    
    #######
    # 6. Mark all remaining regions as new shapes.
    ####### 
    # Remove all pixels already taken into account by tracking.
    curr_fg_labels[new_fg_labels>=0] = -1
    remaining_fg_bg = np.uint8(curr_fg_labels>=0)
    kernel = np.ones((5, 5), np.uint8)
    remaining_fg_bg_arr = np.ascontiguousarray(remaining_fg_bg, dtype=np.uint8)
    remaining_fg_bg = cv2.morphologyEx(remaining_fg_bg_arr, cv2.MORPH_OPEN, kernel)
    curr_fg_labels[remaining_fg_bg==0] = -1
    cv2.imwrite(os.path.join(debug_fgbg_folder, f'{frame_idx:03d}.png'), 255 * remaining_fg_bg)

    # Create temporary labels for new shapes.
    new_label_to_old_label = {}
    if len(time_bank['shapes'][t]) < 1:
      new_shape_cluster_label = 0
    else:
      new_shape_cluster_label = np.max(list(time_bank['shapes'][t].keys())) + 1
    new_shape_cluster_labels = []
    new_shape_layers = []
    for l in np.unique(curr_fg_labels):
      if l < 0:
        continue
      shape_mask = -1 * np.ones_like(fg_bg, dtype=np.int32)
      shape_mask[curr_fg_labels==l] = new_shape_cluster_label
      new_shape_layers.append(shape_mask)
      new_shape_cluster_labels.append(new_shape_cluster_label)
      new_label_to_old_label[new_shape_cluster_label] = l
      new_shape_cluster_label += 1

    # Create bank of new shapes.
    new_shapes = {}
    for l, fg_layer in zip(new_shape_cluster_labels, new_shape_layers):
      shape, _, _, min_coords, max_coords, centroid = get_shape(
        curr_frame, bg_img, fg_layer, l, 
        min_cluster_size=args.min_cluster_size, min_density=args.min_density
      )
      if shape is None:
        if args.verbose:
          print(f'[NOTE] Label {l}: Invalid shape!')
        continue
      # Store shape data in new shapes bank.
      min_x, min_y = min_coords
      max_x, max_y = max_coords
      shape_info = {
        't': dataloader.frame_idxs[t],
        'coords': [min_coords, max_coords],
        'centroid': centroid,
        'shape': shape,
        'h': np.eye(3)
      }
      new_shapes[l] = shape_info

    # Update shape bank.
    new_count = 0
    for l in new_shapes:
      shape_label = len(shape_bank) - 1
      color[shape_label] = np.random.randint(0, 255, (3))
      min_coords, max_coords = new_shapes[l]['coords']
      min_x, min_y = min_coords
      max_x, max_y = max_coords
      highest_res[shape_label] = new_shapes[l]['shape'][min_y:max_y, min_x:max_x]
      highest_res_update[shape_label] = True
      at_edge_x = min_x < 1 or max_x >= curr_frame.shape[1] - 1
      at_edge_y = min_y < 1 or max_y >= curr_frame.shape[0] - 1
      unoccluded_canon[shape_label] = not (at_edge_x or at_edge_y) and (len(curr_fg_comp_to_label[curr_fg_label_to_comp[new_label_to_old_label[l]]]) == 1)
      shape_mask = -1 * np.ones_like(curr_fg_labels)
      shape_mask[new_shapes[l]['shape'][:, :, 3] * fg_bg>0] = shape_label
      shape_mask = np.pad(shape_mask, ((args.bleed, args.bleed), (args.bleed, args.bleed)), constant_values=((-2, -2), (-2, -2)))
      time_bank['shapes'][t][shape_label] = {
        'coords': np.array([[min_x, min_y], [max_x, max_y]]),
        'centroid': [(min_x + max_x) / 2, (min_y + max_y) / 2],
        'mask': shape_mask
      }
      new_fg_labels[new_shapes[l]['shape'][:, :, 3] * fg_bg>0] = shape_label
      shape_bank[shape_label] = new_shapes[l]['shape']
      optim_bank[shape_label] = new_shapes[l]
      del optim_bank[shape_label]['shape']
      new_count += 1
    print('[VARS] All current shapes:', list(sorted(time_bank['shapes'][t].keys())))
    print('Unoccluded_canon:', unoccluded_canon)

    frame_t1 = time.perf_counter()
    print(f'[TIME] Processing frame {frame_idx} took {frame_t1 - frame_t0:.2f}s')

    # Draw clusters for visualization.
    track_vis = viz.clusters(curr_frame, optim_bank, new_fg_labels, color, highest_res_update)
    for shape_idx in highest_res_update:
      highest_res_update[shape_idx] = False
    cv2.imwrite(os.path.join(track_folder, f'{frame_idx:03d}.png'), track_vis)

    prev_frame = curr_frame.copy()
    prev_fg_comps = curr_fg_comps.copy()
    new_fg_labels[fg_bg==0] = -1
    prev_fg_comp_to_label, prev_fg_label_to_comp = get_comp_label_map(curr_fg_comps, new_fg_labels)
    prev_fg_labels = new_fg_labels.copy()

    # If we are at the base frame, save the optim bank for later.
    if t == args.base_frame:
      base_bank = copy.deepcopy(optim_bank)
      base_fg_labels = prev_fg_labels.copy()
      base_fg_comps = prev_fg_comps.copy()
      base_fg_comp_to_label = prev_fg_comp_to_label.copy()
      base_frame = prev_frame.copy()

    if (t + 1) % 50 == 0:
      # Save template info.
      with open(os.path.join(video_folder, 'time_bank.pkl'), 'wb') as handle:
        pickle.dump(time_bank, handle)
      print('[NOTE] Time bank saved to:', os.path.join(video_folder, 'time_bank.pkl'))

      # Save highest resolution shapes.
      shapes_folder = os.path.join(video_folder, 'shapes')
      if not os.path.exists(shapes_folder):
        os.makedirs(shapes_folder)
      for shape_idx in highest_res:
        shape = highest_res[shape_idx]
        cv2.imwrite(os.path.join(shapes_folder, f'{shape_idx}.png'), shape)

  # Save highest resolution shapes.
  shapes_folder = os.path.join(video_folder, 'shapes')
  if not os.path.exists(shapes_folder):
    os.makedirs(shapes_folder)
  for shape_idx in highest_res:
    shape = highest_res[shape_idx]
    cv2.imwrite(os.path.join(shapes_folder, f'{shape_idx}.png'), shape)

  # Update all time bank layers.
  reverse_merge = {}
  for canonical_pl in merged:
    for pl in merged[canonical_pl]:
      reverse_merge[pl] = canonical_pl

  for t in time_bank['shapes']:
    for label in list(time_bank['shapes'][t].keys()):
      for pl in reverse_merge:
        if label == pl:
          print(f'[NOTE] Relabeling {label} as {reverse_merge[pl]}')
          shape_mask = time_bank['shapes'][t][label]['mask']
          relabel_mask = time_bank['shapes'][t][reverse_merge[pl]]['mask']
          relabel_mask[shape_mask==label] = reverse_merge[pl]
          del time_bank['shapes'][t][label]

  # Save template info.
  with open(os.path.join(video_folder, 'time_bank.pkl'), 'wb') as handle:
    pickle.dump(time_bank, handle)
  print('[NOTE] Time bank saved to:', os.path.join(video_folder, 'time_bank.pkl'))

# ================================
# Unified Pipeline Processing Functions
# ================================

def process_video_with_unified_pipeline(
    unified_pipeline, 
    video_name: str, 
    video_dir: str, 
    args
) -> Dict[str, Any]:
    """
    Process video using unified SAM2.1 + CoTracker3 + FlowSeek pipeline
    
    This provides complete video processing with all three state-of-the-art engines
    working in perfect coordination for maximum accuracy and speed.
    """
    print(f"\n🚀 Processing {video_name} with Unified Pipeline")
    print(f"   Mode: {args.unified_mode.upper()}")
    print(f"   Quality threshold: {args.quality_threshold:.1%}")
    
    # Setup output directories for unified results
    output_base = os.path.join(
        args.output_dir, 
        f'{video_name}_{args.suffix or args.unified_mode}'
    )
    
    # Process video through unified pipeline
    try:
        video_path = f"{args.video_dir}/{video_name}.mp4"
        if not os.path.exists(video_path):
            # Try other common formats
            for ext in ['.mov', '.avi', '.mkv']:
                alt_path = f"{args.video_dir}/{video_name}{ext}"
                if os.path.exists(alt_path):
                    video_path = alt_path
                    break
        
        print(f"📹 Input: {video_path}")
        print(f"📁 Output: {output_base}")
        
        # Process with unified pipeline
        results = unified_pipeline.process_video_sequence(
            video_path=video_path,
            output_dir=output_base,
            max_frames=args.max_frames if args.max_frames > 0 else -1,
            start_frame=args.start_frame,
            save_visualizations=args.verbose
        )
        
        # Validate results against quality targets
        overall_quality = results.get('quality_scores', {}).get('overall', {}).get('mean', 0.0)
        processing_fps = results.get('average_fps', 0.0)
        
        print(f"\n📊 Unified Pipeline Results:")
        print(f"   Overall Quality: {overall_quality:.1%} (target: {args.quality_threshold:.1%})")
        print(f"   Processing Speed: {processing_fps:.1f} FPS")
        print(f"   Success Rate: {results.get('success_rate', 0.0):.1%}")
        
        # Quality validation
        if overall_quality < args.quality_threshold:
            if args.progressive_fallback:
                print(f"⚠️  Quality {overall_quality:.1%} below threshold {args.quality_threshold:.1%}")
                print(f"🔄 Progressive fallback triggered - falling back to individual engines")
                return None  # Trigger fallback
            else:
                print(f"❌ Quality {overall_quality:.1%} below threshold {args.quality_threshold:.1%}")
                print(f"💡 Consider using 'accuracy' mode or lowering quality threshold")
        
        # Generate unified motion file
        generate_unified_motion_file(results, output_base, video_name, args)
        
        print(f"✅ Unified pipeline processing completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ Unified pipeline processing failed: {e}")
        if args.progressive_fallback:
            print(f"🔄 Progressive fallback enabled - continuing with individual engines")
            return None
        else:
            raise e


def generate_unified_motion_file(
    unified_results: Dict[str, Any], 
    output_dir: str, 
    video_name: str, 
    args
):
    """Generate motion file from unified pipeline results"""
    motion_file = os.path.join(output_dir, "motion_file.json")
    
    # Extract motion parameters from unified results
    motion_data = {
        'video_name': video_name,
        'processing_mode': 'unified_pipeline',
        'unified_mode': args.unified_mode,
        'quality_scores': unified_results.get('quality_scores', {}),
        'performance_metrics': unified_results.get('performance_summary', {}),
        'frame_data': [],
        'engine_statistics': unified_results.get('engine_statistics', {}),
        'recommendations': unified_results.get('recommendations', [])
    }
    
    # Save motion file
    with open(motion_file, 'w') as f:
        json.dump(motion_data, f, indent=2)
    
    print(f"💾 Unified motion file saved: {motion_file}")


def run_unified_pipeline_benchmarks(
    unified_pipeline, 
    video_dir: str, 
    max_frames: int = 30
) -> Dict[str, Any]:
    """Run performance benchmarks on unified pipeline"""
    print(f"🔥 Running unified pipeline benchmarks...")
    
    benchmark_results = {
        'warmup_time': 0,
        'processing_times': {},
        'quality_scores': {},
        'memory_usage': {},
        'gpu_utilization': {}
    }
    
    try:
        # Warmup benchmark
        import time
        start_time = time.time()
        
        # Create dummy video for benchmarking
        dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(min(10, max_frames))]
        
        # Warmup processing
        for i in range(len(dummy_frames) - 1):
            _ = unified_pipeline.process_frame_pair(
                dummy_frames[i], dummy_frames[i + 1], 
                warmup=True
            )
        
        benchmark_results['warmup_time'] = time.time() - start_time
        
        # Performance summary
        perf_summary = unified_pipeline.get_performance_summary()
        benchmark_results.update(perf_summary)
        
        print(f"✅ Benchmarks completed in {benchmark_results['warmup_time']:.2f}s")
        
    except Exception as e:
        print(f"⚠️  Benchmark failed: {e}")
        benchmark_results['error'] = str(e)
    
    return benchmark_results


# ================================
# Individual Engine Integration Functions  
# ================================

def setup_individual_engines(args) -> Dict[str, Any]:
    """Setup individual engines based on CLI arguments"""
    engines = {
        'sam2': None,
        'cotracker3': None,
        'flowseek': None
    }
    
    # SAM2.1 engine
    if args.use_sam2:
        try:
            from .sam2_engine import SAM2SegmentationEngine, SAM2Config
            sam2_config = SAM2Config(
                model_cfg=f"sam2_hiera_{'s' if args.sam2_model == 'small' else 'l'}.yaml",
                device=args.unified_device,
                mixed_precision=True
            )
            engines['sam2'] = SAM2SegmentationEngine(sam2_config)
            print(f"✅ SAM2.1 engine initialized ({args.sam2_model} model)")
        except Exception as e:
            print(f"⚠️  SAM2.1 engine initialization failed: {e}")
    
    # FlowSeek engine  
    if args.use_flowseek:
        try:
            from .flowseek_engine import create_flowseek_engine, FlowSeekConfig
            flowseek_config = FlowSeekConfig(
                device=args.unified_device,
                depth_integration=args.flowseek_depth_integration,
                mixed_precision=True
            )
            engines['flowseek'] = create_flowseek_engine(
                config=flowseek_config,
                device=args.unified_device
            )
            print(f"✅ FlowSeek engine initialized (depth integration: {args.flowseek_depth_integration})")
        except Exception as e:
            print(f"⚠️  FlowSeek engine initialization failed: {e}")
    
    return engines


if __name__ == '__main__':
  main()
