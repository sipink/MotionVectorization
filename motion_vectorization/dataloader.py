import os
import numpy as np
import cv2
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, List
# import torch
# from scipy import interpolate

from .utils import get_numbers

# Configure logging for better error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader():
  # Folder path attributes - set during initialization
  frame_folder: str
  labels_folder: str
  fgbg_folder: str
  comps_folder: str
  forw_flow_folder: str
  back_flow_folder: str
  
  def __init__(self, video_dir: str, max_frames: int = -1):
    """Initialize DataLoader with comprehensive validation.
    
    Args:
      video_dir: Path to video directory containing required subdirectories
      max_frames: Maximum frames to load (-1 for all frames)
      
    Raises:
      ValueError: If video_dir is invalid or required directories missing
      FileNotFoundError: If video_dir doesn't exist
    """
    self.dir = self._validate_video_dir(video_dir)
    
    # Validate max_frames parameter
    if not isinstance(max_frames, int):
      raise TypeError(f"max_frames must be an integer, got {type(max_frames)}")
    if max_frames < -1:
      raise ValueError(f"max_frames must be >= -1, got {max_frames}")

    # Required subdirectories with validation
    self.required_dirs = {
      'frame_folder': 'rgb',
      'labels_folder': 'labels', 
      'fgbg_folder': 'fgbg',
      'comps_folder': 'comps',
      'forw_flow_folder': os.path.join('flow', 'forward'),
      'back_flow_folder': os.path.join('flow', 'backward')
    }
    
    # Initialize and validate all directories
    self._initialize_directories()
    
    # Get total number of frames with validation
    try:
      self.frame_idxs = get_numbers(self.frame_folder)
      if not self.frame_idxs:
        raise ValueError(f"No frame files found in {self.frame_folder}")
      logger.info(f"Found {len(self.frame_idxs)} frames in {self.frame_folder}")
    except Exception as e:
      raise ValueError(f"Failed to get frame indices from {self.frame_folder}: {e}")
    
    self.pos = 0
    if max_frames >= 0:
      original_count = len(self.frame_idxs)
      self.frame_idxs = self.frame_idxs[:max_frames]
      logger.info(f"Limited frames from {original_count} to {len(self.frame_idxs)}")
    
    # Validate frame consistency across directories
    self._validate_frame_consistency()
    
  def _validate_video_dir(self, video_dir: str) -> str:
    """Validate video directory path."""
    if not isinstance(video_dir, (str, Path)):
      raise TypeError(f"video_dir must be a string or Path, got {type(video_dir)}")
    
    video_path = Path(video_dir)
    if not video_path.exists():
      raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    
    if not video_path.is_dir():
      raise ValueError(f"Video path is not a directory: {video_dir}")
      
    return str(video_path.absolute())
  
  def _initialize_directories(self):
    """Initialize and validate all required directories."""
    missing_dirs = []
    
    for attr_name, subdir in self.required_dirs.items():
      full_path = os.path.join(self.dir, subdir)
      setattr(self, attr_name, full_path)
      
      if not os.path.exists(full_path):
        missing_dirs.append(f"{subdir} -> {full_path}")
      elif not os.path.isdir(full_path):
        raise ValueError(f"Required path is not a directory: {full_path}")
    
    if missing_dirs:
      missing_str = "\n  ".join(missing_dirs)
      raise FileNotFoundError(
        f"Missing required directories in {self.dir}:\n  {missing_str}\n\n"
        f"Expected structure:\n"
        f"  {self.dir}/\n"
        f"    ├── rgb/\n"
        f"    ├── labels/\n"
        f"    ├── fgbg/\n"
        f"    ├── comps/\n"
        f"    └── flow/\n"
        f"        ├── forward/\n"
        f"        └── backward/"
      )
  
  def _validate_frame_consistency(self):
    """Validate that all required files exist for each frame index."""
    logger.info("Validating frame consistency across directories...")
    
    missing_files = []
    example_frame_idx = None
    for i, frame_idx in enumerate(self.frame_idxs[:5]):  # Check first 5 frames for validation
      if example_frame_idx is None:
        example_frame_idx = frame_idx
      frame_files = self._get_frame_file_paths(frame_idx)
      
      for file_type, file_path in frame_files.items():
        if not os.path.exists(file_path):
          missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
      missing_str = "\n  ".join(missing_files[:10])  # Show first 10 missing files
      more_msg = f"\n  ... and {len(missing_files) - 10} more" if len(missing_files) > 10 else ""
      # Use example frame index for error message formatting
      example_idx = example_frame_idx if example_frame_idx is not None else 0
      raise FileNotFoundError(
        f"Missing required files for frames (checked first 5):\n  {missing_str}{more_msg}\n\n"
        f"Each frame requires:\n"
        f"  - rgb/{example_idx:03d}.png\n"
        f"  - labels/{example_idx:03d}.npy\n"
        f"  - fgbg/{example_idx:03d}.npy\n"
        f"  - comps/{example_idx:03d}.npy\n"
        f"  - flow/forward/{example_idx:03d}.npy (except frame 0)\n"
        f"  - flow/backward/{example_idx:03d}.npy (except frame 0)"
      )
    
    logger.info("✅ Frame consistency validation passed")
  
  def _get_frame_file_paths(self, frame_idx: int) -> dict:
    """Get all file paths for a given frame index."""
    return {
      'frame': os.path.join(self.frame_folder, f'{frame_idx:03d}.png'),
      'labels': os.path.join(self.labels_folder, f'{frame_idx:03d}.npy'),
      'comps': os.path.join(self.comps_folder, f'{frame_idx:03d}.npy'),
      'fgbg': os.path.join(self.fgbg_folder, f'{frame_idx:03d}.npy'),
      'forw_flow': os.path.join(self.forw_flow_folder, f'{frame_idx:03d}.npy'),
      'back_flow': os.path.join(self.back_flow_folder, f'{frame_idx:03d}.npy')
    }

  def load_data(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate data for processing frame i.
    
    Args:
      i: Frame index (0-based) within self.frame_idxs
      
    Returns:
      Tuple of (frame, labels, fg_bg, comps, forw_flow, back_flow)
      
    Raises:
      IndexError: If frame index is out of range
      ValueError: If loaded data fails validation
      IOError: If files cannot be loaded
    """
    # Validate frame index
    if not isinstance(i, int):
      raise TypeError(f"Frame index must be an integer, got {type(i)}")
    if i < 0 or i >= len(self.frame_idxs):
      raise IndexError(f"Frame index {i} out of range [0, {len(self.frame_idxs)-1}]")
    
    frame_idx = self.frame_idxs[i]
    logger.debug(f"Loading frame {i} (index {frame_idx})")
    
    try:
      # Load and validate frame image
      frame = self._load_frame_image(frame_idx)
      
      # Load and validate numpy arrays
      labels = self._load_labels_array(frame_idx, frame.shape[:2])
      comps = self._load_comps_array(frame_idx, frame.shape[:2])
      fg_bg = self._load_fgbg_array(frame_idx, frame.shape[:2])
      
      # Load or create flow arrays
      if i == 0:
        # First frame: create zero flow
        forw_flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
        back_flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
        logger.debug(f"Created zero flow for first frame: {forw_flow.shape}")
      else:
        # Load flow from previous frame
        flow_frame_idx = self.frame_idxs[i - 1]
        forw_flow = self._load_flow_array(flow_frame_idx, frame.shape[:2], 'forward')
        back_flow = self._load_flow_array(flow_frame_idx, frame.shape[:2], 'backward')
      
      # Final validation - ensure all data is consistent
      self._validate_data_consistency(frame, labels, fg_bg, comps, forw_flow, back_flow, frame_idx)
      
      logger.debug(f"✅ Successfully loaded frame {i} (index {frame_idx})")
      return frame, labels, fg_bg, comps, forw_flow, back_flow
      
    except Exception as e:
      error_msg = f"Failed to load frame {i} (index {frame_idx}): {e}"
      logger.error(error_msg)
      raise IOError(error_msg) from e
  
  def _load_frame_image(self, frame_idx: int) -> np.ndarray:
    """Load and validate frame image."""
    frame_path = os.path.join(self.frame_folder, f'{frame_idx:03d}.png')
    
    if not os.path.exists(frame_path):
      raise FileNotFoundError(f"Frame image not found: {frame_path}")
    
    # Load image with error checking
    frame = cv2.imread(frame_path)
    
    if frame is None:
      # Common causes: corrupted file, unsupported format, permission issues
      file_size = os.path.getsize(frame_path) if os.path.exists(frame_path) else 0
      raise ValueError(
        f"Failed to load frame image: {frame_path}\n"
        f"File size: {file_size} bytes\n"
        f"Common causes:\n"
        f"  - Corrupted or truncated image file\n"
        f"  - Unsupported image format\n"
        f"  - Insufficient memory\n"
        f"  - File permission issues"
      )
    
    # Validate frame properties
    if len(frame.shape) != 3:
      raise ValueError(f"Expected 3D image (H,W,C), got shape {frame.shape}")
    
    if frame.shape[2] != 3:
      raise ValueError(f"Expected 3-channel BGR image, got {frame.shape[2]} channels")
    
    if frame.shape[0] < 1 or frame.shape[1] < 1:
      raise ValueError(f"Invalid frame dimensions: {frame.shape[:2]}")
    
    # Check for reasonable image size (prevent memory issues)
    max_pixels = 50_000_000  # ~50MP limit
    if frame.shape[0] * frame.shape[1] > max_pixels:
      warnings.warn(
        f"Very large frame detected: {frame.shape[:2]} "
        f"({frame.shape[0] * frame.shape[1]:,} pixels)\n"
        f"This may cause memory issues."
      )
    
    logger.debug(f"Loaded frame: {frame.shape}, dtype: {frame.dtype}")
    return frame
  
  def _load_labels_array(self, frame_idx: int, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Load and validate labels array."""
    labels_path = os.path.join(self.labels_folder, f'{frame_idx:03d}.npy')
    
    try:
      labels = np.load(labels_path)
    except Exception as e:
      raise IOError(f"Failed to load labels array from {labels_path}: {e}")
    
    # Validate labels properties
    if labels.shape != frame_shape:
      raise ValueError(
        f"Labels shape {labels.shape} doesn't match frame shape {frame_shape}\n"
        f"File: {labels_path}"
      )
    
    # Ensure labels are integer type
    if not np.issubdtype(labels.dtype, np.integer):
      warnings.warn(f"Labels array has non-integer dtype {labels.dtype}, converting to int32")
      labels = labels.astype(np.int32)
    
    # Validate label value ranges
    min_label, max_label = labels.min(), labels.max()
    if min_label < -1:
      raise ValueError(f"Invalid label values: minimum {min_label} < -1 (file: {labels_path})")
    
    if max_label > 10000:  # Reasonable upper limit
      warnings.warn(f"Very high label values detected: max {max_label}")
    
    logger.debug(f"Loaded labels: {labels.shape}, dtype: {labels.dtype}, range: [{min_label}, {max_label}]")
    return labels
  
  def _load_comps_array(self, frame_idx: int, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Load and validate components array."""
    comps_path = os.path.join(self.comps_folder, f'{frame_idx:03d}.npy')
    
    try:
      comps = np.load(comps_path)
    except Exception as e:
      raise IOError(f"Failed to load comps array from {comps_path}: {e}")
    
    # Validate comps properties
    if comps.shape != frame_shape:
      raise ValueError(
        f"Comps shape {comps.shape} doesn't match frame shape {frame_shape}\n"
        f"File: {comps_path}"
      )
    
    # Ensure comps are integer type
    if not np.issubdtype(comps.dtype, np.integer):
      warnings.warn(f"Comps array has non-integer dtype {comps.dtype}, converting to int32")
      comps = comps.astype(np.int32)
    
    # Validate component value ranges
    min_comp, max_comp = comps.min(), comps.max()
    if min_comp < -1:
      raise ValueError(f"Invalid component values: minimum {min_comp} < -1 (file: {comps_path})")
    
    logger.debug(f"Loaded comps: {comps.shape}, dtype: {comps.dtype}, range: [{min_comp}, {max_comp}]")
    return comps
  
  def _load_fgbg_array(self, frame_idx: int, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Load and validate foreground/background array."""
    fgbg_path = os.path.join(self.fgbg_folder, f'{frame_idx:03d}.npy')
    
    try:
      fg_bg = np.load(fgbg_path)
    except Exception as e:
      raise IOError(f"Failed to load fgbg array from {fgbg_path}: {e}")
    
    # Validate fg_bg properties
    if fg_bg.shape != frame_shape:
      raise ValueError(
        f"FG/BG shape {fg_bg.shape} doesn't match frame shape {frame_shape}\n"
        f"File: {fgbg_path}"
      )
    
    # Ensure fg_bg are integer type
    if not np.issubdtype(fg_bg.dtype, np.integer):
      warnings.warn(f"FG/BG array has non-integer dtype {fg_bg.dtype}, converting to int32")
      fg_bg = fg_bg.astype(np.int32)
    
    # Validate fg/bg value ranges (typically -1 for background, >= 0 for foreground)
    min_fgbg, max_fgbg = fg_bg.min(), fg_bg.max()
    if min_fgbg < -1:
      raise ValueError(f"Invalid FG/BG values: minimum {min_fgbg} < -1 (file: {fgbg_path})")
    
    logger.debug(f"Loaded fg_bg: {fg_bg.shape}, dtype: {fg_bg.dtype}, range: [{min_fgbg}, {max_fgbg}]")
    return fg_bg
  
  def _load_flow_array(self, frame_idx: int, frame_shape: Tuple[int, int], direction: str) -> np.ndarray:
    """Load and validate optical flow array."""
    if direction == 'forward':
      flow_folder = self.forw_flow_folder
    elif direction == 'backward':
      flow_folder = self.back_flow_folder
    else:
      raise ValueError(f"Invalid flow direction: {direction}. Must be 'forward' or 'backward'")
    
    flow_path = os.path.join(flow_folder, f'{frame_idx:03d}.npy')
    
    try:
      flow = np.load(flow_path)
    except Exception as e:
      raise IOError(f"Failed to load {direction} flow from {flow_path}: {e}")
    
    # Validate flow properties
    expected_shape = (frame_shape[0], frame_shape[1], 2)
    if flow.shape != expected_shape:
      raise ValueError(
        f"{direction.capitalize()} flow shape {flow.shape} doesn't match expected {expected_shape}\n"
        f"File: {flow_path}"
      )
    
    # Ensure flow is float type
    if not np.issubdtype(flow.dtype, np.floating):
      warnings.warn(f"{direction.capitalize()} flow has non-float dtype {flow.dtype}, converting to float32")
      flow = flow.astype(np.float32)
    
    # Validate flow value ranges (check for reasonable motion vectors)
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    max_flow = flow_magnitude.max()
    
    # Warn about extremely large flow values (might indicate corruption)
    max_reasonable_flow = max(frame_shape) * 0.5  # Half frame dimension
    if max_flow > max_reasonable_flow:
      warnings.warn(
        f"Very large {direction} flow values detected: max magnitude {max_flow:.1f} "
        f"(> {max_reasonable_flow:.1f})\n"
        f"This might indicate corrupted flow data."
      )
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(flow)):
      nan_count = np.sum(~np.isfinite(flow))
      raise ValueError(
        f"{direction.capitalize()} flow contains {nan_count} NaN/infinite values\n"
        f"File: {flow_path}"
      )
    
    logger.debug(f"Loaded {direction} flow: {flow.shape}, dtype: {flow.dtype}, max magnitude: {max_flow:.2f}")
    return flow
  
  def _validate_data_consistency(self, frame: np.ndarray, labels: np.ndarray, 
                                fg_bg: np.ndarray, comps: np.ndarray,
                                forw_flow: np.ndarray, back_flow: np.ndarray,
                                frame_idx: int):
    """Validate consistency between all loaded data arrays."""
    # All spatial dimensions should match
    frame_h, frame_w = frame.shape[:2]
    
    spatial_arrays = {
      'labels': labels,
      'fg_bg': fg_bg, 
      'comps': comps,
      'forw_flow': forw_flow[:, :, 0],  # Check spatial dims only
      'back_flow': back_flow[:, :, 0]
    }
    
    for name, array in spatial_arrays.items():
      if array.shape[:2] != (frame_h, frame_w):
        raise ValueError(
          f"Spatial dimension mismatch for {name}: {array.shape[:2]} != {(frame_h, frame_w)} "
          f"(frame {frame_idx})"
        )
    
    # Check logical consistency between labels and components
    unique_labels = set(np.unique(labels))
    unique_comps = set(np.unique(comps))
    
    # Components should generally correspond to labels
    if len(unique_labels) > 0 and len(unique_comps) > 0:
      # Allow some flexibility, but warn if drastically different
      label_comp_ratio = len(unique_labels) / len(unique_comps)
      if label_comp_ratio > 10 or label_comp_ratio < 0.1:
        warnings.warn(
          f"Unusual label/component ratio: {len(unique_labels)} labels vs {len(unique_comps)} components "
          f"(frame {frame_idx})"
        )
    
    # Check memory usage and warn if excessive
    total_memory_mb = sum(
      array.nbytes for array in [frame, labels, fg_bg, comps, forw_flow, back_flow]
    ) / (1024 * 1024)
    
    if total_memory_mb > 500:  # > 500MB per frame
      warnings.warn(
        f"High memory usage for frame {frame_idx}: {total_memory_mb:.1f} MB\n"
        f"Consider reducing frame resolution for large datasets."
      )
    
    logger.debug(f"✅ Data consistency validation passed for frame {frame_idx}")
  
  def get_frame_count(self) -> int:
    """Get total number of available frames."""
    return len(self.frame_idxs)
  
  def get_frame_shape(self, i: int = 0) -> Tuple[int, int, int]:
    """Get frame shape by loading the first frame.
    
    Args:
      i: Frame index to check (default: 0)
      
    Returns:
      Frame shape as (height, width, channels)
    """
    if i < 0 or i >= len(self.frame_idxs):
      raise IndexError(f"Frame index {i} out of range [0, {len(self.frame_idxs)-1}]")
    
    frame_idx = self.frame_idxs[i]
    frame_path = os.path.join(self.frame_folder, f'{frame_idx:03d}.png')
    
    # Try to get shape without loading full image (more efficient)
    try:
      import cv2
      temp_frame = cv2.imread(frame_path)
      if temp_frame is None:
        raise ValueError(f"Cannot read frame to get shape: {frame_path}")
      return temp_frame.shape
    except Exception as e:
      raise IOError(f"Failed to get frame shape from {frame_path}: {e}")
  
  def __len__(self) -> int:
    """Support len() operation."""
    return len(self.frame_idxs)
  
  def __repr__(self) -> str:
    """String representation."""
    return (
      f"DataLoader(\n"
      f"  video_dir='{self.dir}'\n"
      f"  frames={len(self.frame_idxs)}\n"
      f"  indices={self.frame_idxs[:3]}{'...' if len(self.frame_idxs) > 3 else ''}\n"
      f")"
    )