"""
SAM2-CoTracker3 Integration Bridge
Seamlessly connects SAM2.1 segmentation with CoTracker3 point tracking for motion graphics

This bridge enables:
- Automatic contour point extraction from SAM2.1 masks
- Conversion of tracking data to motion parameters
- Object lifecycle management (appearance/disappearance)
- Temporal consistency across frames
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from collections import defaultdict, OrderedDict
import time

from .sam2_engine import SAM2SegmentationEngine, SAM2Config
from .cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config


@dataclass
class BridgeConfig:
    """Configuration for SAM2-CoTracker3 integration bridge"""
    # Contour extraction
    contour_point_density: int = 30  # Points per object contour
    min_contour_points: int = 8  # Minimum points for valid tracking
    max_contour_points: int = 100  # Maximum points per object
    contour_simplification: float = 2.0  # Douglas-Peucker epsilon
    
    # Point selection strategy
    point_selection_mode: str = "adaptive"  # "uniform", "curvature", "adaptive"
    curvature_threshold: float = 0.1  # For curvature-based selection
    corner_detection: bool = True  # Use Harris corner detection
    
    # Tracking integration
    temporal_window: int = 30  # Frames to look back for consistency
    visibility_threshold: float = 0.5  # Point visibility threshold
    motion_smoothing: float = 0.8  # Motion parameter smoothing factor
    
    # Object lifecycle management
    object_persistence: int = 10  # Frames to keep disappeared objects
    reappearance_threshold: float = 0.7  # IoU threshold for reappearance
    merge_threshold: float = 0.8  # IoU threshold for object merging
    
    # Performance optimization
    batch_processing: bool = True
    memory_efficient: bool = True
    cache_features: bool = True


class SAM2CoTrackerBridge:
    """
    Integration bridge between SAM2.1 segmentation and CoTracker3 tracking
    
    Provides seamless workflow:
    1. SAM2.1 segments objects in video frames
    2. Extract contour points from each object mask
    3. CoTracker3 tracks these points across frames
    4. Convert tracking to motion parameters and shape deformation
    """
    
    def __init__(
        self,
        sam2_config: Optional[SAM2Config] = None,
        cotracker_config: Optional[CoTracker3Config] = None,
        bridge_config: Optional[BridgeConfig] = None
    ):
        self.sam2_config = sam2_config or SAM2Config()
        self.cotracker_config = cotracker_config or CoTracker3Config()
        self.bridge_config = bridge_config or BridgeConfig()
        
        # Initialize engines
        self.sam2_engine = SAM2SegmentationEngine(self.sam2_config)
        self.cotracker_engine = CoTracker3TrackerEngine(self.cotracker_config)
        
        # State tracking
        self.object_registry = {}  # object_id -> metadata
        self.point_mappings = {}  # object_id -> point indices
        self.motion_history = defaultdict(list)  # object_id -> motion params
        self.visibility_history = defaultdict(list)  # object_id -> visibility
        
        # Performance monitoring
        self.processing_stats = {
            'frames_processed': 0,
            'objects_tracked': 0,
            'total_points': 0,
            'processing_time': 0.0,
            'accuracy_metrics': []
        }
        
        print(f"🌉 SAM2-CoTracker3 Bridge initialized")
        print(f"   Contour density: {self.bridge_config.contour_point_density} points/object")
        print(f"   Tracking mode: {self.cotracker_config.model_variant}")
        
    def process_video(
        self,
        video: torch.Tensor,
        prompts: Optional[Dict] = None,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline: segmentation → tracking → motion analysis
        
        Args:
            video: Input video tensor (B, T, C, H, W)
            prompts: SAM2.1 prompts for segmentation guidance
            return_intermediate: Return intermediate results for debugging
            
        Returns:
            Complete motion analysis results
        """
        start_time = time.perf_counter()
        B, T, C, H, W = video.shape
        
        print(f"🎬 Processing video: {T} frames at {H}×{W}")
        
        # Step 1: SAM2.1 Segmentation
        print("🔍 Step 1: SAM2.1 Segmentation...")
        # Convert video tensor to list of frames for batch processing
        frames_list = []
        frame_indices = []
        for t in range(T):
            frame = video[0, t].permute(1, 2, 0).cpu().numpy()
            frames_list.append(frame)
            frame_indices.append(t)
        
        masks_list, metadata = self.sam2_engine.segment_video_batch(
            frames_list, frame_indices, prompts
        )
        
        # Convert back to tensor format
        if masks_list:
            masks = torch.stack([torch.from_numpy(mask) for mask in masks_list], dim=1).unsqueeze(0)
        else:
            masks = torch.zeros(B, T, H, W, dtype=torch.long)
        
        segmentation_results = {
            'masks': masks,
            'metadata': metadata
        }
        
        masks = segmentation_results['masks']  # (B, T, H, W)
        object_ids = segmentation_results.get('object_ids', [])
        
        # Step 2: Extract contour points for each object
        print("📍 Step 2: Contour point extraction...")
        contour_points, object_metadata = self._extract_all_contour_points(
            masks, object_ids
        )
        
        # Step 3: CoTracker3 point tracking
        print("🎯 Step 3: CoTracker3 point tracking...")
        tracking_results = self._track_all_objects(
            video, contour_points, object_metadata
        )
        
        # Step 4: Motion parameter extraction and analysis
        print("📊 Step 4: Motion analysis...")
        motion_analysis = self._analyze_motion_parameters(
            tracking_results, object_metadata
        )
        
        # Step 5: Temporal consistency and refinement
        print("🔧 Step 5: Temporal refinement...")
        refined_results = self._apply_temporal_consistency(
            motion_analysis, video.shape
        )
        
        # Compile final results
        processing_time = time.perf_counter() - start_time
        self._update_processing_stats(processing_time, refined_results)
        
        final_results = {
            'motion_parameters': refined_results,
            'object_tracks': tracking_results,
            'segmentation_masks': masks,
            'object_metadata': object_metadata,
            'performance_stats': self.get_performance_report(),
            'processing_time': processing_time,
            'fps': T / processing_time
        }
        
        if return_intermediate:
            final_results.update({
                'contour_points': contour_points,
                'segmentation_raw': segmentation_results,
                'motion_history': dict(self.motion_history),
                'visibility_history': dict(self.visibility_history)
            })
        
        print(f"✅ Video processing complete: {T/processing_time:.1f} FPS")
        return final_results
        
    def _extract_all_contour_points(
        self,
        masks: torch.Tensor,
        object_ids: List[int]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict]]:
        """Extract contour points for all objects across all frames"""
        B, T, H, W = masks.shape
        
        contour_points = {}
        object_metadata = {}
        
        for obj_id in object_ids:
            print(f"   Extracting contours for object {obj_id}...")
            
            # Find frames where object is visible
            obj_mask = (masks == obj_id)
            visible_frames = torch.nonzero(obj_mask.any(dim=(2, 3)), as_tuple=False)
            
            if len(visible_frames) == 0:
                continue
                
            # Extract contour from first visible frame
            first_frame_idx = int(visible_frames[0, 1].item())
            first_mask = obj_mask[0, first_frame_idx].cpu().numpy().astype(np.uint8) * 255
            
            # Extract and refine contour points
            points = self._extract_object_contour_points(
                first_mask, obj_id
            )
            
            if points is not None and len(points) >= self.bridge_config.min_contour_points:
                contour_points[obj_id] = points
                
                # Store object metadata
                object_metadata[obj_id] = {
                    'first_appearance': first_frame_idx,
                    'last_appearance': visible_frames[-1, 1].item(),
                    'total_visible_frames': len(visible_frames),
                    'num_points': len(points[0]),
                    'contour_area': cv2.contourArea(self._get_largest_contour(first_mask)),
                    'bounding_box': self._get_object_bbox(first_mask)
                }
                
        return contour_points, object_metadata
        
    def _extract_object_contour_points(
        self,
        mask: np.ndarray,
        obj_id: int
    ) -> Optional[torch.Tensor]:
        """Extract optimized contour points for a single object"""
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = self._get_largest_contour(mask)
        
        if len(largest_contour) < self.bridge_config.min_contour_points:
            return None
            
        # Simplify contour using Douglas-Peucker algorithm
        epsilon = self.bridge_config.contour_simplification
        simplified_contour = cv2.approxPolyDP(
            largest_contour, epsilon, closed=True
        )
        
        # Select points based on strategy
        if self.bridge_config.point_selection_mode == "uniform":
            selected_points = self._uniform_point_selection(
                simplified_contour, self.bridge_config.contour_point_density
            )
        elif self.bridge_config.point_selection_mode == "curvature":
            selected_points = self._curvature_based_selection(
                simplified_contour, self.bridge_config.contour_point_density
            )
        else:  # adaptive
            selected_points = self._adaptive_point_selection(
                simplified_contour, mask, self.bridge_config.contour_point_density
            )
            
        # Convert to tensor format (1, N, 2) for CoTracker3
        if len(selected_points) > 0:
            points_tensor = torch.tensor(
                selected_points[None, :, :], 
                dtype=torch.float32,
                device=self.cotracker_engine.device
            )
            return points_tensor
            
        return None
        
    def _uniform_point_selection(
        self, 
        contour: np.ndarray, 
        num_points: int
    ) -> np.ndarray:
        """Uniform sampling along contour perimeter"""
        contour_points = contour.reshape(-1, 2)
        
        if len(contour_points) <= num_points:
            return contour_points
            
        # Sample uniformly along contour
        indices = np.linspace(0, len(contour_points) - 1, num_points, dtype=int)
        return contour_points[indices]
        
    def _curvature_based_selection(
        self,
        contour: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Select points based on contour curvature (corners and high-curvature regions)"""
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(contour_points) <= num_points:
            return contour_points
            
        # Calculate curvature at each point
        curvatures = self._calculate_contour_curvature(contour_points)
        
        # Select high-curvature points
        curvature_threshold = np.percentile(curvatures, 100 - (num_points / len(contour_points) * 100))
        high_curvature_indices = np.where(curvatures >= curvature_threshold)[0]
        
        # If we have too many, select top ones
        if len(high_curvature_indices) > num_points:
            top_indices = np.argsort(curvatures[high_curvature_indices])[-num_points:]
            selected_indices = high_curvature_indices[top_indices]
        else:
            # Fill remaining with uniform sampling
            remaining = num_points - len(high_curvature_indices)
            uniform_indices = np.linspace(
                0, len(contour_points) - 1, remaining, dtype=int
            )
            # Remove duplicates
            all_indices = np.unique(np.concatenate([high_curvature_indices, uniform_indices]))[:num_points]
            selected_indices = all_indices
            
        return contour_points[selected_indices]
        
    def _adaptive_point_selection(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Adaptive point selection combining curvature and corner detection"""
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(contour_points) <= num_points:
            return contour_points
            
        # 1. Detect corners using Harris corner detection
        corners = []
        if self.bridge_config.corner_detection:
            corners_response = cv2.cornerHarris(mask, 2, 3, 0.04)
            corner_coords = np.where(corners_response > 0.01 * corners_response.max())
            corners = list(zip(corner_coords[1], corner_coords[0]))  # (x, y) format
            
        # 2. Calculate curvature
        curvatures = self._calculate_contour_curvature(contour_points)
        
        # 3. Combine corner and curvature scores
        scores = curvatures.copy()
        
        # Boost scores near detected corners
        for corner in corners:
            distances = np.sqrt(np.sum((contour_points - corner)**2, axis=1))
            close_mask = distances < 10  # Within 10 pixels of corner
            scores[close_mask] *= 2.0
            
        # 4. Select top scoring points
        top_indices = np.argsort(scores)[-num_points:]
        
        return contour_points[top_indices]
        
    def _calculate_contour_curvature(self, points: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point along the contour"""
        n = len(points)
        curvatures = np.zeros(n)
        
        for i in range(n):
            # Get neighboring points (with wrapping)
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            
            p1 = points[prev_idx]
            p2 = points[i]
            p3 = points[next_idx]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 1e-6:
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures[i] = angle
            else:
                curvatures[i] = 0.0
                
        return curvatures
        
    def _get_largest_contour(self, mask: np.ndarray) -> np.ndarray:
        """Get the largest contour from a binary mask"""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.array([])
            
        return max(contours, key=cv2.contourArea)
        
    def _get_object_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of object in mask"""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
            
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max, y_max)
        
    def _track_all_objects(
        self,
        video: torch.Tensor,
        contour_points: Dict[int, torch.Tensor],
        object_metadata: Dict[int, Dict]
    ) -> Dict[int, Dict]:
        """Track all objects using CoTracker3"""
        tracking_results = {}
        
        for obj_id, points in contour_points.items():
            print(f"   Tracking object {obj_id} with {points.shape[1]} points...")
            
            # Track points for this object
            tracks, visibility = self.cotracker_engine.track_video_grid(
                video, custom_points=points
            )
            
            # Store tracking results
            tracking_results[obj_id] = {
                'tracks': tracks,  # (B, T, N, 2)
                'visibility': visibility,  # (B, T, N, 1)
                'num_points': points.shape[1],
                'metadata': object_metadata[obj_id]
            }
            
            # Update point mappings
            self.point_mappings[obj_id] = list(range(points.shape[1]))
            
        return tracking_results
        
    def _analyze_motion_parameters(
        self,
        tracking_results: Dict[int, Dict],
        object_metadata: Dict[int, Dict]
    ) -> Dict[int, Dict]:
        """Extract motion parameters from tracking results"""
        motion_analysis = {}
        
        for obj_id, track_data in tracking_results.items():
            tracks = track_data['tracks']
            visibility = track_data['visibility']
            
            # Extract motion parameters using CoTracker3 engine
            motion_params = self.cotracker_engine.extract_motion_parameters(
                tracks, visibility
            )
            
            # Add object-specific analysis
            motion_analysis[obj_id] = {
                'translation': motion_params['translation'],
                'rotation': motion_params['rotation'],
                'scale': motion_params['scale'],
                'affine_matrix': motion_params['affine_matrix'],
                'velocity': self._calculate_velocity(motion_params['translation']),
                'acceleration': self._calculate_acceleration(motion_params['translation']),
                'angular_velocity': self._calculate_angular_velocity(motion_params['rotation']),
                'deformation': self._analyze_shape_deformation(tracks, visibility),
                'quality_score': self._calculate_tracking_quality(tracks, visibility)
            }
            
            # Store in history for temporal consistency
            self.motion_history[obj_id].append(motion_analysis[obj_id])
            self.visibility_history[obj_id].append(visibility.mean().item())
            
        return motion_analysis
        
    def _calculate_velocity(self, translation: torch.Tensor) -> torch.Tensor:
        """Calculate velocity from translation parameters"""
        return torch.diff(translation, dim=1)
        
    def _calculate_acceleration(self, translation: torch.Tensor) -> torch.Tensor:
        """Calculate acceleration from translation parameters"""
        velocity = self._calculate_velocity(translation)
        return torch.diff(velocity, dim=1)
        
    def _calculate_angular_velocity(self, rotation: torch.Tensor) -> torch.Tensor:
        """Calculate angular velocity from rotation parameters"""
        return torch.diff(rotation, dim=1)
        
    def _analyze_shape_deformation(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze shape deformation beyond rigid body motion"""
        B, T, N, _ = tracks.shape
        
        deformation_metrics = {}
        reference_distances = None
        
        # Calculate inter-point distances for deformation analysis
        for t in range(T):
            valid_mask = visibility[0, t, :, 0] > self.bridge_config.visibility_threshold
            valid_points = tracks[0, t, valid_mask]
            
            if len(valid_points) < 3:
                continue
                
            # Calculate pairwise distances
            distances = torch.cdist(valid_points, valid_points)
            
            if t == 0:
                reference_distances = distances
                deformation_metrics['reference_distances'] = reference_distances
            else:
                # Compare with reference to detect deformation
                if reference_distances is not None and distances.shape == reference_distances.shape:
                    relative_change = (distances - reference_distances) / (reference_distances + 1e-6)
                    deformation_magnitude = torch.mean(torch.abs(relative_change))
                    
                    if 'deformation_magnitude' not in deformation_metrics:
                        deformation_metrics['deformation_magnitude'] = []
                    deformation_metrics['deformation_magnitude'].append(deformation_magnitude)
                    
        # Convert lists to tensors
        for key, value in deformation_metrics.items():
            if isinstance(value, list) and value:
                deformation_metrics[key] = torch.stack(value)
                
        return deformation_metrics
        
    def _calculate_tracking_quality(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor
    ) -> float:
        """Calculate overall tracking quality score"""
        # Visibility-based quality
        visibility_score = visibility.mean().item()
        
        # Motion smoothness quality
        velocities = torch.diff(tracks, dim=1)
        accelerations = torch.diff(velocities, dim=1)
        smoothness_score = 1.0 / (1.0 + torch.std(accelerations).item())
        
        # Temporal consistency quality
        temporal_score = self._calculate_temporal_consistency_score(tracks)
        
        # Combined quality score
        quality_score = 0.4 * visibility_score + 0.3 * smoothness_score + 0.3 * temporal_score
        
        return float(quality_score)
        
    def _calculate_temporal_consistency_score(self, tracks: torch.Tensor) -> float:
        """Calculate temporal consistency of tracking"""
        # Check for sudden jumps or inconsistencies
        velocities = torch.diff(tracks, dim=1)
        velocity_magnitudes = torch.norm(velocities, dim=-1)
        
        # Sudden changes indicate poor tracking
        velocity_changes = torch.diff(velocity_magnitudes, dim=1)
        consistency_score = 1.0 / (1.0 + torch.std(velocity_changes).item())
        
        return float(consistency_score)
        
    def _apply_temporal_consistency(
        self,
        motion_analysis: Dict[int, Dict],
        video_shape: Tuple[int, ...]
    ) -> Dict[int, Dict]:
        """Apply temporal smoothing and consistency constraints"""
        refined_analysis = {}
        
        for obj_id, motion_data in motion_analysis.items():
            refined_motion = motion_data.copy()
            
            # Apply smoothing to motion parameters
            smoothing_factor = self.bridge_config.motion_smoothing
            
            for param_name in ['translation', 'rotation', 'scale']:
                if param_name in refined_motion:
                    param_data = refined_motion[param_name]
                    
                    # Apply exponential moving average for smoothing
                    smoothed_param = param_data.clone()
                    for t in range(1, param_data.shape[1]):
                        smoothed_param[:, t] = (
                            smoothing_factor * smoothed_param[:, t-1] + 
                            (1 - smoothing_factor) * param_data[:, t]
                        )
                    
                    refined_motion[f'{param_name}_smoothed'] = smoothed_param
                    
            # Apply outlier detection and correction
            refined_motion = self._correct_motion_outliers(refined_motion)
            
            refined_analysis[obj_id] = refined_motion
            
        return refined_analysis
        
    def _correct_motion_outliers(self, motion_data: Dict) -> Dict:
        """Detect and correct motion parameter outliers"""
        corrected_data = motion_data.copy()
        
        for param_name in ['translation', 'rotation', 'scale']:
            if param_name in motion_data:
                param_tensor = motion_data[param_name]
                
                # Calculate z-scores for outlier detection
                mean_val = torch.mean(param_tensor, dim=1, keepdim=True)
                std_val = torch.std(param_tensor, dim=1, keepdim=True)
                z_scores = torch.abs((param_tensor - mean_val) / (std_val + 1e-6))
                
                # Mark outliers (z-score > 3)
                outlier_mask = z_scores > 3.0
                
                # Replace outliers with interpolated values
                if outlier_mask.any():
                    corrected_tensor = param_tensor.clone()
                    
                    # Simple linear interpolation for outliers
                    for b in range(param_tensor.shape[0]):
                        for dim in range(param_tensor.shape[-1]):
                            outliers = outlier_mask[b, :, dim] if param_tensor.dim() == 3 else outlier_mask[b, :]
                            
                            if outliers.any():
                                valid_indices = (~outliers).nonzero(as_tuple=False).squeeze()
                                outlier_indices = outliers.nonzero(as_tuple=False).squeeze()
                                
                                if len(valid_indices) >= 2 and len(outlier_indices) > 0:
                                    # Interpolate
                                    if param_tensor.dim() == 3:
                                        valid_values = param_tensor[b, valid_indices, dim]
                                        interp_values = torch.nn.functional.interpolate(
                                            valid_values.unsqueeze(0).unsqueeze(0),
                                            size=len(outlier_indices),
                                            mode='linear',
                                            align_corners=True
                                        ).squeeze()
                                        corrected_tensor[b, outlier_indices, dim] = interp_values
                                    else:
                                        valid_values = param_tensor[b, valid_indices]
                                        interp_values = torch.nn.functional.interpolate(
                                            valid_values.unsqueeze(0).unsqueeze(0),
                                            size=len(outlier_indices),
                                            mode='linear',
                                            align_corners=True
                                        ).squeeze()
                                        corrected_tensor[b, outlier_indices] = interp_values
                    
                    corrected_data[f'{param_name}_corrected'] = corrected_tensor
                    
        return corrected_data
        
    def _update_processing_stats(self, processing_time: float, results: Dict):
        """Update processing performance statistics"""
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['processing_time'] += processing_time
        self.processing_stats['objects_tracked'] += len(results)
        
        # Calculate total points tracked
        total_points = sum(
            obj_data.get('num_points', 0) 
            for obj_data in results.values()
        )
        self.processing_stats['total_points'] += total_points
        
        # Calculate average accuracy
        accuracies = [
            obj_data.get('quality_score', 0.0)
            for obj_data in results.values()
        ]
        if accuracies:
            self.processing_stats['accuracy_metrics'].extend(accuracies)
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        stats = self.processing_stats.copy()
        
        # Calculate derived metrics
        if stats['frames_processed'] > 0:
            stats['average_processing_time'] = stats['processing_time'] / stats['frames_processed']
            stats['average_objects_per_frame'] = stats['objects_tracked'] / stats['frames_processed']
            stats['average_points_per_frame'] = stats['total_points'] / stats['frames_processed']
        else:
            stats['average_processing_time'] = 0.0
            stats['average_objects_per_frame'] = 0.0
            stats['average_points_per_frame'] = 0.0
            
        if stats['accuracy_metrics']:
            stats['average_accuracy'] = np.mean(stats['accuracy_metrics'])
            stats['accuracy_std'] = np.std(stats['accuracy_metrics'])
        else:
            stats['average_accuracy'] = 0.0
            stats['accuracy_std'] = 0.0
            
        # Performance rating
        stats['performance_rating'] = self._calculate_bridge_performance_rating(stats)
        
        return stats
        
    def _calculate_bridge_performance_rating(self, stats: Dict) -> str:
        """Calculate overall bridge performance rating"""
        accuracy = stats['average_accuracy']
        processing_time = stats['average_processing_time']
        
        if accuracy >= 0.95 and processing_time <= 0.05:  # < 50ms per frame
            return "EXCELLENT"
        elif accuracy >= 0.90 and processing_time <= 0.1:  # < 100ms per frame
            return "GOOD"
        elif accuracy >= 0.80 and processing_time <= 0.2:  # < 200ms per frame
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"
            
    def reset_state(self):
        """Reset bridge state for new video processing"""
        self.object_registry.clear()
        self.point_mappings.clear()
        self.motion_history.clear()
        self.visibility_history.clear()
        
        # Reset engine states
        self.cotracker_engine.reset_tracking_state()
        
    def cleanup(self):
        """Cleanup bridge resources"""
        self.reset_state()
        self.sam2_engine.cleanup()
        self.cotracker_engine.cleanup()
        print("🧹 SAM2-CoTracker3 bridge cleaned up")


# Factory function for easy creation
def create_sam2_cotracker_bridge(
    sam2_accuracy: str = "high",
    cotracker_mode: str = "offline",
    contour_density: int = 30,
    device: str = "auto",
    **kwargs
) -> SAM2CoTrackerBridge:
    """
    Factory function to create optimized SAM2-CoTracker3 bridge
    
    Args:
        sam2_accuracy: "high", "medium", "fast"
        cotracker_mode: "offline", "online"
        contour_density: Points per object contour
        device: "auto", "cuda", "cpu"
        **kwargs: Additional configuration
        
    Returns:
        Configured SAM2CoTrackerBridge instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # SAM2 configuration based on accuracy
    sam2_configs = {
        "high": {"model_cfg": "sam2_hiera_l.yaml", "mixed_precision": True},
        "medium": {"model_cfg": "sam2_hiera_b_plus.yaml", "mixed_precision": True},
        "fast": {"model_cfg": "sam2_hiera_s.yaml", "mixed_precision": True}
    }
    
    sam2_config = SAM2Config(
        device=device,
        **sam2_configs.get(sam2_accuracy, sam2_configs["high"])
    )
    
    # CoTracker3 configuration
    cotracker_config = CoTracker3Config(
        model_variant=f"cotracker3_{cotracker_mode}",
        device=device,
        mixed_precision=True
    )
    
    # Bridge configuration
    bridge_config = BridgeConfig(
        contour_point_density=contour_density,
        **kwargs
    )
    
    return SAM2CoTrackerBridge(sam2_config, cotracker_config, bridge_config)

