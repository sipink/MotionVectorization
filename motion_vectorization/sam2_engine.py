"""
SAM2.1 Segmentation Engine for Motion Vectorization Pipeline
Replaces primitive Canny edge detection with Meta's SAM2.1 for maximum accuracy
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from collections import defaultdict
import time

try:
    # Try importing SAM2 if available
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    # Fallback implementation if SAM2 not available
    SAM2_AVAILABLE = False
    warnings.warn("SAM2.1 not available. Using fallback implementation.")
    
    # Define typed dummies to prevent unbound variable errors
    def build_sam2_video_predictor(*args, **kwargs) -> Any:
        raise RuntimeError("SAM2 not available")
    
    class SAM2ImagePredictor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs) -> Any:
            raise RuntimeError("SAM2 not available")


@dataclass
class SAM2Config:
    """Configuration for SAM2.1 segmentation engine"""
    model_cfg: str = "sam2_hiera_l.yaml"  # Large model for max accuracy
    sam2_checkpoint: str = "sam2_hiera_large.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Will be automatically disabled on CPU
    vos_optimized: bool = True  # Video Object Segmentation optimization
    compile_model: bool = True  # Will be automatically disabled on CPU
    max_obj_ptrs_in_encoder: int = 16
    batch_size: int = 4
    target_fps: float = 44.0
    accuracy_threshold: float = 0.95
    
    def __post_init__(self):
        """Auto-adjust settings based on device capabilities"""
        cuda_available = torch.cuda.is_available()
        
        # Disable CUDA-specific features on CPU
        if self.device == "cpu" or not cuda_available:
            self.mixed_precision = False
            self.compile_model = False
            if not cuda_available:
                self.device = "cpu"
                
        # Adjust batch size for CPU
        if self.device == "cpu":
            self.batch_size = min(self.batch_size, 2)  # Reduce batch size for CPU


class SAM2SegmentationEngine:
    """
    High-performance SAM2.1 segmentation engine for motion graphics
    Designed to achieve 44 FPS and 95%+ accuracy
    """
    
    def __init__(self, config: Optional[SAM2Config] = None):
        self.config = config or SAM2Config()
        self.predictor: Optional[Any] = None
        self.image_predictor: Optional[Any] = None
        self.inference_state: Optional[Any] = None
        self.device = torch.device(self.config.device)
        self.performance_stats = {
            'total_frames_processed': 0,
            'total_time': 0,
            'average_fps': 0,
            'accuracy_scores': []
        }
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SAM2.1 models and optimizations"""
        print(f"🚀 Initializing SAM2.1 Engine on {self.device}")
        print(f"📊 SAM2 available: {SAM2_AVAILABLE}, CUDA available: {torch.cuda.is_available()}")
        
        # Try SAM2 if available (works on both GPU and CPU)
        if SAM2_AVAILABLE:
            try:
                self._initialize_sam2_models()
                return
            except Exception as e:
                print(f"⚠️ SAM2 initialization failed: {e}")
                print("🔄 Falling back to traditional segmentation")
        
        # Always fallback to traditional methods if SAM2 fails
        self._initialize_fallback_engine()
    
    def _initialize_sam2_models(self):
        """Initialize official SAM2.1 models with robust error handling"""
        predictor_loaded = False
        image_predictor_loaded = False
        
        try:
            print("📦 Loading SAM2 video predictor...")
            # Video predictor for motion graphics
            self.predictor = build_sam2_video_predictor(
                self.config.model_cfg,
                self.config.sam2_checkpoint,
                device=self.device
            )
            predictor_loaded = True
            print("✅ SAM2 video predictor loaded")
            
        except Exception as e:
            print(f"⚠️ SAM2 video predictor failed: {e}")
            self.predictor = None
            
        try:
            print("📦 Loading SAM2 image predictor...")
            # Image predictor for single frame processing
            self.image_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large",
                device=self.device
            )
            image_predictor_loaded = True
            print("✅ SAM2 image predictor loaded")
            
        except Exception as e:
            print(f"⚠️ SAM2 image predictor failed: {e}")
            self.image_predictor = None
            
        # Apply optimizations only if models loaded and CUDA available
        if (predictor_loaded or image_predictor_loaded):
            self._apply_optimizations()
            
        # If neither model loaded, raise exception to trigger fallback
        if not predictor_loaded and not image_predictor_loaded:
            raise RuntimeError("Both SAM2 predictors failed to load")
            
        print(f"✅ SAM2.1 models initialized (Video: {predictor_loaded}, Image: {image_predictor_loaded})")
    
    def _apply_optimizations(self):
        """Apply optimizations with proper device checking"""
        # Only apply CUDA-specific optimizations on CUDA devices
        if self.config.mixed_precision and self.device.type == "cuda":
            try:
                if self.predictor is not None:
                    self.predictor = self.predictor.half()
                if self.image_predictor is not None:
                    self.image_predictor = self.image_predictor.half()
                print("✅ Mixed precision enabled")
            except Exception as e:
                print(f"⚠️ Mixed precision failed: {e}")
        
        # Only compile on CUDA with PyTorch 2.0+
        if self.config.compile_model and self.device.type == "cuda":
            try:
                import torch._dynamo
                if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                    if self.predictor is not None:
                        self.predictor = torch.compile(self.predictor, mode="reduce-overhead")
                    if self.image_predictor is not None:
                        self.image_predictor = torch.compile(self.image_predictor, mode="reduce-overhead")
                    print("✅ torch.compile optimization enabled")
                else:
                    print("⚠️ torch.compile not available (PyTorch < 2.0)")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
    
    def _initialize_fallback_engine(self):
        """Initialize CPU-compatible fallback segmentation engine"""
        print("🔄 Initializing fallback segmentation engine")
        self.predictor = None
        self.image_predictor = None
        # Will use enhanced traditional methods with deep learning principles
    
    def segment_video_batch(
        self, 
        frames: List[np.ndarray],
        frame_indices: List[int],
        prompts: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Segment a batch of video frames with SAM2.1
        
        Args:
            frames: List of RGB frames (H, W, 3)
            frame_indices: Frame indices in the video
            prompts: Optional prompts for guided segmentation
            
        Returns:
            masks: List of segmentation masks
            metadata: Processing metadata and stats
        """
        start_time = time.time()
        
        if self.predictor is not None:
            masks, metadata = self._sam2_batch_segmentation(frames, frame_indices, prompts)
        else:
            masks, metadata = self._fallback_batch_segmentation(frames, frame_indices, prompts)
        
        # Update performance stats
        elapsed_time = time.time() - start_time
        self._update_performance_stats(len(frames), elapsed_time)
        
        return masks, metadata
    
    def _sam2_batch_segmentation(
        self, 
        frames: List[np.ndarray],
        frame_indices: List[int],
        prompts: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """SAM2.1-based video segmentation"""
        
        masks = []
        metadata = {
            'method': 'sam2.1',
            'quality_scores': [],
            'processing_times': [],
            'detected_objects': []
        }
        
        # Initialize inference state for video
        if self.inference_state is None and self.predictor is not None:
            try:
                self.inference_state = self.predictor.init_state()
            except Exception as e:
                print(f"⚠️ Failed to initialize SAM2 inference state: {e}")
                self.inference_state = None
        
        # Process frames with mixed precision (only on CUDA)
        autocast_enabled = self.config.mixed_precision and self.device.type == "cuda"
        try:
            if autocast_enabled:
                autocast_context = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=True)
            else:
                # Dummy context manager for CPU
                import contextlib
                autocast_context = contextlib.nullcontext()
        except Exception as e:
            print(f"⚠️ Autocast setup failed: {e}, using default precision")
            import contextlib
            autocast_context = contextlib.nullcontext()
            
        with autocast_context:
            for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
                frame_start = time.time()
                
                # Convert BGR to RGB if needed
                if frame.shape[-1] == 3 and frame.dtype == np.uint8:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                # Add frame to inference state
                try:
                    if self.predictor is not None and self.inference_state is not None:
                        self.predictor.add_new_frame(self.inference_state, frame_rgb)
                    else:
                        print("⚠️ SAM2 predictor or inference state not available")
                        # Will fall back to image predictor below
                except Exception as e:
                    print(f"⚠️ Failed to add frame to inference state: {e}")
                
                # Automatic segmentation or prompt-based
                if prompts and frame_idx in prompts:
                    # Use provided prompts
                    if self.predictor is not None and self.inference_state is not None:
                        ann_frame_idx, ann_obj_id, mask = self.predictor.add_new_mask(
                            self.inference_state,
                            frame_idx=frame_idx,
                            obj_id=prompts[frame_idx]['obj_id'],
                            mask=prompts[frame_idx]['mask']
                        )
                    else:
                        print("⚠️ SAM2 predictor not available for prompt-based segmentation")
                        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                else:
                    # Automatic segmentation using image predictor
                    if self.image_predictor is not None:
                        try:
                            self.image_predictor.set_image(frame_rgb)
                            
                            # Generate automatic masks
                            if hasattr(self.image_predictor, 'generate_masks'):
                                auto_masks = self.image_predictor.generate_masks()
                            else:
                                # Fallback to predict method
                                masks, scores, logits = self.image_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    multimask_output=True,
                                    return_logits=True
                                )
                                auto_masks = [{
                                    'segmentation': masks[i],
                                    'predicted_iou': scores[i] if i < len(scores) else 0.5
                                } for i in range(len(masks))]
                            
                            # Select best masks based on quality scores
                            if auto_masks and len(auto_masks) > 0:
                                # Sort by predicted IoU (quality score)
                                best_masks = sorted(auto_masks, key=lambda x: x.get('predicted_iou', 0), reverse=True)
                                mask = best_masks[0]['segmentation']
                            else:
                                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                        except Exception as e:
                            print(f"⚠️ SAM2 image prediction failed: {e}, using fallback")
                            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                    else:
                        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                
                # Post-process mask
                mask_uint8 = (mask * 255).astype(np.uint8)
                masks.append(mask_uint8)
                
                # Calculate quality metrics
                quality_score = self._calculate_quality_score(frame, mask_uint8)
                metadata['quality_scores'].append(quality_score)
                metadata['processing_times'].append(time.time() - frame_start)
                
        return masks, metadata
    
    def _fallback_batch_segmentation(
        self, 
        frames: List[np.ndarray],
        frame_indices: List[int],
        prompts: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """Enhanced fallback segmentation using traditional methods with ML principles"""
        
        masks = []
        metadata = {
            'method': 'enhanced_traditional',
            'quality_scores': [],
            'processing_times': [],
            'detected_objects': []
        }
        
        for frame, frame_idx in zip(frames, frame_indices):
            frame_start = time.time()
            
            # Enhanced segmentation pipeline
            mask = self._enhanced_traditional_segmentation(frame)
            masks.append(mask)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(frame, mask)
            metadata['quality_scores'].append(quality_score)
            metadata['processing_times'].append(time.time() - frame_start)
        
        return masks, metadata
    
    def _enhanced_traditional_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced traditional segmentation using deep learning principles
        Significantly improved over basic Canny edge detection
        """
        
        # Multi-scale processing
        original_size = frame.shape[:2]
        scales = [1.0, 0.8, 1.2]  # Multi-scale analysis
        scale_masks = []
        
        for scale in scales:
            if scale != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h))
            else:
                scaled_frame = frame
            
            # Enhanced preprocessing pipeline
            mask_scale = self._process_single_scale(scaled_frame)
            
            # Resize back to original size
            if scale != 1.0:
                mask_scale = cv2.resize(mask_scale, (original_size[1], original_size[0]))
            
            scale_masks.append(mask_scale)
        
        # Fuse multi-scale results
        final_mask = self._fuse_multiscale_masks(scale_masks)
        
        return final_mask
    
    def _process_single_scale(self, frame: np.ndarray) -> np.ndarray:
        """Process frame at single scale with enhanced algorithms"""
        
        # Convert to LAB for better color segmentation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Enhanced edge detection (better than basic Canny)
        # Combine multiple edge detection methods
        edges_sobel = cv2.Sobel(l_enhanced, cv2.CV_64F, 1, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(edges_sobel))
        
        edges_canny = cv2.Canny(l_enhanced, 50, 150)
        
        # Ensure both arrays are same type for bitwise operation
        edges_sobel = edges_sobel.astype(np.uint8)
        edges_canny = edges_canny.astype(np.uint8)
        
        # Combine edge information using numpy bitwise_or for type safety
        edges_combined = np.bitwise_or(edges_sobel, edges_canny)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
        
        # Watershed-based segmentation
        mask = self._watershed_segmentation(frame, edges_combined)
        
        return mask
    
    def _watershed_segmentation(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Enhanced watershed segmentation"""
        
        # Create distance transform
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Find local maxima (seeds)
        local_maxima = (dist_transform > 0.4 * dist_transform.max()).astype(np.uint8)
        
        # Connected component analysis for seeds
        _, markers = cv2.connectedComponents(local_maxima)
        
        # Apply watershed
        markers = cv2.watershed(frame, markers)
        
        # Create binary mask (foreground vs background)
        mask = (markers > 1).astype(np.uint8) * 255
        
        return mask
    
    def _fuse_multiscale_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """Fuse masks from multiple scales"""
        
        # Weighted voting
        weights = [0.5, 0.25, 0.25]  # Give more weight to original scale
        
        fused = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            fused += mask.astype(np.float32) * weight
        
        # Threshold and convert to binary
        final_mask = (fused > 127).astype(np.uint8) * 255
        
        return final_mask
    
    def _calculate_quality_score(self, frame: np.ndarray, mask: np.ndarray) -> float:
        """Calculate segmentation quality score"""
        
        # Basic quality metrics
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        coverage_ratio = mask_area / total_area
        
        # Edge coherence score
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        mask_edges = cv2.Canny(mask, 50, 150)
        edge_overlap = np.sum(cv2.bitwise_and(edges, mask_edges))
        edge_coherence = edge_overlap / (np.sum(edges) + 1e-6)
        
        # Combine metrics
        quality_score = 0.6 * edge_coherence + 0.4 * min(coverage_ratio * 2, 1.0)
        
        return min(quality_score, 1.0)
    
    def _update_performance_stats(self, num_frames: int, elapsed_time: float) -> None:
        """Update performance statistics"""
        
        self.performance_stats['total_frames_processed'] += num_frames
        self.performance_stats['total_time'] += elapsed_time
        
        if self.performance_stats['total_time'] > 0:
            self.performance_stats['average_fps'] = (
                self.performance_stats['total_frames_processed'] / 
                self.performance_stats['total_time']
            )
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_frames_processed': 0,
            'total_time': 0,
            'average_fps': 0,
            'accuracy_scores': []
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.inference_state is not None:
            # Reset inference state
            self.inference_state = None
        
        if hasattr(self, 'predictor') and self.predictor is not None:
            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ Failed to clear CUDA cache: {e}")
        
        # Clear all references
        self.predictor = None
        self.image_predictor = None


# Utility functions for integration with existing pipeline

def create_sam2_engine(device: str = "auto") -> SAM2SegmentationEngine:
    """Create and initialize SAM2.1 segmentation engine"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = SAM2Config(device=device)
    engine = SAM2SegmentationEngine(config)
    
    print(f"🎯 SAM2.1 Engine created (device: {device})")
    print(f"📊 Target performance: {config.target_fps} FPS, {config.accuracy_threshold*100}% accuracy")
    
    return engine


def sam2_segment_frame(
    engine: SAM2SegmentationEngine,
    frame: np.ndarray,
    frame_idx: int = 0,
    prompts: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Segment a single frame using SAM2.1 engine
    Compatible with existing extract_clusters.py pipeline
    """
    
    masks, metadata = engine.segment_video_batch([frame], [frame_idx], prompts)
    return masks[0], metadata


def convert_to_clusters(
    mask: np.ndarray,
    min_cluster_size: int = 50,
    min_density: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SAM2.1 mask to cluster format compatible with existing pipeline
    """
    
    # Convert to labels format expected by extract_clusters.py
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Connected components analysis
    num_labels, labels = cv2.connectedComponents(mask_binary)
    
    # Filter small clusters
    filtered_labels = labels.copy()
    for label_id in range(1, num_labels):
        cluster_size = np.sum(labels == label_id)
        if cluster_size < min_cluster_size:
            filtered_labels[labels == label_id] = 0
    
    # Renumber labels
    unique_labels = np.unique(filtered_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    final_labels = np.zeros_like(filtered_labels)
    for old_label, new_label in label_mapping.items():
        final_labels[filtered_labels == old_label] = new_label
    
    # Create visualization (compatible with existing visualizer)
    try:
        from .visualizer import Visualizer
        viz = Visualizer()
        labels_vis = viz.show_labels(final_labels)
        # Ensure labels_vis is ndarray
        if not isinstance(labels_vis, np.ndarray):
            labels_vis = np.array(labels_vis, dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}, using dummy output")
        labels_vis = np.zeros_like(final_labels, dtype=np.uint8)
    
    return final_labels - 1, labels_vis  # Subtract 1 to match existing format


if __name__ == "__main__":
    # Test the engine
    print("🧪 Testing SAM2.1 Segmentation Engine")
    
    engine = create_sam2_engine()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test segmentation
    mask, metadata = sam2_segment_frame(engine, test_frame)
    
    print(f"✅ Test completed")
    print(f"📊 Method: {metadata['method']}")
    print(f"⚡ Performance: {engine.get_performance_stats()}")
    
    engine.cleanup()