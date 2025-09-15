"""
Advanced SVG Generation System for Motion Vectorization
=========================================================

Converts motion_file.json and PNG shape masks to editable SVG animations
with true vector graphics (not bitmap sprites embedded in SVG).

Key Features:
- PNG mask to SVG path conversion using contour detection and curve fitting
- 7-parameter affine transformations to SVG animateTransform elements
- Proper z-ordering and timeline management
- Smooth curve approximation with configurable quality
- Full compatibility with vector graphics editors
- Optimized output with minimal file sizes

Transformation Parameters:
- cx, cy: Center coordinates (pixels)
- sx, sy: Scale factors
- theta: Rotation angle (radians)
- kx, ky: Shear parameters  
- z: Z-order for layering

Output: Complete SVG documents with embedded animations
"""

import os
import json
import numpy as np
import cv2
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import base64
from dataclasses import dataclass, field
import logging

# Optional dependencies with robust fallbacks
try:
    import drawsvg as draw
    DRAWSVG_AVAILABLE = True
except ImportError:
    DRAWSVG_AVAILABLE = False
    draw = None

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    measure = None

# For curve fitting (optional advanced feature)
try:
    from scipy.interpolate import splprep, splev
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    splprep = splev = cdist = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SVGGenerationConfig:
    """Configuration for SVG generation with quality/performance tradeoffs"""
    
    # Path generation quality
    contour_approximation_epsilon: float = 0.02  # Lower = more accurate paths
    curve_smoothing: bool = True
    min_contour_area: int = 100  # Minimum area for valid contours
    max_path_points: int = 200  # Maximum points per path for optimization
    
    # SVG output settings
    viewbox_padding: int = 0  # Padding around content
    coordinate_precision: int = 2  # Decimal places for coordinates
    frame_rate: float = 60.0  # Animation frame rate
    
    # Performance optimization
    enable_path_optimization: bool = True
    merge_close_contours: bool = True
    simplify_animations: bool = False  # Keep all keyframes vs interpolation
    
    # Quality vs file size tradeoffs
    quality_mode: str = "balanced"  # "speed", "balanced", "quality"
    include_debug_info: bool = False  # Add debug comments to SVG
    
    def __post_init__(self):
        """Apply quality mode presets"""
        if self.quality_mode == "speed":
            self.contour_approximation_epsilon = 0.05
            self.curve_smoothing = False
            self.max_path_points = 100
            self.enable_path_optimization = True
            self.simplify_animations = True
        elif self.quality_mode == "quality":
            self.contour_approximation_epsilon = 0.005
            self.curve_smoothing = True
            self.max_path_points = 500
            self.enable_path_optimization = False
            self.simplify_animations = False


class SVGPathConverter:
    """Converts PNG masks to optimized SVG paths using contour detection"""
    
    def __init__(self, config: SVGGenerationConfig):
        self.config = config
        
    def png_to_svg_path(self, png_path: str) -> List[str]:
        """
        Convert PNG mask to SVG path strings
        
        Args:
            png_path: Path to PNG mask file
            
        Returns:
            List of SVG path strings (one per contour)
        """
        if not os.path.exists(png_path):
            logger.warning(f"PNG file not found: {png_path}")
            return []
            
        # Load image and extract alpha channel
        image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning(f"Could not load image: {png_path}")
            return []
            
        # Handle different image formats
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA image - use alpha channel
            mask = image[:, :, 3]
        elif len(image.shape) == 3:
            # RGB image - convert to grayscale and threshold
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            # Grayscale image
            _, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
        return self._mask_to_svg_paths(mask)
    
    def _mask_to_svg_paths(self, mask: np.ndarray) -> List[str]:
        """Convert binary mask to SVG path strings"""
        paths = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue
                
            # Simplify contour
            epsilon = self.config.contour_approximation_epsilon * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to SVG path
            path_string = self._contour_to_svg_path(simplified)
            if path_string:
                paths.append(path_string)
                
        return paths
    
    def _contour_to_svg_path(self, contour: np.ndarray) -> str:
        """Convert OpenCV contour to SVG path string"""
        if len(contour) < 3:
            return ""
            
        # Extract coordinates
        points = contour.reshape(-1, 2)
        
        # Limit number of points for optimization
        if len(points) > self.config.max_path_points:
            # Subsample points uniformly
            indices = np.linspace(0, len(points)-1, self.config.max_path_points, dtype=int)
            points = points[indices]
        
        # Build SVG path string
        path_parts = []
        
        # Start with move command
        x, y = points[0]
        path_parts.append(f"M {x:.{self.config.coordinate_precision}f} {y:.{self.config.coordinate_precision}f}")
        
        if self.config.curve_smoothing and SCIPY_AVAILABLE and len(points) > 4:
            # Use smooth curves
            path_parts.extend(self._create_smooth_curves(points))
        else:
            # Use straight lines
            for point in points[1:]:
                x, y = point
                path_parts.append(f"L {x:.{self.config.coordinate_precision}f} {y:.{self.config.coordinate_precision}f}")
        
        # Close path
        path_parts.append("Z")
        
        return " ".join(path_parts)
    
    def _create_smooth_curves(self, points: np.ndarray) -> List[str]:
        """Create smooth cubic Bezier curves from points"""
        if not SCIPY_AVAILABLE:
            # Fallback to linear segments
            path_parts = []
            for point in points[1:]:
                x, y = point
                path_parts.append(f"L {x:.{self.config.coordinate_precision}f} {y:.{self.config.coordinate_precision}f}")
            return path_parts
        
        try:
            # Fit spline to points
            if splprep is None or splev is None:
                # SCIPY not available - fallback to linear segments
                path_parts = []
                for point in points[1:]:
                    x, y = point
                    path_parts.append(f"L {x:.{self.config.coordinate_precision}f} {y:.{self.config.coordinate_precision}f}")
                return path_parts
            
            tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=True)
            
            # Generate smooth curve points
            u_fine = np.linspace(0, 1, len(points) * 2)
            smooth_points = np.array(splev(u_fine, tck)).T
            
            # Convert to cubic Bezier curves
            return self._points_to_bezier_curves(smooth_points)
            
        except Exception as e:
            logger.warning(f"Smooth curve generation failed: {e}")
            # Fallback to linear segments
            path_parts = []
            for point in points[1:]:
                x, y = point
                path_parts.append(f"L {x:.{self.config.coordinate_precision}f} {y:.{self.config.coordinate_precision}f}")
            return path_parts
    
    def _points_to_bezier_curves(self, points: np.ndarray) -> List[str]:
        """Convert points to cubic Bezier curve commands"""
        path_parts = []
        
        # Simple implementation: use every 4th point as control points
        for i in range(1, len(points), 3):
            if i + 2 < len(points):
                cp1 = points[i]
                cp2 = points[i + 1] 
                end = points[i + 2]
                
                path_parts.append(
                    f"C {cp1[0]:.{self.config.coordinate_precision}f} {cp1[1]:.{self.config.coordinate_precision}f} "
                    f"{cp2[0]:.{self.config.coordinate_precision}f} {cp2[1]:.{self.config.coordinate_precision}f} "
                    f"{end[0]:.{self.config.coordinate_precision}f} {end[1]:.{self.config.coordinate_precision}f}"
                )
            else:
                # Fallback to line
                end = points[i]
                path_parts.append(f"L {end[0]:.{self.config.coordinate_precision}f} {end[1]:.{self.config.coordinate_precision}f}")
        
        return path_parts


class SVGAnimationGenerator:
    """Generates SVG animation elements from motion parameters"""
    
    def __init__(self, config: SVGGenerationConfig):
        self.config = config
        
    def create_transform_animations(self, motion_data: Dict[str, List[float]], 
                                   frame_times: List[int], frame_rate: float) -> List[Dict[str, Any]]:
        """
        Create SVG animateTransform elements from motion parameters
        
        Args:
            motion_data: Dictionary with keys cx, cy, sx, sy, theta, kx, ky, z
            frame_times: Frame indices for keyframes
            frame_rate: Animation frame rate
            
        Returns:
            List of animation dictionaries
        """
        animations = []
        
        if not frame_times or len(frame_times) < 2:
            return animations
            
        # Calculate timing
        duration = (frame_times[-1] - frame_times[0]) / frame_rate
        
        # Generate keyTimes (normalized 0-1)
        key_times = [(t - frame_times[0]) / (frame_times[-1] - frame_times[0]) for t in frame_times]
        key_times_str = ";".join(f"{t:.3f}" for t in key_times)
        
        # Translation animation
        if 'cx' in motion_data and 'cy' in motion_data:
            translations = [f"{cx} {cy}" for cx, cy in zip(motion_data['cx'], motion_data['cy'])]
            animations.append({
                'type': 'animateTransform',
                'attributeName': 'transform',
                'attributeType': 'XML',
                'type_attr': 'translate',
                'values': ";".join(translations),
                'keyTimes': key_times_str,
                'dur': f"{duration}s",
                'calcMode': 'spline' if not self.config.simplify_animations else 'linear',
                'fill': 'freeze'
            })
        
        # Scale animation
        if 'sx' in motion_data and 'sy' in motion_data:
            scales = [f"{sx} {sy}" for sx, sy in zip(motion_data['sx'], motion_data['sy'])]
            animations.append({
                'type': 'animateTransform',
                'attributeName': 'transform',
                'attributeType': 'XML',
                'type_attr': 'scale',
                'values': ";".join(scales),
                'keyTimes': key_times_str,
                'dur': f"{duration}s",
                'calcMode': 'spline' if not self.config.simplify_animations else 'linear',
                'fill': 'freeze'
            })
        
        # Rotation animation
        if 'theta' in motion_data:
            # Convert radians to degrees
            rotations = [f"{math.degrees(theta)}" for theta in motion_data['theta']]
            animations.append({
                'type': 'animateTransform',
                'attributeName': 'transform',
                'attributeType': 'XML',
                'type_attr': 'rotate',
                'values': ";".join(rotations),
                'keyTimes': key_times_str,
                'dur': f"{duration}s",
                'calcMode': 'spline' if not self.config.simplify_animations else 'linear',
                'fill': 'freeze'
            })
        
        # Shear animations (using skewX and skewY)
        if 'kx' in motion_data:
            skew_x_values = [f"{math.degrees(math.atan(kx))}" for kx in motion_data['kx']]
            animations.append({
                'type': 'animateTransform',
                'attributeName': 'transform',
                'attributeType': 'XML',
                'type_attr': 'skewX',
                'values': ";".join(skew_x_values),
                'keyTimes': key_times_str,
                'dur': f"{duration}s",
                'calcMode': 'spline' if not self.config.simplify_animations else 'linear',
                'fill': 'freeze'
            })
            
        if 'ky' in motion_data:
            skew_y_values = [f"{math.degrees(math.atan(ky))}" for ky in motion_data['ky']]
            animations.append({
                'type': 'animateTransform',
                'attributeName': 'transform',
                'attributeType': 'XML',
                'type_attr': 'skewY',
                'values': ";".join(skew_y_values),
                'keyTimes': key_times_str,
                'dur': f"{duration}s",
                'calcMode': 'spline' if not self.config.simplify_animations else 'linear',
                'fill': 'freeze'
            })
        
        return animations


class SVGDocumentBuilder:
    """Builds complete SVG documents with proper structure and animations"""
    
    def __init__(self, config: SVGGenerationConfig):
        self.config = config
        
    def create_svg_document(self, width: int, height: int, 
                           total_duration: float, background_color: str = "#FFFFFF") -> str:
        """Create base SVG document structure"""
        
        # Calculate viewBox with padding
        padding = self.config.viewbox_padding
        viewbox = f"{-padding} {-padding} {width + 2*padding} {height + 2*padding}"
        
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" 
     height="{height}"
     viewBox="{viewbox}"
     data-duration="{total_duration:.3f}"
     data-frame-rate="{self.config.frame_rate}">
'''
        
        # Add background
        background = f'  <rect width="{width}" height="{height}" fill="{background_color}"/>\n'
        
        # Debug info
        debug_comment = ""
        if self.config.include_debug_info:
            debug_comment = f"  <!-- Generated by Motion Vectorization SVG Generator -->\n"
            debug_comment += f"  <!-- Quality mode: {self.config.quality_mode} -->\n"
        
        return svg_header + debug_comment + background
    
    def create_shape_group(self, shape_id: int, paths: List[str], 
                          animations: List[Dict[str, Any]], 
                          start_time: float = 0.0, 
                          fill_color: str = "#000000",
                          z_index: int = 0) -> str:
        """Create SVG group for a shape with animations"""
        
        group_start = f'  <g id="shape_{shape_id}" data-z-index="{z_index}">\n'
        
        # Create paths
        paths_svg = ""
        for i, path in enumerate(paths):
            path_id = f"shape_{shape_id}_path_{i}"
            paths_svg += f'    <path id="{path_id}" d="{path}" fill="{fill_color}"/>\n'
        
        # Add animations
        animations_svg = ""
        for anim in animations:
            if anim['type'] == 'animateTransform':
                animations_svg += f'''    <animateTransform
      attributeName="{anim['attributeName']}"
      attributeType="{anim['attributeType']}"
      type="{anim['type_attr']}"
      values="{anim['values']}"
      keyTimes="{anim['keyTimes']}"
      dur="{anim['dur']}"
      calcMode="{anim['calcMode']}"
      fill="{anim['fill']}"
      begin="{start_time}s"/>
'''
        
        group_end = "  </g>\n"
        
        return group_start + paths_svg + animations_svg + group_end
    
    def finalize_svg_document(self, svg_content: str) -> str:
        """Add closing SVG tag"""
        return svg_content + "</svg>"


class MotionVectorizationSVGGenerator:
    """
    Main class for generating SVG animations from motion vectorization data
    
    Converts motion_file.json + shape masks → editable SVG animations
    """
    
    def __init__(self, config: Optional[SVGGenerationConfig] = None):
        self.config = config or SVGGenerationConfig()
        self.path_converter = SVGPathConverter(self.config)
        self.animation_generator = SVGAnimationGenerator(self.config)
        self.document_builder = SVGDocumentBuilder(self.config)
    
    def generate_svg_from_motion_file(self, motion_file_path: str, 
                                     shapes_dir: str, 
                                     output_svg_path: str) -> bool:
        """
        Generate SVG animation from motion_file.json and shape masks
        
        Args:
            motion_file_path: Path to motion_file.json
            shapes_dir: Directory containing shape PNG masks  
            output_svg_path: Output SVG file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load motion file
            with open(motion_file_path, 'r') as f:
                motion_data = json.load(f)
            
            # Extract metadata
            if '-1' not in motion_data:
                logger.error("Motion file missing metadata (key '-1')")
                return False
                
            metadata = motion_data['-1']
            width = metadata.get('width', 1920)
            height = metadata.get('height', 1080)
            
            # Calculate total duration
            all_times = []
            for shape_id, shape_data in motion_data.items():
                if shape_id != '-1' and 'time' in shape_data:
                    all_times.extend(shape_data['time'])
            
            if not all_times:
                logger.error("No timing data found in motion file")
                return False
                
            total_duration = (max(all_times) - min(all_times)) / self.config.frame_rate
            
            # Determine background color
            bg_color = "#FFFFFF"
            if 'bg_color' in metadata and metadata['bg_color']:
                bg_rgb = metadata['bg_color'][0]  # First frame background
                bg_color = f"rgb({bg_rgb[0]},{bg_rgb[1]},{bg_rgb[2]})"
            
            # Start building SVG
            svg_content = self.document_builder.create_svg_document(
                width, height, total_duration, bg_color
            )
            
            # Process each shape in z-order
            shapes_by_z = []
            for shape_id, shape_data in motion_data.items():
                if shape_id == '-1':
                    continue
                    
                try:
                    shape_id_int = int(shape_id)
                    z_values = shape_data.get('z', [0])
                    avg_z = sum(z_values) / len(z_values) if z_values else 0
                    shapes_by_z.append((avg_z, shape_id_int, shape_data))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid shape ID: {shape_id}")
                    continue
            
            # Sort by z-order (background to foreground)
            shapes_by_z.sort(key=lambda x: x[0])
            
            # Generate SVG for each shape
            for avg_z, shape_id, shape_data in shapes_by_z:
                shape_svg = self._generate_shape_svg(shape_id, shape_data, shapes_dir, avg_z)
                if shape_svg:
                    svg_content += shape_svg
            
            # Finalize document
            svg_content = self.document_builder.finalize_svg_document(svg_content)
            
            # Write to file
            with open(output_svg_path, 'w') as f:
                f.write(svg_content)
            
            logger.info(f"Successfully generated SVG: {output_svg_path}")
            return True
            
        except Exception as e:
            logger.error(f"SVG generation failed: {e}")
            return False
    
    def _generate_shape_svg(self, shape_id: int, shape_data: Dict[str, Any], 
                           shapes_dir: str, z_index: float) -> str:
        """Generate SVG for a single shape"""
        try:
            # Find shape mask file
            shape_path = None
            if 'shape' in shape_data:
                # Use path from motion data
                shape_path = shape_data['shape']
                if not os.path.isabs(shape_path):
                    shape_path = os.path.join(shapes_dir, os.path.basename(shape_path))
            else:
                # Default naming convention
                shape_path = os.path.join(shapes_dir, f"{shape_id}.png")
            
            if not os.path.exists(shape_path):
                logger.warning(f"Shape file not found: {shape_path}")
                return ""
            
            # Convert PNG to SVG paths
            svg_paths = self.path_converter.png_to_svg_path(shape_path)
            if not svg_paths:
                logger.warning(f"No paths generated for shape {shape_id}")
                return ""
            
            # Generate animations
            animations = self.animation_generator.create_transform_animations(
                shape_data, shape_data.get('time', []), self.config.frame_rate
            )
            
            # Calculate start time
            start_time = 0.0
            if 'time' in shape_data and shape_data['time']:
                start_time = shape_data['time'][0] / self.config.frame_rate
            
            # Generate fill color (simple black for now, could be enhanced)
            fill_color = "#000000"
            
            # Create shape group
            return self.document_builder.create_shape_group(
                shape_id, svg_paths, animations, start_time, fill_color, int(z_index)
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate SVG for shape {shape_id}: {e}")
            return ""


def create_svg_from_motion_file(motion_file_path: str, shapes_dir: str, 
                               output_svg_path: str,
                               config: Optional[SVGGenerationConfig] = None) -> bool:
    """
    Convenience function to generate SVG from motion vectorization data
    
    Args:
        motion_file_path: Path to motion_file.json
        shapes_dir: Directory containing shape PNG masks
        output_svg_path: Output SVG file path
        config: Optional configuration (uses defaults if None)
        
    Returns:
        True if successful, False otherwise
    """
    generator = MotionVectorizationSVGGenerator(config)
    return generator.generate_svg_from_motion_file(
        motion_file_path, shapes_dir, output_svg_path
    )


# CLI interface when run as script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SVG animations from motion vectorization data")
    parser.add_argument("motion_file", help="Path to motion_file.json")
    parser.add_argument("shapes_dir", help="Directory containing shape PNG masks")
    parser.add_argument("output_svg", help="Output SVG file path")
    parser.add_argument("--quality", choices=["speed", "balanced", "quality"], 
                       default="balanced", help="Quality mode")
    parser.add_argument("--frame-rate", type=float, default=60.0, help="Animation frame rate")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SVGGenerationConfig(
        quality_mode=args.quality,
        frame_rate=args.frame_rate
    )
    
    # Generate SVG
    success = create_svg_from_motion_file(
        args.motion_file, args.shapes_dir, args.output_svg, config
    )
    
    if success:
        print(f"✅ SVG generated successfully: {args.output_svg}")
    else:
        print("❌ SVG generation failed")
        exit(1)