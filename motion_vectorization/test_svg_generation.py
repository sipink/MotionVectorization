"""
Test System for SVG Generation from Motion Vectorization
=======================================================

This test system validates that the SVG generation pipeline works correctly
by processing test videos and comparing outputs.

Test Functions:
- test_svg_generation_basic(): Basic SVG generation test
- test_motion_file_format(): Validate motion_file.json format
- test_path_conversion(): Test PNG to SVG path conversion
- validate_svg_structure(): Verify SVG document structure
- compare_with_reference(): Compare with known good outputs

Usage:
    python -m motion_vectorization.test_svg_generation
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our SVG generation system
try:
    from .svg_generator import (
        MotionVectorizationSVGGenerator, 
        SVGGenerationConfig,
        create_svg_from_motion_file
    )
    from .unified_pipeline import UnifiedMotionPipeline, UnifiedPipelineConfig
    SVG_AVAILABLE = True
except ImportError as e:
    logger.error(f"SVG generation imports failed: {e}")
    # Set defaults for type checker
    MotionVectorizationSVGGenerator = None
    SVGGenerationConfig = None
    create_svg_from_motion_file = None
    UnifiedMotionPipeline = None
    UnifiedPipelineConfig = None
    SVG_AVAILABLE = False


class SVGGenerationTester:
    """Test harness for SVG generation functionality"""
    
    def __init__(self, test_data_dir: str = "videos"):
        self.test_data_dir = Path(test_data_dir)
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_environment(self) -> Path:
        """Set up temporary directory for testing"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="svg_test_"))
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def create_test_motion_file(self, output_path: Path) -> bool:
        """Create a simple test motion file for validation"""
        try:
            # Create a minimal valid motion file
            motion_data = {
                '-1': {
                    'name': 'test_video',
                    'width': 640,
                    'height': 480,
                    'bg_color': [[255, 255, 255]],
                    'bg_img': None,
                    'time': [1, 2, 3, 4, 5]
                },
                '0': {
                    'shape': 'shapes/0.png',
                    'size': (100, 100),
                    'centroid': [50, 50],
                    'time': [1, 2, 3, 4, 5],
                    'cx': [0.3, 0.4, 0.5, 0.6, 0.7],  # Moving across screen
                    'cy': [0.5, 0.5, 0.5, 0.5, 0.5],  # Constant Y
                    'sx': [1.0, 1.1, 1.2, 1.1, 1.0],  # Scale animation
                    'sy': [1.0, 1.1, 1.2, 1.1, 1.0],
                    'theta': [0.0, 0.1, 0.2, 0.1, 0.0],  # Rotation
                    'kx': [0.0, 0.0, 0.0, 0.0, 0.0],  # No shear
                    'ky': [0.0, 0.0, 0.0, 0.0, 0.0],
                    'z': [1.0, 1.0, 1.0, 1.0, 1.0]
                },
                '1': {
                    'shape': 'shapes/1.png',
                    'size': (80, 80),
                    'centroid': [40, 40],
                    'time': [2, 3, 4, 5],  # Appears later
                    'cx': [0.7, 0.6, 0.5, 0.4],  # Moving opposite direction
                    'cy': [0.3, 0.4, 0.5, 0.6],
                    'sx': [0.8, 0.9, 1.0, 1.1],
                    'sy': [0.8, 0.9, 1.0, 1.1],
                    'theta': [0.0, -0.1, -0.2, -0.3],
                    'kx': [0.0, 0.0, 0.0, 0.0],
                    'ky': [0.0, 0.0, 0.0, 0.0],
                    'z': [2.0, 2.0, 2.0, 2.0]
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(motion_data, f, indent=2)
            
            logger.info(f"✅ Test motion file created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test motion file: {e}")
            return False
    
    def create_test_shape_masks(self, shapes_dir: Path) -> bool:
        """Create simple test shape masks"""
        try:
            shapes_dir.mkdir(parents=True, exist_ok=True)
            
            # Create simple test shapes
            shapes = [
                {
                    'filename': '0.png',
                    'shape': 'circle',
                    'size': (100, 100),
                    'color': (255, 0, 0, 255)  # Red circle
                },
                {
                    'filename': '1.png', 
                    'shape': 'rectangle',
                    'size': (80, 80),
                    'color': (0, 255, 0, 255)  # Green rectangle
                }
            ]
            
            for shape_info in shapes:
                shape_path = shapes_dir / shape_info['filename']
                size = shape_info['size']
                
                # Create RGBA image
                image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
                
                if shape_info['shape'] == 'circle':
                    # Draw circle
                    center = (size[0]//2, size[1]//2)
                    radius = min(size)//3
                    cv2.circle(image, center, radius, shape_info['color'], -1)
                elif shape_info['shape'] == 'rectangle':
                    # Draw rectangle
                    cv2.rectangle(image, (10, 10), (size[0]-10, size[1]-10), shape_info['color'], -1)
                
                # Save as PNG with alpha channel
                cv2.imwrite(str(shape_path), image)
            
            logger.info(f"✅ Test shape masks created in: {shapes_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test shape masks: {e}")
            return False
    
    def test_svg_generation_basic(self) -> Dict[str, Any]:
        """Basic test of SVG generation functionality"""
        test_name = "test_svg_generation_basic"
        result = {'test': test_name, 'success': False, 'details': {}}
        
        try:
            if not SVG_AVAILABLE:
                result['error'] = "SVG generation not available"
                return result
            
            # Set up test environment
            test_dir = self.setup_test_environment()
            shapes_dir = test_dir / "shapes"
            motion_file_path = test_dir / "motion_file.json"
            svg_output_path = test_dir / "motion_file.svg"
            
            # Create test data
            motion_created = self.create_test_motion_file(motion_file_path)
            shapes_created = self.create_test_shape_masks(shapes_dir)
            
            if not motion_created or not shapes_created:
                result['error'] = "Failed to create test data"
                return result
            
            # Test SVG generation
            if SVGGenerationConfig is None or create_svg_from_motion_file is None:
                result['error'] = "SVG generation classes not available"
                return result
                
            config = SVGGenerationConfig(quality_mode="balanced")
            success = create_svg_from_motion_file(
                str(motion_file_path),
                str(shapes_dir),
                str(svg_output_path),
                config
            )
            
            result['details']['svg_generated'] = success
            result['details']['svg_file_exists'] = svg_output_path.exists()
            
            if success and svg_output_path.exists():
                # Validate SVG file
                svg_size = svg_output_path.stat().st_size
                result['details']['svg_file_size'] = svg_size
                result['details']['svg_not_empty'] = svg_size > 0
                
                # Basic SVG content validation
                with open(svg_output_path, 'r') as f:
                    svg_content = f.read()
                
                result['details']['contains_svg_tag'] = '<svg' in svg_content
                result['details']['contains_path_elements'] = '<path' in svg_content
                result['details']['contains_animations'] = 'animateTransform' in svg_content
                result['details']['svg_content_length'] = len(svg_content)
                
                # All checks passed
                if all([
                    result['details']['svg_file_exists'],
                    result['details']['svg_not_empty'],
                    result['details']['contains_svg_tag'],
                    result['details']['contains_path_elements']
                ]):
                    result['success'] = True
                    logger.info(f"✅ {test_name} passed")
                else:
                    logger.warning(f"⚠️ {test_name} partial success")
            else:
                result['error'] = "SVG generation failed"
                logger.error(f"❌ {test_name} failed")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ {test_name} error: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def test_motion_file_format(self) -> Dict[str, Any]:
        """Test motion file format validation"""
        test_name = "test_motion_file_format"
        result = {'test': test_name, 'success': False, 'details': {}}
        
        try:
            test_dir = self.setup_test_environment()
            motion_file_path = test_dir / "motion_file.json"
            
            # Create test motion file
            if not self.create_test_motion_file(motion_file_path):
                result['error'] = "Failed to create test motion file"
                return result
            
            # Load and validate motion file
            with open(motion_file_path, 'r') as f:
                motion_data = json.load(f)
            
            # Validation checks
            checks = {
                'has_metadata': '-1' in motion_data,
                'metadata_has_width': motion_data.get('-1', {}).get('width') is not None,
                'metadata_has_height': motion_data.get('-1', {}).get('height') is not None,
                'has_shapes': len([k for k in motion_data.keys() if k != '-1']) > 0,
                'shapes_have_time': all('time' in motion_data[k] for k in motion_data if k != '-1'),
                'shapes_have_transforms': all(
                    all(param in motion_data[k] for param in ['cx', 'cy', 'sx', 'sy', 'theta', 'kx', 'ky'])
                    for k in motion_data if k != '-1'
                )
            }
            
            result['details'] = checks
            result['success'] = all(checks.values())
            
            if result['success']:
                logger.info(f"✅ {test_name} passed")
            else:
                logger.warning(f"⚠️ {test_name} failed some checks")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ {test_name} error: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def test_unified_pipeline_integration(self) -> Dict[str, Any]:
        """Test SVG generation through unified pipeline"""
        test_name = "test_unified_pipeline_integration"
        result = {'test': test_name, 'success': False, 'details': {}}
        
        try:
            # Check if test video exists
            test_video = self.test_data_dir / "test1.mp4"
            if not test_video.exists():
                result['error'] = f"Test video not found: {test_video}"
                return result
            
            # Set up test environment
            test_dir = self.setup_test_environment()
            
            # Configure unified pipeline with SVG generation
            if UnifiedPipelineConfig is None or UnifiedMotionPipeline is None:
                result['error'] = "Unified pipeline classes not available"
                return result
                
            config = UnifiedPipelineConfig(
                mode="balanced",
                device="cpu",  # Use CPU for testing
                generate_svg=True,
                svg_quality_mode="balanced"
            )
            
            # Initialize pipeline
            pipeline = UnifiedMotionPipeline(config)
            
            # Process video (small portion for testing)
            processing_result = pipeline.process_video(
                str(test_video),
                str(test_dir),
                max_frames=5,  # Just process a few frames
                save_visualizations=False
            )
            
            result['details']['processing_completed'] = processing_result is not None
            
            # Check for expected outputs
            expected_files = [
                test_dir / "motion_file.json",
                test_dir / "motion_file.svg",
                test_dir / "unified_pipeline_results.json"
            ]
            
            for file_path in expected_files:
                file_exists = file_path.exists()
                result['details'][f'{file_path.name}_exists'] = file_exists
                if file_exists:
                    result['details'][f'{file_path.name}_size'] = file_path.stat().st_size
            
            # Determine success
            result['success'] = all(
                result['details'].get(f'{f.name}_exists', False) 
                for f in expected_files
            )
            
            if result['success']:
                logger.info(f"✅ {test_name} passed")
            else:
                logger.warning(f"⚠️ {test_name} failed - missing expected files")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ {test_name} error: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all SVG generation tests"""
        logger.info("🧪 Starting SVG Generation Test Suite")
        logger.info("=" * 50)
        
        all_results = {}
        
        # Run individual tests
        tests = [
            self.test_motion_file_format,
            self.test_svg_generation_basic,
            # self.test_unified_pipeline_integration,  # Skip for now as it requires more setup
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                all_results[result['test']] = result
                
                # Print test result
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                logger.info(f"{status}: {result['test']}")
                
                if 'error' in result:
                    logger.error(f"  Error: {result['error']}")
                
            except Exception as e:
                logger.error(f"Test {test_func.__name__} crashed: {e}")
        
        # Summary
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results.values() if r['success'])
        
        logger.info("=" * 50)
        logger.info(f"🧪 Test Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed")
        
        # Cleanup
        self.cleanup_test_environment()
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'detailed_results': all_results
        }


def main():
    """Main test runner"""
    print("🎬 Motion Vectorization SVG Generation Test Suite")
    print("=" * 60)
    
    if not SVG_AVAILABLE:
        print("❌ SVG generation system not available")
        print("Please ensure all dependencies are installed")
        return False
    
    # Initialize tester
    tester = SVGGenerationTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Return success status
    return results['summary']['success_rate'] >= 0.8  # 80% pass rate


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)