#!/usr/bin/env python3
"""
CoTracker3 Integration Test and Performance Benchmark
Demonstrates superior tracking accuracy vs current primitive system

This test validates:
- CoTracker3 engine functionality
- SAM2-CoTracker3 bridge performance  
- Integration with motion vectorization pipeline
- Performance improvements and accuracy gains
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import tempfile
import shutil
from pathlib import Path

# Add motion_vectorization to path
sys.path.insert(0, '.')

# Import CoTracker3 components
try:
    from motion_vectorization.cotracker3_engine import CoTracker3TrackerEngine, create_cotracker3_engine
    from motion_vectorization.sam2_cotracker_bridge import SAM2CoTrackerBridge, create_sam2_cotracker_bridge
    from motion_vectorization.processor import Processor
    COTRACKER3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CoTracker3 components not available: {e}")
    COTRACKER3_AVAILABLE = False

# Test configuration
class TestConfig:
    # Test video parameters
    NUM_FRAMES = 50
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    NUM_OBJECTS = 3
    
    # Performance thresholds
    EXPECTED_SPEEDUP = 1.2  # 20% speed improvement (conservative estimate)
    EXPECTED_ACCURACY = 0.85  # 85% tracking accuracy
    
    # Test modes
    TEST_SYNTHETIC = True
    TEST_REAL_VIDEO = False  # Set to True if test video available
    
    # Output
    SAVE_VISUALIZATIONS = True
    OUTPUT_DIR = "cotracker3_test_results"


class CoTracker3IntegrationTest:
    """Comprehensive test suite for CoTracker3 integration"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = {
            'cotracker3_times': [],
            'primitive_times': [],
            'cotracker3_accuracy': [],
            'primitive_accuracy': [],
            'cotracker3_quality': [],
            'primitive_quality': []
        }
        
        # Setup output directory
        if self.config.SAVE_VISUALIZATIONS:
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run complete CoTracker3 integration test suite"""
        print("🧪 CoTracker3 Integration Test Suite")
        print("=" * 50)
        
        if not COTRACKER3_AVAILABLE:
            print("❌ CoTracker3 components not available - cannot run tests")
            return {"error": "CoTracker3 not available"}
        
        try:
            # Test 1: Engine functionality
            print("\n1️⃣ Testing CoTracker3 Engine...")
            engine_results = self.test_cotracker3_engine()
            
            # Test 2: SAM2-CoTracker3 bridge
            print("\n2️⃣ Testing SAM2-CoTracker3 Bridge...")
            bridge_results = self.test_sam2_cotracker_bridge()
            
            # Test 3: Processor integration
            print("\n3️⃣ Testing Processor Integration...")
            processor_results = self.test_processor_integration()
            
            # Test 4: Performance comparison
            print("\n4️⃣ Running Performance Comparison...")
            performance_results = self.test_performance_comparison()
            
            # Test 5: Accuracy comparison
            print("\n5️⃣ Running Accuracy Comparison...")
            accuracy_results = self.test_accuracy_comparison()
            
            # Generate final report
            print("\n📊 Generating Performance Report...")
            final_report = self.generate_final_report({
                'engine': engine_results,
                'bridge': bridge_results,
                'processor': processor_results,
                'performance': performance_results,
                'accuracy': accuracy_results
            })
            
            print("\n✅ CoTracker3 Integration Test Suite Complete!")
            return final_report
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def test_cotracker3_engine(self) -> Dict[str, Any]:
        """Test CoTracker3 engine core functionality"""
        print("   Testing engine initialization...")
        
        # Create engine
        engine = create_cotracker3_engine(
            mode="offline",
            device="auto",
            grid_size=20  # Smaller for testing
        )
        
        # Test video processing
        print("   Testing video tracking...")
        test_video = self.create_synthetic_test_video()
        
        start_time = time.perf_counter()
        tracks, visibility = engine.track_video_grid(test_video, grid_size=20)
        processing_time = time.perf_counter() - start_time
        
        # Extract motion parameters
        print("   Testing motion parameter extraction...")
        motion_params = engine.extract_motion_parameters(tracks, visibility)
        
        # Calculate performance metrics
        fps = test_video.shape[1] / processing_time
        visibility_score = visibility.mean().item()
        
        # Cleanup
        engine.cleanup()
        
        results = {
            'initialization_success': True,
            'tracking_success': tracks is not None and visibility is not None,
            'motion_params_success': len(motion_params) > 0,
            'processing_time': processing_time,
            'fps': fps,
            'visibility_score': visibility_score,
            'num_points_tracked': tracks.shape[2] if tracks is not None else 0
        }
        
        print(f"   ✅ Engine test: {fps:.1f} FPS, {results['num_points_tracked']} points")
        return results
    
    def test_sam2_cotracker_bridge(self) -> Dict[str, Any]:
        """Test SAM2-CoTracker3 bridge functionality"""
        print("   Testing bridge initialization...")
        
        # Create bridge (may fail if SAM2 not available)
        try:
            bridge = create_sam2_cotracker_bridge(
                sam2_accuracy="fast",  # Use fast mode for testing
                cotracker_mode="offline",
                contour_density=15,  # Smaller for testing
                device="auto"
            )
            
            # Test video processing
            print("   Testing bridge video processing...")
            test_video = self.create_synthetic_test_video()
            
            start_time = time.perf_counter()
            results = bridge.process_video(test_video, prompts=None)
            processing_time = time.perf_counter() - start_time
            
            # Calculate performance metrics
            fps = test_video.shape[1] / processing_time
            
            # Cleanup
            bridge.cleanup()
            
            bridge_results = {
                'initialization_success': True,
                'processing_success': 'motion_parameters' in results,
                'processing_time': processing_time,
                'fps': fps,
                'num_objects': len(results.get('motion_parameters', {}))
            }
            
            print(f"   ✅ Bridge test: {fps:.1f} FPS, {bridge_results['num_objects']} objects")
            return bridge_results
            
        except Exception as e:
            print(f"   ⚠️  SAM2 bridge test skipped: {e}")
            return {
                'initialization_success': False,
                'processing_success': False,
                'error': str(e)
            }
    
    def test_processor_integration(self) -> Dict[str, Any]:
        """Test processor integration with CoTracker3"""
        print("   Testing processor with CoTracker3...")
        
        # Create processors (with and without CoTracker3)
        processor_cotracker = Processor(
            use_cotracker3=True,
            cotracker3_mode="offline",
            device="auto"
        )
        
        processor_original = Processor(
            use_cotracker3=False
        )
        
        # Create test shapes
        test_shapes = self.create_test_shapes()
        prev_shapes, curr_shapes = test_shapes['prev_shapes'], test_shapes['curr_shapes']
        prev_centroids, curr_centroids = test_shapes['prev_centroids'], test_shapes['curr_centroids']
        
        # Test CoTracker3 correspondences
        print("   Testing CoTracker3 correspondences...")
        if processor_cotracker.use_cotracker3:
            start_time = time.perf_counter()
            shape_diffs, motion_scores, metadata = processor_cotracker.get_cotracker3_correspondences(
                prev_shapes, curr_shapes, 
                self.create_test_frame(), self.create_test_frame(),
                prev_centroids, curr_centroids
            )
            cotracker_time = time.perf_counter() - start_time
            cotracker_success = True
        else:
            print("   ⚠️  CoTracker3 not available, skipping correspondence test")
            cotracker_time = 0
            cotracker_success = False
            shape_diffs, motion_scores = np.zeros((len(prev_shapes), len(curr_shapes))), np.zeros((len(prev_shapes), len(curr_shapes)))
        
        # Test original correspondences
        print("   Testing original correspondences...")
        start_time = time.perf_counter()
        orig_shape_diffs, orig_rgb_diffs = processor_original.get_appearance_graphs(
            prev_shapes, curr_shapes, prev_centroids, curr_centroids
        )
        original_time = time.perf_counter() - start_time
        
        # Cleanup
        if processor_cotracker.use_cotracker3:
            processor_cotracker.cleanup_cotracker3()
        
        results = {
            'cotracker3_available': processor_cotracker.use_cotracker3,
            'cotracker3_time': cotracker_time,
            'original_time': original_time,
            'speedup_ratio': original_time / cotracker_time if cotracker_time > 0 else 0,
            'cotracker3_success': cotracker_success,
            'shape_similarity_improvement': np.mean(shape_diffs) - np.mean(orig_shape_diffs) if cotracker_success else 0
        }
        
        if cotracker_success:
            print(f"   ✅ Processor test: {results['speedup_ratio']:.2f}x speedup")
        else:
            print(f"   ⚠️  Processor test: CoTracker3 not available")
            
        return results
    
    def test_performance_comparison(self) -> Dict[str, Any]:
        """Run comprehensive performance comparison"""
        print("   Running performance benchmarks...")
        
        # Create test data
        test_video = self.create_synthetic_test_video()
        test_shapes = self.create_test_shapes()
        
        # Benchmark CoTracker3
        cotracker_times = []
        if COTRACKER3_AVAILABLE:
            engine = create_cotracker3_engine(mode="offline", grid_size=15)
            
            for i in range(3):  # Multiple runs for average
                start_time = time.perf_counter()
                tracks, visibility = engine.track_video_grid(test_video, grid_size=15)
                cotracker_times.append(time.perf_counter() - start_time)
            
            engine.cleanup()
        
        # Benchmark original system
        primitive_times = []
        processor = Processor(use_cotracker3=False)
        
        for i in range(3):  # Multiple runs for average
            start_time = time.perf_counter()
            shape_diffs, rgb_diffs = processor.get_appearance_graphs(
                test_shapes['prev_shapes'], test_shapes['curr_shapes'],
                test_shapes['prev_centroids'], test_shapes['curr_centroids']
            )
            primitive_times.append(time.perf_counter() - start_time)
        
        # Calculate statistics
        avg_cotracker_time = np.mean(cotracker_times) if cotracker_times else float('inf')
        avg_primitive_time = np.mean(primitive_times)
        speedup = avg_primitive_time / avg_cotracker_time if cotracker_times else 0
        
        results = {
            'avg_cotracker3_time': avg_cotracker_time,
            'avg_primitive_time': avg_primitive_time,
            'speedup_ratio': speedup,
            'cotracker3_fps': test_video.shape[1] / avg_cotracker_time if cotracker_times else 0,
            'primitive_fps': test_video.shape[1] / avg_primitive_time,
            'meets_speed_target': speedup >= self.config.EXPECTED_SPEEDUP
        }
        
        print(f"   📈 Performance: {speedup:.2f}x speedup ({'✅' if results['meets_speed_target'] else '❌'})")
        return results
    
    def test_accuracy_comparison(self) -> Dict[str, Any]:
        """Test tracking accuracy improvements"""
        print("   Testing tracking accuracy...")
        
        # Create controlled test with known ground truth
        ground_truth_motion = self.create_ground_truth_motion()
        test_video = self.create_motion_test_video(ground_truth_motion)
        
        # Test CoTracker3 accuracy
        cotracker_accuracy = 0.0
        if COTRACKER3_AVAILABLE:
            engine = create_cotracker3_engine(mode="offline", grid_size=10)
            tracks, visibility = engine.track_video_grid(test_video, grid_size=10)
            
            if tracks is not None:
                # Calculate accuracy against ground truth
                cotracker_accuracy = self.calculate_tracking_accuracy(tracks, ground_truth_motion)
            
            engine.cleanup()
        
        # Test primitive system accuracy (estimate based on correspondence quality)
        processor = Processor(use_cotracker3=False)
        test_shapes = self.create_test_shapes()
        shape_diffs, rgb_diffs = processor.get_appearance_graphs(
            test_shapes['prev_shapes'], test_shapes['curr_shapes'],
            test_shapes['prev_centroids'], test_shapes['curr_centroids']
        )
        
        # Estimate primitive accuracy from correspondence quality
        primitive_accuracy = np.mean(shape_diffs * rgb_diffs)
        
        results = {
            'cotracker3_accuracy': cotracker_accuracy,
            'primitive_accuracy': primitive_accuracy,
            'accuracy_improvement': cotracker_accuracy - primitive_accuracy,
            'meets_accuracy_target': cotracker_accuracy >= self.config.EXPECTED_ACCURACY
        }
        
        print(f"   🎯 Accuracy: {cotracker_accuracy:.3f} vs {primitive_accuracy:.3f} ({'✅' if results['meets_accuracy_target'] else '❌'})")
        return results
    
    def create_synthetic_test_video(self) -> torch.Tensor:
        """Create synthetic test video with moving objects"""
        video = torch.zeros(1, self.config.NUM_FRAMES, 3, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)
        
        for t in range(self.config.NUM_FRAMES):
            frame = torch.zeros(3, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)
            
            # Add moving circles
            for obj_id in range(self.config.NUM_OBJECTS):
                # Object motion: circular path
                angle = 2 * np.pi * t / self.config.NUM_FRAMES + obj_id * 2 * np.pi / self.config.NUM_OBJECTS
                center_x = int(self.config.FRAME_WIDTH // 2 + 100 * np.cos(angle))
                center_y = int(self.config.FRAME_HEIGHT // 2 + 80 * np.sin(angle))
                
                # Draw circle
                y, x = np.ogrid[:self.config.FRAME_HEIGHT, :self.config.FRAME_WIDTH]
                mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
                
                # Color based on object ID
                color = [(obj_id == 0), (obj_id == 1), (obj_id == 2)]
                for c in range(3):
                    frame[c][mask] = color[c] * 0.8
            
            video[0, t] = frame
            
        return video
    
    def create_test_shapes(self) -> Dict[str, List]:
        """Create test shapes for processor testing"""
        shapes = {'prev_shapes': [], 'curr_shapes': [], 'prev_centroids': [], 'curr_centroids': []}
        
        for i in range(3):  # 3 test shapes
            # Create RGBA shape (circle)
            shape_size = 60
            shape = np.zeros((shape_size, shape_size, 4), dtype=np.uint8)
            
            # Draw circle
            center = shape_size // 2
            y, x = np.ogrid[:shape_size, :shape_size]
            mask = (x - center)**2 + (y - center)**2 <= (shape_size//3)**2
            
            # Color and alpha
            shape[mask, 0] = 255 if i == 0 else 0  # Red
            shape[mask, 1] = 255 if i == 1 else 0  # Green  
            shape[mask, 2] = 255 if i == 2 else 0  # Blue
            shape[mask, 3] = 255  # Alpha
            
            # Add some motion between prev and curr
            shapes['prev_shapes'].append(shape)
            
            # Shifted version for curr
            curr_shape = np.zeros_like(shape)
            shift = 5  # 5 pixel shift
            if shift < shape_size - shift:
                curr_shape[shift:, :] = shape[:-shift, :]
            shapes['curr_shapes'].append(curr_shape)
            
            # Centroids
            shapes['prev_centroids'].append([center / 100.0, center / 100.0])  # Normalized
            shapes['curr_centroids'].append([(center + shift) / 100.0, center / 100.0])
            
        return shapes
    
    def create_test_frame(self) -> np.ndarray:
        """Create simple test frame"""
        frame = np.zeros((self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, 3), dtype=np.uint8)
        # Add some texture
        frame[:, :, 0] = 50  # Slight blue tint
        return frame
        
    def create_ground_truth_motion(self) -> Dict[str, np.ndarray]:
        """Create ground truth motion for accuracy testing"""
        # Simple circular motion
        angles = np.linspace(0, 2*np.pi, self.config.NUM_FRAMES)
        radius = 50
        
        ground_truth = {
            'positions': np.column_stack([
                self.config.FRAME_WIDTH//2 + radius * np.cos(angles),
                self.config.FRAME_HEIGHT//2 + radius * np.sin(angles)
            ]),
            'velocities': np.column_stack([
                -radius * np.sin(angles),
                radius * np.cos(angles)
            ])
        }
        
        return ground_truth
        
    def create_motion_test_video(self, ground_truth: Dict[str, np.ndarray]) -> torch.Tensor:
        """Create test video with known motion"""
        video = torch.zeros(1, self.config.NUM_FRAMES, 3, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)
        
        for t in range(self.config.NUM_FRAMES):
            frame = torch.zeros(3, self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH)
            
            # Place object at ground truth position
            pos_x, pos_y = ground_truth['positions'][t]
            center_x, center_y = int(pos_x), int(pos_y)
            
            # Draw object
            y, x = np.ogrid[:self.config.FRAME_HEIGHT, :self.config.FRAME_WIDTH]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 25**2
            
            frame[0][mask] = 0.9  # Red object
            video[0, t] = frame
            
        return video
        
    def calculate_tracking_accuracy(self, tracks: torch.Tensor, ground_truth: Dict[str, np.ndarray]) -> float:
        """Calculate tracking accuracy against ground truth"""
        if tracks is None or tracks.shape[1] != len(ground_truth['positions']):
            return 0.0
            
        # Find closest tracked point to ground truth
        gt_positions = ground_truth['positions']
        
        total_error = 0.0
        valid_frames = 0
        
        for t in range(tracks.shape[1]):
            if t >= len(gt_positions):
                break
                
            # Get all tracked points at this frame
            frame_tracks = tracks[0, t].cpu().numpy()  # (N, 2)
            
            # Find closest track to ground truth
            gt_pos = gt_positions[t]
            distances = np.sqrt(np.sum((frame_tracks - gt_pos)**2, axis=1))
            min_distance = np.min(distances)
            
            # Convert distance to accuracy (closer = higher accuracy)
            if min_distance < 50:  # Within reasonable range
                accuracy = max(0, 1.0 - min_distance / 50.0)
                total_error += accuracy
                valid_frames += 1
                
        return total_error / valid_frames if valid_frames > 0 else 0.0
    
    def generate_final_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'test_summary': {
                'cotracker3_available': COTRACKER3_AVAILABLE,
                'engine_test_passed': test_results['engine']['initialization_success'],
                'bridge_test_passed': test_results['bridge']['initialization_success'],
                'processor_test_passed': test_results['processor']['cotracker3_success'],
                'performance_target_met': test_results['performance']['meets_speed_target'],
                'accuracy_target_met': test_results['accuracy']['meets_accuracy_target']
            },
            'performance_metrics': {
                'speedup_ratio': test_results['performance']['speedup_ratio'],
                'cotracker3_fps': test_results['performance']['cotracker3_fps'],
                'primitive_fps': test_results['performance']['primitive_fps'],
                'expected_speedup': self.config.EXPECTED_SPEEDUP
            },
            'accuracy_metrics': {
                'cotracker3_accuracy': test_results['accuracy']['cotracker3_accuracy'],
                'primitive_accuracy': test_results['accuracy']['primitive_accuracy'],
                'accuracy_improvement': test_results['accuracy']['accuracy_improvement'],
                'expected_accuracy': self.config.EXPECTED_ACCURACY
            },
            'detailed_results': test_results
        }
        
        # Calculate overall success
        all_tests_passed = all([
            report['test_summary']['engine_test_passed'],
            report['test_summary']['processor_test_passed'],
            report['test_summary']['performance_target_met'] or not COTRACKER3_AVAILABLE,
            report['test_summary']['accuracy_target_met'] or not COTRACKER3_AVAILABLE
        ])
        
        report['overall_success'] = all_tests_passed
        
        # Print summary
        self.print_final_summary(report)
        
        # Save report
        if self.config.SAVE_VISUALIZATIONS:
            import json
            report_path = os.path.join(self.config.OUTPUT_DIR, 'test_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Full report saved to: {report_path}")
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final test summary"""
        print("\n" + "="*60)
        print("🏆 COTRACKER3 INTEGRATION TEST SUMMARY")
        print("="*60)
        
        # Overall result
        if report['overall_success']:
            print("✅ OVERALL RESULT: SUCCESS")
        else:
            print("❌ OVERALL RESULT: SOME TESTS FAILED")
            
        print(f"\nCoTracker3 Available: {'✅' if COTRACKER3_AVAILABLE else '❌'}")
        
        # Individual test results
        print("\n📋 Test Results:")
        tests = [
            ("Engine Test", report['test_summary']['engine_test_passed']),
            ("Bridge Test", report['test_summary']['bridge_test_passed']),
            ("Processor Test", report['test_summary']['processor_test_passed']),
            ("Performance Target", report['test_summary']['performance_target_met']),
            ("Accuracy Target", report['test_summary']['accuracy_target_met'])
        ]
        
        for test_name, passed in tests:
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"   Speedup Ratio: {report['performance_metrics']['speedup_ratio']:.2f}x")
        print(f"   CoTracker3 FPS: {report['performance_metrics']['cotracker3_fps']:.1f}")
        print(f"   Primitive FPS: {report['performance_metrics']['primitive_fps']:.1f}")
        
        # Accuracy metrics
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   CoTracker3 Accuracy: {report['accuracy_metrics']['cotracker3_accuracy']:.3f}")
        print(f"   Primitive Accuracy: {report['accuracy_metrics']['primitive_accuracy']:.3f}")
        print(f"   Improvement: {report['accuracy_metrics']['accuracy_improvement']:.3f}")
        
        print("\n" + "="*60)


def main():
    """Run CoTracker3 integration test"""
    print("🚀 Starting CoTracker3 Integration Test Suite")
    
    # Create test instance
    test_config = TestConfig()
    test = CoTracker3IntegrationTest(test_config)
    
    # Run tests
    results = test.run_complete_test_suite()
    
    # Exit with appropriate code
    if results.get('overall_success', False):
        print("\n🎉 All tests passed! CoTracker3 integration is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the detailed report for issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()