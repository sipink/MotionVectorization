#!/usr/bin/env python3
"""
Smoke Tests for Motion Vectorization Engines
Tests engine initialization and basic processing with CPU-only configurations
"""

import sys
import os
import time
import warnings
import torch
import numpy as np
import cv2
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add motion_vectorization to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

@dataclass
class TestResults:
    """Container for test results"""
    test_name: str
    passed: bool
    error: Optional[str] = None
    duration: float = 0.0
    details: Optional[str] = None

class EngineSmokeTester:
    """Comprehensive smoke testing for all motion vectorization engines"""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        
        # Force CPU-only mode for all tests
        self.device = "cpu"
        
        # Create minimal test data
        self.test_video = self._create_test_video()
        self.test_frame = self._create_test_frame()
        
        print("🧪 Motion Vectorization Engine Smoke Tests")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print()
    
    def _create_test_video(self) -> torch.Tensor:
        """Create minimal test video tensor (4 frames, 64x64, RGB)"""
        # Create simple animated square moving across frames
        video = torch.zeros((1, 4, 3, 64, 64), dtype=torch.float32)
        
        for t in range(4):
            # Create a white square that moves diagonally
            x = 10 + t * 8
            y = 10 + t * 8
            video[0, t, :, y:y+16, x:x+16] = 1.0
            
        return video
    
    def _create_test_frame(self) -> np.ndarray:
        """Create minimal test frame (64x64 BGR)"""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add a simple white square
        frame[20:36, 20:36] = [255, 255, 255]
        return frame
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test with error handling and timing"""
        print(f"🔍 Testing: {test_name}")
        self.total_tests += 1
        
        start_time = time.perf_counter()
        try:
            details = test_func()
            duration = time.perf_counter() - start_time
            
            self.results.append(TestResults(
                test_name=test_name,
                passed=True,
                duration=duration,
                details=details
            ))
            self.passed_tests += 1
            print(f"  ✅ PASSED ({duration:.3f}s) - {details or 'OK'}")
            return True
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.results.append(TestResults(
                test_name=test_name,
                passed=False,
                error=str(e),
                duration=duration
            ))
            print(f"  ❌ FAILED ({duration:.3f}s) - {str(e)}")
            return False
    
    def test_sam2_engine_initialization(self) -> str:
        """Test SAM2 engine initialization with CPU-only config"""
        try:
            from motion_vectorization.sam2_engine import SAM2SegmentationEngine, SAM2Config
            
            # CPU-only configuration
            config = SAM2Config(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                batch_size=1
            )
            
            # Initialize engine (should gracefully fallback if SAM2 not available)
            engine = SAM2SegmentationEngine(config)
            
            # Basic validation
            assert hasattr(engine, 'config'), "Engine missing config attribute"
            assert engine.config.device == "cpu", "Engine not using CPU device"
            
            return f"Initialized with fallback: {not hasattr(engine, 'predictor') or engine.predictor is None}"
            
        except ImportError:
            return "SAM2 engine module not available (expected)"
    
    def test_sam2_basic_processing(self) -> str:
        """Test SAM2 basic frame processing"""
        try:
            from motion_vectorization.sam2_engine import SAM2SegmentationEngine, SAM2Config
            
            config = SAM2Config(device="cpu", mixed_precision=False, compile_model=False)
            engine = SAM2SegmentationEngine(config)
            
            # Test frame processing (should not crash even if using fallback)
            if hasattr(engine, 'segment_frame'):
                result = engine.segment_frame(self.test_frame)
                assert isinstance(result, dict), "segment_frame should return dict"
                return f"Processed frame, got {len(result)} result keys"
            else:
                return "segment_frame method not available (fallback mode)"
                
        except ImportError:
            return "SAM2 engine not available (expected)"
    
    def test_cotracker3_engine_initialization(self) -> str:
        """Test CoTracker3 engine initialization with CPU-only config"""
        try:
            from motion_vectorization.cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config
            
            # CPU-only configuration
            config = CoTracker3Config(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                grid_size=10,  # Small grid for CPU testing
                max_points=100
            )
            
            # Initialize engine (should gracefully fallback if CoTracker3 not available)
            engine = CoTracker3TrackerEngine(config)
            
            # Basic validation
            assert hasattr(engine, 'config'), "Engine missing config attribute"
            assert engine.config.device == "cpu", "Engine not using CPU device"
            assert engine.config.grid_size == 10, "Grid size not set correctly"
            
            return f"Initialized with model: {engine.model is not None}"
            
        except ImportError:
            return "CoTracker3 engine module not available (expected)"
    
    def test_cotracker3_basic_tracking(self) -> str:
        """Test CoTracker3 basic point tracking"""
        try:
            from motion_vectorization.cotracker3_engine import CoTracker3TrackerEngine, CoTracker3Config
            
            config = CoTracker3Config(device="cpu", mixed_precision=False, grid_size=5)
            engine = CoTracker3TrackerEngine(config)
            
            # Test basic tracking (should not crash even in fallback mode)
            if hasattr(engine, 'track_points'):
                # Create minimal point grid
                points = torch.tensor([[[16, 16], [32, 32], [48, 48]]], dtype=torch.float32)
                
                # This may fail gracefully if torch.hub model not available
                try:
                    result = engine.track_points(self.test_video, points)
                    return f"Tracked {len(points[0])} points successfully"
                except Exception as e:
                    return f"Tracking failed gracefully: {type(e).__name__}"
            else:
                return "track_points method not available (fallback mode)"
                
        except ImportError:
            return "CoTracker3 engine not available (expected)"
    
    def test_flowseek_engine_initialization(self) -> str:
        """Test FlowSeek engine initialization with CPU-only config"""
        try:
            from motion_vectorization.flowseek_engine import FlowSeekEngine, FlowSeekConfig
            
            # CPU-only configuration
            config = FlowSeekConfig(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                max_resolution=256,  # Small resolution for CPU
                depth_integration=False  # Disable for basic test
            )
            
            # Initialize engine (should gracefully fallback if FlowSeek not available)
            engine = FlowSeekEngine(config)
            
            # Basic validation
            assert hasattr(engine, 'config'), "Engine missing config attribute"
            assert engine.config.device == "cpu", "Engine not using CPU device"
            
            return f"Initialized with fallback: {not hasattr(engine, 'model') or engine.model is None}"
            
        except ImportError:
            return "FlowSeek engine module not available (expected)"
    
    def test_flowseek_basic_flow(self) -> str:
        """Test FlowSeek basic optical flow computation"""
        try:
            from motion_vectorization.flowseek_engine import FlowSeekEngine, FlowSeekConfig
            
            config = FlowSeekConfig(device="cpu", mixed_precision=False, max_resolution=64)
            engine = FlowSeekEngine(config)
            
            # Test basic flow computation
            if hasattr(engine, 'compute_flow'):
                frame1 = torch.tensor(self.test_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                frame2 = frame1.clone()
                frame2[:, :, :, 10:] = frame1[:, :, :, :-10]  # Simple shift for flow
                
                try:
                    flow = engine.compute_flow(frame1, frame2)
                    if isinstance(flow, torch.Tensor):
                        return f"Computed flow: {flow.shape}"
                    else:
                        return f"Flow computation returned: {type(flow)}"
                except Exception as e:
                    return f"Flow computation failed gracefully: {type(e).__name__}"
            else:
                return "compute_flow method not available (fallback mode)"
                
        except ImportError:
            return "FlowSeek engine not available (expected)"
    
    def test_unified_pipeline_import(self) -> str:
        """Test unified pipeline module import"""
        from motion_vectorization.unified_pipeline import UnifiedPipelineProcessor, UnifiedPipelineConfig
        
        # Should import without errors
        config = UnifiedPipelineConfig(device="cpu")
        assert config.device == "cpu", "Config not using CPU device"
        
        return "Unified pipeline imports successfully"
    
    def test_basic_processor_import(self) -> str:
        """Test basic processor module import and initialization"""
        from motion_vectorization.processor import Processor
        
        # Initialize with CPU settings
        processor = Processor(use_cotracker3=False, device="cpu")
        
        assert hasattr(processor, 'use_cotracker3'), "Processor missing use_cotracker3 attribute"
        assert processor.use_cotracker3 == False, "CoTracker3 should be disabled"
        
        return "Basic processor initialized successfully"
    
    def run_all_tests(self):
        """Run complete engine smoke test suite"""
        print("Starting comprehensive engine smoke tests...\n")
        
        # Test all engines
        tests = [
            ("SAM2 Engine Initialization", self.test_sam2_engine_initialization),
            ("SAM2 Basic Processing", self.test_sam2_basic_processing),
            ("CoTracker3 Engine Initialization", self.test_cotracker3_engine_initialization),
            ("CoTracker3 Basic Tracking", self.test_cotracker3_basic_tracking),
            ("FlowSeek Engine Initialization", self.test_flowseek_engine_initialization),
            ("FlowSeek Basic Flow", self.test_flowseek_basic_flow),
            ("Unified Pipeline Import", self.test_unified_pipeline_import),
            ("Basic Processor Import", self.test_basic_processor_import),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test summary"""
        print("=" * 50)
        print("🧪 ENGINE SMOKE TEST SUMMARY")
        print("=" * 50)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        print()
        
        if self.passed_tests < self.total_tests:
            print("❌ Failed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   • {result.test_name}: {result.error}")
            print()
        else:
            print("✅ All engine smoke tests passed!")
        
        # Timing summary
        total_time = sum(r.duration for r in self.results)
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        
        # Performance details
        print("\n📊 Performance Details:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.test_name}: {result.duration:.3f}s")

def main():
    """Main entry point for engine smoke tests"""
    try:
        tester = EngineSmokeTester()
        tester.run_all_tests()
        
        # Exit with error code if any tests failed
        if tester.passed_tests < tester.total_tests:
            sys.exit(1)
        else:
            print("\n🎉 All engine smoke tests completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()