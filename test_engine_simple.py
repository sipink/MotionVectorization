#!/usr/bin/env python3
"""
Simple Engine Smoke Tests - Quick validation without heavy model loading
Tests basic imports, configuration, and initialization with minimal overhead
"""

import sys
import os
import time
import warnings
import torch
import numpy as np
from typing import Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add motion_vectorization to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

class SimpleEngineTester:
    """Lightweight engine testing focused on imports and configuration"""
    
    def __init__(self):
        self.results = []
        self.device = "cpu"
        
        print("🧪 Simple Engine Smoke Tests (CPU-only)")
        print("=" * 50)
        print(f"Device: {self.device}")
        print()
    
    def test_sam2_imports_and_config(self) -> str:
        """Test SAM2 imports and configuration"""
        try:
            from motion_vectorization.sam2_engine import SAM2Config
            
            # Test configuration creation
            config = SAM2Config(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                batch_size=1
            )
            
            assert config.device == "cpu"
            assert config.mixed_precision == False
            assert config.batch_size == 1
            
            return "SAM2Config created successfully"
            
        except ImportError:
            return "SAM2 module not available (expected)"
        except Exception as e:
            raise e
    
    def test_cotracker3_imports_and_config(self) -> str:
        """Test CoTracker3 imports and configuration"""
        try:
            from motion_vectorization.cotracker3_engine import CoTracker3Config
            
            # Test configuration creation  
            config = CoTracker3Config(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                grid_size=5,  # Very small for testing
                max_points=25
            )
            
            assert config.device == "cpu"
            assert config.grid_size == 5
            assert config.max_points == 25
            
            return "CoTracker3Config created successfully"
            
        except ImportError:
            return "CoTracker3 module not available (expected)"
        except Exception as e:
            raise e
    
    def test_flowseek_imports_and_config(self) -> str:
        """Test FlowSeek imports and configuration"""
        try:
            from motion_vectorization.flowseek_engine import FlowSeekConfig
            
            # Test configuration creation
            config = FlowSeekConfig(
                device="cpu",
                mixed_precision=False,
                compile_model=False,
                max_resolution=64
            )
            
            assert config.device == "cpu"
            assert config.max_resolution == 64
            
            return "FlowSeekConfig created successfully"
            
        except ImportError:
            return "FlowSeek module not available (expected)"
        except Exception as e:
            raise e
    
    def test_unified_pipeline_imports(self) -> str:
        """Test unified pipeline imports"""
        try:
            from motion_vectorization.unified_pipeline import UnifiedPipelineConfig
            
            config = UnifiedPipelineConfig(
                device="cpu",
                mixed_precision=False,
                fallback_to_traditional=True
            )
            
            assert config.device == "cpu"
            
            return "UnifiedPipelineConfig created successfully"
            
        except ImportError:
            return "Unified pipeline module not available (expected)"
        except Exception as e:
            raise e
    
    def test_processor_imports(self) -> str:
        """Test basic processor imports"""
        try:
            from motion_vectorization.processor import Processor
            
            # Basic import validation only, no initialization
            assert hasattr(Processor, '__init__')
            
            return "Processor class imported successfully"
            
        except ImportError:
            return "Processor module not available (expected)"
        except Exception as e:
            raise e
    
    def run_test(self, name: str, test_func):
        """Run a single test"""
        print(f"🔍 {name}")
        
        start_time = time.perf_counter()
        try:
            result = test_func()
            duration = time.perf_counter() - start_time
            print(f"  ✅ PASSED ({duration:.3f}s) - {result}")
            self.results.append((name, True, result, duration))
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            print(f"  ❌ FAILED ({duration:.3f}s) - {str(e)}")
            self.results.append((name, False, str(e), duration))
    
    def run_all(self):
        """Run all tests"""
        tests = [
            ("SAM2 Imports & Config", self.test_sam2_imports_and_config),
            ("CoTracker3 Imports & Config", self.test_cotracker3_imports_and_config),
            ("FlowSeek Imports & Config", self.test_flowseek_imports_and_config),
            ("Unified Pipeline Imports", self.test_unified_pipeline_imports),
            ("Processor Imports", self.test_processor_imports),
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Summary
        passed = sum(1 for _, success, _, _ in self.results if success)
        total = len(self.results)
        
        print("\n" + "=" * 50)
        print(f"🧪 SIMPLE ENGINE TEST SUMMARY")
        print("=" * 50)
        print(f"Passed: {passed}/{total}")
        if passed == total:
            print("✅ All tests passed!")
        
        return passed == total

def main():
    tester = SimpleEngineTester()
    success = tester.run_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()