#!/usr/bin/env python3
"""
Test script to verify engine status and configuration
"""

import sys
import os
sys.path.insert(0, '/home/runner/workspace')

# Test engine status module
print("=" * 80)
print("TESTING ENGINE STATUS MODULE")
print("=" * 80)

try:
    from motion_vectorization.engine_status import check_engine_availability, get_active_engines
    
    print("\n✅ Engine status module imported successfully")
    
    # Check engine availability
    status = check_engine_availability()
    
    print("\n📊 Engine Availability Summary:")
    print(f"  • SAM2: {'✅ Available' if status['sam2']['available'] else '❌ Not Available'}")
    print(f"  • CoTracker3: {'✅ Available' if status['cotracker3']['available'] else '❌ Not Available'}")
    print(f"  • FlowSeek: {'✅ Available' if status['flowseek']['available'] else '❌ Not Available'}")
    print(f"  • GPU: {'✅ Available' if status['gpu']['available'] else '❌ Not Available'}")
    
    active_engines = get_active_engines(status)
    print(f"\n🚀 Active Engines: {', '.join(active_engines) if active_engines else 'None'}")
    
except Exception as e:
    print(f"\n❌ Error testing engine status: {e}")
    import traceback
    traceback.print_exc()

# Test default configuration
print("\n" + "=" * 80)
print("TESTING DEFAULT CONFIGURATION")
print("=" * 80)

try:
    # Parse default arguments to check configuration
    import argparse
    from motion_vectorization.extract_shapes import parser
    
    # Get default arguments
    args = parser.parse_args([
        '--video_file', 'test.mp4',
        '--video_dir', 'videos'
    ])
    
    print("\n📝 Default Configuration:")
    print(f"  • Max frames: {args.max_frames} (should be 200)")
    print(f"  • Use unified pipeline: {args.use_unified_pipeline} (should be True)")
    print(f"  • Use SAM2: {args.use_sam2} (should be True)")
    print(f"  • Use CoTracker3: {args.use_cotracker3} (should be True)")
    print(f"  • Use FlowSeek: {args.use_flowseek} (should be True)")
    print(f"  • Quality threshold: {args.quality_threshold} (should be 0.95)")
    print(f"  • Unified mode: {args.unified_mode} (should be 'accuracy')")
    print(f"  • Use GPU: {args.use_gpu} (should be True)")
    
    # Check if all defaults are correct
    all_correct = (
        args.max_frames == 200 and
        args.use_unified_pipeline == True and
        args.use_sam2 == True and
        args.use_cotracker3 == True and
        args.use_flowseek == True and
        args.quality_threshold == 0.95 and
        args.unified_mode == 'accuracy' and
        args.use_gpu == True
    )
    
    if all_correct:
        print("\n✅ ALL DEFAULT CONFIGURATIONS ARE CORRECT!")
        print("   System is configured for MAXIMUM ACCURACY by default.")
    else:
        print("\n⚠️ Some default configurations may need adjustment.")
        
except Exception as e:
    print(f"\n❌ Error testing configuration: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)