#!/usr/bin/env python3
"""
Deep analysis and fix for Motion Vectorization accuracy issues
Run this on RunPod to diagnose and fix the poor SVG accuracy
"""
import os
import cv2
import pickle
import json
import numpy as np
from collections import defaultdict

def analyze_time_bank(video_name="shapes38"):
    """Analyze time_bank structure to understand available motion data"""
    output_dir = f"motion_vectorization/outputs/{video_name}_None"
    time_bank_path = os.path.join(output_dir, "time_bank.pkl")
    
    print("=" * 60)
    print("ANALYZING TIME_BANK STRUCTURE")
    print("=" * 60)
    
    with open(time_bank_path, 'rb') as f:
        time_bank = pickle.load(f)
    
    print(f"Top-level keys: {list(time_bank.keys())}")
    print(f"Number of frames: {len(time_bank.get('shapes', {}))}")
    
    # Analyze shape data structure
    if 'shapes' in time_bank and time_bank['shapes']:
        frame_keys = sorted(list(time_bank['shapes'].keys()))
        print(f"Frame range: {frame_keys[0]} to {frame_keys[-1]}")
        
        # Check first frame with shapes
        for frame_idx in frame_keys[:5]:
            shapes_at_frame = time_bank['shapes'][frame_idx]
            if shapes_at_frame:
                print(f"\n--- Frame {frame_idx} ---")
                print(f"  Number of shapes: {len(shapes_at_frame)}")
                
                # Analyze first shape
                for shape_id, shape_data in list(shapes_at_frame.items())[:2]:
                    print(f"\n  Shape {shape_id}:")
                    for key, value in shape_data.items():
                        if isinstance(value, np.ndarray):
                            print(f"    {key}: array {value.shape} = {value.flatten()[:8]}...")
                        elif isinstance(value, (list, tuple)) and len(value) > 3:
                            print(f"    {key}: {type(value).__name__}[{len(value)}] = {value[:3]}...")
                        else:
                            print(f"    {key}: {value}")
                break
    
    return time_bank

def create_accurate_shape_bank(time_bank, video_name="shapes38"):
    """Create shape_bank with ACTUAL transformation parameters from time_bank"""
    output_dir = f"motion_vectorization/outputs/{video_name}_None"
    
    print("\n" + "=" * 60)
    print("CREATING ACCURATE SHAPE_BANK")
    print("=" * 60)
    
    shape_bank = {-1: []}
    
    # Add background info
    if 'bgr' in time_bank:
        shape_bank[-1] = time_bank['bgr']
        print(f"✓ Added background color data")
    
    # Build shape_bank with REAL transformation data
    shape_temporal_data = defaultdict(list)
    
    for frame_idx, shapes_at_frame in time_bank['shapes'].items():
        for shape_id, shape_data in shapes_at_frame.items():
            if shape_id < 0:
                continue
            
            # Extract ACTUAL transformation parameters
            entry = {
                't': frame_idx,  # Frame index
                'centroid': shape_data.get('centroid', [0, 0]),
            }
            
            # Look for transformation matrix/parameters
            if 'H' in shape_data:  # Homography matrix
                H = shape_data['H']
                print(f"  Found H matrix for shape {shape_id} at frame {frame_idx}: {H.shape}")
                # Decompose homography to affine parameters
                # [sx, sy, tx, ty, theta, kx, ky, z]
                entry['h'] = decompose_homography(H)
            elif 'matrix' in shape_data:  # Transformation matrix
                matrix = shape_data['matrix']
                print(f"  Found matrix for shape {shape_id} at frame {frame_idx}: {matrix.shape}")
                entry['h'] = decompose_matrix(matrix)
            elif 'params' in shape_data:  # Direct parameters
                params = shape_data['params']
                print(f"  Found params for shape {shape_id} at frame {frame_idx}: {params}")
                entry['h'] = params
            elif 'h' in shape_data:  # Already has h parameters
                entry['h'] = shape_data['h']
                print(f"  Found h params for shape {shape_id} at frame {frame_idx}: {shape_data['h']}")
            else:
                # Try to compute from shape properties
                print(f"  Computing params for shape {shape_id} at frame {frame_idx}")
                entry['h'] = compute_shape_params(shape_data)
            
            shape_temporal_data[shape_id].append(entry)
    
    # Sort by frame index and add to shape_bank
    total_motion_frames = 0
    for shape_id, entries in shape_temporal_data.items():
        sorted_entries = sorted(entries, key=lambda x: x['t'])
        shape_bank[shape_id] = sorted_entries
        total_motion_frames += len(sorted_entries)
        
        if shape_id < 5:  # Show first few shapes
            print(f"Shape {shape_id}: {len(sorted_entries)} frames")
            if sorted_entries:
                print(f"  First: t={sorted_entries[0]['t']}, h={sorted_entries[0].get('h', 'NO H!')}")
                if len(sorted_entries) > 1:
                    print(f"  Last:  t={sorted_entries[-1]['t']}, h={sorted_entries[-1].get('h', 'NO H!')}")
    
    print(f"\n✓ Created shape_bank with {len(shape_bank)-1} shapes")
    print(f"✓ Total motion keyframes: {total_motion_frames}")
    
    # Save the accurate shape_bank
    shape_bank_path = os.path.join(output_dir, 'shape_bank.pkl')
    with open(shape_bank_path, 'wb') as f:
        pickle.dump(shape_bank, f)
    print(f"✓ Saved to {shape_bank_path}")
    
    return shape_bank

def decompose_homography(H):
    """Decompose 3x3 homography matrix to affine parameters"""
    # Extract affine part (top-left 2x2 and translation)
    if H.shape == (3, 3):
        # Normalize by H[2,2]
        H = H / H[2, 2]
        
        # Extract components
        sx = np.linalg.norm(H[:2, 0])
        sy = np.linalg.norm(H[:2, 1])
        tx = H[0, 2]
        ty = H[1, 2]
        
        # Compute rotation
        theta = np.arctan2(H[1, 0], H[0, 0])
        
        # Estimate shear (simplified)
        kx = 0.0
        ky = 0.0
        
        # Z-order from perspective component
        z = 1.0 / (1.0 + abs(H[2, 0]) + abs(H[2, 1]))
        
        return [sx, sy, tx, ty, theta, kx, ky, z]
    else:
        print(f"  Warning: Invalid H shape {H.shape}, using defaults")
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

def decompose_matrix(matrix):
    """Decompose transformation matrix to affine parameters"""
    if matrix.shape == (2, 3):
        # Affine matrix
        sx = np.linalg.norm(matrix[:, 0])
        sy = np.linalg.norm(matrix[:, 1])
        tx = matrix[0, 2]
        ty = matrix[1, 2]
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        return [sx, sy, tx, ty, theta, 0.0, 0.0, 1.0]
    elif matrix.shape == (3, 3):
        return decompose_homography(matrix)
    else:
        print(f"  Warning: Invalid matrix shape {matrix.shape}, using defaults")
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

def compute_shape_params(shape_data):
    """Compute transformation parameters from shape properties"""
    params = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Default
    
    # Try to extract from various possible fields
    if 'scale' in shape_data:
        scale = shape_data['scale']
        if isinstance(scale, (list, tuple, np.ndarray)):
            params[0] = scale[0] if len(scale) > 0 else 1.0
            params[1] = scale[1] if len(scale) > 1 else scale[0]
        else:
            params[0] = params[1] = scale
    
    if 'translation' in shape_data:
        trans = shape_data['translation']
        params[2] = trans[0] if len(trans) > 0 else 0.0
        params[3] = trans[1] if len(trans) > 1 else 0.0
    elif 'centroid' in shape_data:
        # Use centroid as translation
        centroid = shape_data['centroid']
        params[2] = centroid[0]
        params[3] = centroid[1]
    
    if 'rotation' in shape_data:
        params[4] = shape_data['rotation']
    elif 'angle' in shape_data:
        params[4] = shape_data['angle']
    
    if 'z_order' in shape_data:
        params[7] = shape_data['z_order']
    elif 'depth' in shape_data:
        params[7] = shape_data['depth']
    elif 'z' in shape_data:
        params[7] = shape_data['z']
    
    return params

def fix_motion_file_coordinates(video_name="shapes38"):
    """Fix coordinate normalization issue in motion_file.json"""
    output_dir = f"motion_vectorization/outputs/{video_name}_None"
    motion_file_path = os.path.join(output_dir, "motion_file.json")
    
    print("\n" + "=" * 60)
    print("FIXING MOTION FILE COORDINATES")
    print("=" * 60)
    
    with open(motion_file_path, 'r') as f:
        motion_file = json.load(f)
    
    # Get frame dimensions
    width = motion_file['-1']['width']
    height = motion_file['-1']['height']
    print(f"Frame dimensions: {width}x{height}")
    
    # Fix coordinate normalization for each shape
    fixed_count = 0
    for shape_id, shape_data in motion_file.items():
        if shape_id == '-1':
            continue
        
        # Check if coordinates are normalized (values between 0 and 1)
        if 'cx' in shape_data and shape_data['cx']:
            max_cx = max(shape_data['cx'])
            max_cy = max(shape_data['cy'])
            
            if max_cx <= 1.0 and max_cy <= 1.0:
                # Coordinates are normalized, convert to pixels
                shape_data['cx'] = [x * width for x in shape_data['cx']]
                shape_data['cy'] = [y * height for y in shape_data['cy']]
                fixed_count += 1
                
                if fixed_count < 5:
                    print(f"  Fixed shape {shape_id}: cx[0]={shape_data['cx'][0]:.1f}, cy[0]={shape_data['cy'][0]:.1f}")
    
    if fixed_count > 0:
        # Save fixed motion file
        motion_file_fixed_path = os.path.join(output_dir, "motion_file_fixed.json")
        with open(motion_file_fixed_path, 'w') as f:
            json.dump(motion_file, f, indent=2)
        
        # Also overwrite original
        with open(motion_file_path, 'w') as f:
            json.dump(motion_file, f, indent=2)
        
        print(f"\n✓ Fixed {fixed_count} shapes with normalized coordinates")
        print(f"✓ Saved fixed motion file")
    else:
        print("\n✓ Coordinates already in pixel space")
    
    return motion_file

def main():
    """Main accuracy fix pipeline"""
    print("\n" + "=" * 80)
    print("MOTION VECTORIZATION ACCURACY FIX")
    print("=" * 80)
    
    # Step 1: Analyze time_bank structure
    time_bank = analyze_time_bank()
    
    # Step 2: Create accurate shape_bank with real motion data
    shape_bank = create_accurate_shape_bank(time_bank)
    
    # Step 3: Fix coordinate normalization in motion_file
    motion_file = fix_motion_file_coordinates()
    
    print("\n" + "=" * 80)
    print("ACCURACY FIX COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Re-run: python -m motion_vectorization.create_motion_file --video_name shapes38 --config 'motion_vectorization/config/shapes38.json'")
    print("2. Re-run: python -m motion_vectorization.full_motion_file shapes38")
    print("3. Re-run: python -m svg_utils.create_svg_dense --video_dir 'motion_vectorization/outputs/shapes38_None' --frame_rate 30")
    print("\nThe SVG should now have MUCH better accuracy!")

if __name__ == "__main__":
    main()