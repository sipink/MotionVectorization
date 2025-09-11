#!/usr/bin/env python3
"""
Advanced recovery script to create proper shape_bank.pkl structure from time_bank.pkl
"""
import os
import cv2
import pickle
import numpy as np
from collections import defaultdict

video_name = "shapes38"
output_dir = f"motion_vectorization/outputs/{video_name}_None"
shapes_folder = os.path.join(output_dir, "shapes")
time_bank_path = os.path.join(output_dir, "time_bank.pkl")

# Load time_bank first
if not os.path.exists(time_bank_path):
    print(f"❌ time_bank.pkl not found at {time_bank_path}")
    exit(1)

with open(time_bank_path, 'rb') as f:
    time_bank = pickle.load(f)

print(f"Loaded time_bank with {len(time_bank['shapes'])} frames")

# Create proper shape_bank structure from time_bank
shape_bank = {-1: []}

# Add background color info
if 'bgr' in time_bank and len(time_bank['bgr']) > 0:
    shape_bank[-1] = time_bank['bgr']
    print(f"Added background color info: {len(shape_bank[-1])} entries")

# Build shape_bank from time_bank['shapes']
shape_info_dict = defaultdict(list)

for frame_idx, shapes_at_frame in time_bank['shapes'].items():
    for shape_idx, shape_data in shapes_at_frame.items():
        if shape_idx < 0:
            continue
            
        # Create shape info entry for this frame
        shape_entry = {
            't': frame_idx,  # Frame index
            'centroid': shape_data.get('centroid', [0, 0]),
            'h': [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Default transformation
        }
        
        # If we have actual transformation data, use it
        if 'h' in shape_data:
            shape_entry['h'] = shape_data['h']
        elif 'coords' in shape_data:
            # Calculate centroid from coords if not provided
            coords = shape_data['coords']
            if isinstance(coords, np.ndarray) and coords.shape == (2, 2):
                min_x, min_y = coords[0]
                max_x, max_y = coords[1]
                shape_entry['centroid'] = [(min_x + max_x) / 2, (min_y + max_y) / 2]
        
        shape_info_dict[shape_idx].append(shape_entry)

# Sort entries by frame index and add to shape_bank
for shape_idx, entries in shape_info_dict.items():
    sorted_entries = sorted(entries, key=lambda x: x['t'])
    shape_bank[shape_idx] = sorted_entries
    print(f"Shape {shape_idx}: {len(sorted_entries)} frames")

# Save the properly structured shape_bank
with open(os.path.join(output_dir, 'shape_bank.pkl'), 'wb') as f:
    pickle.dump(shape_bank, f)

print(f"✅ Successfully saved shape_bank.pkl with {len(shape_bank)-1} shapes")
print(f"   Each shape has temporal data across frames")

# Verify structure
for shape_idx in list(shape_bank.keys())[:5]:
    if shape_idx >= 0 and shape_bank[shape_idx]:
        first_entry = shape_bank[shape_idx][0]
        print(f"   Shape {shape_idx} first entry: t={first_entry.get('t')}, centroid={first_entry.get('centroid')}")