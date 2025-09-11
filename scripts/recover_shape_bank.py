#!/usr/bin/env python3
"""
Emergency recovery script to create shape_bank.pkl from existing shapes folder
"""
import os
import cv2
import pickle
import numpy as np

video_name = "shapes38"
output_dir = f"motion_vectorization/outputs/{video_name}_None"
shapes_folder = os.path.join(output_dir, "shapes")

# Create shape_bank from the saved shape images
shape_bank = {-1: []}

if os.path.exists(shapes_folder):
    print(f"Recovering shape_bank from {shapes_folder}")
    
    for shape_file in os.listdir(shapes_folder):
        if shape_file.endswith('.png'):
            shape_idx = int(shape_file.split('.')[0])
            shape_img = cv2.imread(os.path.join(shapes_folder, shape_file), cv2.IMREAD_UNCHANGED)
            
            if shape_img is not None:
                shape_bank[shape_idx] = shape_img
                print(f"  Recovered shape {shape_idx}: {shape_img.shape}")
    
    # Save the recovered shape_bank
    with open(os.path.join(output_dir, 'shape_bank.pkl'), 'wb') as f:
        pickle.dump(shape_bank, f)
    
    print(f"✅ Successfully saved shape_bank.pkl with {len(shape_bank)-1} shapes")
else:
    print(f"❌ Shapes folder not found at {shapes_folder}")
    print("You'll need to re-run the extract_shapes step")