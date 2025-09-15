# Overview

This project implements motion vectorization for editing motion graphics videos through a program transformation approach using cutting-edge 2024-2025 AI technologies. The system extracts and analyzes motion patterns from video content, converts them into parametric representations, and generates editable SVG animations that exactly match the original video. The pipeline includes video preprocessing, advanced segmentation using SAM2.1, point tracking using CoTracker3, optical flow computation using FlowSeek, motion optimization, and comprehensive SVG generation for output.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Processing Pipeline
The system follows a multi-stage pipeline architecture:
- **Video Preprocessing**: Extract frames and prepare video data for analysis
- **Optical Flow Computation**: Uses FlowSeek (ICCV 2025) for state-of-the-art dense optical flow estimation with 10-15% accuracy improvement
- **Shape Extraction**: Identifies and segments moving objects using clustering and connected components
- **Motion Tracking**: Tracks shape correspondences across frames using shape context descriptors
- **Motion Optimization**: Fits parametric motion models to tracked shapes using gradient descent
- **SVG Generation**: Converts motion data and shape masks to editable SVG animations with exact visual fidelity

## Data Flow Architecture
The system processes video data through a dataloader that manages:
- RGB frames stored as PNG sequences
- Foreground/background segmentation masks
- Component labels for shape identification
- Forward and backward optical flow fields

## Shape Representation
Objects are represented using affine transformation parameters:
- Translation (cx, cy): Center position coordinates
- Scale (sx, sy): Scaling factors in x and y directions  
- Rotation (theta): Rotation angle
- Shear (kx, ky): Shear transformation parameters
- Z-order: Depth ordering for compositing

## Optimization Framework
Motion parameter optimization uses PyTorch-based gradient descent:
- Differentiable sampling and compositing operations
- Perceptual loss functions comparing rendered and original frames
- Support for GPU acceleration when available
- Configurable optimization constraints and regularization

## Configuration System
JSON-based configuration management allows per-video parameter tuning:
- Clustering and segmentation thresholds
- Matching and tracking parameters
- Optimization settings and constraints
- Output format specifications

## Modular Design
The codebase is organized into specialized modules:
- `dataloader.py`: Video data management and frame loading
- `processor.py`: Shape matching and correspondence algorithms
- `compositing.py`: Differentiable rendering and blending operations
- `sampling.py`: Geometric transformation and sampling utilities
- `svg_generator.py`: Comprehensive SVG generation with exact video matching
- `unified_pipeline.py`: Orchestrated processing pipeline with AI engine integration
- `visualizer.py`: Debug visualization and progress monitoring

## SVG Generation System
Advanced SVG animation generation with exact video fidelity:
- **Path Conversion**: PNG shape masks converted to optimized SVG paths
- **Transform Accuracy**: 7-parameter affine transformations (translation, scale, rotation, shear) with mathematical precision
- **Visual Fidelity**: Image-based rendering with clipPath for exact texture preservation
- **Quality Modes**: Speed, balanced, and quality modes for different accuracy/performance needs
- **Validation Framework**: Automated SSIM/PSNR validation ensures exact video matching
- **Full Editability**: Generated SVG animations are fully editable in standard design tools

# External Dependencies

## Core Dependencies
- **PyTorch**: Deep learning framework for optimization and GPU acceleration
- **FlowSeek**: State-of-the-art optical flow model (ICCV 2025) with 10-15% accuracy improvement
- **OpenCV**: Computer vision operations and video processing
- **Kornia**: Differentiable computer vision operations for PyTorch
- **scikit-image**: Image processing and feature extraction
- **NetworkX**: Graph algorithms for shape matching

## Specialized Libraries
- **pymatting**: Alpha matting for shape extraction
- **pyefd**: Elliptic Fourier descriptors for shape analysis
- **drawSvg**: SVG generation and manipulation
- **easing-functions**: Animation easing for motion interpolation

## Scientific Computing
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing and optimization algorithms
- **matplotlib**: Plotting and visualization

## Optional Components
- **tkinter**: GUI for manual correction and visualization
- **Pillow (PIL)**: Additional image processing capabilities
- **tqdm**: Progress bars for long-running operations

The system requires CUDA-compatible hardware for optimal performance, with fallback CPU support available for all operations.