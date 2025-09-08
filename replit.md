# Motion Vectorization Project

## Project Overview
This is a research project for "Editing Motion Graphics Video via Motion Vectorization and Program Transformation". The system converts videos into animated SVG graphics by analyzing motion patterns, extracting shapes, and creating smooth vector animations.

## Current Status
- **Setup Complete**: All dependencies installed and configured for Replit environment
- **RAFT Models**: Downloaded and configured for optical flow estimation
- **Web Interface**: Running on port 5000 with user-friendly controls
- **CPU Optimized**: Modified for CPU-only execution (CUDA not available in Replit)

## Project Architecture

### Core Components
1. **RAFT Module** (`RAFT/`)
   - Optical flow estimation using deep learning
   - Pre-trained models for different datasets (Things, Sintel, KITTI, etc.)
   - Modified for CPU execution and proper imports

2. **Motion Vectorization** (`motion_vectorization/`)
   - Main processing pipeline for shape extraction and tracking
   - Configuration files for different videos (`config/`)
   - GUI tool for manual corrections (`correction_gui.py`)

3. **SVG Utils** (`svg_utils/`)
   - Converts motion data to animated SVG files
   - JavaScript utilities for SVG visualization

4. **Processing Scripts** (`scripts/`)
   - Bash scripts orchestrating the full pipeline
   - Parallel processing support for multiple videos

### Web Interface (`app.py`)
- Flask-based web application on port 5000
- Provides access to:
  - RAFT optical flow demonstration
  - Video listing and processing
  - Pipeline status monitoring

## File Structure
```
├── RAFT/                    # Optical flow estimation
│   ├── models/             # Pre-trained model files
│   ├── core/               # Core RAFT implementation
│   └── demo-frames/        # Sample frames for testing
├── motion_vectorization/   # Main processing pipeline
│   ├── config/            # Video-specific configurations
│   └── *.py               # Processing modules
├── svg_utils/             # SVG generation utilities
├── scripts/               # Pipeline orchestration scripts
├── videos/                # Input videos and processing outputs
├── app.py                 # Web interface
└── templates/             # HTML templates
```

## Usage Instructions

### Web Interface (Recommended)
1. Navigate to the web preview (port 5000)
2. Use the interface to:
   - Run RAFT demo on sample frames
   - List available videos
   - Process videos through the full pipeline

### Command Line Usage
1. **RAFT Demo**: `cd RAFT && python demo.py --model=models/raft-things.pth --path=demo-frames`
2. **Full Pipeline**: `./scripts/script.sh videos/test.txt`
3. **SVG Generation**: `./scripts/convert_to_svg.sh test1 30`

### Adding New Videos
1. Place MP4/MOV files in `videos/` directory
2. Create corresponding JSON config in `motion_vectorization/config/`
3. Add video name to a text file in `videos/`
4. Run processing pipeline

## Dependencies Installed
- **Core Libraries**: torch, torchvision, opencv-python, numpy, scipy
- **Image Processing**: pillow, scikit-image, matplotlib
- **Motion Analysis**: kornia, networkx
- **Utilities**: tqdm, pyefd, easing-functions, drawsvg
- **Web Interface**: flask

## Environment Optimizations
- Modified RAFT for CPU execution (no CUDA available)
- Fixed import paths for module compatibility
- Added proper device detection and mapping
- Created web interface for easier access

## Performance Notes
- Processing is CPU-intensive and may take significant time
- RAFT demo runs slowly on CPU (expected behavior)
- Full video processing can take hours depending on video length
- Use shorter videos or reduced frame counts for testing

## Recent Changes
- **September 8, 2025**: Initial Replit setup completed
- Fixed RAFT imports and CUDA/CPU compatibility
- Created Flask web interface for user-friendly access
- Installed all required dependencies
- Modified demo scripts for proper execution

## User Preferences
- Web interface preferred for ease of use
- CPU-only execution environment
- Comprehensive status monitoring and logging

## Next Steps
1. Test full pipeline with sample videos
2. Optimize processing for better performance
3. Add more interactive features to web interface
4. Implement result visualization tools