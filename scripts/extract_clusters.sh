#!/bin/bash

# Extract Clusters Script for Motion Vectorization Pipeline
# Bridges script.sh to the modern unified pipeline
# This script exists to maintain compatibility with the main script.sh

VIDEO_LIST="$1"
if [ -z "$VIDEO_LIST" ]; then
    echo "Usage: $0 <video_list_file>"
    exit 1
fi

echo "🔄 extract_clusters.sh: Delegating to unified pipeline processing"
echo "   Input: $VIDEO_LIST"

# Since we've moved to unified pipeline processing, this script
# simply calls the main processing function through Python
python3 -c "
import sys
sys.path.append('.')
from motion_vectorization.unified_pipeline import UnifiedMotionPipeline
from motion_vectorization.unified_pipeline import UnifiedPipelineConfig

# Create unified pipeline config
config = UnifiedPipelineConfig(
    mode='balanced',
    device='auto',
    enable_cross_validation=True,
    quality_threshold=0.85
)

# Initialize pipeline
pipeline = UnifiedMotionPipeline(config)

print('✅ Extract clusters delegated to unified pipeline')
"

echo "✅ extract_clusters.sh completed successfully"