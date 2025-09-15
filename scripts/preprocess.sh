#!/bin/bash

# Preprocessing wrapper script for Motion Vectorization Pipeline
# Calls the Python preprocess module with proper argument handling

# Check arguments
if [ $# -lt 1 ]; then
    echo "❌ Error: Missing video list file"
    echo "Usage: $0 <video_list> [threshold] [max_frames]"
    exit 1
fi

VIDEO_LIST="$1"
THRESHOLD="${2:-0.0001}"
MAX_FRAMES="${3:-500}"

# Ensure video list exists
if [ ! -f "$VIDEO_LIST" ]; then
    echo "❌ Error: Video list file '$VIDEO_LIST' not found"
    exit 1
fi

echo "🔧 PREPROCESS: Processing video list '$VIDEO_LIST'"
echo "  📊 Threshold: $THRESHOLD"
echo "  🎯 Max frames: $MAX_FRAMES"

# Process each video in the list
while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^#.* ]]; then
        echo "📹 Processing video: $line"
        
        python3 motion_vectorization/preprocess.py \
            --video_file "$line" \
            --video_dir "videos" \
            --thresh "$THRESHOLD" \
            --max_frames "$MAX_FRAMES" \
            --use_ai_engines \
            --generate_labels \
            --generate_flow \
            --device "auto"
            
        if [ $? -ne 0 ]; then
            echo "❌ Preprocessing failed for $line"
            exit 1
        fi
        
        echo "✅ Preprocessing completed for $line"
    fi
done < "$VIDEO_LIST"

echo "✅ All preprocessing completed successfully"