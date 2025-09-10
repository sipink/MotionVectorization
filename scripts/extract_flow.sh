#!/bin/bash

# FlowSeek Flow Extraction Script (ICCV 2025) - State-of-the-Art Optical Flow
# Usage: ./extract_flow.sh <batch_file> <max_frames> [additional_options]

BATCH_FILE=$1
MAX_FRAMES=$2
FLOW_ENGINE="flowseek"  # Only FlowSeek supported - RAFT removed for cleanup
ADDITIONAL_ARGS="${@:3}"      # Additional arguments passed to flow extraction

echo "🎬 FLOWSEEK FLOW EXTRACTION (ICCV 2025)"
echo "========================================"
echo "📂 Batch file: $BATCH_FILE"
echo "🎯 Max frames: $MAX_FRAMES"
echo "🚀 Engine: FlowSeek (state-of-the-art)"
echo ""
echo "✨ FlowSeek Advantages:"
echo "   • 10-15% accuracy improvement over legacy methods"
echo "   • 8x less hardware requirements"
echo "   • Superior cross-dataset generalization"
echo "   • Depth-aware motion understanding"
echo "   • ICCV 2025 state-of-the-art optical flow"

echo "================================"

while read -r line
do
        if [[ $line =~ ^#.*  ]]; then
                continue
        else
                EXT="${line##*.}"
                VID_NAME="${line%.*}"
                echo "🎥 Processing video: $VID_NAME"
                
                # FlowSeek processing with state-of-the-art features
                echo "🚀 FlowSeek processing: videos/${VID_NAME}"
                python3 -m motion_vectorization.flowseek_engine \
                    --path "videos/${VID_NAME}" \
                    --max_frames $MAX_FRAMES \
                    --add_backward_flow \
                    --depth_integration \
                    --adaptive_complexity \
                    --compile_model \
                    --max_resolution 1024 \
                    --mixed_precision \
                    --device auto \
                    $ADDITIONAL_ARGS
                    
                if [ $? -eq 0 ]; then
                    echo "✅ FlowSeek processing completed for $VID_NAME"
                    echo "📊 State-of-the-art optical flow generated"
else
                    echo "❌ FlowSeek processing failed for $VID_NAME"
                    echo "💡 Check FlowSeek installation and GPU availability"
                    exit 1
                fi
                
                echo "---"
        fi
done < "$BATCH_FILE"

echo "🎉 FlowSeek batch processing completed!"
echo "📊 Processed videos from: $BATCH_FILE"
echo "🚀 State-of-the-art optical flow generated with FlowSeek (ICCV 2025)"
echo "✨ Ready for motion vectorization pipeline"
