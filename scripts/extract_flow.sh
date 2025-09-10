#!/bin/bash

# Enhanced Flow Extraction Script - Supports RAFT and FlowSeek (ICCV 2025)
# Usage: ./extract_flow.sh <batch_file> <max_frames> [flowseek|raft] [additional_options]

BATCH_FILE=$1
MAX_FRAMES=$2
FLOW_ENGINE=${3:-"flowseek"}  # Default to FlowSeek for state-of-the-art accuracy
ADDITIONAL_ARGS="${@:4}"      # Additional arguments passed to flow extraction

echo "🎬 ENHANCED FLOW EXTRACTION"
echo "================================"
echo "📂 Batch file: $BATCH_FILE"
echo "🎯 Max frames: $MAX_FRAMES"
echo "🚀 Flow engine: $FLOW_ENGINE"

if [ "$FLOW_ENGINE" = "flowseek" ]; then
    echo "✨ Using FlowSeek (ICCV 2025) - State-of-the-art Optical Flow"
    echo "   • 10-15% accuracy improvement over SEA-RAFT"
    echo "   • 8x less hardware requirements"
    echo "   • Superior cross-dataset generalization"
    echo "   • Depth-aware motion understanding"
elif [ "$FLOW_ENGINE" = "raft" ]; then
    echo "🔄 Using RAFT (Classic) - Legacy mode for compatibility"
else
    echo "⚠️  Unknown flow engine: $FLOW_ENGINE"
    echo "📖 Supported engines: flowseek, raft"
    exit 1
fi

echo "================================"

while read -r line
do
        if [[ $line =~ ^#.*  ]]; then
                continue
        else
                EXT="${line##*.}"
                VID_NAME="${line%.*}"
                echo "🎥 Processing video: $VID_NAME"
                
                if [ "$FLOW_ENGINE" = "flowseek" ]; then
                    # FlowSeek mode with enhanced options
                    echo "🚀 FlowSeek processing: videos/${VID_NAME}"
                    python3 -m RAFT.extract_flow \
                        --path "videos/${VID_NAME}" \
                        --max_frames $MAX_FRAMES \
                        --add_back \
                        --use_flowseek \
                        --flowseek_depth_integration \
                        --flowseek_adaptive_complexity \
                        --flowseek_compile_model \
                        --flowseek_max_resolution 1024 \
                        --mixed_precision \
                        $ADDITIONAL_ARGS
                        
                    if [ $? -eq 0 ]; then
                        echo "✅ FlowSeek processing completed for $VID_NAME"
                    else
                        echo "❌ FlowSeek processing failed for $VID_NAME"
                        echo "🔄 Attempting RAFT fallback..."
                        python3 -m RAFT.extract_flow \
                            --path "videos/${VID_NAME}" \
                            --model RAFT/models/raft-sintel.pth \
                            --max_frames $MAX_FRAMES \
                            --add_back \
                            --force_raft \
                            $ADDITIONAL_ARGS
                    fi
                    
                elif [ "$FLOW_ENGINE" = "raft" ]; then
                    # RAFT mode (legacy)
                    echo "🔄 RAFT processing: videos/${VID_NAME}"
                    python3 -m RAFT.extract_flow \
                        --path "videos/${VID_NAME}" \
                        --model RAFT/models/raft-sintel.pth \
                        --max_frames $MAX_FRAMES \
                        --add_back \
                        --force_raft \
                        $ADDITIONAL_ARGS
                        
                    if [ $? -eq 0 ]; then
                        echo "✅ RAFT processing completed for $VID_NAME"
                    else
                        echo "❌ RAFT processing failed for $VID_NAME"
                    fi
                fi
                
                echo "---"
        fi
done < "$BATCH_FILE"

echo "🎉 Flow extraction batch processing completed!"
echo "📊 Processed videos from: $BATCH_FILE"
echo "🚀 Engine used: $FLOW_ENGINE"
