#!/usr/bin/env python3
"""
Motion Vectorization Web Interface
Provides a simple web interface to access the motion vectorization tools.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import sys
import threading
import json

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with project information and tools."""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """RAFT demo page."""
    return render_template('demo.html')

@app.route('/api/videos')
def list_videos():
    """List available videos."""
    videos_dir = 'videos'
    videos = []
    if os.path.exists(videos_dir):
        for file in os.listdir(videos_dir):
            if file.endswith(('.mp4', '.mov')):
                videos.append(file)
    return jsonify(videos)

@app.route('/api/run_raft_demo', methods=['POST'])
def run_raft_demo():
    """Run RAFT demo on demo frames."""
    try:
        # Run RAFT demo
        result = subprocess.run([
            'python', 'RAFT/demo.py', 
            '--model=RAFT/models/raft-things.pth', 
            '--path=RAFT/demo-frames'
        ], capture_output=True, text=True, cwd='.')
        
        return jsonify({
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Process a video with motion vectorization."""
    data = request.json
    video_name = data.get('video_name', 'test1')
    
    try:
        # Create video list file
        video_list = f'videos/{video_name}.txt'
        with open(video_list, 'w') as f:
            f.write(f'{video_name}.mp4\n')
        
        # Run motion vectorization script
        result = subprocess.run([
            './scripts/script.sh', video_list
        ], capture_output=True, text=True, cwd='.')
        
        return jsonify({
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)