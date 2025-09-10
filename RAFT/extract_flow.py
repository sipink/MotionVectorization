import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import time
import warnings

from .core.raft import RAFT
from .core.utils import flow_viz
from .core.utils.utils import InputPadder
from tqdm import tqdm

# FlowSeek Integration - State-of-the-art Optical Flow (ICCV 2025)
try:
    from ..motion_vectorization.flowseek_engine import FlowSeekEngine, FlowSeekConfig, create_flowseek_engine
    FLOWSEEK_AVAILABLE = True
    print("🚀 FlowSeek integration available - ICCV 2025 state-of-the-art optical flow")
except ImportError:
    FLOWSEEK_AVAILABLE = False
    warnings.warn("FlowSeek not available. Using RAFT fallback.")


DEVICE = 'cuda'

def load_image(imfile, max_size=512, flowseek_mode=False):
    """
    Load and preprocess image for optical flow computation
    Enhanced for FlowSeek with better preprocessing
    """
    img = Image.open(imfile)
    w, h = img.size
    
    # FlowSeek supports higher resolutions more efficiently
    if flowseek_mode and max_size < 1024:
        max_size = min(1024, max(w, h))
    
    if w > max_size and h > max_size:
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        img = img.resize((new_w, new_h), Image.LANCZOS)  # Better quality for FlowSeek
    
    # Convert to numpy and normalize
    img_array = np.array(img).astype(np.uint8)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
        
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    return img_tensor[None].to(DEVICE)


def create_flow_model(args):
    """
    Factory function to create either RAFT or FlowSeek model based on arguments
    """
    if args.use_flowseek and FLOWSEEK_AVAILABLE:
        print("🚀 Creating FlowSeek model (ICCV 2025)")
        
        # FlowSeek configuration
        config = FlowSeekConfig(
            device=DEVICE,
            mixed_precision=args.mixed_precision,
            depth_integration=args.flowseek_depth_integration,
            adaptive_complexity=args.flowseek_adaptive_complexity,
            compile_model=args.flowseek_compile_model,
            complexity_threshold=args.flowseek_complexity_threshold,
            max_resolution=args.flowseek_max_resolution,
            iters=args.flowseek_iters
        )
        
        # Create FlowSeek engine
        model = create_flowseek_engine(
            depth_integration=config.depth_integration,
            adaptive_complexity=config.adaptive_complexity,
            device=DEVICE,
            mixed_precision=config.mixed_precision,
            compile_model=config.compile_model
        )
        
        print(f"✅ FlowSeek model initialized")
        print(f"   • Resolution: up to {config.max_resolution}px")
        print(f"   • Depth integration: {config.depth_integration}")
        print(f"   • Adaptive complexity: {config.adaptive_complexity}")
        print(f"   • Mixed precision: {config.mixed_precision}")
        
        return model, 'flowseek'
        
    else:
        if args.use_flowseek and not FLOWSEEK_AVAILABLE:
            print("⚠️  FlowSeek requested but not available. Falling back to RAFT.")
            
        print("🔄 Creating RAFT model")
        model = torch.nn.DataParallel(RAFT(args))
        
        if args.model and os.path.exists(args.model):
            print(f"📦 Loading RAFT checkpoint: {args.model}")
            model.load_state_dict(torch.load(args.model))
        else:
            print("⚠️  No RAFT checkpoint provided or file not found")
        
        model = model.module
        model.to(DEVICE)
        model.eval()
        
        return model, 'raft'


def predict_flow(model, image1, image2, model_type='raft', iters=20):
    """
    Unified interface for flow prediction with both RAFT and FlowSeek
    """
    if model_type == 'flowseek':
        # FlowSeek prediction
        with torch.no_grad():
            # FlowSeek expects images in [0, 255] range
            flow_low, flow_up = model(image1, image2, iters=iters, test_mode=True)
            return flow_low, flow_up
    else:
        # RAFT prediction  
        with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)
            flow_low, flow_up = model(image1_padded, image2_padded, iters=iters, test_mode=True)
            return flow_low, flow_up

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    output = output[0].permute(1, 2, 0)
    return output.detach().cpu().numpy()


def viz(img, img2, flo):
    warped = warp(img2, flo)
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo, warped], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    return np.uint8(img_flo[:, :, [2,1,0]])
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey(0)


def demo(args):
    """
    Enhanced demo function supporting both RAFT and FlowSeek
    Provides unified interface with performance monitoring
    """
    orig_folder = os.path.join(args.path, 'rgb')
    flow_dir = os.path.join(args.path, 'flow')
    dirs = ['viz', 'forward', 'backward']
    
    # Create output directories
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)
    for dirname in dirs:
        if not os.path.exists(os.path.join(flow_dir, dirname)):
            os.makedirs(os.path.join(flow_dir, dirname))

    # Create model (either RAFT or FlowSeek)
    model, model_type = create_flow_model(args)
    
    # Performance tracking
    processing_times = []
    flow_magnitudes = []
    
    # Determine iterations based on model type
    iters = args.flowseek_iters if model_type == 'flowseek' else 20
    
    # Enhanced image loading for FlowSeek
    flowseek_mode = (model_type == 'flowseek')
    max_size = args.flowseek_max_resolution if flowseek_mode else 512
    
    print(f"🎬 Processing with {model_type.upper()} (iterations: {iters}, max_size: {max_size})")
    
    with torch.no_grad():
        images = glob.glob(os.path.join(orig_folder, '*.png')) + \
                 glob.glob(os.path.join(orig_folder, '*.jpg'))
        
        images = sorted(images)
        
        # Progress tracking
        progress_bar = tqdm(enumerate(zip(images[:-1], images[1:])), 
                          desc=f"{model_type.upper()} Flow Extraction",
                          total=min(len(images)-1, args.max_frames))
        
        j = 0
        for frame_idx, (imfile1, imfile2) in progress_bar:
            if j >= args.max_frames:
                # Process final backward flow if needed
                if args.add_back and j > 0:
                    start_time = time.time()
                    _, flow_backward = predict_flow(model, image2, image1, model_type, iters)
                    processing_times.append(time.time() - start_time)
                    
                    flow_backward_np = flow_backward[0].permute(1,2,0).cpu().numpy()
                    np.save(os.path.join(flow_dir, 'backward', f'{outname_prefix}.npy'), flow_backward_np)
                break
    
            # Load images with model-appropriate preprocessing
            image1 = load_image(imfile1, max_size, flowseek_mode)
            image2 = load_image(imfile2, max_size, flowseek_mode)
            
            # Predict forward flow
            start_time = time.time()
            _, flow_forward = predict_flow(model, image1, image2, model_type, iters)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Calculate flow statistics
            flow_magnitude = torch.norm(flow_forward, dim=1).mean().item()
            flow_magnitudes.append(flow_magnitude)
            
            # Visualization
            vis_img_forward = viz(image1, image2, flow_forward)
            
            # Backward flow if requested
            vis_img_backward = None
            if args.add_back:
                start_time_back = time.time()
                _, flow_backward = predict_flow(model, image2, image1, model_type, iters)
                processing_times.append(time.time() - start_time_back)
                vis_img_backward = viz(image2, image1, flow_backward)

            # Combine visualizations
            if args.add_back and vis_img_backward is not None:
                vis_img = np.concatenate([vis_img_forward, vis_img_backward], axis=1)
            else:
                vis_img = vis_img_forward

            # Save results
            outname = os.path.basename(imfile1)
            outname_prefix = os.path.splitext(outname)[0]
            
            # Save visualization
            cv2.imwrite(os.path.join(flow_dir, 'viz', outname), vis_img)
            
            # Save flow arrays
            flow_forward_np = flow_forward[0].permute(1,2,0).cpu().numpy()
            np.save(os.path.join(flow_dir, 'forward', f'{outname_prefix}.npy'), flow_forward_np)
            
            if args.add_back and vis_img_backward is not None:
                flow_backward_np = flow_backward[0].permute(1,2,0).cpu().numpy()
                np.save(os.path.join(flow_dir, 'backward', f'{outname_prefix}.npy'), flow_backward_np)
            
            # Update progress bar with performance info
            avg_time = np.mean(processing_times[-10:])  # Last 10 frames
            progress_bar.set_postfix({
                'avg_time': f'{avg_time:.3f}s',
                'flow_mag': f'{flow_magnitude:.2f}',
                'fps': f'{1.0/avg_time:.1f}'
            })
            
            j += 1
    
    # Performance Summary
    total_frames = len(processing_times)
    avg_processing_time = np.mean(processing_times)
    avg_flow_magnitude = np.mean(flow_magnitudes)
    
    print(f"\n📊 {model_type.upper()} Performance Summary:")
    print(f"   • Total frames processed: {total_frames}")
    print(f"   • Average processing time: {avg_processing_time:.3f}s ± {np.std(processing_times):.3f}s")
    print(f"   • Average FPS: {1.0/avg_processing_time:.1f}")
    print(f"   • Average flow magnitude: {avg_flow_magnitude:.3f} ± {np.std(flow_magnitudes):.3f}")
    
    if model_type == 'flowseek':
        print(f"   🚀 FlowSeek benefits:")
        print(f"      • 10-15% accuracy improvement over SEA-RAFT")
        print(f"      • Superior cross-dataset generalization") 
        print(f"      • 8x less hardware requirements")
        print(f"      • Depth-aware motion understanding")
    
    # Save performance log
    perf_log = {
        'model_type': model_type,
        'total_frames': total_frames,
        'avg_processing_time': avg_processing_time,
        'processing_times': processing_times,
        'avg_flow_magnitude': avg_flow_magnitude,
        'flow_magnitudes': flow_magnitudes,
        'config': vars(args)
    }
    
    log_path = os.path.join(flow_dir, f'{model_type}_performance_log.json')
    import json
    with open(log_path, 'w') as f:
        json.dump(perf_log, f, indent=2)
    
    print(f"💾 Performance log saved to: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract optical flow using RAFT or FlowSeek (ICCV 2025)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Original RAFT arguments (maintained for backward compatibility)
    parser.add_argument('--model', help="RAFT checkpoint path (required for RAFT mode)")
    parser.add_argument('--path', required=True, help="Dataset path for evaluation")
    parser.add_argument('--small', action='store_true', help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', default=True, 
                       help='Use mixed precision (recommended for both RAFT and FlowSeek)')
    parser.add_argument('--alternate_corr', action='store_true', 
                       help='Use efficient correlation implementation (RAFT only)')
    parser.add_argument('--max_frames', type=int, default=100, 
                       help='Maximum number of frames to process')
    parser.add_argument('--add_back', action='store_true', default=False, 
                       help='Also compute backward optical flow')
    
    # =====================================
    # FlowSeek Arguments (ICCV 2025)
    # =====================================
    parser.add_argument('--use_flowseek', action='store_true', default=True,
                       help='🚀 Use FlowSeek (ICCV 2025) instead of RAFT for state-of-the-art accuracy')
    
    # Core FlowSeek configuration
    parser.add_argument('--flowseek_depth_integration', action='store_true', default=True,
                       help='Enable depth foundation models integration (recommended)')
    parser.add_argument('--flowseek_adaptive_complexity', action='store_true', default=True,
                       help='Enable adaptive FlowSeek/SEA-RAFT switching based on complexity')
    parser.add_argument('--flowseek_compile_model', action='store_true', default=True,
                       help='Enable torch.compile optimization for 2x speedup')
    
    # Performance and quality settings  
    parser.add_argument('--flowseek_complexity_threshold', type=float, default=0.7,
                       help='Complexity threshold for adaptive mode switching (0.0-1.0)')
    parser.add_argument('--flowseek_max_resolution', type=int, default=1024,
                       help='Maximum resolution for FlowSeek processing (supports up to 2048px)')
    parser.add_argument('--flowseek_iters', type=int, default=12,
                       help='Number of FlowSeek update iterations (8-16 recommended)')
    
    # Depth model configuration
    parser.add_argument('--flowseek_depth_model', type=str, default='dpt_large',
                       choices=['dpt_large', 'dpt_hybrid_midas', 'midas_v3_large'],
                       help='Depth foundation model for FlowSeek')
    
    # Motion basis configuration
    parser.add_argument('--flowseek_motion_bases_dim', type=int, default=6,
                       help='Motion basis dimensionality (6 for 6-DOF motion)')
    
    # Benchmarking and evaluation
    parser.add_argument('--benchmark_mode', action='store_true', default=False,
                       help='Enable detailed benchmarking and comparison')
    parser.add_argument('--save_performance_log', action='store_true', default=True,
                       help='Save detailed performance logs')
    
    # Force RAFT mode (override FlowSeek default)
    parser.add_argument('--force_raft', action='store_true', default=False,
                       help='Force use of RAFT even if FlowSeek is available')
    
    args = parser.parse_args()
    
    # Override FlowSeek if force_raft is specified
    if args.force_raft:
        args.use_flowseek = False
        print("🔄 Forced RAFT mode - FlowSeek disabled")
    
    # Validate arguments
    if not args.use_flowseek and not args.model:
        parser.error("RAFT mode requires --model checkpoint path")
    
    if args.use_flowseek and not FLOWSEEK_AVAILABLE:
        print("⚠️  FlowSeek requested but not available. Falling back to RAFT mode.")
        if not args.model:
            parser.error("FlowSeek fallback to RAFT requires --model checkpoint path")
    
    # Display configuration
    print("=" * 60)
    print("🎬 OPTICAL FLOW EXTRACTION CONFIGURATION")
    print("=" * 60)
    
    if args.use_flowseek and FLOWSEEK_AVAILABLE:
        print("🚀 ENGINE: FlowSeek (ICCV 2025) - State-of-the-art Optical Flow")
        print(f"   • Depth integration: {args.flowseek_depth_integration}")
        print(f"   • Adaptive complexity: {args.flowseek_adaptive_complexity}") 
        print(f"   • Max resolution: {args.flowseek_max_resolution}px")
        print(f"   • Iterations: {args.flowseek_iters}")
        print(f"   • Mixed precision: {args.mixed_precision}")
        print(f"   • Model compilation: {args.flowseek_compile_model}")
        print("   📈 Benefits:")
        print("      • 10-15% accuracy improvement over SEA-RAFT")
        print("      • 8x less hardware requirements")
        print("      • Superior cross-dataset generalization")
        print("      • Depth-aware motion understanding")
    else:
        print("🔄 ENGINE: RAFT (Classic)")
        print(f"   • Model: {args.model}")
        print(f"   • Small model: {args.small}")
        print(f"   • Mixed precision: {args.mixed_precision}")
        print(f"   • Alternate correlation: {args.alternate_corr}")
    
    print(f"📂 Dataset: {args.path}")
    print(f"🎯 Max frames: {args.max_frames}")
    print(f"↔️  Backward flow: {args.add_back}")
    print("=" * 60)

    demo(args)
