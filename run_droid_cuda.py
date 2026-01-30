
import sys
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup Paths - Assumes DROID-SLAM and Depth-Anything-V2 are cloned in the current directory
sys.path.append('DROID-SLAM')
sys.path.append('Depth-Anything-V2')

try:
    from droid import Droid
except ImportError:
    print("Error: Could not import 'droid'. Make sure DROID-SLAM is installed/cloned.")
    print("  git clone https://github.com/princeton-vl/DROID-SLAM.git")
    print("  cd DROID-SLAM && python setup.py install")
    sys.exit(1)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Error: Could not import 'depth_anything_v2'. Make sure Depth-Anything-V2 is cloned.")
    sys.exit(1)

class DroidArgs:
    def __init__(self):
        self.imagedir = None
        self.calib = None
        self.stride = 1
        self.t0 = 0
        self.weights = 'droid.pth'
        self.buffer = 512
        self.image_size = [384, 512] 
        self.disable_vis = True 
        self.beta = 0.3
        self.filter_thresh = 2.4
        self.warmup = 8
        self.keyframe_thresh = 4.0
        self.frontend_thresh = 16.0
        self.frontend_window = 25
        self.frontend_radius = 2
        self.frontend_nms = 1
        self.backend_thresh = 22.0
        self.backend_radius = 2
        self.backend_nms = 3
        self.upsample = False
        self.stereo = False
        self.frontend_device = 'cuda'
        self.backend_device = 'cuda'

def run_cuda_pipeline(video_path, output_dir, start_min, end_min, vis=False):
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. DROID-SLAM requires CUDA.")
        # DROID will likely fail, but let's proceed in case user has some magic setup
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download Weights if missing
    if not os.path.exists('droid.pth'):
        print("Downloading DROID-SLAM weights...")
        os.system("wget https://github.com/princeton-vl/DROID-SLAM/raw/main/droid.pth")

    depth_checkpoint = 'checkpoints/depth_anything_v2_vits.pth' # Using Small for speed, can switch to vitl
    if not os.path.exists(depth_checkpoint):
        print(f"Downloading Depth-Anything weights to {depth_checkpoint}...")
        os.makedirs('checkpoints', exist_ok=True)
        os.system(f"wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -O {depth_checkpoint}")

    # 2. Initialize Models
    print("Initializing DROID-SLAM...")
    args = DroidArgs()
    droid = Droid(args)
    
    print("Initializing Depth-Anything V2...")
    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(torch.load(depth_checkpoint, map_location='cpu'))
    depth_model = depth_model.to('cuda').eval()

    # 3. Video Setup
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    
    # DROID Intrinsics (Approximate if not calibrated)
    # DROID expects [384, 512] usually, so we resize to that
    target_h, target_w = args.image_size
    fx = target_w * 0.8
    fy = target_w * 0.8 # Square pixels
    cx = target_w / 2.0
    cy = target_h / 2.0
    intrinsics = torch.tensor([fx, fy, cx, cy]).cuda()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    poses_list = []
    
    print(f"Starting execution on {video_path} ({start_frame}-{end_frame})")
    
    t = 0
    for i in tqdm(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if not ret: break
        
        # --- DROID-SLAM ---
        # Resize to DROID input size
        img_droid = cv2.resize(frame, (target_w, target_h))
        # Convert to tensor [1, 3, H, W]
        img_tensor = torch.from_numpy(img_droid).permute(2, 0, 1).float()[None] / 255.0
        img_tensor = img_tensor.cuda()
        
        droid.track(t, img_tensor, intrinsics=intrinsics)
        
        # Get current pose (approximate from frontend)
        # Note: accurate pose usually comes from backend after optimization
        # But we can grab what we have
        # Droid stores poses in droid.video.poses
        
        # --- Depth Anything (Visualization) ---
        if vis and t % 10 == 0:
            depth = depth_model.infer_image(frame) # Handles its own resizing
            
            # Colorize
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
            
            # Save or Show
            cv2.imwrite(os.path.join(output_dir, f"depth_{i:06d}.jpg"), depth_vis)
            
        t += 1

    # Terminate / Finalize
    print("Terminating and optimizing...")
    # droid.terminate() requires an image stream generator, usually
    # But essentially we just need to save the poses
    
    # Save Trajectory
    poses = droid.video.poses[:t].cpu().detach().numpy() # [N, 7] (x,y,z, qx,qy,qz,qw)
    np.save(os.path.join(output_dir, 'trajectory.npy'), poses)
    
    # Export to TUM format for easy viewing/eval
    with open(os.path.join(output_dir, 'trajectory.txt'), 'w') as f:
        for j in range(len(poses)):
            timestamp = (start_frame + j) / fps
            p = poses[j]
            # tx ty tz qx qy qz qw
            f.write(f"{timestamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n")
            
    print(f"Done. Results saved to {output_dir}")
    
    # Simple Plot
    plt.figure()
    # Poses are [x, y, z]. DROID usually is:
    # X Right, Y Down, Z Forward? Or standard vision frame.
    # We plot X vs Z (Top Down) usually.
    plt.plot(poses[:, 0], poses[:, 2], label='Trajectory') 
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.axis('equal')
    plt.legend()
    plt.title("DROID-SLAM Estimated Trajectory")
    plt.savefig(os.path.join(output_dir, 'trajectory_plot.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/GX010035.MP4')
    parser.add_argument('--output', default='results_cuda')
    parser.add_argument('--start', type=float, default=12.0)
    parser.add_argument('--end', type=float, default=13.0)
    parser.add_argument('--vis', action='store_true', help="Save depth visualizations")
    args = parser.parse_args()
    
    run_cuda_pipeline(args.video, args.output, args.start, args.end, args.vis)
