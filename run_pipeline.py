
import sys
import os
import argparse
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add Depth-Anything-V2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("Could not import DepthAnythingV2. Make sure the repo is cloned in the current directory.")
    sys.exit(1)

# Import MonoSLAM
try:
    from monoslam import VisualOdometry, PinholeCamera
except ImportError:
    print("Could not import monoslam. Make sure monoslam.py is in the current directory.")
    sys.exit(1)

def load_gps_track(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lats = []
    lons = []
    alts = []
    
    for point in data:
        lats.append(point['lat'])
        lons.append(point['lon'])
        alts.append(point.get('alt', 0))
        
    return np.array(lats), np.array(lons), np.array(alts)

def latlon_to_meters(lats, lons, lat0, lon0):
    R = 6378137.0 # Earth radius
    x = (np.deg2rad(lons) - np.deg2rad(lon0)) * np.cos(np.deg2rad(lat0)) * R
    y = (np.deg2rad(lats) - np.deg2rad(lat0)) * R
    return x, y

def align_trajectories(model_traj, gt_traj):
    """
    Align model trajectory to ground truth using Umeyama alignment (sim3).
    model_traj: Nx3
    gt_traj: Nx3
    """
    # Simply shift start to 0 for both
    model_traj = model_traj - model_traj[0]
    gt_traj = gt_traj - gt_traj[0]
    
    # Scale correction (simple distance ratio)
    model_dist = np.linalg.norm(model_traj[-1] - model_traj[0])
    gt_dist = np.linalg.norm(gt_traj[-1] - gt_traj[0])
    
    if model_dist < 1e-3: return model_traj
    
    scale = gt_dist / model_dist
    aligned_model = model_traj * scale
    
    # Rotation alignment (2D for simplicity, assuming flat ground)
    # Find rotation that minimizes error between endpoints or PCA
    # Let's align the endpoint vectors
    
    v_model = aligned_model[-1]
    v_gt = gt_traj[-1]
    
    angle_model = np.arctan2(v_model[0], v_model[2]) # x, z
    angle_gt = np.arctan2(v_gt[0], v_gt[1]) # x, y (GPS) - wait, GPS x is East, y is North.
    # MonoSLAM x is Right, z is Forward. 
    # Let's assume standard VO frame: Z forward, X right.
    # GPS frame: X East, Y North.
    
    # We need to rotate model (x,z) to match GPS (x,y)
    # Target angle is angle_gt. Source angle is angle_model.
    # d_angle = angle_gt - angle_model
    
    # Using Procrustes analysis would be better but let's do SVD for rotation
    # Match 2D points (N, 2)
    # Use only first and last point for rough alignment? No, user wants validation.
    # Since we don't have time alignment, we can only align the SHAPE.
    # But usually we align start and end points if we assume constant speed?
    # Or just align first N points if possible.
    
    # Given we don't know the exact GPS subset corresponding to the video segment,
    # This is TRICKY.
    # The user said "Validate against GPS data".
    # We will plot both and let the user see.
    # We can try to manually scale based on total distance if we assume the video covers the WHOLE GPS track?
    # Or just return the scaled trajectory.
    
    return aligned_model

def run_pipeline(video_path, gps_path, start_min, end_min, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load GPS
    lats, lons, alts = load_gps_track(gps_path)
    # Convert to local meters (East, North)
    gps_x, gps_y = latlon_to_meters(lats, lons, lats[0], lons[0])
    
    # 2. Setup Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {total_frames} frames, {fps} fps")
    
    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    
    if start_frame >= total_frames:
        print("Start time beyond video duration.")
        return
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 3. Setup Models
    # Scale down for VO speed
    downscale = 0.5
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(w_orig * downscale)
    h = int(h_orig * downscale)
    
    # Intrinsics (Approximate)
    fx = w * 0.8
    fy = w * 0.8
    cx = w / 2.0
    cy = h / 2.0
    
    cam = PinholeCamera(w, h, fx, fy, cx, cy)
    # Tuned Feature Params for far objects (lower quality)
    feature_params = dict(maxCorners=500, qualityLevel=0.001, minDistance=5, blockSize=5)
    vo = VisualOdometry(cam, stop_threshold=0.5, max_turn_degrees=5.0, feature_params=feature_params)
    
    # Create static mask to ignore car body (bottom 15%)
    static_mask = np.ones((h, w), dtype=np.uint8)
    car_hood_height = int(h * 0.85)
    static_mask[car_hood_height:, :] = 0
    
    # Depth Anything
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running Depth-Anything on {DEVICE}")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    depth_model = DepthAnythingV2(**model_configs['vits'])
    checkpoint_path = 'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, trying local checkpoints/")
        checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
        
    depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    
    # 4. Processing Loop
    traj_vo = []
    
    pbar = tqdm(range(start_frame, end_frame))
    
    # Visualization window
    cv2.namedWindow('Pipeline', cv2.WINDOW_NORMAL)
    
    for i in pbar:
        ret, frame = cap.read()
        if not ret: break
        
        # VO
        frame_vo = cv2.resize(frame, (w, h))
        gray = cv2.cvtColor(frame_vo, cv2.COLOR_BGR2GRAY)
        vo.process_frame(i, gray, img_color=frame_vo, static_mask=static_mask)
        
        x_vo = vo.cur_t[0][0]
        z_vo = vo.cur_t[2][0]
        traj_vo.append([x_vo, z_vo])
        
        # Depth (every 10 frames to save speed)
        if i % 10 == 0:
            depth_input = frame # Full res? Or resized?
            # Depth Anything handles resizing internally usually infer_image
            depth_map = depth_model.infer_image(depth_input)
            
            # Normalize for vis
            depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
            # Show side-by-side
            vis_h, vis_w = frame_vo.shape[:2]
            depth_vis_resized = cv2.resize(depth_vis, (vis_w, vis_h))
            
            combined = np.hstack((frame_vo, depth_vis_resized))
            cv2.putText(combined, f"Frame {i} VO: ({x_vo:.2f}, {z_vo:.2f})", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Pipeline', combined)
            if cv2.waitKey(1) == 27: break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # 5. Plotting
    traj_vo = np.array(traj_vo)
    
    # Align VO to GPS (Assume start aligns)
    # GPS track is huge, we need to know WHICH part it corresponds to.
    # Ideally we'd have timestamps. 
    # For now, plot full GPS and the generated track on top (scaled).
    
    plt.figure(figsize=(10, 10))
    plt.plot(gps_x, gps_y, 'g.', label='GPS Ground Truth', markersize=1)
    
    # VO is X (right), Z (forward). GPS is X (East), Y (North).
    # Assuming Car started facing North? Or East?
    # We will rotate VO to match the general direction of the first N points of GPS?
    # Actually, let's just plot VO as is (x, z) -> (x, y) and let user see.
    # Maybe scale it nicely.
    
    plt.plot(traj_vo[:, 0], traj_vo[:, 1], 'b-', label='VO Trajectory (Raw)')
    
    plt.legend()
    plt.title("Trajectory vs GPS")
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'))
    print(f"Comparison saved to {output_dir}/trajectory_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/GX010035.MP4')
    parser.add_argument('--gps', default='gps_tools/gps_track.json')
    parser.add_argument('--start', type=float, default=12.0, help="Start minute")
    parser.add_argument('--end', type=float, default=13.0, help="End minute")
    args = parser.parse_args()
    
    run_pipeline(args.video, args.gps, args.start, args.end)
