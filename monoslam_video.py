import numpy as np
import cv2
import sys
import os
import argparse
from tqdm import tqdm

# Import core SLAM classes from the original script
# Ensure monoslam.py is in the same directory or python path
try:
    from monoslam import MultiClassSegmentation, RoadAreaDetector, PinholeCamera, VisualOdometry
except ImportError as e:
    print(f"Error importing monoslam: {e}")
    sys.exit(1)

class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")
        
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        
        print(f"Video Loaded: {video_path}")
        print(f"Resolution: {self.width}x{self.height}, Frames: {self.num_frames}, FPS: {self.fps}")

    def __len__(self):
        return self.num_frames
    
    def get_image(self, index):
        # Optimized for sequential access
        # If the requested index is the next frame, just read
        if index == self.current_frame_idx:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_idx += 1
                return frame
            else:
                return None
        
        # If requested index is exactly what we just read (re-read case?), or backwards, or skipped ahead
        # We need to seek.
        # Note: set(CV_CAP_PROP_POS_FRAMES, index) is 0-indexed
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = index + 1
            return frame
        else:
            return None

    def release(self):
        self.cap.release()

def parse_hhmmss(time_str):
    """Converts hhmmss or hh:mm:ss string to total seconds."""
    if not time_str:
        return None
    time_str = time_str.replace(':', '')
    if len(time_str) != 6:
        raise ValueError("Time must be in hhmmss format (e.g., 000130 for 1 min 30 sec)")
    
    h = int(time_str[0:2])
    m = int(time_str[2:4])
    s = int(time_str[4:6])
    return h * 3600 + m * 60 + s

def run_video(video_path, seg_model=None, road_detector=None, traj_img_size=800, downscale=0.5, fov=None, start_time=None, end_time=None, stop_threshold=1.0, view_mode="global", max_turn_degrees=10.0, mask_path=None):
    sequence_name = video_path.split("/")[-1]
    print(f"--- Running on Video: {sequence_name} (Scale: {downscale}) ---")
    
    try:
        loader = VideoLoader(video_path)
    except IOError as e:
        print(e)
        return

    # Calculate start and end frames
    start_frame = 0
    end_frame = len(loader)
    
    if start_time:
        try:
            seconds = parse_hhmmss(start_time)
            start_frame = int(seconds * loader.fps)
            print(f"Start Time: {start_time} -> Frame {start_frame}")
        except ValueError as e:
            print(f"Error parsing start time: {e}")
            return

    if end_time:
        try:
            seconds = parse_hhmmss(end_time)
            end_frame = int(seconds * loader.fps)
            print(f"End Time: {end_time} -> Frame {end_frame}")
        except ValueError as e:
            print(f"Error parsing end time: {e}")
            return
            
    if start_frame >= len(loader):
        print(f"Error: Start frame {start_frame} is beyond video length {len(loader)}")
        return
        
    # Seek to start frame
    loader.current_frame_idx = start_frame
    loader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    first_img = loader.get_image(start_frame)
    if first_img is None: 
        print("Could not read first frame (at start time)")
        return
    
    # Reset loader to start_frame logic (since get_image advances it)
    # We want the loop to start processing FROM start_frame
    loader.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    loader.current_frame_idx = start_frame
    
    h_orig, w_orig = first_img.shape[:2]
    
    # Calculate Intrinsics
    if fov:
        # fx = (w/2) / tan(fov/2)
        fov_rad = np.radians(float(fov))
        fx_orig = (w_orig / 2.0) / np.tan(fov_rad / 2.0)
        fy_orig = fx_orig # Square pixels assumption
        print(f"Using provided FOV {fov} deg -> fx={fx_orig:.2f}")
    else:
        # Default Logic
        if w_orig > 1000: # Assuming HD/FullHD
            fx_orig = w_orig * 0.8 
            fy_orig = w_orig * 0.8 
            print(f"Using default assumption (HD): fx ~ 0.8*w = {fx_orig:.2f} (~64 deg FOV)")
        else:
            # Fallback to KITTI defaults if low res 
            fx_orig = 718.856
            fy_orig = 718.856
            print(f"Using KITTI default: fx={fx_orig:.2f}")

    cx_orig = w_orig / 2.0
    cy_orig = h_orig / 2.0
    
    fx = fx_orig * downscale
    fy = fy_orig * downscale
    cx = cx_orig * downscale
    cy = cy_orig * downscale
    w = int(w_orig * downscale)
    h = int(h_orig * downscale)
    
    cam = PinholeCamera(w, h, fx, fy, cx, cy)
    vo = VisualOdometry(cam, stop_threshold=stop_threshold, max_turn_degrees=max_turn_degrees)
    if seg_model:
        vo.set_segmentation_model(seg_model)
    
    # Load static mask (e.g., for car body exclusion)
    static_mask = None
    if mask_path:
        if os.path.exists(mask_path):
            static_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if static_mask_full is not None:
                # Resize mask to match processing resolution
                static_mask = cv2.resize(static_mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
                # Normalize: 255 = valid, 0 = masked
                static_mask = (static_mask > 127).astype(np.uint8)
                print(f"Loaded mask from: {mask_path}")
                print(f"Masked pixels: {np.sum(static_mask == 0)} / {w * h}")
            else:
                print(f"Warning: Could not read mask file: {mask_path}")
        else:
            print(f"Warning: Mask file not found: {mask_path}")
    
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    
    cv2.namedWindow(f'MonoSLAM - {sequence_name}', cv2.WINDOW_NORMAL)
    
    # Loop from start_frame to end_frame
    # We use range(end_frame - start_frame) and use i relative for display, 
    # but loader must simply respond to calls.
    # Actually, simpler to just loop `for i in range(start_frame, end_frame):`
    # and passed `i` to process_frame (it uses it for ID mainly).
    
    loop_range = range(start_frame, end_frame)
    
    last_heading = 0.0
    last_flow_mag = 0.0
    
    for i in tqdm(loop_range, desc="Processing Frames"):
        img_full = loader.get_image(i)
        if img_full is None: break
        
        img = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
        
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        road_vp = None
        if road_detector:
            display_img = road_detector.detect_and_draw(img)
            road_vp = road_detector.get_vanishing_point()
        else:
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()

        vo.process_frame(i, img_gray, img_color=img, road_vp=road_vp, static_mask=static_mask)
        
        # Visualize static mask if present
        if static_mask is not None:
            # Draw masked region with semi-transparent red
            mask_overlay = display_img.copy()
            mask_overlay[static_mask == 0] = (0, 0, 180)  # Red tint on masked areas
            cv2.addWeighted(mask_overlay, 0.3, display_img, 0.7, 0, display_img)
        
        estimated_heading = vo.get_heading()
        turn_rate = estimated_heading - last_heading
        # Handle wrap-around if needed (though headings typically don't jump 360 in one frame)
        last_heading = estimated_heading
        
        cv2.putText(display_img, f"Est Heading: {estimated_heading:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # --- Turn Indicator (Visual Validation) ---
        # Draw a bar at the bottom center
        bar_center_x = w // 2
        bar_y = h - 20
        bar_width = 200
        # Sensitivity: full bar width = 5 degrees per frame?
        # turn_rate is deg/frame.
        
        offset = int(turn_rate * 20) # 1 deg turn -> 20 pixels
        cv2.rectangle(display_img, (bar_center_x - bar_width//2, bar_y - 5), (bar_center_x + bar_width//2, bar_y + 5), (50, 50, 50), -1)
        # Center marker
        cv2.line(display_img, (bar_center_x, bar_y - 5), (bar_center_x, bar_y + 5), (255, 255, 255), 1)
        # Current Turn Bar
        if abs(offset) > 0:
            color = (0, 255, 0) # Green for turn
            if abs(turn_rate) > 2.0: color = (0, 0, 255) # Red for high rate
            # Clip
            offset = np.clip(offset, -bar_width//2, bar_width//2)
            cv2.rectangle(display_img, (bar_center_x, bar_y - 3), (bar_center_x + offset, bar_y + 3), color, -1)
            
        cv2.putText(display_img, f"Turn: {turn_rate:.2f}", (bar_center_x - 30, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if vo.is_stopped:
            cv2.putText(display_img, "STOPPED", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Acceleration Indicator (Vertical Bar) ---
        # Accel proxy: Change in optical flow magnitude
        curr_flow_mag = vo.current_flow_mag
        accel = curr_flow_mag - last_flow_mag
        last_flow_mag = curr_flow_mag
        
        # Draw vertical bar on right side
        vb_center_x = w - 30
        vb_center_y = h // 2
        vb_height = 200 # Total height (100 up, 100 down)
        vb_width = 15
        
        # Background
        cv2.rectangle(display_img, (vb_center_x - vb_width//2, vb_center_y - vb_height//2), 
                                   (vb_center_x + vb_width//2, vb_center_y + vb_height//2), (50, 50, 50), -1)
        # Center Line
        cv2.line(display_img, (vb_center_x - vb_width//2, vb_center_y), (vb_center_x + vb_width//2, vb_center_y), (255, 255, 255), 1)
        
        # Accel Bar
        # Sensitivity: 1.0 pixel change is huge. 0.1 is significant.
        # Scale: 0.1 delta -> 20 pixels?
        accel_scale = 100.0 # 1.0 delta -> 100 pixels
        val_px = int(accel * accel_scale)
        val_px = np.clip(val_px, -vb_height//2, vb_height//2)
        
        if val_px != 0:
            color_acc = (0, 255, 0) if val_px > 0 else (0, 0, 255) # Green Up, Red Down
            # If val_px > 0: draw from center_y - val_px to center_y
            # If val_px < 0: draw from center_y to center_y - val_px (double negative -> positive offset)
            # rectangle takes (pt1, pt2).
            # Y increases downwards.
            # Up means Y decreases.
            
            top_y = vb_center_y - val_px
            # If val_px > 0 (Accel), top_y < center_y. Rect: (top_y, center_y)
            # If val_px < 0 (Decel), top_y > center_y. Rect: (center_y, top_y)
            
            cv2.rectangle(display_img, (vb_center_x - vb_width//2, vb_center_y),
                                       (vb_center_x + vb_width//2, top_y), color_acc, -1)
            
        cv2.putText(display_img, "Accel", (vb_center_x - 20, vb_center_y - vb_height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if vo.kp1 is not None:
            for pt in vo.kp1:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        
        # --- Trajectory Drawing ---
        traj_points = np.array(vo.get_trajectory())
        
        # Create canvas
        traj_img.fill(0)
        draw_scale = 1.0 # Adjust as needed
        cx, cy = traj_img_size // 2, traj_img_size // 2
        
        curr_x = vo.cur_t[0][0]
        curr_z = vo.cur_t[2][0]
        curr_heading_rad = np.radians(estimated_heading)
        
        if len(traj_points) > 1:
            if view_mode == "ego":
                # --- Egocentric (Track Up, Vehicle Fixed) ---
                # 1. Translate to make current pos (0,0)
                rel_points = traj_points - np.array([curr_x, curr_z])
                
                # 2. Rotate so current heading aligns with Up (Z-axis)
                c, s = np.cos(-curr_heading_rad), np.sin(-curr_heading_rad)
                R = np.array(((c, -s), (s, c)))
                
                # rot_points = rel_points @ R.T
                rot_points = rel_points @ R.T
                
                # 3. Map to Screen
                screen_points = np.zeros_like(rot_points, dtype=np.int32)
                screen_points[:, 0] = (cx + rot_points[:, 0] * draw_scale).astype(np.int32)
                screen_points[:, 1] = (cy - rot_points[:, 1] * draw_scale).astype(np.int32) # Z is Up on screen (negative Y pixel)
                
                cv2.polylines(traj_img, [screen_points], isClosed=False, color=(255, 0, 0), thickness=1)
                
                # Draw Fixed Car Arrow at Center (Points UP)
                arrow_len = 30
                tip_u = cx
                tip_v = cy - arrow_len
                cv2.arrowedLine(traj_img, (cx, cy), (tip_u, tip_v), (0, 255, 255), 3, tipLength=0.3)
                
            else: 
                # --- Global (North Up) ---
                # Draw Absolute t values.
                # Center the map at the current vehicle position to keep it in view?
                # Or keep it fixed at 0,0?
                # User wants "Total Route". Fixed at 0,0 helps see shape, but car might drive off.
                # Let's Center on Vehicle but NOT rotate.
                
                # 1. Translate to make current pos (0,0) (So vehicle is always at center of screen)
                rel_points = traj_points - np.array([curr_x, curr_z])
                
                # 2. No Rotation.
                
                # 3. Map to Screen
                screen_points = np.zeros_like(rel_points, dtype=np.int32)
                screen_points[:, 0] = (cx + rel_points[:, 0] * draw_scale).astype(np.int32)
                screen_points[:, 1] = (cy - rel_points[:, 1] * draw_scale).astype(np.int32)
                
                cv2.polylines(traj_img, [screen_points], isClosed=False, color=(255, 0, 0), thickness=1)
                
                # Draw Car Arrow at Center (Rotates based on heading)
                # Heading 0 = Up (Negative Y). 
                # x = sin(h), y = -cos(h)
                arrow_len = 30
                tip_u = int(cx + arrow_len * np.sin(curr_heading_rad))
                tip_v = int(cy - arrow_len * np.cos(curr_heading_rad))
                cv2.arrowedLine(traj_img, (cx, cy), (tip_u, tip_v), (0, 255, 255), 3, tipLength=0.3)
                
        # -------------------------------------

        scale_traj = display_img.shape[0] / traj_img.shape[0]
        traj_img_resized = cv2.resize(traj_img, (int(traj_img.shape[1] * scale_traj), display_img.shape[0]))
        final_vis = np.hstack((display_img, traj_img_resized))
        
        cv2.imshow(f'MonoSLAM - {sequence_name}', final_vis)
        
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
            
    print(f"Finished {sequence_name}")
    loader.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run MonoSLAM on a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input mp4 video file.")
    parser.add_argument("--no_road_detection", action="store_true", help="Disable road area detection.")
    parser.add_argument("--no_segmentation", action="store_true", help="Disable semantic segmentation (faster).")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor for processing (default 0.5).")
    parser.add_argument("--fov", type=float, help="Horizontal Field of View (FOV) in degrees (e.g., 90, 120).")
    parser.add_argument("--start", type=str, help="Start timestamp in hhmmss format (e.g., 000130 for 00:01:30)")
    parser.add_argument("--end", type=str, help="End timestamp in hhmmss format.")
    parser.add_argument("--stop_threshold", type=float, default=1.0, help="Movement threshold in pixels (default 1.0). Increase if false stopped detection.")
    parser.add_argument("--view_mode", type=str, default="global", choices=["global", "ego"], help="Map view mode: 'global' (North Up, default) or 'ego' (Vehicle Up).")
    parser.add_argument("--max_turn", type=float, default=10.0, help="Maximum allowed turn rate (deg/frame) to reject outliers.")
    parser.add_argument("--mask", type=str, default=None, help="Path to mask image (PNG). Use mask_editor.py to create one.")

    args = parser.parse_args()

    seg_model = None
    if not args.no_segmentation:
        print("Initializing Segmentation Model (might take a moment)...")
        try:
            seg_model = MultiClassSegmentation()
        except Exception as e:
            print(f"Failed to load DL model: {e}")
            seg_model = None
            
    road_detector = None
    if not args.no_road_detection:
        road_detector = RoadAreaDetector()
    
    run_video(args.video_path, seg_model, road_detector, downscale=args.scale, fov=args.fov, start_time=args.start, end_time=args.end, stop_threshold=args.stop_threshold, view_mode=args.view_mode, max_turn_degrees=args.max_turn, mask_path=args.mask)

if __name__ == "__main__":
    main()
