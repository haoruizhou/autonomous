import numpy as np
import cv2
import sys
import argparse
import os
from tqdm import tqdm
import torch
from torchvision import models, transforms

# ==========================================
# PART 1: Core SLAM Classes (from monoslam.py)
# ==========================================

class MultiClassSegmentation:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Check for MPS (Mac) just in case running locally, though Colab is usually CUDA/CPU
        if torch.backends.mps.is_available(): 
             self.device = torch.device('mps')
        
        print(f"Loading DeepLabV3 on {self.device}...")
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.dynamic_classes = [2, 6, 7, 14, 15, 19] 
        
    def get_mask(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        mask_resized = cv2.resize(output_predictions, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        binary_mask = np.ones_like(mask_resized, dtype=np.uint8)
        for cls_id in self.dynamic_classes:
            binary_mask[mask_resized == cls_id] = 0
        return binary_mask

class RoadAreaDetector:
    def __init__(self):
        self.vanishing_point = None
        
    def detect_and_draw(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]
        seed_point = (width // 2, height - 50)
        
        mask_shape = (height + 2, width + 2)
        mask = np.zeros(mask_shape, np.uint8)
        
        loDiff = (15, 15, 15) 
        upDiff = (15, 15, 15)
        
        flooded = img.copy()
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(flooded, mask, seed_point, (0, 255, 0), loDiff, upDiff, flags)
        
        road_mask = mask[1:-1, 1:-1]
        kernel = np.ones((7,7), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # Estimate Vanishing Point (Centroid of top part of the mask)
        # We look at the top 60% of the image to find the "far" road
        top_h = int(height * 0.6)
        road_mask_top = road_mask[:top_h, :]
        
        M = cv2.moments(road_mask_top)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.vanishing_point = (cX, cY) # Note: cY is relative to top 0
        else:
            self.vanishing_point = (width // 2, height // 3) # Default logic

        overlay = img.copy()
        overlay[road_mask > 0] = (0, 255, 0)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        
        # Draw VP
        if self.vanishing_point:
             cv2.circle(img, self.vanishing_point, 5, (255, 0, 0), -1) # Blue dot
        
        return img
    
    def get_vanishing_point(self):
        return self.vanishing_point

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.pp = (cx, cy)
        self.K_inv = np.linalg.inv(self.K)

class VisualOdometry:
    def __init__(self, cam, stop_threshold=1.0, max_turn_degrees=10.0):
        self.cam = cam
        self.stop_threshold = stop_threshold
        self.max_turn_degrees = max_turn_degrees
        self.curr_frame = None
        self.last_frame = None
        self.kp1 = None
        self.des1 = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.traj = [] 
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.lk_params = dict(winSize=(15, 15), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.seg_model = None
        self.is_stopped = False
        self.current_flow_mag = 0.0

    def set_segmentation_model(self, model):
        self.seg_model = model

    def process_frame(self, frame_id, new_frame, img_color=None, road_vp=None):
        if self.last_frame is None:
            self.last_frame = new_frame
            self.kp1 = self.detector.detect(new_frame, None)
            self.kp1 = np.array([x.pt for x in self.kp1], dtype=np.float32)
            self.current_flow_mag = 0.0
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, new_frame, self.kp1, None, **self.lk_params)
        
        st_flat = st.flatten()
        good_old = self.kp1[st_flat == 1]
        good_new = p1[st_flat == 1]
        
        if self.seg_model is not None and img_color is not None:
            mask = self.seg_model.get_mask(img_color)
            valid_indices = []
            for idx, pt in enumerate(good_new):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] == 1: 
                        valid_indices.append(idx)
                        
            good_old = good_old[valid_indices]
            good_new = good_new[valid_indices]

        if len(good_new) < 50:
             kp = self.detector.detect(new_frame, None)
             self.kp1 = np.array([x.pt for x in kp], dtype=np.float32)
             self.last_frame = new_frame
             self.current_flow_mag = 0.0
             return

        E, mask = cv2.findEssentialMat(good_new, good_old, self.cam.fx, self.cam.pp, cv2.RANSAC, 0.999, 1.0)
        
        if E is not None:
             _, R, t, mask = cv2.recoverPose(E, good_new, good_old, self.cam.K)
             
             flow_mag = np.mean(np.linalg.norm(good_new - good_old, axis=1))
             self.current_flow_mag = flow_mag
             
             # Calculate rotation angle strength
             trace = np.trace(R)
             # trace = 1 + 2cos(theta) -> cos(theta) = (trace - 1)/2
             cos_theta = (trace - 1.0) / 2.0
             cos_theta = np.clip(cos_theta, -1.0, 1.0)
             angle_deg = np.degrees(np.arccos(cos_theta))
             
             # OUTLIER REJECTION: Sudden "Teleport" Turns
             if abs(angle_deg) > self.max_turn_degrees:
                 # Reject this update
                 t = np.zeros((3, 1))
                 R = np.eye(3)
                 
             else:
                 # STRICTER STOP DETECTION
                 if flow_mag < self.stop_threshold:  # Use configurable threshold
                     t = np.zeros((3, 1))
                     R = np.eye(3)
                     self.is_stopped = True
                 else:
                     self.is_stopped = False
                 
                 # --- ROAD FUSION ---
                 if road_vp is not None:
                     # road_vp is (u, v)
                     # Convert to normalized vector
                     # t_vp = inv(K) * [u, v, 1]
                     vp_homog = np.array([road_vp[0], road_vp[1], 1.0])
                     t_vp = self.cam.K_inv.dot(vp_homog)
                     
                     # Normalize
                     t_vp = t_vp / np.linalg.norm(t_vp)
                     t_vp = t_vp.reshape((3, 1))
                     
                     # Blend t from VO and t from Road
                     # VO t is unit vector. t_vp is unit vector.
                     # t_vp direction is "Towards the road center".
                     # If we are driving straight, t ~ t_vp.
                     # Fusion Factor:
                     alpha = 0.2 # 20% weight to road direction (gentle bias)
                     
                     # Only apply if we are moving roughly fwd (dot product > 0.7)
                     if np.dot(t.flatten(), t_vp.flatten()) > 0.7:
                         t_fused = (1 - alpha) * t + alpha * t_vp
                         t_fused = t_fused / np.linalg.norm(t_fused) # Renormalize
                         t = t_fused
                 # -------------------
              
             absolute_scale = 1.0 
             if np.sum(t) != 0 and (t[2] > t[0] and t[2] > t[1]): 
                 self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                 self.cur_R = self.cur_R.dot(R)
        
        self.traj.append((self.cur_t[0][0], self.cur_t[2][0]))
        self.last_frame = new_frame
        self.kp1 = good_new

    def get_trajectory(self):
        return self.traj

    def get_heading(self):
        R = self.cur_R
        forward_x = R[0, 2]
        forward_z = R[2, 2]
        yaw = np.arctan2(forward_x, forward_z)
        return np.degrees(yaw)

# ==========================================
# PART 2: Video Loading & Helper Functions (from monoslam_video.py)
# ==========================================

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

# ==========================================
# PART 3: Main Execution (Colab Friendly)
# ==========================================

def run_video(video_path, output_path, seg_model=None, road_detector=None, traj_img_size=800, downscale=0.5, fov=None, start_time=None, end_time=None, stop_threshold=1.0, view_mode="global", max_turn_degrees=10.0):
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
    
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    
    # --- VIDEO WRITER SETUP ---
    # We need to know the final visualization size.
    # The final visualization is hstack(display_img, traj_img_resized).
    # display_img size is (h, w).
    # traj_img_resized size: height = h, width = traj_img.width * (h / traj_img.height) = traj_img_size * (h / traj_img_size) = h? 
    # Actually: scale_traj = display_img.shape[0] / traj_img.shape[0] (which is h / traj_img_size)
    # traj_img_resized width = int(traj_img.shape[1] * scale_traj)
    
    scale_traj = h / traj_img_size
    traj_w_resized = int(traj_img_size * scale_traj)
    final_w = w + traj_w_resized
    final_h = h
    
    print(f"Output Video Resolution: {final_w}x{final_h} @ {loader.fps} FPS")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, loader.fps, (final_w, final_h))
    
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

        vo.process_frame(i, img_gray, img_color=img, road_vp=road_vp)
        
        estimated_heading = vo.get_heading()
        turn_rate = estimated_heading - last_heading
        last_heading = estimated_heading
        
        cv2.putText(display_img, f"Est Heading: {estimated_heading:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # --- Turn Indicator (Visual Validation) ---
        bar_center_x = w // 2
        bar_y = h - 20
        bar_width = 200
        offset = int(turn_rate * 20) 
        cv2.rectangle(display_img, (bar_center_x - bar_width//2, bar_y - 5), (bar_center_x + bar_width//2, bar_y + 5), (50, 50, 50), -1)
        cv2.line(display_img, (bar_center_x, bar_y - 5), (bar_center_x, bar_y + 5), (255, 255, 255), 1)
        if abs(offset) > 0:
            color = (0, 255, 0)
            if abs(turn_rate) > 2.0: color = (0, 0, 255) 
            offset = np.clip(offset, -bar_width//2, bar_width//2)
            cv2.rectangle(display_img, (bar_center_x, bar_y - 3), (bar_center_x + offset, bar_y + 3), color, -1)
            
        cv2.putText(display_img, f"Turn: {turn_rate:.2f}", (bar_center_x - 30, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if vo.is_stopped:
            cv2.putText(display_img, "STOPPED", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Acceleration Indicator ---
        curr_flow_mag = vo.current_flow_mag
        accel = curr_flow_mag - last_flow_mag
        last_flow_mag = curr_flow_mag
        
        vb_center_x = w - 30
        vb_center_y = h // 2
        vb_height = 200 
        vb_width = 15
        
        cv2.rectangle(display_img, (vb_center_x - vb_width//2, vb_center_y - vb_height//2), 
                                   (vb_center_x + vb_width//2, vb_center_y + vb_height//2), (50, 50, 50), -1)
        cv2.line(display_img, (vb_center_x - vb_width//2, vb_center_y), (vb_center_x + vb_width//2, vb_center_y), (255, 255, 255), 1)
        
        accel_scale = 100.0 
        val_px = int(accel * accel_scale)
        val_px = np.clip(val_px, -vb_height//2, vb_height//2)
        
        if val_px != 0:
            color_acc = (0, 255, 0) if val_px > 0 else (0, 0, 255) 
            top_y = vb_center_y - val_px
            cv2.rectangle(display_img, (vb_center_x - vb_width//2, vb_center_y),
                                       (vb_center_x + vb_width//2, top_y), color_acc, -1)
            
        cv2.putText(display_img, "Accel", (vb_center_x - 20, vb_center_y - vb_height//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if vo.kp1 is not None:
            for pt in vo.kp1:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        
        # --- Trajectory Drawing ---
        traj_points = np.array(vo.get_trajectory())
        
        traj_img.fill(0)
        draw_scale = 1.0 
        cx_traj, cy_traj = traj_img_size // 2, traj_img_size // 2
        
        curr_x = vo.cur_t[0][0]
        curr_z = vo.cur_t[2][0]
        curr_heading_rad = np.radians(estimated_heading)
        
        if len(traj_points) > 1:
            if view_mode == "ego":
                # Egocentric
                rel_points = traj_points - np.array([curr_x, curr_z])
                c, s = np.cos(-curr_heading_rad), np.sin(-curr_heading_rad)
                R = np.array(((c, -s), (s, c)))
                rot_points = rel_points @ R.T
                
                screen_points = np.zeros_like(rot_points, dtype=np.int32)
                screen_points[:, 0] = (cx_traj + rot_points[:, 0] * draw_scale).astype(np.int32)
                screen_points[:, 1] = (cy_traj - rot_points[:, 1] * draw_scale).astype(np.int32)
                
                cv2.polylines(traj_img, [screen_points], isClosed=False, color=(255, 0, 0), thickness=1)
                
                arrow_len = 30
                tip_u = cx_traj
                tip_v = cy_traj - arrow_len
                cv2.arrowedLine(traj_img, (cx_traj, cy_traj), (tip_u, tip_v), (0, 255, 255), 3, tipLength=0.3)
                
            else: 
                # Global
                rel_points = traj_points - np.array([curr_x, curr_z])
                screen_points = np.zeros_like(rel_points, dtype=np.int32)
                screen_points[:, 0] = (cx_traj + rel_points[:, 0] * draw_scale).astype(np.int32)
                screen_points[:, 1] = (cy_traj - rel_points[:, 1] * draw_scale).astype(np.int32)
                
                cv2.polylines(traj_img, [screen_points], isClosed=False, color=(255, 0, 0), thickness=1)
                
                arrow_len = 30
                tip_u = int(cx_traj + arrow_len * np.sin(curr_heading_rad))
                tip_v = int(cy_traj - arrow_len * np.cos(curr_heading_rad))
                cv2.arrowedLine(traj_img, (cx_traj, cy_traj), (tip_u, tip_v), (0, 255, 255), 3, tipLength=0.3)
                
        traj_img_resized = cv2.resize(traj_img, (traj_w_resized, display_img.shape[0]))
        final_vis = np.hstack((display_img, traj_img_resized))
        
        # WRITE TO VIDEO
        out.write(final_vis)
            
    print(f"Finished {sequence_name}")
    print(f"Saved video to: {output_path}")
    loader.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run MonoSLAM on a video file (Colab Compatible).")
    parser.add_argument("video_path", type=str, help="Path to the input mp4 video file.")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to the output video file (default: output.mp4).")
    parser.add_argument("--no_road_detection", action="store_true", help="Disable road area detection.")
    parser.add_argument("--no_segmentation", action="store_true", help="Disable semantic segmentation (faster).")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor for processing (default 0.5).")
    parser.add_argument("--fov", type=float, help="Horizontal Field of View (FOV) in degrees (e.g., 90, 120).")
    parser.add_argument("--start", type=str, help="Start timestamp in hhmmss format (e.g., 000130 for 00:01:30)")
    parser.add_argument("--end", type=str, help="End timestamp in hhmmss format.")
    parser.add_argument("--stop_threshold", type=float, default=1.0, help="Movement threshold in pixels (default 1.0). Increase if false stopped detection.")
    parser.add_argument("--view_mode", type=str, default="global", choices=["global", "ego"], help="Map view mode: 'global' (North Up, default) or 'ego' (Vehicle Up).")
    parser.add_argument("--max_turn", type=float, default=10.0, help="Maximum allowed turn rate (deg/frame) to reject outliers.")

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
    
    run_video(args.video_path, args.output, seg_model, road_detector, downscale=args.scale, fov=args.fov, start_time=args.start, end_time=args.end, stop_threshold=args.stop_threshold, view_mode=args.view_mode, max_turn_degrees=args.max_turn)

if __name__ == "__main__":
    main()
