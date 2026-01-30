import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import torch
from torchvision import models, transforms

class MultiClassSegmentation:
    def __init__(self):
        self.device = torch.device('cpu') 
        if torch.backends.mps.is_available(): 
             self.device = torch.device('mps')
        
        print(f"Loading YOLOP (hustvl/yolop) on {self.device}...")
        try:
            # Load YOLOP from torch hub
            self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading YOLOP: {e}")
            raise e
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def get_mask(self, img_bgr):
        # YOLOP expects RGB, 640x640 (or multiple of 32). Resize logic might be needed.
        # It's robust to size but 640 width is standard.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Resize to multiple of 32 (YOLOP requirement)
        h, w = img_rgb.shape[:2]
        new_h = (h + 31) // 32 * 32
        new_w = (w + 31) // 32 * 32
        
        if new_h != h or new_w != w:
            img_input = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_input = img_rgb
        
        img_tensor = self.transform(img_input).to(self.device)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        with torch.no_grad():
            # YOLOP returns: (det_out, da_seg_out, ll_seg_out)
            # da_seg_out: Drivable Area
            # ll_seg_out: Lane Line
            _, da_seg_out, _ = self.model(img_tensor)
            
        # da_seg_out shape: [1, 2, H, W] -> argmax(1) -> [1, H, W]
        # Or sometimes sigmoid?
        # Typically YOLOP segmentation output is raw logits.
        
        # Resize output back to original image size
        da_seg_out = torch.nn.functional.interpolate(da_seg_out, size=(h, w), mode='bilinear', align_corners=True)
        
        _, da_seg_mask = torch.max(da_seg_out, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        
        # da_seg_mask: 1 = Road (Drivable), 0 = Background
        
        return da_seg_mask


class DataLoader:
    def __init__(self, data_path, subfolder):
        self.image_path = os.path.join(data_path, subfolder, "data")
        self.images = sorted(os.listdir(self.image_path))
        if not self.images:
            print(f"Warning: No images found in {self.image_path}")
            self.images = []
        
    def __len__(self):
        return len(self.images)
    
    def get_image(self, index):
        if index >= len(self.images): return None
        img_name = self.images[index]
        img = cv2.imread(os.path.join(self.image_path, img_name), cv2.IMREAD_COLOR)
        return img 

class OxtsLoader:
    def __init__(self, data_path):
        self.oxts_path = os.path.join(data_path, "oxts", "data")
        self.files = sorted(os.listdir(self.oxts_path))
        if not self.files:
            print(f"Warning: No oxts data found in {self.oxts_path}")
    
    def get_yaw(self, index):
        if index >= len(self.files): return 0
        file_path = os.path.join(self.oxts_path, self.files[index])
        with open(file_path, 'r') as f:
            content = f.read().strip()
            values = content.split()
            yaw_rad = float(values[5])
            return yaw_rad

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
        self.traj = [] 
        # Using Good Features To Track (Shi-Tomasi) instead of FAST
        # We don't initialize a single detector object because GFTT is a function call
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.seg_model = None
        self.latest_seg_mask = None # Store latest mask for visualization
        self.curr_road_vp = None # Store latest VP for visualization
        self.is_stopped = False
        self.current_flow_mag = 0.0

    def detect_features_grid(self, img, grid_size=(4, 4), max_feats_total=1000):
        """
        Detect features in a grid to ensure distribution across the image.
        Useful for low-texture roads where features might cluster on the horizon.
        """
        h, w = img.shape[:2]
        ny, nx = grid_size
        dy, dx = h // ny, w // nx
        
        feats_per_cell = max_feats_total // (nx * ny)
        
        all_keypoints = []
        
        for y in range(0, h, dy):
            for x in range(0, w, dx):
                # Define cell ROI
                y_end = min(y + dy, h)
                x_end = min(x + dx, w)
                
                # Sanity check
                if x >= w or y >= h: continue
                
                roi = img[y:y_end, x:x_end]
                
                # Detect in ROI
                # Using Good Features To Track (Shi-Tomasi)
                # It's more sensitive to corners than FAST
                p0 = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
                
                if p0 is not None:
                    # Offset points to global coordinates
                    p0 = p0.reshape(-1, 2)
                    p0[:, 0] += x
                    p0[:, 1] += y
                    
                    # If we found too many, take strongest? GFTT already returns strongest.
                    # Just cap the number
                    if len(p0) > feats_per_cell:
                        p0 = p0[:feats_per_cell]
                        
                    all_keypoints.extend(p0)
                    
        return np.array(all_keypoints, dtype=np.float32)

    def set_segmentation_model(self, model):
        self.seg_model = model

    def process_frame(self, frame_id, new_frame, img_color=None, static_mask=None):
        road_vp = None # Initialize to avoid UnboundLocalError
        
        if self.last_frame is None:
            self.last_frame = new_frame
        if self.last_frame is None:
            self.last_frame = new_frame
            self.kp1 = self.detect_features_grid(new_frame)

            
            # Filter keypoints by static mask on first frame
            if static_mask is not None and len(self.kp1) > 0:
                valid_kp = []
                for pt in self.kp1:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < static_mask.shape[0] and 0 <= x < static_mask.shape[1]:
                        if static_mask[y, x] == 1:  # 1 = valid, 0 = masked
                            valid_kp.append(pt)
                self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)

            # Filter keypoints by SEGMENTATION mask on first frame
            if self.seg_model is not None and img_color is not None and len(self.kp1) > 0:
                mask = self.seg_model.get_mask(img_color)
                
                # Apply static mask logic if needed (consistency)
                if static_mask is not None:
                    if mask.shape != static_mask.shape:
                        static_mask_for_op = cv2.resize(static_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        static_mask_for_op = static_mask
                    mask = cv2.bitwise_and(mask.astype(np.uint8), static_mask_for_op.astype(np.uint8))
                
                valid_kp = []
                for pt in self.kp1:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                         if mask[y, x] == 1: # Road only
                             valid_kp.append(pt)
                self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)
                
                # Log init
                print(f"Features initialized on road: {len(self.kp1)}")
            
            self.current_flow_mag = 0.0
            return

        # Ensure we have features to track
        if self.kp1 is None or len(self.kp1) == 0:
             self.last_frame = new_frame
             self.kp1 = self.detect_features_grid(new_frame)
             
             # Also filter new keypoints by static mask
             if static_mask is not None and len(self.kp1) > 0:
                 valid_kp = []
                 for pt in self.kp1:
                     x, y = int(pt[0]), int(pt[1])
                     if 0 <= y < static_mask.shape[0] and 0 <= x < static_mask.shape[1]:
                         if static_mask[y, x] == 1:
                             valid_kp.append(pt)
                 self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)
            
             # Filter keypoints by SEGMENTATION mask (Top Block)
             if self.seg_model is not None and img_color is not None and len(self.kp1) > 0:
                mask = self.seg_model.get_mask(img_color)
                # Apply static mask logic if needed
                if static_mask is not None:
                    if mask.shape != static_mask.shape:
                        static_mask_for_op = cv2.resize(static_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        static_mask_for_op = static_mask
                    mask = cv2.bitwise_and(mask.astype(np.uint8), static_mask_for_op.astype(np.uint8))
                
                valid_kp = []
                for pt in self.kp1:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                         if mask[y, x] == 1: 
                             valid_kp.append(pt)
                self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)
                print(f"Features replenished (top) on road: {len(self.kp1)}")

             self.current_flow_mag = 0.0
             return

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.last_frame, new_frame, self.kp1, None, **self.lk_params)
        
        st_flat = st.flatten()
        good_old = self.kp1[st_flat == 1]
        good_new = p1[st_flat == 1]
        
        # Apply static mask (e.g., car body exclusion)
        if static_mask is not None:
            valid_indices = []
            for idx, pt in enumerate(good_new):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < static_mask.shape[0] and 0 <= x < static_mask.shape[1]:
                    if static_mask[y, x] == 1:  # 1 = valid, 0 = masked
                        valid_indices.append(idx)
            good_old = good_old[valid_indices]
            good_new = good_new[valid_indices]
        
        # Apply semantic segmentation mask (dynamic objects / road segmentation)
        # With YOLOP: 1 = Drivable Area, 0 = Background
        # We want to TRACK on Drivable Area (1).
        if self.seg_model is not None and img_color is not None:
            mask = self.seg_model.get_mask(img_color)
            
            # Apply static mask to YOLOP mask to remove car body from "Road"
            if static_mask is not None:
                # static_mask is 1 for valid, 0 for masked
                # YOLOP mask is 1 for road, 0 for bg
                # We want 1 only if BOTH are 1
                
                # Ensure shapes match exactly (resize static_mask to mask if needed)
                if mask.shape != static_mask.shape:
                    # Note: cv2.resize expects (width, height), shape is (height, width)
                    static_mask_for_op = cv2.resize(static_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    static_mask_for_op = static_mask
                    
                mask = cv2.bitwise_and(mask.astype(np.uint8), static_mask_for_op.astype(np.uint8))
            
            self.latest_seg_mask = mask # Store for visualization
            
            valid_indices = []
            for idx, pt in enumerate(good_new):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    # YOLOP: Keep if 1 (Road)
                    if mask[y, x] == 1: 
                        valid_indices.append(idx)
            
            # DEBUG: Log feature survival rate
            print(f"Features on road: {len(valid_indices)} / {len(good_new)}")
                        
            good_old = good_old[valid_indices]
            good_new = good_new[valid_indices]

            # --- CALCULATE VP FROM YOLOP MASK ---
            # Use the centroid of the top portion of the road mask
            # This turns the segmentation into a heading/turn signal
            h_mask, w_mask = mask.shape
            # Analyze top 60% of the mask to find "far" road
            top_h_limit = int(h_mask * 0.6)
            mask_top = mask[:top_h_limit, :]
            
            # Convert to uint8 for moments
            mask_uint8 = (mask_top == 1).astype(np.uint8) * 255
            M = cv2.moments(mask_uint8)
            
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY_rel = int(M["m01"] / M["m00"])
                # cY is relative to the top of the image
                yolop_vp = (cX, cY_rel)
                
                # OVERRIDE the passed road_vp
                road_vp = yolop_vp
                self.curr_road_vp = yolop_vp
            else:
                self.curr_road_vp = None
        else:
            self.curr_road_vp = None
            
        if len(good_new) < 50:
             self.kp1 = self.detect_features_grid(new_frame)

             # Also filter new keypoints by static mask
             if static_mask is not None and len(self.kp1) > 0:
                 valid_kp = []
                 for pt in self.kp1:
                     x, y = int(pt[0]), int(pt[1])
                     if 0 <= y < static_mask.shape[0] and 0 <= x < static_mask.shape[1]:
                         if static_mask[y, x] == 1:
                             valid_kp.append(pt)
                 self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)

             # Filter keypoints by SEGMENTATION mask (Bottom Block)
             # Reuse latest_seg_mask if available
             if self.seg_model is not None and len(self.kp1) > 0:
                 # If we are here, we likely processed the frame and have a mask
                 mask = self.latest_seg_mask
                 if mask is None and img_color is not None:
                     mask = self.seg_model.get_mask(img_color)
                     # (Skip static mask merge redundancy if we just computed it raw, but keeping it simple)
                 
                 if mask is not None:
                    valid_kp = []
                    for pt in self.kp1:
                        x, y = int(pt[0]), int(pt[1])
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                             if mask[y, x] == 1: 
                                 valid_kp.append(pt)
                    self.kp1 = np.array(valid_kp, dtype=np.float32) if valid_kp else np.array([], dtype=np.float32).reshape(-1, 2)
                    print(f"Features replenished (bottom) on road: {len(self.kp1)}")

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

def run_sequence(data_dir, sequence_name, seg_model=None, traj_img_size=800, downscale=0.5):
    print(f"--- Running Sequence: {sequence_name} (Scale: {downscale}) ---")
    loader = DataLoader(data_dir, sequence_name)
    gt_loader = OxtsLoader(data_dir)
    
    if len(loader) == 0:
        return

    first_img = loader.get_image(0)
    if first_img is None: return
    
    h_orig, w_orig = first_img.shape[:2]
    fx_orig = 718.856
    fy_orig = 718.856
    cx_orig = w_orig / 2.0
    cy_orig = h_orig / 2.0
    
    fx = fx_orig * downscale
    fy = fy_orig * downscale
    cx = cx_orig * downscale
    cy = cy_orig * downscale
    w = int(w_orig * downscale)
    h = int(h_orig * downscale)
    
    cam = PinholeCamera(w, h, fx, fy, cx, cy)
    vo = VisualOdometry(cam)
    if seg_model:
        vo.set_segmentation_model(seg_model)
    
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    
    cv2.namedWindow(f'MonoSLAM - {sequence_name}', cv2.WINDOW_NORMAL)
    
    initial_gt_yaw = None
    
    for i in tqdm(range(len(loader)), desc=sequence_name):
        img_full = loader.get_image(i)
        if img_full is None: break
        
        img = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
        
        current_gt_yaw_rad = gt_loader.get_yaw(i)
        if initial_gt_yaw is None:
            initial_gt_yaw = current_gt_yaw_rad
            
        diff_gt = current_gt_yaw_rad - initial_gt_yaw
        rel_gt_yaw_deg = -np.degrees(diff_gt)
        
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                 display_img = img.copy()

        vo.process_frame(i, img_gray, img_color=img)
        
        estimated_heading = vo.get_heading()
        
        cv2.putText(display_img, f"Est Heading: {estimated_heading:.2f} deg", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_img, f"GT Heading:  {rel_gt_yaw_deg:.2f} deg", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_img, f"Seq: {sequence_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # STOPPED indicator
        if vo.is_stopped:
            cv2.putText(display_img, "STOPPED", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        if vo.kp1 is not None:
            for pt in vo.kp1:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        
        trajectory_points = vo.get_trajectory()
        draw_scale = 1.0
        draw_x_offset = traj_img_size // 2
        draw_y_offset = traj_img_size // 2 
        
        traj_img.fill(0)
        curr_u, curr_v = 0, 0
        for j, (tx, tz) in enumerate(trajectory_points):
            u = int(tx * draw_scale) + draw_x_offset
            v = int(-tz * draw_scale) + draw_y_offset
            
            if 0 <= u < traj_img_size and 0 <= v < traj_img_size:
                 cv2.circle(traj_img, (u, v), 1, (255, 0, 0), 1)
            
            if j == len(trajectory_points) - 1:
                curr_u, curr_v = u, v
        
        arrow_len = 50
        
        est_rad = np.radians(estimated_heading) 
        tip_u = int(curr_u + arrow_len * np.sin(est_rad))
        tip_v = int(curr_v - arrow_len * np.cos(est_rad))
        if 0 <= curr_u < traj_img_size and 0 <= curr_v < traj_img_size:
            cv2.arrowedLine(traj_img, (curr_u, curr_v), (tip_u, tip_v), (0, 255, 255), 3, tipLength=0.3)

        gt_rad = np.radians(rel_gt_yaw_deg)
        gt_tip_u = int(curr_u + arrow_len * np.sin(gt_rad))
        gt_tip_v = int(curr_v - arrow_len * np.cos(gt_rad))
        
        if 0 <= curr_u < traj_img_size and 0 <= curr_v < traj_img_size:
            cv2.arrowedLine(traj_img, (curr_u, curr_v), (gt_tip_u, gt_tip_v), (0, 255, 0), 2, tipLength=0.3)

        scale_traj = display_img.shape[0] / traj_img.shape[0]
        traj_img_resized = cv2.resize(traj_img, (int(traj_img.shape[1] * scale_traj), display_img.shape[0]))
        final_vis = np.hstack((display_img, traj_img_resized))
        
        cv2.imshow(f'MonoSLAM - {sequence_name}', final_vis)
        
        key = cv2.waitKey(1)
        if key == 27: 
            cv2.destroyAllWindows()
            sys.exit(0)
            
        if i == len(loader) - 1:
            cv2.imwrite(f"monoslam_result_{sequence_name}_Integrated.png", final_vis)
            
    print(f"Finished {sequence_name}")

def main():
    data_dir = "data/2011_09_26_drive_0051_sync"
    sequences = ["image_03"] 
    
    print("Initializing Models...")
    try:
        seg_model = MultiClassSegmentation()
    except Exception as e:
        print(f"Failed to load DL model: {e}")
        seg_model = None
        
    
    downscale_factor = 0.5 
    
    for seq in sequences:
        run_sequence(data_dir, seq, seg_model, downscale=downscale_factor)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
