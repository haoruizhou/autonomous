import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import torch
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

class DepthAnythingV2:
    """Wrapper for Depth Anything V2 monocular depth estimation."""
    
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        # Note: MPS doesn't support all ops needed by DinoV2 backbone (upsample_bicubic2d)
        # So we use CUDA if available, otherwise fall back to CPU
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        # MPS not supported for this model due to missing ops
        
        print(f"Loading Depth Anything V2 on {self.device}...")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Depth Anything V2 loaded.")
        
    def estimate_depth(self, img_bgr):
        """
        Estimate depth from a BGR image.
        
        Args:
            img_bgr: OpenCV BGR image (H, W, 3)
            
        Returns:
            depth_map: numpy array (H, W) with relative depth values (higher = closer)
        """
        h, w = img_bgr.shape[:2]
        
        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Preprocess
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()
        
        return depth_map

class MultiClassSegmentation:
    def __init__(self):
        self.device = torch.device('cpu') 
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
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.lk_params = dict(winSize=(15, 15), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.seg_model = None
        self.depth_model = None
        self.is_stopped = False
        self.current_flow_mag = 0.0

    def set_segmentation_model(self, model):
        self.seg_model = model

    def set_depth_model(self, model):
        self.depth_model = model

    def _backproject_points(self, points_2d, depth_map):
        """
        Backproject 2D points to 3D using depth map and camera intrinsics.
        
        Args:
            points_2d: Nx2 array of 2D points
            depth_map: HxW depth map
            
        Returns:
            points_3d: Nx3 array of 3D points in camera frame
        """
        points_3d = []
        valid_indices = []
        
        for idx, pt in enumerate(points_2d):
            x, y = int(pt[0]), int(pt[1])
            
            # Check bounds
            if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
                continue
                
            d = depth_map[y, x]
            
            # Skip invalid depth (too close or too far)
            if d <= 0.1 or d > 1000:
                continue
            
            # Backproject: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
            # Depth Anything gives relative depth, so we scale to reasonable range
            # Invert: higher value = closer in Depth Anything, but we want further = higher Z
            Z = 100.0 / (d + 1e-6)  # Convert relative to approximate metric
            Z = np.clip(Z, 1.0, 200.0)  # Reasonable driving distances
            
            X = (pt[0] - self.cam.cx) * Z / self.cam.fx
            Y = (pt[1] - self.cam.cy) * Z / self.cam.fy
            
            points_3d.append([X, Y, Z])
            valid_indices.append(idx)
        
        return np.array(points_3d, dtype=np.float64), valid_indices

    def process_frame(self, frame_id, new_frame, img_color=None, road_vp=None, depth_map=None):
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

        # --- POSE ESTIMATION ---
        R = np.eye(3)
        t = np.zeros((3, 1))
        used_pnp = False
        
        # Try PnP with depth first (gives proper scale)
        if depth_map is not None and len(good_old) >= 6:
            pts_3d, valid_idx = self._backproject_points(good_old, depth_map)
            
            if len(pts_3d) >= 6:
                # Filter 2D points to match valid 3D points
                pts_2d = good_new[valid_idx].astype(np.float64)
                
                # Solve PnP
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d, pts_2d, self.cam.K, None,
                    iterationsCount=100,
                    reprojectionError=8.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success and inliers is not None and len(inliers) >= 4:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec
                    used_pnp = True
        
        # Fallback to Essential Matrix (no scale)
        if not used_pnp:
            E, mask = cv2.findEssentialMat(good_new, good_old, self.cam.fx, self.cam.pp, cv2.RANSAC, 0.999, 1.0)
            
            if E is not None:
                _, R, t, mask = cv2.recoverPose(E, good_new, good_old, self.cam.K)
        
        # Calculate flow magnitude for stop detection
        flow_mag = np.mean(np.linalg.norm(good_new - good_old, axis=1))
        self.current_flow_mag = flow_mag
        
        # Calculate rotation angle strength
        trace = np.trace(R)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_theta))
        
        # OUTLIER REJECTION: Sudden "Teleport" Turns
        if abs(angle_deg) > self.max_turn_degrees:
            t = np.zeros((3, 1))
            R = np.eye(3)
        else:
            # STOP DETECTION
            if flow_mag < self.stop_threshold:
                t = np.zeros((3, 1))
                R = np.eye(3)
                self.is_stopped = True
            else:
                self.is_stopped = False
            
            # --- ROAD FUSION (only if not using PnP) ---
            if not used_pnp and road_vp is not None:
                vp_homog = np.array([road_vp[0], road_vp[1], 1.0])
                t_vp = self.cam.K_inv.dot(vp_homog)
                t_vp = t_vp / np.linalg.norm(t_vp)
                t_vp = t_vp.reshape((3, 1))
                
                alpha = 0.2
                if np.dot(t.flatten(), t_vp.flatten()) > 0.7:
                    t_fused = (1 - alpha) * t + alpha * t_vp
                    t_fused = t_fused / np.linalg.norm(t_fused)
                    t = t_fused
        
        # Apply pose update
        if used_pnp:
            # PnP gives camera pose, so t is already scaled (in world units)
            # t from PnP is the camera translation, we update directly
            t_norm = np.linalg.norm(t)
            if t_norm > 0.01 and t_norm < 50:  # Sanity check on translation magnitude
                self.cur_t = self.cur_t + self.cur_R.dot(t)
                self.cur_R = self.cur_R.dot(R)
        else:
            # Essential matrix gives unit translation
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

def run_sequence(data_dir, sequence_name, seg_model=None, road_detector=None, traj_img_size=800, downscale=0.5):
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
            
        road_vp = None
        if road_detector:
            display_img = road_detector.detect_and_draw(img) # Draw on scaled image
            road_vp = road_detector.get_vanishing_point()
        else:
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                 display_img = img.copy()

        vo.process_frame(i, img_gray, img_color=img, road_vp=road_vp)
        
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
        
    road_detector = RoadAreaDetector()
    
    downscale_factor = 0.5 
    
    for seq in sequences:
        run_sequence(data_dir, seq, seg_model, road_detector, downscale=downscale_factor)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
