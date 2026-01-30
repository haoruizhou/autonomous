"""
Interactive Mask Editor GUI

Draw a polygon to mask out areas (like the car's hood/body) from optical flow tracking.

Usage:
    python mask_editor.py <video_path>
    python mask_editor.py <video_path> --output mask.png
    python mask_editor.py <video_path> --load existing_mask.png  # Edit existing mask
    
Controls:
    - Left Click: Add polygon vertex
    - Right Click: Close polygon and add it to mask
    - 'c': Clear current incomplete polygon
    - 'r': Reset entire mask
    - 'i': Invert mask (toggle what is masked)
    - 's': Save and exit
    - 'q' / ESC: Quit without saving
"""

import cv2
import numpy as np
import argparse
import os


class MaskEditor:
    def __init__(self, frame, existing_mask=None):
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        
        # Mask: 255 = valid (process this area), 0 = masked (ignore this area)
        if existing_mask is not None:
            self.mask = existing_mask.copy()
        else:
            self.mask = np.ones((self.h, self.w), dtype=np.uint8) * 255
        
        self.current_polygon = []  # Points of currently being drawn polygon
        self.polygons = []  # List of completed polygons (for undo capability)
        self.inverted = False  # If true, polygon areas are KEPT instead of masked
        
        self.window_name = "Mask Editor - Draw polygon to mask car body"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current polygon
            self.current_polygon.append((x, y))
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Close current polygon and apply to mask
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                self._apply_polygon_to_mask(self.current_polygon)
                self.current_polygon = []
                self.update_display()
                
    def _apply_polygon_to_mask(self, polygon):
        """Apply a polygon to the mask"""
        pts = np.array(polygon, dtype=np.int32)
        if self.inverted:
            # Inverted mode: polygon areas are KEPT (255), rest is 0
            # First time in inverted, start with all masked
            if len(self.polygons) == 1:  # First polygon in inverted mode
                self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillPoly(self.mask, [pts], 255)
        else:
            # Normal mode: polygon areas are MASKED (0)
            cv2.fillPoly(self.mask, [pts], 0)
            
    def update_display(self):
        """Update the display with current mask and polygon being drawn"""
        self.frame = self.original_frame.copy()
        
        # Overlay the mask (red tint on masked areas)
        overlay = self.frame.copy()
        masked_area = self.mask == 0
        overlay[masked_area] = overlay[masked_area] * 0.3 + np.array([0, 0, 200]) * 0.7
        
        self.frame = overlay.astype(np.uint8)
        
        # Draw current incomplete polygon
        if len(self.current_polygon) > 0:
            pts = np.array(self.current_polygon, dtype=np.int32)
            cv2.polylines(self.frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
            for pt in self.current_polygon:
                cv2.circle(self.frame, pt, 5, (0, 255, 0), -1)
                
        # Draw instructions
        instructions = [
            "Left Click: Add vertex",
            "Right Click: Close polygon",
            "'c': Clear current",
            "'r': Reset all",
            "'i': Invert mode",
            "'s': Save & Exit",
            "'q'/ESC: Quit"
        ]
        
        y_offset = 25
        for i, text in enumerate(instructions):
            cv2.putText(self.frame, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        mode_text = "MODE: Keep polygons" if self.inverted else "MODE: Mask polygons"
        cv2.putText(self.frame, mode_text, (10, self.h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    def run(self):
        """Run the interactive editor"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            cv2.imshow(self.window_name, self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Clear current polygon
                self.current_polygon = []
                self.update_display()
                
            elif key == ord('r'):
                # Reset entire mask
                self.mask = np.ones((self.h, self.w), dtype=np.uint8) * 255
                self.polygons = []
                self.current_polygon = []
                self.inverted = False
                self.update_display()
                
            elif key == ord('i'):
                # Toggle inverted mode
                self.inverted = not self.inverted
                print(f"Inverted mode: {self.inverted}")
                self.update_display()
                
            elif key == ord('s'):
                # Save and exit
                cv2.destroyAllWindows()
                return self.mask
                
            elif key == ord('q') or key == 27:
                # Quit without saving
                cv2.destroyAllWindows()
                return None
                
        return None


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


def load_frame_at_time(video_path, time_str=None):
    """Load a frame from a video file at the specified timestamp"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    if time_str:
        try:
            seconds = parse_hhmmss(time_str)
            frame_idx = int(seconds * fps)
            if frame_idx >= total_frames:
                print(f"Warning: Timestamp {time_str} is beyond video length, using last frame")
                frame_idx = total_frames - 1
            print(f"Seeking to {time_str} -> Frame {frame_idx}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        except ValueError as e:
            print(f"Error parsing time: {e}, using first frame")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise IOError(f"Cannot read frame from: {video_path}")
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Interactive mask editor for optical flow")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", default=None, help="Output mask file path (default: <video_name>_mask.png)")
    parser.add_argument("--load", "-l", default=None, help="Load existing mask to edit")
    parser.add_argument("--time", "-t", default=None, help="Timestamp in hhmmss format to load specific frame (e.g., 000920)")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_path = f"{video_name}_mask.png"
    
    print(f"Loading video: {args.video_path}")
    frame = load_frame_at_time(args.video_path, args.time)
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Load existing mask if specified
    existing_mask = None
    if args.load and os.path.exists(args.load):
        print(f"Loading existing mask: {args.load}")
        existing_mask = cv2.imread(args.load, cv2.IMREAD_GRAYSCALE)
        # Resize if needed
        if existing_mask.shape[:2] != frame.shape[:2]:
            existing_mask = cv2.resize(existing_mask, (frame.shape[1], frame.shape[0]))
    
    # Run editor
    editor = MaskEditor(frame, existing_mask)
    mask = editor.run()
    
    if mask is not None:
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to: {output_path}")
        print(f"Use with: python monoslam_video.py <video> --mask {output_path}")
    else:
        print("Editor cancelled, mask not saved.")


if __name__ == "__main__":
    main()
